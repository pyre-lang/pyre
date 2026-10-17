/// llsupport/regalloc.py + x86/regalloc.py: Register allocation.
///
/// Classes:
///   Lifetime           — regalloc.py
///   FixedRegisterPositions — regalloc.py
///   LifetimeManager    — regalloc.py
///   FrameManager       — regalloc.py
///   RegisterManager    — regalloc.py
///   RegAlloc           — x86/regalloc.py
///
/// Functions:
///   compute_vars_longevity — regalloc.py
///   valid_addressing_size  — regalloc.py
///   get_scale              — regalloc.py
use indexmap::IndexMap;
use majit_backend::BackendError;
use majit_ir::IndexMapExt;
use rustc_hash::FxBuildHasher;

use crate::arch::*;
use crate::gcmap::{allocate_gcmap, gcmap_set_bit};
use crate::j2plan::{GuardKind, IntBinKind, IntUnaryKind, LirOp, LoadKind, StoreKind};
use crate::regloc::*;
use majit_ir::{InputArg, InputArgRc, Op, OpCode, OpRc, OpRef, OpTypeIndex, Type, descr_identity};
use smallvec::SmallVec;

// ── Constants ──────────────────────────────────────────────────────

/// regalloc.py:909
pub const UNDEF_POS: i32 = -42;

/// regalloc.py:26-28
pub const SAVE_DEFAULT_REGS: u8 = 0;
pub const SAVE_GCREF_REGS: u8 = 1;
pub const SAVE_ALL_REGS: u8 = 2;

/// `save_all_regs` for the CALL_MALLOC_NURSERY_VARSIZE spill. x86
/// `consider_call_malloc_nursery_varsize` uses the default: the
/// `malloc_slowpath_varsize` helper saves every register but ecx/edx into
/// the jitframe. The aarch64 emitter still expects a live length in its
/// frame slot, so it keeps spilling every binding.
#[cfg(target_arch = "x86_64")]
const VARSIZE_MALLOC_SAVE_REGS: u8 = SAVE_DEFAULT_REGS;
#[cfg(not(target_arch = "x86_64"))]
const VARSIZE_MALLOC_SAVE_REGS: u8 = SAVE_ALL_REGS;

/// aarch64/regalloc.py DEFAULT_IMM_SIZE
const DEFAULT_IMM_SIZE: i64 = 4096;

/// aarch64/regalloc.py check_imm_arg
fn check_imm_arg(arg: i64) -> bool {
    (0..DEFAULT_IMM_SIZE).contains(&arg)
}

// Large-immediate scratch register:
//   aarch64 → registers.ip1 (`aarch64/registers.py`)
//   x86_64  → regloc.X86_64_SCRATCH_REG (`x86/regloc.py`)
#[cfg(target_arch = "aarch64")]
use crate::aarch64::registers::IP1 as LARGE_IMM_SCRATCH;
#[cfg(target_arch = "x86_64")]
use crate::regloc::X86_64_SCRATCH_REG as LARGE_IMM_SCRATCH;

// ── Lifetime ───────────────────────────────────────────────────────

/// regalloc.py Lifetime — liveness information for a single variable.
pub struct Lifetime {
    /// regalloc.py:916 position where the variable is defined
    pub definition_pos: i32,
    /// regalloc.py:918 position where the variable is last used (including failargs/jump)
    pub last_usage: i32,
    /// `Lifetime.real_usages` — *real* usages (as op argument, excluding
    /// jump/failargs). Four inline slots cover the common SSA use-count
    /// so the first `push` does not heap-grow 16 B per live box.
    pub real_usages: Option<SmallVec<[i32; 4]>>,
    /// regalloc.py:925 positions requiring specific registers
    pub fixed_positions: Option<Vec<(i32, RegLoc)>>,
    /// regalloc.py another Lifetime that wants to share a register
    pub share_with: Option<OpRef>,
    /// regalloc.py:934
    _definition_pos_shared: i32,
    /// regalloc.py current register index into RegisterManager.all_regs (-1 = not in reg)
    pub current_register_index: i32,
    /// regalloc.py:940 frame location where the box currently lives
    pub current_frame_loc: Option<FrameLoc>,
    /// regalloc.py:943 hinted frame location (at the end of the trace)
    pub hint_frame_pos: i32,
}

impl Lifetime {
    pub fn new(definition_pos: i32, last_usage: i32) -> Self {
        Lifetime {
            definition_pos,
            last_usage,
            real_usages: None,
            fixed_positions: None,
            share_with: None,
            _definition_pos_shared: UNDEF_POS,
            current_register_index: -1,
            current_frame_loc: None,
            hint_frame_pos: -1,
        }
    }

    /// regalloc.py:946
    pub fn last_usage_including_sharing(&self, longevity: &LifetimeManager) -> i32 {
        let mut result = self.last_usage;
        let mut share = self.share_with;
        while let Some(opref) = share {
            if let Some(lt) = longevity.get(opref) {
                result = lt.last_usage;
                share = lt.share_with;
            } else {
                break;
            }
        }
        result
    }

    /// regalloc.py:951
    pub fn is_last_real_use_before(&self, position: i32) -> bool {
        match &self.real_usages {
            None => true,
            Some(usages) => *usages.last().unwrap() <= position,
        }
    }

    /// regalloc.py:956 binary search for next real usage after position.
    pub fn next_real_usage(&self, position: i32) -> i32 {
        debug_assert!(position >= self.definition_pos);
        let l = self.real_usages.as_ref().unwrap();
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

    /// regalloc.py:972
    pub fn definition_pos_shared(&self) -> i32 {
        if self._definition_pos_shared != UNDEF_POS {
            self._definition_pos_shared
        } else {
            self.definition_pos
        }
    }

    /// regalloc.py:978
    pub fn fixed_register(&mut self, position: i32, reg: RegLoc) -> i32 {
        debug_assert!(self.definition_pos <= position && position <= self.last_usage);
        let res;
        if let Some(positions) = self.fixed_positions.as_ref() {
            debug_assert!(position > positions.last().unwrap().0);
            res = positions.last().unwrap().0;
        } else {
            self.fixed_positions = Some(Vec::new());
            res = self.definition_pos_shared();
        }
        self.fixed_positions.as_mut().unwrap().push((position, reg));
        res
    }

    /// regalloc.py:992
    pub fn find_fixed_register(&self, opindex: i32, longevity: &LifetimeManager) -> Option<RegLoc> {
        if let Some(ref positions) = self.fixed_positions {
            for &(index, reg) in positions {
                if opindex <= index {
                    return Some(reg);
                }
            }
        }
        if let Some(share_opref) = self.share_with
            && let Some(share_lt) = longevity.get(share_opref)
        {
            return share_lt.find_fixed_register(opindex, longevity);
        }
        None
    }

    /// regalloc.py:1001
    pub fn check_invariants(&self) {
        debug_assert!(self.definition_pos <= self.last_usage);
        if let Some(ref usages) = self.real_usages {
            let mut sorted = usages.clone();
            sorted.sort();
            debug_assert_eq!(usages, &sorted);
            debug_assert!(self.last_usage >= *usages.iter().max().unwrap());
            debug_assert!(self.definition_pos < *usages.iter().min().unwrap());
        }
    }
}

impl std::fmt::Debug for Lifetime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{:?}({})",
            self.definition_pos, self.real_usages, self.last_usage
        )?;
        if let Some(ref fp) = self.fixed_positions {
            for &(index, reg) in fp {
                write!(f, " @{} in {:?}", index, reg)?;
            }
        }
        if self.current_register_index >= 0 {
            write!(f, " reg_idx:{}", self.current_register_index)?;
        }
        if self.current_frame_loc.is_some() {
            write!(f, " frame:{:?}", self.current_frame_loc)?;
        }
        if self.hint_frame_pos >= 0 {
            write!(f, " hint:{}", self.hint_frame_pos)?;
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

/// regalloc.py LifetimeManager — manages Lifetime info for all variables.
pub struct LifetimeManager {
    // Insertion-ordered map (`IndexMap`) rather than the Vec-backed
    // `IndexMap`: every register-allocation location query
    // (`RegisterManager::loc`/`_sync_var_to_stack`/`spill_*`,
    // `FrameManager::get`/`bind`/`get_new_loc`) resolves a variable through
    // `lifetimes.get(v)`, and a Vec-backed `get` is O(n) — making regalloc
    // O(n^2) on very large traces (a whole-program loop spends
    // ~all its backend time in `IndexMap::get_index_of` here). regalloc.py:1054
    // keys longevity by a dict (O(1)); `IndexMap` restores that O(1) lookup
    // while keeping the insertion-ordered iteration `IndexMap` provided.
    lifetimes: indexmap::IndexMap<OpRef, Lifetime, FxBuildHasher>,
    /// regalloc.py maps register → FixedRegisterPositions
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
            lifetimes: indexmap::IndexMap::default(),
            fixed_register_use: IndexMap::with_hasher(FxBuildHasher),
        }
    }

    pub fn with_capacity(n: usize) -> Self {
        LifetimeManager {
            lifetimes: indexmap::IndexMap::with_capacity_and_hasher(n, Default::default()),
            fixed_register_use: IndexMap::with_hasher(FxBuildHasher),
        }
    }

    pub fn contains(&self, v: OpRef) -> bool {
        self.lifetimes.contains_key(&v)
    }

    pub fn get(&self, v: OpRef) -> Option<&Lifetime> {
        self.lifetimes.get(&v)
    }

    pub fn get_mut(&mut self, v: OpRef) -> Option<&mut Lifetime> {
        self.lifetimes.get_mut(&v)
    }

    pub fn set(&mut self, v: OpRef, lifetime: Lifetime) {
        self.lifetimes.insert(v, lifetime);
    }

    /// Iterate over all (OpRef, Lifetime) pairs.
    pub fn lifetimes_iter(&self) -> impl Iterator<Item = (&OpRef, &Lifetime)> {
        self.lifetimes.iter()
    }

    /// regalloc.py:1066
    pub fn fixed_register(&mut self, opindex: i32, register: RegLoc, var: Option<OpRef>) {
        let definition_pos = match var {
            None => opindex,
            Some(v) => {
                let lt = self.lifetimes.get_mut(&v).unwrap();
                lt.fixed_register(opindex, register)
            }
        };
        self.fixed_register_use
            .entry_or_insert_with(register, || FixedRegisterPositions::new(register))
            .fixed_register(opindex, definition_pos);
    }

    /// regalloc.py:1079
    pub fn try_use_same_register(&mut self, v0: OpRef, v1: OpRef) {
        let def0 = self.lifetimes[&v0].definition_pos;
        let last0 = self.lifetimes[&v0].last_usage;
        let def1 = self.lifetimes[&v1].definition_pos;
        debug_assert!(def0 < def1);
        if last0 != def1 {
            return; // not supported for now
        }
        let def0_shared = self.lifetimes[&v0].definition_pos_shared();
        self.lifetimes.get_mut(&v0).unwrap().share_with = Some(v1);
        self.lifetimes.get_mut(&v1).unwrap()._definition_pos_shared = def0_shared;
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
        v: OpRef,
        free_regs: &[RegLoc],
    ) -> Option<RegLoc> {
        let longevity_var = &self.lifetimes[&v];
        let last_usage = longevity_var.last_usage_including_sharing(self);
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
        v: OpRef,
        free_regs: &[RegLoc],
    ) -> Option<RegLoc> {
        if free_regs.is_empty() {
            return None;
        }
        let longevity_var = &self.lifetimes[&v];
        // check whether there is a fixed register and whether it's free
        let fixed_reg = longevity_var.find_fixed_register(position, self);
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
pub fn compute_vars_longevity<T: AsRef<Op>, A: AsRef<InputArg>>(
    inputargs: &[A],
    operations: &[T],
) -> LifetimeManager {
    // Pre-size for one result per op plus one entry per input argument.
    let mut longevity = LifetimeManager::with_capacity(operations.len() + inputargs.len());

    // regalloc.py:1179 iterate operations in REVERSE
    for i in (0..operations.len()).rev() {
        let op = operations[i].as_ref();
        let opnum = op.opcode;
        let opref = op.pos().get();
        let i = i as i32;

        if !longevity.contains(opref) {
            // regalloc.py:1183-1186: if op.type != 'v' and has_no_side_effect:
            // result not used, operation has no side-effect → dead code
            if opnum.result_type() != Type::Void && opnum.has_no_side_effect() {
                continue;
            }
            longevity.set(opref, Lifetime::new(i, i));
        } else {
            longevity.get_mut(opref).unwrap().definition_pos = i;
        }

        // regalloc.py:1190-1201 process arguments
        for j in 0..op.num_args() {
            let arg = op.arg(j).to_opref();
            if arg.is_constant() {
                continue;
            }
            if !longevity.contains(arg) {
                longevity.set(arg, Lifetime::new(UNDEF_POS, i));
            }
            let lifetime = longevity.get_mut(arg).unwrap();
            // real_usages: exclude JUMP and LABEL
            if opnum != OpCode::Jump && opnum != OpCode::Label {
                if lifetime.real_usages.is_none() {
                    lifetime.real_usages = Some(SmallVec::new());
                }
                lifetime.real_usages.as_mut().unwrap().push(i);
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
                let arg = arg.to_opref();
                if !longevity.contains(arg) {
                    longevity.set(arg, Lifetime::new(UNDEF_POS, i));
                }
            }
        }
    }

    // regalloc.py:1210-1213 input arguments
    for iarg in inputargs {
        let opref = iarg.as_ref().opref();
        if !longevity.contains(opref) {
            longevity.set(opref, Lifetime::new(-1, -1));
        }
    }

    // regalloc.py:1224-1231 reverse real_usages and check invariants
    // We need to iterate over all lifetimes; collect keys first to avoid borrow issues
    let keys: Vec<OpRef> = longevity.lifetimes.keys().copied().collect();
    for opref in keys {
        let lifetime = longevity.get_mut(opref).unwrap();
        if let Some(ref mut usages) = lifetime.real_usages {
            usages.reverse();
        }
        #[cfg(debug_assertions)]
        lifetime.check_invariants();
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
    pub boxes_in_frame: Vec<Option<OpRef>>,
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
    pub fn get(&self, v: OpRef, longevity: &LifetimeManager) -> Option<FrameLoc> {
        longevity.get(v).and_then(|lt| lt.current_frame_loc)
    }

    /// regalloc.py:95
    pub fn loc(
        &mut self,
        v: OpRef,
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
    pub fn get_new_loc(&mut self, v: OpRef, tp: Type, longevity: &mut LifetimeManager) -> FrameLoc {
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
    pub fn bind(&mut self, v: OpRef, loc: FrameLoc, tp: Type, longevity: &mut LifetimeManager) {
        let pos = loc.get_position();
        let size = Self::frame_size(tp);
        if pos + size > self.current_frame_depth {
            self._increase_frame_depth(pos + size - self.current_frame_depth);
        }
        debug_assert!(self.boxes_in_frame[pos].is_none());
        for index in pos..pos + size {
            self.boxes_in_frame[index] = Some(v);
        }
        let lifetime = longevity
            .get_mut(v)
            .expect("bind: variable not in longevity");
        lifetime.current_frame_loc = Some(loc);
    }

    /// regalloc.py:181
    pub fn finish_binding(&self) {
        #[cfg(debug_assertions)]
        self._check_invariants();
    }

    /// regalloc.py:185
    pub fn mark_as_free(&mut self, v: OpRef, tp: Type, longevity: &mut LifetimeManager) {
        let loc = match self.get(v, longevity) {
            Some(loc) => loc,
            None => return, // not in frame
        };
        let lifetime = longevity.get_mut(v).unwrap();
        debug_assert!(lifetime.current_frame_loc.is_some());
        lifetime.current_frame_loc = None;
        let pos = loc.get_position();
        let size = Self::frame_size(tp);
        debug_assert!(self.boxes_in_frame[pos] == Some(v));
        for index in pos..pos + size {
            self.boxes_in_frame[index] = None;
        }
    }

    /// regalloc.py:201
    pub fn add_frame_pos_hint(&self, v: OpRef, loc: FrameLoc, longevity: &mut LifetimeManager) {
        if let Some(lifetime) = longevity.get_mut(v) {
            lifetime.hint_frame_pos = loc.get_position() as i32;
        }
    }

    /// regalloc.py:206
    pub fn get_frame_pos_hint(&self, v: OpRef, longevity: &LifetimeManager) -> i32 {
        longevity.get(v).map(|lt| lt.hint_frame_pos).unwrap_or(-1)
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
    MALLOC_NURSERY_CLOBBER, MALLOC_NURSERY_RESULT, all_core_regs, all_float_regs, call_result_fpr,
    call_result_gpr, core_reg_index, frame_reg, save_around_call_core_regs,
};
// aarch64 `_prepare_op_cond_call` forces extras into `argument_regs`.
// x86 `consider_cond_call` captures current locations instead.
#[cfg(target_arch = "aarch64")]
use arch_regalloc::cond_call_argument_regs;

// Per-arch reghint pass.  Upstream RPython ships reghint only under
// `rpython/jit/backend/x86/`; there is no AArch64 counterpart here.
// Strict line-by-line parity: run the
// reghint pre-pass only on x86_64.
#[cfg(target_arch = "x86_64")]
use crate::x86::reghint as arch_reghint;

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
    /// regalloc.py:371 register index → OpRef (None = free)
    pub reg_bindings_list: Vec<Option<OpRef>>,
    /// regalloc.py:373
    pub temp_boxes: Vec<OpRef>,
    /// regalloc.py:375
    pub box_currently_in_frame_reg: Option<OpRef>,
    /// regalloc.py:376
    pub position: i32,
    /// Spill moves generated by _sync_var_to_stack.
    /// RPython calls self.assembler.regalloc_mov() directly; in Rust we
    /// collect moves here and flush them to the assembler's output.
    pub spill_moves: Vec<(Loc, Loc)>,
    /// `X86XMMRegisterManager` / `VFPRegisterManager`
    /// `assembler.datablockwrapper`. `0` on the GPR manager. The assembler
    /// installs the pointer on the XMM / VFP manager after `prepare_loop` /
    /// `prepare_bridge` (those rebuild this manager) and after
    /// `emit_check_frame_depth`, immediately before `walk_operations`.
    datablockwrapper: usize,
    /// First `MachineDataBlockWrapper.malloc_aligned` failure.
    /// `open_malloc` raises `MemoryError`; `walk_operations` returns it as
    /// `BackendError::CompilationFailed` so `consider_*` stays infallible.
    pool_error: Option<std::io::Error>,
}

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
fn const_bits_or_panic(v: OpRef, constants: &indexmap::IndexMap<u32, i64>, where_: &str) -> i64 {
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

/// `box.type` for a regalloc operand, given its `OpTypeIndex` lookup result.
///
/// RPython reads `v.type` directly during register allocation; every real
/// Box pins its type at construction (history.py ConstInt / :261
/// ConstFloat / :307 ConstPtr), so `OpTypeIndex::opref_type` returns it.
/// The only type-less operand is a regalloc-scratch `TempVar`, which
/// RPython's `_check_type` (regalloc.py) exempts from the type
/// check via `isinstance(v, TempVar)`. Pyre allocates every `TempVar` as
/// GP-integer scratch (`fresh_temp_var`, all call sites mirror `TempInt()`),
/// so a type-less operand resolves to `Type::Int`.
///
/// Any *other* type-less opref is a real Box that lost its type tag — a
/// Box-identity violation. `_check_type` is debug-only (`not
/// we_are_translated()`), so this mirrors it with a `debug_assert!`: the
/// release build keeps RPython's untyped flow, the debug build surfaces the
/// violation at the boundary instead of silently picking `Int`.
fn box_type_or_temp(known: Option<Type>, v: OpRef) -> Type {
    match known {
        Some(tp) => tp,
        None => {
            debug_assert!(
                matches!(v, OpRef::TempVar(_)),
                "regalloc operand {v:?} has no box.type and is not a TempVar \
                 (real Boxes carry box.type intrinsically)"
            );
            Type::Int
        }
    }
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
            datablockwrapper: 0,
            pool_error: None,
        }
    }

    /// Attach `X86XMMRegisterManager` / `VFPRegisterManager`
    /// `assembler.datablockwrapper` for this walk.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_datablockwrapper(&mut self, wrapper: &mut majit_backend::MachineDataBlockWrapper) {
        self.datablockwrapper = wrapper as *mut majit_backend::MachineDataBlockWrapper as usize;
    }

    // ── RegBindingsDict equivalent ──

    fn _register_index(&self, reg: RegLoc) -> usize {
        self.all_regs
            .iter()
            .position(|r| *r == reg)
            .expect("register not in all_regs")
    }

    /// regalloc.py / RegBindingsDict.get
    pub fn reg_bindings_get(&self, v: OpRef, longevity: &LifetimeManager) -> Option<RegLoc> {
        if let Some(lifetime) = longevity.get(v) {
            let index = lifetime.current_register_index;
            if index >= 0 {
                return Some(self.all_regs[index as usize]);
            }
        }
        None
    }

    /// RegBindingsDict.__contains__
    pub fn reg_bindings_contains(&self, v: OpRef, longevity: &LifetimeManager) -> bool {
        self.reg_bindings_get(v, longevity).is_some()
    }

    /// RegBindingsDict.__setitem__
    ///
    /// Overwrites the reverse slot (`reg_bindings_list[new]`) but leaves any
    /// previous occupant's `current_register_index` alone. `loc()` reads that
    /// index, so two liveboxes recovered from the same physical register
    /// (duplicate `rd_locs` entries) stay findable. Clearing the old occupant
    /// here made `RegisterManager.loc` panic on the first box.
    pub fn reg_bindings_set(&mut self, v: OpRef, reg: RegLoc, longevity: &mut LifetimeManager) {
        let new_index = self._register_index(reg);
        // Clear this variable's previous reverse slot first.
        let old_index = longevity
            .get(v)
            .map(|lt| lt.current_register_index)
            .unwrap_or(-1);
        if old_index >= 0 {
            self.reg_bindings_list[old_index as usize] = None;
        }
        let lifetime = longevity
            .get_mut(v)
            .expect("reg_bindings_set: not in longevity");
        lifetime.current_register_index = new_index as i32;
        self.reg_bindings_list[new_index] = Some(v);
    }

    /// regalloc.py RegBindingsDict.__delitem__
    pub fn reg_bindings_del(&mut self, v: OpRef, longevity: &mut LifetimeManager) {
        let lifetime = longevity
            .get_mut(v)
            .expect("reg_bindings_del: not in longevity");
        let index = lifetime.current_register_index;
        assert!(
            index >= 0,
            "reg_bindings_del: {:?} not in register (index={})",
            v,
            index
        );
        self.reg_bindings_list[index as usize] = None;
        lifetime.current_register_index = -1;
    }

    /// regalloc.py RegBindingsDict.pop
    pub fn reg_bindings_pop(&mut self, v: OpRef, longevity: &mut LifetimeManager) -> RegLoc {
        let lifetime = longevity
            .get_mut(v)
            .expect("reg_bindings_pop: not in longevity");
        let index = lifetime.current_register_index;
        assert!(
            index >= 0,
            "reg_bindings_pop: {:?} not in register (index={})",
            v,
            index
        );
        let reg = self.all_regs[index as usize];
        self.reg_bindings_list[index as usize] = None;
        lifetime.current_register_index = -1;
        reg
    }

    /// Iterate (OpRef, RegLoc) bindings — RegBindingsIterItems equivalent.
    pub fn reg_bindings_iter(&self) -> impl Iterator<Item = (OpRef, RegLoc)> + '_ {
        self.reg_bindings_list
            .iter()
            .enumerate()
            .filter_map(|(i, opt)| opt.map(|v| (v, self.all_regs[i])))
    }

    /// Collect current register bindings as Vec.
    pub fn reg_bindings_items(&self) -> Vec<(OpRef, RegLoc)> {
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
    pub fn is_still_alive(&self, v: OpRef, longevity: &LifetimeManager) -> bool {
        longevity
            .get(v)
            .map(|lt| lt.last_usage >= self.position)
            .unwrap_or(false)
    }

    /// regalloc.py:385
    pub fn stays_alive(&self, v: OpRef, longevity: &LifetimeManager) -> bool {
        longevity
            .get(v)
            .map(|lt| lt.last_usage > self.position)
            .unwrap_or(false)
    }

    /// regalloc.py:390
    pub fn next_instruction(&mut self, incr: i32) {
        self.position += incr;
    }

    /// regalloc.py:409
    pub fn possibly_free_var(
        &mut self,
        v: OpRef,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        v_type: Type,
    ) {
        if v.is_constant() {
            return;
        }
        let should_free =
            !longevity.contains(v) || longevity.get(v).unwrap().last_usage <= self.position;
        if should_free {
            if let Some(reg) = self.reg_bindings_get(v, longevity) {
                self.free_regs.push(reg);
                self.reg_bindings_del(v, longevity);
            }
            if self.box_currently_in_frame_reg == Some(v) {
                self.box_currently_in_frame_reg = None;
            }
            fm.mark_as_free(v, v_type, longevity);
        }
    }

    /// regalloc.py:427
    pub fn possibly_free_vars(
        &mut self,
        vars: &[OpRef],
        types: &[Type],
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
    ) {
        for (v, tp) in vars.iter().zip(types.iter()) {
            self.possibly_free_var(*v, longevity, fm, *tp);
        }
    }

    /// regalloc.py:437
    pub fn free_temp_vars(&mut self, longevity: &mut LifetimeManager, fm: &mut FrameManager) {
        let temps: Vec<OpRef> = std::mem::take(&mut self.temp_boxes);
        for v in temps {
            // Temp vars are always INT type
            self.possibly_free_var(v, longevity, fm, Type::Int);
        }
    }

    /// regalloc.py:460
    pub fn try_allocate_reg(
        &mut self,
        v: OpRef,
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
        longevity: &mut LifetimeManager,
    ) -> Option<RegLoc> {
        debug_assert!(!v.is_constant());

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
        forbidden_vars: &[OpRef],
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
        let loc = self.reg_bindings_get(v_to_spill, longevity).unwrap();
        let tp = self._type_of(v_to_spill);
        self._sync_var_to_stack(v_to_spill, tp, longevity, fm);
        self.reg_bindings_del(v_to_spill, longevity);
        loc
    }

    /// regalloc.py _pick_variable_to_spill
    pub fn _pick_variable_to_spill(
        &self,
        forbidden_vars: &[OpRef],
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
        regs: Option<&[OpRef]>,
        longevity: &LifetimeManager,
    ) -> OpRef {
        let binding_keys: Vec<OpRef>;
        let regs = match regs {
            Some(r) => r,
            None => {
                binding_keys = self.reg_bindings_list.iter().filter_map(|x| *x).collect();
                &binding_keys
            }
        };

        let position = self.position;
        let mut cur_max_use_distance: i32 = -1;
        let mut candidate: Option<OpRef> = None;
        let mut cur_max_age_failargs: i32 = -1;
        let mut candidate_from_failargs: Option<OpRef> = None;

        for &next in regs {
            let reg = self.reg_bindings_get(next, longevity).unwrap();
            if forbidden_vars.contains(&next) {
                continue;
            }
            // regalloc.py:544
            if self.temp_boxes.contains(&next) {
                continue;
            }
            if let Some(sel) = selected_reg {
                if reg == sel {
                    return next;
                }
                continue;
            }
            if need_lower_byte && self.no_lower_byte_regs.contains(&reg) {
                continue;
            }
            let lifetime = longevity.get(next).unwrap();
            if lifetime.is_last_real_use_before(position) {
                // regalloc.py:554-561
                let max_age = lifetime.last_usage;
                if cur_max_age_failargs < max_age {
                    cur_max_age_failargs = max_age;
                    candidate_from_failargs = Some(next);
                }
            } else {
                // regalloc.py:563-566
                let use_distance = lifetime.next_real_usage(position) - position;
                if cur_max_use_distance < use_distance {
                    cur_max_use_distance = use_distance;
                    candidate = Some(next);
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
        v: OpRef,
        forbidden_vars: &[OpRef],
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
    ) -> RegLoc {
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
    pub fn force_allocate_frame_reg(&mut self, v: OpRef) {
        debug_assert!(self.box_currently_in_frame_reg.is_none());
        self.box_currently_in_frame_reg = Some(v);
    }

    /// regalloc.py:602
    pub fn force_spill_var(
        &mut self,
        var: OpRef,
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
    fn is_float_constant(&self, v: OpRef) -> bool {
        matches!(v.ty(), Some(Type::Float))
    }

    /// regalloc.py:611
    pub fn loc(
        &mut self,
        v: OpRef,
        tp: Type,
        must_exist: bool,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        constants: &indexmap::IndexMap<u32, i64>,
    ) -> Loc {
        if v.is_constant() {
            // `RegisterManager.loc`: a Const goes through `convert_to_imm`.
            // With `datablockwrapper` set this is
            // `X86XMMRegisterManager.convert_to_imm` /
            // `VFPRegisterManager.convert_to_imm` (`ConstFloatLoc`).
            // The GPR manager has no wrapper and keeps an `ImmedLoc`.
            if self.datablockwrapper != 0 {
                return self.convert_to_imm(v, constants);
            }
            // history.py/268/314 — inline-Const variants carry the
            // value directly; legacy pool-indexed variants resolve through
            // the constants snapshot.
            let val = const_bits_or_panic(v, constants, "loc");
            if tp == Type::Float || self.is_float_constant(v) {
                return Loc::immed_float(val);
            }
            return Loc::immed(val);
        }
        if let Some(reg) = self.reg_bindings_get(v, longevity) {
            return Loc::Reg(reg);
        }
        if self.box_currently_in_frame_reg == Some(v) {
            return Loc::Reg(self.frame_reg);
        }
        // regalloc.py:622 → frame_manager.loc(box, must_exist)
        if let Some(frame_loc) = fm.get(v, longevity) {
            return Loc::Frame(frame_loc);
        }
        // Fallback: check constants map. The optimizer may leave constants
        // as plain OpRefs (without CONST_BIT) when they originate from
        // short preamble or constant folding.
        if let Some(val) = v
            .inline_const_bits()
            .or_else(|| constants.get(&v.raw()).copied())
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
        v: OpRef,
        forbidden_vars: &[OpRef],
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
        v: OpRef,
        tp: Type,
        forbidden_vars: &[OpRef],
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
        from_v: OpRef,
        to_v: OpRef,
        longevity: &mut LifetimeManager,
    ) -> RegLoc {
        let reg = self.reg_bindings_pop(from_v, longevity);
        self.reg_bindings_set(to_v, reg, longevity);
        reg
    }

    /// regalloc.py:667
    pub fn force_result_in_reg(
        &mut self,
        result_v: OpRef,
        v: OpRef,
        tp: Type,
        forbidden_vars: &[OpRef],
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
        let v_keeps_living = longevity.get(v).unwrap().last_usage > self.position;
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
        v: OpRef,
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
        v: OpRef,
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
    pub fn before_call<T: AsRef<Op>, A: AsRef<InputArg>>(
        &mut self,
        force_store: &[OpRef],
        save_all_regs: u8,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        pending_moves: &mut Vec<(Loc, Loc)>,
        type_index: &OpTypeIndex<'_, T, A>,
    ) {
        self.spill_or_move_registers_before_call(
            &self.save_around_call_regs.clone(),
            force_store,
            save_all_regs,
            longevity,
            fm,
            pending_moves,
            type_index,
        );
    }

    /// regalloc.py:714
    pub fn spill_or_move_registers_before_call<T: AsRef<Op>, A: AsRef<InputArg>>(
        &mut self,
        save_sublist: &[RegLoc],
        force_store: &[OpRef],
        save_all_regs: u8,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        _pending_moves: &mut Vec<(Loc, Loc)>,
        type_index: &OpTypeIndex<'_, T, A>,
    ) {
        let mut new_free_regs: Vec<RegLoc> = Vec::new();
        let mut move_or_spill: Vec<OpRef> = Vec::new();

        // regalloc.py iterate current bindings
        let items: Vec<(OpRef, RegLoc)> = self.reg_bindings_items();
        for (v, reg) in items {
            let max_age = longevity.get(v).unwrap().last_usage;
            // regalloc.py reads `v.type` directly; box.type is
            // intrinsic on every real Box (history.py:220). A type-less
            // binding is only a regalloc-scratch TempVar (`_check_type`
            // exemption, regalloc.py:405), resolved to Type::Int by
            // `box_type_or_temp`.
            let v_type = if self.position >= 0 {
                box_type_or_temp(type_index.opref_type_at(v, self.position as usize), v)
            } else {
                box_type_or_temp(type_index.opref_type(v), v)
            };
            if !force_store.contains(&v) && max_age <= self.position {
                // variable dies
                self.reg_bindings_del(v, longevity);
                new_free_regs.push(reg);
                continue;
            }

            if save_all_regs == SAVE_ALL_REGS {
                // regalloc.py:782-784
                self._bc_spill(v, v_type, &mut new_free_regs, longevity, fm);
            } else if save_all_regs == SAVE_GCREF_REGS && v_type == Type::Ref {
                // regalloc.py:786-788 — spill GC ptrs only
                self._bc_spill(v, v_type, &mut new_free_regs, longevity, fm);
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
                self._bc_spill(v, self._type_of(v), &mut new_free_regs, longevity, fm);
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
                let old_reg = self.reg_bindings_get(*v, longevity).unwrap();
                self.spill_moves
                    .push((Loc::Reg(old_reg), Loc::Reg(new_reg)));
                self.reg_bindings_set(*v, new_reg, longevity);
                new_free_regs.push(old_reg);
            }
        }

        // regalloc.py:826-827 re-add in reverse
        while let Some(r) = new_free_regs.pop() {
            self.free_regs.push(r);
        }
    }

    /// regalloc.py:829
    pub fn after_call(&mut self, v: OpRef, longevity: &mut LifetimeManager) -> RegLoc {
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
    pub fn convert_to_imm(&mut self, v: OpRef, constants: &indexmap::IndexMap<u32, i64>) -> Loc {
        debug_assert!(v.is_constant());
        let val = const_bits_or_panic(v, constants, "convert_to_imm");
        if self.datablockwrapper != 0 {
            return self.float_const_loc(val as u64);
        }
        // x86/regalloc.py:58-61: a non-null `ConstPtr` whose object can
        // still move must never be baked as an immediate — `remove_constptr`
        // (rewrite.py:1100) routes those through the gc_table. Null and
        // non-moving (prebuilt / old-gen / no active GC) ConstPtrs are baked
        // directly. `rgc.can_move` parity via `majit_gc::can_move`; the
        // `we_are_translated()` clause is subsumed because `can_move` is
        // false when no moving GC is active.
        if let Some(g) = v.as_const_ptr()
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
        if self.datablockwrapper == 0 {
            return Loc::immed_float(bits as i64);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let wrapper = unsafe {
                &mut *(self.datablockwrapper as *mut majit_backend::MachineDataBlockWrapper)
            };
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
        {
            let _ = bits;
            Loc::immed_float(bits as i64)
        }
    }

    /// x86/regalloc.py call_result_location → eax (GPR) or xmm0 (XMM).
    pub fn call_result_location(&self) -> RegLoc {
        self.call_result_reg
    }

    /// Infer type for an OpRef (INT for GPR manager, FLOAT for XMM).
    fn _type_of(&self, _v: OpRef) -> Type {
        match &self.box_types {
            Some(types) if types.contains(&Type::Float) => Type::Float,
            _ => Type::Int,
        }
    }
}

// ── RegAlloc ───────────────────────────────────────────────────────

/// x86/regalloc.py walk_operations output: one entry per operation.
/// The assembler uses these to generate machine code.
#[derive(Debug)]
pub enum RegAllocOp {
    /// regalloc_perform(op_index, arglocs, result_loc)
    ///
    /// `gcmap` is a precomputed GC bitmap pointer captured at regalloc
    /// time. Non-None for collecting calls (e.g., CallMallocNursery) so
    /// the emit path can store it into `jf_gcmap` before a potentially
    /// collecting call. None for ordinary ops — emit uses its own
    /// gcmap path (guards compute theirs from faillocs).
    Perform {
        op_index: usize,
        /// Slice of [`RegAlloc::arglocs_arena`]. A 2-loc `Vec<Loc>` was
        /// the 32 B class on the regex and/or compile path
        /// (`x86/regalloc.py perform(op, [loc, argloc], res)`).
        arglocs_start: u32,
        arglocs_len: u32,
        result_loc: Option<Loc>,
        gcmap: Option<usize>,
    },
    /// One-loc `perform` without a 16 B `Vec`.
    Perform1 {
        op_index: usize,
        loc: Loc,
        result_loc: Option<Loc>,
        gcmap: Option<usize>,
    },
    /// regalloc_perform_guard(op_index, arglocs, result_loc, faillocs)
    ///
    /// `faillocs` live in [`RegAlloc::faillocs_arena`]; start/len are
    /// the slice. `locs_for_fail` writes them from `getfailargs()`.
    PerformGuard {
        op_index: usize,
        arglocs_start: u32,
        arglocs_len: u32,
        result_loc: Option<Loc>,
        faillocs_start: u32,
        faillocs_len: u32,
    },
    /// One-loc `perform_guard` without a 16 B `Vec`.
    PerformGuard1 {
        op_index: usize,
        loc: Loc,
        result_loc: Option<Loc>,
        faillocs_start: u32,
        faillocs_len: u32,
    },
    /// regalloc_perform_discard(op_index, arglocs)
    PerformDiscard { op_index: usize, arglocs: Vec<Loc> },
    /// `GC_STORE` `[value, base, ofs, imm(size)]` without a 64 B `Vec`.
    /// Three locs plus `size` stay inside the 72 B `RegAllocOp` envelope.
    PerformDiscardGcStore {
        op_index: usize,
        value: Loc,
        base: Loc,
        ofs: Loc,
        size: i64,
    },
    /// `GC_LOAD` `[base, ofs, res, imm(nsize)]` without a 64 B `Vec`.
    /// Three locs plus `nsize` stay inside the 72 B `RegAllocOp` envelope.
    PerformGcLoad {
        op_index: usize,
        base: Loc,
        ofs: Loc,
        res: Loc,
        nsize: i64,
    },
    /// Register move: src → dst (from spill/reload/register-register moves)
    Move { src: Loc, dst: Loc },
    /// Skip (dead operation, no-op)
    Skip,
}

/// x86/regalloc.py RegAlloc — main x86 register allocator.
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
    inputargs: &'a [InputArgRc],
    /// Trace operations — borrowed for `opref_type` lookups (reads
    /// `op.type_` directly, RPython `box.type` parity).
    pub(crate) operations: &'a [OpRc],
    /// `arg.index` raw -> idx in inputargs. Mirrors
    /// `OpTypeIndex::inputarg_pos`.
    inputarg_pos: majit_ir::PosIndex,
    /// `op.pos.raw()` -> idx in operations, skipping Void/None ops.
    /// Mirrors `OpTypeIndex::op_pos`.
    op_pos: majit_ir::PosIndex,
    /// x86/regalloc.py:1305 self.jump_target_descr — TargetToken descriptor
    /// of the closing JUMP. PyPy keys it by Python object identity; we use
    /// the underlying `Arc<dyn Descr>` allocation address.
    jump_target_descr: Option<usize>,
    /// x86/regalloc.py:1407 self.final_jump_op — paired (descr identity,
    /// jump arglist) so consider_label can recognize a self-loop and emit
    /// frame position hints via add_frame_pos_hint.
    final_jump_args: Option<(usize, Vec<OpRef>)>,
    /// x86/regalloc.py:1270 self.final_jump_op_position — position of the
    /// closing JUMP, used by `_compute_hint_locations_from_descr` to pin
    /// reg hints via `longevity.fixed_register(position, reg, box)`.
    final_jump_op_position: i32,
    /// j2-lowered operations, indexed by original operation index. The main
    /// dispatch path consumes these; legacy opcode dispatch is a guard rail
    /// only if a plan entry is missing.
    j2_ops: Vec<LirOp>,
    /// Per-trace counter feeding `OpRef::fresh_temp_var()` — RPython
    /// `TempVar()` / `TempInt()` allocate a fresh Python object per
    /// call (`x86/regalloc.py:470,514,521,605`,
    /// `aarch64/regalloc.py`); pyre's flat-OpRef encoding needs a
    /// counter to feed the unique raw payload reservation.
    temp_var_counter: u32,
    /// Failarg locations for every guard in this walk. `locs_for_fail`
    /// appends here so `PerformGuard` stores a (start, len) instead of
    /// a 96 B `Vec` per guard.
    faillocs_arena: Vec<Option<Loc>>,
    /// Loc lists for `Perform` / `PerformGuard`. `perform(op, [a, b], res)`
    /// is a 2-element list upstream; a per-op `Vec<Loc>` was 32 B on the
    /// regex and/or compile path (`push_perform` / `consider_int_ri_j2`).
    arglocs_arena: Vec<Loc>,
}

fn is_math_sqrt_call(op: &Op) -> bool {
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
            j2_ops: Vec::new(),
            temp_var_counter: 0,
            faillocs_arena: Vec::new(),
            arglocs_arena: Vec::new(),
        }
    }

    /// RPython `TempVar()` / `TempInt()` 1:1 parity. Each call returns a
    /// fresh `OpRef::TempVar` for use as a regalloc scratch box. Lifetime
    /// is single-instruction: caller registers it in `longevity` for the
    /// current `position`, calls `force_allocate_reg`, then
    /// `possibly_free_var`.
    pub fn fresh_temp_var(&mut self) -> OpRef {
        let counter = self.temp_var_counter;
        self.temp_var_counter += 1;
        OpRef::fresh_temp_var(counter)
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
    fn opref_type_at(&self, opref: OpRef, at_op_index: Option<usize>) -> Option<Type> {
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
    fn make_gpr_manager() -> RegisterManager {
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
    fn make_xmm_manager() -> RegisterManager {
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

    /// x86/regalloc.py _prepare
    pub fn _prepare(&mut self) {
        self.longevity = compute_vars_longevity(self.inputargs, self.operations);
        let base_ofs = crate::jitframe::FIRST_ITEM_OFFSET as i32;
        self.fm = FrameManager::new(base_ofs);
        self.rm = Self::make_gpr_manager();
        self.xrm = Self::make_xmm_manager();
        self.jump_target_descr = None;
        self.final_jump_args = None;
        self.final_jump_op_position = -1;
        // aarch64/assembler.py `_walk_operations` / x86/regalloc.py
        // `walk_operations` dispatch on the live ResOp. Materializing a
        // `Vec<LirOp>` here cloned every op (and every guard's fail_args
        // SmallVec) before the walk; lower one op on the stack instead.
        // Tests that inject a replacement plan still write `j2_ops`.
        self.j2_ops.clear();
        // x86/regalloc.py:191 X86RegisterHints().add_hints(longevity, inputargs, operations)
        //
        // Strict parity: upstream `rpython/jit/backend/x86/reghint.py`
        // is x86-only; no backend/aarch64 counterpart exists, so the
        // pre-pass runs on x86_64 alone.  aarch64 loses
        // call-position caller-save blocking — a latent upstream bug
        // that pyre does not paper over.
        #[cfg(target_arch = "x86_64")]
        {
            let hints = arch_reghint::RegisterHints::new(
                self.rm.save_around_call_regs.clone(),
                self.xrm.save_around_call_regs.clone(),
                self.rm.all_regs.clone(),
                self.xrm.all_regs.clone(),
                self.constants.clone(),
            );
            hints.add_hints(&mut self.longevity, self.inputargs, self.operations);
        }
    }

    /// x86/regalloc.py prepare_loop
    pub fn prepare_loop(&mut self) {
        self._prepare();
        let inputargs = self.inputargs;
        self._set_initial_bindings(inputargs);
        // Free input variables that are dead at position 0
        for iarg in inputargs {
            let tp = iarg.tp.get();
            let opref = iarg.opref();
            if tp == Type::Float {
                self.xrm
                    .possibly_free_var(opref, &mut self.longevity, &mut self.fm, tp);
            } else {
                self.rm
                    .possibly_free_var(opref, &mut self.longevity, &mut self.fm, tp);
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

    /// x86/regalloc.py compute_hint_frame_locations — pre-walk scan
    /// that records the closing JUMP so `consider_label` (and the
    /// same-loop branch in this routine) can feed frame/register hints
    /// back to the forward regalloc pass.
    fn compute_hint_frame_locations(&mut self) {
        let operations = self.operations;
        let Some(op) = operations.last() else {
            return;
        };
        if op.opcode != OpCode::Jump {
            return;
        }
        let descr_id = op.getdescr().as_ref().map(descr_identity);
        self.final_jump_args = descr_id.map(|id| {
            (
                id,
                op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect()),
            )
        });
        self.final_jump_op_position = (operations.len() - 1) as i32;
        let Some(descr) = op.getdescr() else {
            return;
        };
        // x86/regalloc.py:1274 descr._ll_loop_code != 0:
        //   target LABEL already compiled (bridge/external jump) → apply
        //   hints now. else: same-loop, wait for consider_label.
        if let Some(target) = descr.as_loop_target_descr()
            && target.ll_loop_code() != 0
        {
            self._compute_hint_locations_from_descr(target);
        }
    }

    /// x86/regalloc.py _compute_hint_locations_from_descr /
    /// aarch64/regalloc.py _compute_hint_frame_locations_from_descr.
    ///
    /// x86 hints both Frame and Reg targets; aarch64 only hints Frame.
    /// Mirror per-arch via #[cfg].
    fn _compute_hint_locations_from_descr(&mut self, descr: &dyn majit_ir::descr::LoopTargetDescr) {
        use majit_ir::descr::TargetArgLoc;
        let Some((_, jump_args)) = self.final_jump_args.as_ref() else {
            return;
        };
        let target_arglocs = descr.target_arglocs();
        // x86/regalloc.py `_compute_hint_locations_from_descr`:
        // `assert len(arglocs) == jump_op.numargs()`. The assembler remap
        // is the same 1:1 walk; a length mismatch here is the JUMP vs
        // LABEL arity bug, not a hint to skip.
        assert_eq!(
            jump_args.len(),
            target_arglocs.len(),
            "JUMP args ({}) != target LABEL args ({})",
            jump_args.len(),
            target_arglocs.len(),
        );
        #[cfg(target_arch = "x86_64")]
        let position = self.final_jump_op_position;
        #[cfg(target_arch = "x86_64")]
        let mut hinted: Vec<OpRef> = Vec::new();
        for (arg, target) in jump_args.iter().zip(target_arglocs.iter()) {
            if arg.is_constant() {
                continue;
            }
            match *target {
                TargetArgLoc::Frame {
                    position: pos,
                    ebp_offset,
                    is_float,
                } => {
                    let f = crate::regloc::FrameLoc::new(pos, ebp_offset, is_float);
                    self.fm.add_frame_pos_hint(*arg, f, &mut self.longevity);
                }
                #[cfg(target_arch = "x86_64")]
                TargetArgLoc::Reg { regnum, is_xmm } => {
                    if !hinted.contains(arg) {
                        hinted.push(*arg);
                        let r = crate::regloc::RegLoc::new(regnum, is_xmm);
                        self.longevity.fixed_register(position, r, Some(*arg));
                    }
                }
                _ => {}
            }
        }
    }

    /// regalloc.py _set_initial_bindings — place all inputargs in frame slots.
    fn _set_initial_bindings(&mut self, inputargs: &[InputArgRc]) {
        for iarg in inputargs {
            let opref = iarg.opref();
            let _loc = self
                .fm
                .get_new_loc(opref, iarg.tp.get(), &mut self.longevity);
        }
    }

    /// x86/regalloc.py _update_bindings — bind bridge inputargs to their locations.
    fn _update_bindings(&mut self, locs: &[Loc], inputargs: &[InputArgRc]) {
        let mut used: IndexMap<RegLoc, (), FxBuildHasher> = IndexMap::with_hasher(FxBuildHasher);

        // x86/regalloc.py:295-312
        for (iarg, loc) in inputargs.iter().zip(locs.iter()) {
            let v = iarg.opref();
            let tp = iarg.tp.get();
            match loc {
                Loc::Reg(reg) => {
                    if tp == Type::Float {
                        self.xrm.reg_bindings_set(v, *reg, &mut self.longevity);
                        used.insert(*reg, ());
                    } else if *reg == self.rm.frame_reg {
                        // x86/regalloc.py:305-307
                        debug_assert!(self.rm.box_currently_in_frame_reg.is_none());
                        self.rm.box_currently_in_frame_reg = Some(v);
                    } else {
                        self.rm.reg_bindings_set(v, *reg, &mut self.longevity);
                        used.insert(*reg, ());
                    }
                }
                Loc::Frame(floc) => {
                    self.fm.bind(v, *floc, tp, &mut self.longevity);
                }
                _ => {
                    self.fm.bind(
                        v,
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
            let opref = iarg.opref();
            self.possibly_free_var(opref, iarg.tp.get());
        }
        // x86/regalloc.py:322
        self.fm.finish_binding();
    }

    // ── Location dispatch (type-based routing to rm or xrm) ──

    /// x86/regalloc.py loc(v)
    pub fn loc(&mut self, v: OpRef, tp: Type) -> Loc {
        self.loc_with_mode(v, tp, false)
    }

    pub fn loc_must_exist(&mut self, v: OpRef, tp: Type) -> Loc {
        self.loc_with_mode(v, tp, true)
    }

    fn loc_with_mode(&mut self, v: OpRef, tp: Type, must_exist: bool) -> Loc {
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
    pub fn possibly_free_var(&mut self, v: OpRef, tp: Type) {
        if tp == Type::Float {
            self.xrm
                .possibly_free_var(v, &mut self.longevity, &mut self.fm, tp);
        } else {
            self.rm
                .possibly_free_var(v, &mut self.longevity, &mut self.fm, tp);
        }
    }

    /// x86/regalloc.py possibly_free_vars_for_op(op) — RPython reads
    /// `arg.type` per box; pyre routes through `self.tp`, which resolves a
    /// real Box's intrinsic type and falls back to `Type::Int` only for a
    /// `TempVar` (the `_check_type` exemption, regalloc.py).
    pub fn possibly_free_vars_for_op(&mut self, op: &Op) {
        for arg in op.args_slice().iter() {
            if !arg.is_constant() && !arg.is_none() {
                let arg = arg.to_opref();
                let tp = self.tp(arg);
                self.possibly_free_var(arg, tp);
            }
        }
    }

    /// x86/regalloc.py make_sure_var_in_reg
    pub fn make_sure_var_in_reg(
        &mut self,
        v: OpRef,
        tp: Type,
        forbidden_vars: &[OpRef],
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
    ) -> Loc {
        // x86/regalloc.py `RegAlloc.make_sure_var_in_reg`: a `ConstFloat` is
        // `FloatImmedLoc(getfloatstorage())` before `xrm.make_sure_var_in_reg`,
        // so `save_into_mem` stores the bits and never a `ConstFloatLoc`.
        // `selected_reg` / `need_lower_byte` do not apply on that path.
        // aarch64/regalloc.py `Regalloc.make_sure_var_in_reg` has no such
        // short-circuit: `ARMRegisterManager.return_constant` loads
        // `VFPRegisterManager.convert_to_imm`'s `ConstFloatLoc` into a VFP
        // scratch (`AssemblerARM64.load`). The walk installs the data block
        // on `xrm`, so that immediate is `Loc::ConstFloat`.
        #[cfg(target_arch = "x86_64")]
        if tp == Type::Float && v.is_constant() {
            return Loc::immed_float(self.const_value(v));
        }
        if tp == Type::Float {
            self.xrm.make_sure_var_in_reg(
                v,
                tp,
                forbidden_vars,
                selected_reg,
                need_lower_byte,
                &mut self.longevity,
                &mut self.fm,
                &self.constants,
                &mut self.pending_moves,
            )
        } else {
            self.rm.make_sure_var_in_reg(
                v,
                tp,
                forbidden_vars,
                selected_reg,
                need_lower_byte,
                &mut self.longevity,
                &mut self.fm,
                &self.constants,
                &mut self.pending_moves,
            )
        }
    }

    /// x86/regalloc.py force_allocate_reg
    pub fn force_allocate_reg(
        &mut self,
        v: OpRef,
        tp: Type,
        forbidden_vars: &[OpRef],
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
    pub fn force_spill_var(&mut self, v: OpRef, tp: Type) {
        if tp == Type::Float {
            self.xrm
                .force_spill_var(v, tp, &mut self.longevity, &mut self.fm);
        } else {
            self.rm
                .force_spill_var(v, tp, &mut self.longevity, &mut self.fm);
        }
    }

    // ── Perform / emit ──

    /// Drain pending_moves as Move entries.
    fn flush_moves(&mut self, output: &mut Vec<RegAllocOp>) {
        // Operand moves and spills share one queue per register manager and
        // run in the order they were queued. Integer and float registers do
        // not alias, so one manager's queue flushes before the other's.
        // `pending_moves` is the out-of-line immediate materialisation from
        // `gc_offset_loc`, queued after those operand moves.
        for (src, dst) in self.rm.spill_moves.drain(..) {
            output.push(RegAllocOp::Move { src, dst });
        }
        for (src, dst) in self.xrm.spill_moves.drain(..) {
            output.push(RegAllocOp::Move { src, dst });
        }
        for (src, dst) in self.pending_moves.drain(..) {
            output.push(RegAllocOp::Move { src, dst });
        }
    }

    /// x86/regalloc.py perform
    pub(crate) fn perform(
        &mut self,
        op_index: usize,
        arglocs: impl AsRef<[Loc]>,
        result_loc: Option<Loc>,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.push_perform(op_index, arglocs.as_ref(), result_loc, None, output);
    }

    fn push_perform(
        &mut self,
        op_index: usize,
        arglocs: &[Loc],
        result_loc: Option<Loc>,
        gcmap: Option<usize>,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.flush_moves(output);
        match *arglocs {
            [loc] => output.push(RegAllocOp::Perform1 {
                op_index,
                loc,
                result_loc,
                gcmap,
            }),
            _ => {
                let (arglocs_start, arglocs_len) = self.push_arglocs(arglocs);
                output.push(RegAllocOp::Perform {
                    op_index,
                    arglocs_start,
                    arglocs_len,
                    result_loc,
                    gcmap,
                });
            }
        }
    }

    fn perform_with_gcmap_ptr(
        &mut self,
        op_index: usize,
        arglocs: impl AsRef<[Loc]>,
        result_loc: Option<Loc>,
        gcmap: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.push_perform(op_index, arglocs.as_ref(), result_loc, Some(gcmap), output);
    }

    /// Variant of `perform` that captures the current regalloc state as a
    /// GC bitmap, for collecting-call ops such as CallMallocNursery. The
    /// emit path stores the returned pointer into `jf_gcmap` before the
    /// slow-path call so the collector can trace live Refs pinned to the
    /// jitframe.
    fn perform_with_gcmap(
        &mut self,
        op_index: usize,
        arglocs: impl AsRef<[Loc]>,
        result_loc: Option<Loc>,
        output: &mut Vec<RegAllocOp>,
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
        self.push_perform(op_index, arglocs.as_ref(), result_loc, Some(gcmap), output);
    }

    /// x86/regalloc.py perform_guard
    fn perform_guard(
        &mut self,
        op: &Op,
        op_index: usize,
        arglocs: impl AsRef<[Loc]>,
        result_loc: Option<Loc>,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.flush_moves(output);
        let (faillocs_start, faillocs_len) = self.locs_for_fail(op);
        self.push_perform_guard(
            op_index,
            arglocs.as_ref(),
            result_loc,
            faillocs_start,
            faillocs_len,
            output,
        );
        // self.possibly_free_vars(guard_op.getfailargs())
        if let Some(fail_args) = op.guard_fail_args() {
            self._possibly_free_j2_vars(fail_args.iter().map(|a| a.to_opref()));
        }
    }

    fn push_perform_guard(
        &mut self,
        op_index: usize,
        arglocs: &[Loc],
        result_loc: Option<Loc>,
        faillocs_start: u32,
        faillocs_len: u32,
        output: &mut Vec<RegAllocOp>,
    ) {
        match *arglocs {
            [loc] => output.push(RegAllocOp::PerformGuard1 {
                op_index,
                loc,
                result_loc,
                faillocs_start,
                faillocs_len,
            }),
            _ => {
                let (arglocs_start, arglocs_len) = self.push_arglocs(arglocs);
                output.push(RegAllocOp::PerformGuard {
                    op_index,
                    arglocs_start,
                    arglocs_len,
                    result_loc,
                    faillocs_start,
                    faillocs_len,
                });
            }
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
            if self.opref_type(v) == Some(Type::Ref) && self.rm.is_still_alive(v, &self.longevity) {
                assert!(!noregs);
                let val = core_reg_index(reg).expect("gcmap: reg not in managed core regs");
                gcmap_set_bit(gcmap, val);
            }
        }
        // regalloc.py BindingsIterItems; regalloc.py bindings_iteritems.
        let mut index = 0;
        while index < self.fm.current_frame_depth {
            let Some(v) = self.fm.boxes_in_frame[index] else {
                index += 1;
                continue;
            };
            let loc = self
                .longevity
                .get(v)
                .and_then(|lifetime| lifetime.current_frame_loc)
                .expect("gcmap frame binding without current_frame_loc");
            debug_assert_eq!(FrameManager::get_loc_index(&loc), index);
            let size = FrameManager::frame_size(self.tp(v));
            if self.opref_type(v) == Some(Type::Ref) && self.rm.is_still_alive(v, &self.longevity) {
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
        arglocs: Vec<Loc>,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.flush_moves(output);
        output.push(RegAllocOp::PerformDiscard { op_index, arglocs });
    }

    fn perform_discard_gc_store(
        &mut self,
        op_index: usize,
        value: Loc,
        base: Loc,
        ofs: Loc,
        size: i64,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.flush_moves(output);
        output.push(RegAllocOp::PerformDiscardGcStore {
            op_index,
            value,
            base,
            ofs,
            size,
        });
    }

    fn perform_gc_load(
        &mut self,
        op_index: usize,
        base: Loc,
        ofs: Loc,
        res: Loc,
        nsize: i64,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.flush_moves(output);
        output.push(RegAllocOp::PerformGcLoad {
            op_index,
            base,
            ofs,
            res,
            nsize,
        });
    }

    /// Slice of [`Self::faillocs_arena`] recorded for one guard.
    pub fn faillocs(&self, start: u32, len: u32) -> &[Option<Loc>] {
        let start = start as usize;
        &self.faillocs_arena[start..start + len as usize]
    }

    /// Slice of [`Self::arglocs_arena`] recorded for one `perform`.
    pub fn arglocs(&self, start: u32, len: u32) -> &[Loc] {
        let start = start as usize;
        &self.arglocs_arena[start..start + len as usize]
    }

    fn push_arglocs(&mut self, locs: &[Loc]) -> (u32, u32) {
        let start = self.arglocs_arena.len() as u32;
        self.arglocs_arena.extend_from_slice(locs);
        (start, locs.len() as u32)
    }

    /// x86/regalloc.py locs_for_fail
    pub fn locs_for_fail(&mut self, guard_op: &Op) -> (u32, u32) {
        let Some(fail_args) = guard_op.guard_fail_args() else {
            return (self.faillocs_arena.len() as u32, 0);
        };
        let start = self.faillocs_arena.len();
        for arg in fail_args.iter() {
            self.push_failloc(arg.to_opref());
        }
        (start as u32, (self.faillocs_arena.len() - start) as u32)
    }

    fn push_failloc(&mut self, arg: OpRef) {
        if arg.is_none() {
            self.faillocs_arena.push(None);
            return;
        }
        let tp = self.tp(arg);
        // x86/regalloc.py `RegAlloc.locs_for_fail` calls `self.loc(arg)`:
        // a box reused from ResumeDataLoopMemo may be live only in this
        // guard and therefore need its frame binding allocated here.
        let loc = self.loc(arg, tp);
        self.faillocs_arena.push(Some(loc));
    }

    // ── Type resolution ──

    /// RPython reads `box.type` directly (history.py ConstInt /
    /// :261 ConstFloat / :307 ConstPtr pin `type` at construction); pyre
    /// delegates to `OpTypeIndex`. A type-less result is only ever a
    /// regalloc-scratch `TempVar` (GP-integer), which [`box_type_or_temp`]
    /// resolves to `Type::Int` while debug-asserting the `_check_type`
    /// (regalloc.py) TempVar exemption.
    pub(crate) fn tp(&self, v: OpRef) -> Type {
        box_type_or_temp(self.opref_type(v), v)
    }

    /// Extract the integer value of a constant OpRef (like RPython's getint()).
    ///
    /// The constants map is keyed by `OpRef.0` (raw value, WITH the
    /// CONST_BIT tag preserved) — see `convert_to_imm` at regalloc.rs
    /// line 1475 and the fail-args handling at line 2079. This helper
    /// must use the same key convention.
    #[track_caller]
    pub(crate) fn const_value(&self, v: OpRef) -> i64 {
        const_bits_or_panic(v, &self.constants, "const_value")
    }

    // ── walk_operations + consider_* ──

    /// x86/regalloc.py walk_operations — main dispatch loop.
    ///
    /// `MachineDataBlockWrapper.malloc_aligned` → `_allocate_next_block` →
    /// `open_malloc` raises `MemoryError`. That becomes
    /// `BackendError::CompilationFailed` for `assemble_loop` / `assemble_bridge`.
    pub fn walk_operations(&mut self) -> Result<Vec<RegAllocOp>, BackendError> {
        self.faillocs_arena.clear();
        self.arglocs_arena.clear();
        // One reserve for the walk. Size to the failargs already on
        // the ops so `locs_for_fail` does not grow mid-guard. `ops * 8`
        // over-reserved a 71-op bridge by ~9 KiB.
        let failarg_slots: usize = self
            .operations
            .iter()
            .filter_map(|op| op.guard_fail_args().map(|fa| fa.len()))
            .sum();
        self.faillocs_arena.reserve(failarg_slots);
        // Enough for the first few 2-loc performs without a 2.8 KiB
        // unused tail per bridge (ops*2 over-reserved and raised B/char).
        self.arglocs_arena.reserve(16);
        let operations: &'a [OpRc] = self.operations;
        let inputargs: &'a [InputArgRc] = self.inputargs;
        // Take the lowering plan so dispatch can borrow each LirOp without cloning it.
        let j2_ops = std::mem::take(&mut self.j2_ops);
        let mut output = Vec::with_capacity(operations.len());

        for (i, op) in operations.iter().enumerate() {
            self.rm.position = i as i32;
            self.xrm.position = i as i32;

            // x86/regalloc.py:383-386 skip dead ops
            if op.opcode.has_no_side_effect() && !self.longevity.contains(op.pos().get()) {
                self._free_op_vars(op);
                output.push(RegAllocOp::Skip);
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

            // x86/regalloc.py:390 dispatch to consider_* method.
            // Tests may inject a replacement `j2_ops` plan; production
            // lowers one op on the stack, matching the upstream walk
            // that never materializes a parallel IR.
            if let Some(j2_op) = j2_ops.get(i) {
                self._dispatch_j2(j2_op, op, i, &mut output);
                self._free_j2_op_vars(j2_op, op);
            } else {
                let j2_op = crate::j2plan::lower_op(op);
                self._dispatch_j2(&j2_op, op, i, &mut output);
                self._free_j2_op_vars(&j2_op, op);
            }
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
            let opref = iarg.opref();
            self.possibly_free_var(opref, iarg.tp.get());
        }

        Ok(output)
    }

    /// Free args and result of an op (x86/regalloc.py:308).
    fn _free_op_vars(&mut self, op: &Op) {
        for arg in op.args_slice().iter() {
            if !arg.is_constant() && !arg.is_none() {
                let arg = arg.to_opref();
                let tp = self.tp(arg);
                self.possibly_free_var(arg, tp);
            }
        }
        if let Some(fail_args) = op.guard_fail_args() {
            for arg in fail_args.iter() {
                if !arg.is_constant() && !arg.is_none() {
                    let arg = arg.to_opref();
                    let tp = self.tp(arg);
                    self.possibly_free_var(arg, tp);
                }
            }
        }
        // Free the result itself if it's dead
        if !op.pos().get().is_none() && !op.pos().get().is_constant() {
            let tp = op.opcode.result_type();
            if tp != Type::Void {
                self.possibly_free_var(op.pos().get(), tp);
            }
        }
    }

    /// Free a lowered operation's uses, fail args, and result in order.
    fn _free_j2_op_vars(&mut self, j2_op: &LirOp, raw_op: &Op) {
        let (def, def_tp) = match j2_op {
            LirOp::Label { args } | LirOp::Jump { args } | LirOp::Finish { args } => {
                self._possibly_free_j2_vars(args.iter().copied());
                (None, raw_op.opcode.result_type())
            }
            LirOp::IntBin { dst, lhs, rhs, .. } | LirOp::IntCmp { dst, lhs, rhs, .. } => {
                self._possibly_free_j2_vars([*lhs, *rhs].into_iter());
                (Some(*dst), Type::Int)
            }
            LirOp::IntUnary { dst, arg, .. } => {
                self._possibly_free_j2_vars(std::iter::once(*arg));
                (Some(*dst), Type::Int)
            }
            // `perform_guard` already freed `getfailargs()`.
            LirOp::Guard { args, .. } => {
                self._possibly_free_j2_vars(args.iter().copied());
                (None, raw_op.opcode.result_type())
            }
            LirOp::Load {
                dst,
                base,
                offset,
                index,
                scale,
                size,
                ..
            } => {
                self._possibly_free_j2_vars(
                    std::iter::once(*base)
                        .chain(offset.iter().copied())
                        .chain(index.iter().copied())
                        .chain(scale.iter().copied())
                        .chain(size.iter().copied()),
                );
                (Some(*dst), raw_op.opcode.result_type())
            }
            LirOp::Store {
                base,
                offset,
                index,
                scale,
                value,
                size,
                ..
            } => {
                self._possibly_free_j2_vars(
                    std::iter::once(*base)
                        .chain(offset.iter().copied())
                        .chain(index.iter().copied())
                        .chain(scale.iter().copied())
                        .chain(std::iter::once(*value))
                        .chain(size.iter().copied()),
                );
                (None, raw_op.opcode.result_type())
            }
            LirOp::Call { dst, args, .. } => {
                self._possibly_free_j2_vars(args.iter().copied());
                (*dst, raw_op.opcode.result_type())
            }
            LirOp::Opcode {
                dst,
                args,
                fail_args,
                ..
            } => {
                self._possibly_free_j2_vars(args.iter().copied());
                self._possibly_free_j2_vars(fail_args.iter().copied());
                (*dst, raw_op.opcode.result_type())
            }
        };
        if let Some(dst) = def
            && !dst.is_none()
            && !dst.is_constant()
            && def_tp != Type::Void
        {
            self.possibly_free_var(dst, def_tp);
        }
    }

    /// Free lowered operands in order without materializing a temporary use list.
    fn _possibly_free_j2_vars(&mut self, uses: impl Iterator<Item = OpRef>) {
        for arg in uses {
            if !arg.is_constant() && !arg.is_none() {
                let tp = self.tp(arg);
                self.possibly_free_var(arg, tp);
            }
        }
    }

    fn _dispatch_j2(&mut self, j2_op: &LirOp, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        match j2_op {
            LirOp::Label { args } => self.consider_label_j2(args, op, i, output),
            LirOp::Jump { args } => self.consider_jump_j2(args, op, i, output),
            LirOp::Finish { args } => self.consider_finish_j2(args, i, output),
            LirOp::IntBin {
                kind,
                dst,
                lhs,
                rhs,
            } => self._dispatch_j2_int_bin(*kind, *dst, *lhs, *rhs, i, output),
            LirOp::IntUnary { kind, dst, arg } => {
                self._dispatch_j2_int_unary(*kind, *dst, *arg, i, output)
            }
            LirOp::IntCmp { dst, lhs, rhs, .. } => {
                self.consider_compop_j2(*dst, *lhs, *rhs, i, output)
            }
            LirOp::Guard {
                kind,
                args,
                fail_args: _,
            } => self._dispatch_j2_guard(*kind, args, op, i, output),
            LirOp::Load {
                kind,
                dst,
                base,
                offset,
                index,
                scale,
                size,
            } => self._dispatch_j2_load(
                *kind, *dst, *base, *offset, *index, *scale, *size, op, i, output,
            ),
            LirOp::Store {
                kind,
                base,
                offset,
                index,
                scale,
                value,
                size,
            } => self._dispatch_j2_store(
                *kind, *base, *offset, *index, *scale, *value, *size, op, i, output,
            ),
            LirOp::Call { opcode, dst, args } => {
                self._dispatch_j2_call(*opcode, *dst, args, op, i, output)
            }
            LirOp::Opcode {
                opcode,
                dst,
                args,
                fail_args,
            } => self._dispatch_j2_opcode(*opcode, *dst, args, fail_args, op, i, output),
        }
    }

    fn _dispatch_j2_int_bin(
        &mut self,
        kind: IntBinKind,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        match kind {
            IntBinKind::Add => self.consider_int_add_j2(dst, lhs, rhs, i, output),
            IntBinKind::Sub => self.consider_int_sub_j2(dst, lhs, rhs, i, output),
            // x86 routes AddOvf/SubOvf to _consider_binop{,_symm} because
            // LEA can't set flags; aarch64 routes to prepare_int_ri /
            // prepare_op_int_sub because adds/subs accept #imm12. Each
            // backend implements consider_int_{add,sub}_ovf_j2 with the
            // arch-correct semantics.
            IntBinKind::AddOvf => self.consider_int_add_ovf_j2(dst, lhs, rhs, i, output),
            IntBinKind::SubOvf => self.consider_int_sub_ovf_j2(dst, lhs, rhs, i, output),
            IntBinKind::Mul
            | IntBinKind::MulOvf
            | IntBinKind::And
            | IntBinKind::Or
            | IntBinKind::Xor => self.consider_binop_symm_j2(dst, lhs, rhs, i, output),
            IntBinKind::LShift | IntBinKind::RShift | IntBinKind::URShift => {
                self.consider_int_lshift_j2(dst, lhs, rhs, i, output)
            }
        }
    }

    fn _dispatch_j2_int_unary(
        &mut self,
        kind: IntUnaryKind,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        match kind {
            IntUnaryKind::Neg | IntUnaryKind::Invert => {
                self.consider_unary_int_j2(dst, arg, i, output)
            }
            // x86/regalloc.py `consider_int_is_true` — arg stays in
            // memory/imm (no force-into-reg), result goes through
            // `force_allocate_reg_or_cc`. AArch64 wraps to its 3-op unary
            // handler since `neg`/`mvn` already keep arg and result in
            // distinct registers.
            IntUnaryKind::IsTrue | IntUnaryKind::IsZero => {
                self.consider_int_is_true_j2(dst, arg, i, output)
            }
        }
    }

    fn _dispatch_j2_guard(
        &mut self,
        kind: GuardKind,
        args: &[OpRef],
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        match kind {
            GuardKind::True | GuardKind::False | GuardKind::NonNull | GuardKind::IsNull => {
                if let Some(&arg) = args.first() {
                    self.consider_guard_cc_j2(arg, op, i, output);
                } else {
                    output.push(RegAllocOp::Skip);
                }
            }
            GuardKind::Value => {
                if args.len() >= 2 {
                    self.consider_guard_value_j2(args[0], args[1], op, i, output);
                } else {
                    output.push(RegAllocOp::Skip);
                }
            }
            GuardKind::Class | GuardKind::GcType | GuardKind::NonNullClass => {
                if args.len() >= 2 {
                    self.consider_guard_class_j2(args[0], args[1], op, i, output);
                } else {
                    output.push(RegAllocOp::Skip);
                }
            }
            GuardKind::Subclass => {
                if args.len() >= 2 {
                    self.consider_guard_subclass_j2(args[0], args[1], op, i, output);
                } else {
                    output.push(RegAllocOp::Skip);
                }
            }
            GuardKind::IsObject => {
                if let Some(&arg) = args.first() {
                    self.consider_guard_is_object_j2(arg, op, i, output);
                } else {
                    output.push(RegAllocOp::Skip);
                }
            }
            GuardKind::NotForced if op.opcode == OpCode::GuardNotForced2 => {
                self.consider_guard_not_forced_2_j2(op, i, output)
            }
            GuardKind::NoException
            | GuardKind::NoOverflow
            | GuardKind::Overflow
            | GuardKind::NotInvalidated
            | GuardKind::FutureCondition
            | GuardKind::NotForced
            | GuardKind::AlwaysFails => self.consider_guard_no_args_j2(op, i, output),
            GuardKind::Exception => self.consider_guard_exception_j2(args, op, i, output),
            GuardKind::Other(_) => self.consider_guard_no_args_j2(op, i, output),
        }
    }

    fn _dispatch_j2_load(
        &mut self,
        kind: LoadKind,
        dst: OpRef,
        base: OpRef,
        offset: Option<OpRef>,
        index: Option<OpRef>,
        scale: Option<OpRef>,
        size: Option<OpRef>,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        match kind {
            LoadKind::Gc | LoadKind::Raw => {
                if let Some(offset) = offset {
                    self.consider_gc_load_j2(dst, base, offset, size, op, i, output);
                } else {
                    output.push(RegAllocOp::Skip);
                }
            }
            LoadKind::GcIndexed => {
                if let (Some(index), Some(scale), Some(offset), Some(size)) =
                    (index, scale, offset, size)
                {
                    self.consider_gc_load_indexed_j2(
                        dst, base, index, scale, offset, size, op, i, output,
                    );
                } else {
                    output.push(RegAllocOp::Skip);
                }
            }
        }
    }

    fn _dispatch_j2_store(
        &mut self,
        kind: StoreKind,
        base: OpRef,
        offset: Option<OpRef>,
        index: Option<OpRef>,
        scale: Option<OpRef>,
        value: OpRef,
        size: Option<OpRef>,
        _op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        match kind {
            StoreKind::Gc | StoreKind::Raw => {
                if let Some(offset) = offset {
                    self.consider_gc_store_j2(base, offset, value, size, i, output);
                } else {
                    output.push(RegAllocOp::Skip);
                }
            }
            StoreKind::GcIndexed => {
                if let (Some(index), Some(scale), Some(offset), Some(size)) =
                    (index, scale, offset, size)
                {
                    self.consider_gc_store_indexed_j2(
                        base, index, value, scale, offset, size, i, output,
                    );
                } else {
                    output.push(RegAllocOp::Skip);
                }
            }
        }
    }

    fn _dispatch_j2_call(
        &mut self,
        opcode: OpCode,
        dst: Option<OpRef>,
        args: &[OpRef],
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        match opcode {
            OpCode::CallI
            | OpCode::CallR
            | OpCode::CallF
            | OpCode::CallN
            | OpCode::CallPureI
            | OpCode::CallPureR
            | OpCode::CallPureF
            | OpCode::CallPureN
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN
            | OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN => {
                // x86/regalloc.py `_consider_real_call`: OS_MATH_SQRT is
                // `_consider_math_sqrt`, not a real call.
                if is_math_sqrt_call(op) {
                    let dst = dst.unwrap_or(op.pos().get());
                    let src = args.get(1).copied().unwrap_or(dst);
                    self.consider_math_sqrt_j2(dst, src, i, output);
                } else {
                    self.consider_call_j2(dst, args, op, i, output, opcode.is_call_may_force(), 1);
                }
            }
            OpCode::CallReleaseGilI | OpCode::CallReleaseGilF | OpCode::CallReleaseGilN => {
                self.consider_call_j2(dst, args, op, i, output, true, 2);
            }
            OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerF
            | OpCode::CallAssemblerN => {
                self.consider_call_assembler_j2(dst, args, op, i, output);
            }
            OpCode::CondCallN => self.consider_cond_call_n(args, i, output),
            OpCode::CondCallValueI | OpCode::CondCallValueR => {
                self.consider_cond_call_value_j2(dst, args, op, i, output);
            }
            OpCode::CallMallocNursery | OpCode::CallMallocNurseryHeaderless => {
                self.consider_call_malloc_nursery_j2(dst, args, i, output)
            }
            OpCode::CallMallocNurseryVarsizeFrame => {
                self.consider_call_malloc_nursery_varsize_frame_j2(dst, args, i, output);
            }
            OpCode::CallMallocNurseryVarsize | OpCode::CallMallocNurseryVarsizeHeaderless => {
                self.consider_call_malloc_nursery_varsize_j2(dst, args, i, output);
            }
            OpCode::CheckMemoryError => self.consider_check_memory_error_j2(args, i, output),
            OpCode::RecordKnownResult => output.push(RegAllocOp::Skip),
            OpCode::New
            | OpCode::NewWithVtable
            | OpCode::NewArray
            | OpCode::NewArrayClear
            | OpCode::Newstr
            | OpCode::Newunicode => {
                self.consider_raw_call_like_j2(dst, args, op, i, output, SAVE_DEFAULT_REGS);
            }
            _ => output.push(RegAllocOp::Skip),
        }
    }

    fn _dispatch_j2_opcode(
        &mut self,
        opcode: OpCode,
        dst: Option<OpRef>,
        args: &[OpRef],
        _fail_args: &[OpRef],
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        match opcode {
            OpCode::NurseryPtrIncrement if args.len() >= 2 => {
                self.consider_int_add_j2(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    args[1],
                    i,
                    output,
                );
            }
            OpCode::IntForceGeZero if !args.is_empty() => {
                self.consider_int_force_ge_zero_j2(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    i,
                    output,
                );
            }
            OpCode::IntFloorDiv | OpCode::IntMod if args.len() >= 2 => {
                self.consider_binop_j2(dst.unwrap_or(op.pos().get()), args[0], args[1], i, output);
            }
            OpCode::UintMulHigh if args.len() >= 2 => {
                self.consider_uint_mul_high_j2(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    args[1],
                    i,
                    output,
                );
            }
            OpCode::IntSignext if args.len() >= 2 => {
                self.consider_int_signext_j2(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    args[1],
                    i,
                    output,
                );
            }
            OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv
                if args.len() >= 2 =>
            {
                self.consider_float_op_j2(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    args[1],
                    i,
                    output,
                );
            }
            OpCode::FloatNeg | OpCode::FloatAbs if !args.is_empty() => {
                self.consider_float_unary_j2(dst.unwrap_or(op.pos().get()), args[0], i, output);
            }
            // aarch64/regalloc.py `prepare_op_cast_float_to_singlefloat` and
            // `prepare_op_cast_singlefloat_to_float` are both `prepare_unary`:
            // the arg in a register of its own kind, then
            // `force_allocate_reg(op)` in a register of the result's kind.
            // CAST_FLOAT_TO_SINGLEFLOAT is ('f','i'); CAST_SINGLEFLOAT_TO_FLOAT
            // is ('i','f') (`resoperation.py` CastOp.cast_types).
            #[cfg(not(target_arch = "x86_64"))]
            OpCode::CastFloatToSinglefloat if !args.is_empty() => {
                self.consider_cast_float_to_int_j2(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    i,
                    output,
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            OpCode::CastSinglefloatToFloat if !args.is_empty() => {
                self.consider_cast_int_to_float_j2(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    i,
                    output,
                );
            }
            // x86/regalloc.py `consider_cast_float_to_singlefloat`.
            #[cfg(target_arch = "x86_64")]
            OpCode::CastFloatToSinglefloat if !args.is_empty() => {
                self.consider_cast_float_to_singlefloat_j2(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    i,
                    output,
                );
            }
            // x86/regalloc.py `consider_cast_singlefloat_to_float`,
            // an alias of `consider_cast_int_to_float`.
            #[cfg(target_arch = "x86_64")]
            OpCode::CastSinglefloatToFloat if !args.is_empty() => {
                self.consider_cast_int_to_float_j2(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    i,
                    output,
                );
            }
            OpCode::FloatLt
            | OpCode::FloatLe
            | OpCode::FloatEq
            | OpCode::FloatNe
            | OpCode::FloatGt
            | OpCode::FloatGe
                if args.len() >= 2 =>
            {
                self.consider_float_cmp_j2(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    args[1],
                    i,
                    output,
                );
            }
            OpCode::CastIntToFloat if !args.is_empty() => {
                self.consider_cast_int_to_float_j2(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    i,
                    output,
                );
            }
            OpCode::CastFloatToInt if !args.is_empty() => {
                self.consider_cast_float_to_int_j2(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    i,
                    output,
                );
            }
            // Same bank split as the non-j2 arm — see the citation there.
            OpCode::ConvertFloatBytesToLonglong if !args.is_empty() => {
                self.consider_cast_float_to_int_j2(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    i,
                    output,
                );
            }
            OpCode::ConvertLonglongBytesToFloat if !args.is_empty() => {
                self.consider_cast_int_to_float_j2(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    i,
                    output,
                );
            }
            OpCode::CastPtrToInt
            | OpCode::CastIntToPtr
            | OpCode::CastOpaquePtr
            | OpCode::SameAsI
            | OpCode::SameAsR
            | OpCode::SameAsF
            | OpCode::LoadFromGcTable
            | OpCode::VirtualRefR
                if !args.is_empty() =>
            {
                self.consider_same_as_j2(dst.unwrap_or(op.pos().get()), args[0], op, i, output);
            }
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetfieldRawI
            | OpCode::GetfieldRawR
            | OpCode::GetfieldRawF
            | OpCode::ArraylenGc
            | OpCode::Strlen
            | OpCode::Unicodelen
                if !args.is_empty() =>
            {
                self.consider_getfield_j2(dst.unwrap_or(op.pos().get()), args[0], op, i, output);
            }
            OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcR
            | OpCode::GetarrayitemGcF
            | OpCode::GetarrayitemGcPureI
            | OpCode::GetarrayitemGcPureR
            | OpCode::GetarrayitemGcPureF
            | OpCode::GetarrayitemRawI
            | OpCode::GetarrayitemRawR
            | OpCode::GetarrayitemRawF
            | OpCode::Strgetitem
            | OpCode::Unicodegetitem
            | OpCode::GetinteriorfieldGcI
            | OpCode::GetinteriorfieldGcR
            | OpCode::GetinteriorfieldGcF
                if args.len() >= 2 =>
            {
                self.consider_getarrayitem_j2(
                    dst.unwrap_or(op.pos().get()),
                    args[0],
                    args[1],
                    op,
                    i,
                    output,
                );
            }
            OpCode::SetfieldGc | OpCode::SetfieldRaw if args.len() >= 2 => {
                self.consider_setfield_j2(args[0], args[1], i, output);
            }
            OpCode::SetarrayitemGc
            | OpCode::SetarrayitemRaw
            | OpCode::SetinteriorfieldGc
            | OpCode::SetinteriorfieldRaw
                if args.len() >= 3 =>
            {
                self.consider_setarrayitem_j2(args[0], args[1], args[2], i, output);
            }
            OpCode::ZeroArray if args.len() >= 3 => {
                self.consider_zero_array_j2(args, op, i, output);
            }
            OpCode::Strsetitem | OpCode::Unicodesetitem if args.len() >= 3 => {
                self.consider_discard_3args_j2(args, i, output);
            }
            OpCode::Copystrcontent | OpCode::Copyunicodecontent => {
                self.consider_discard_nargs_j2(args, i, output);
            }
            OpCode::New
            | OpCode::NewWithVtable
            | OpCode::NewArray
            | OpCode::NewArrayClear
            | OpCode::Newstr
            | OpCode::Newunicode => {
                self.consider_raw_call_like_j2(dst, args, op, i, output, SAVE_DEFAULT_REGS);
            }
            OpCode::ForceToken => {
                self.consider_force_token_j2(dst.unwrap_or(op.pos().get()), i, output)
            }
            OpCode::LoadEffectiveAddress if args.len() >= 4 => {
                self.consider_load_effective_address_j2(
                    dst.unwrap_or(op.pos().get()),
                    args,
                    i,
                    output,
                );
            }
            OpCode::SaveException | OpCode::SaveExcClass => {
                self.consider_no_arg_result_j2(dst.unwrap_or(op.pos().get()), op, i, output);
            }
            OpCode::RestoreException if args.len() >= 2 => {
                self.consider_restore_exception_j2(args, i, output);
            }
            OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                self.consider_cond_call_gc_wb_j2(args, i, output);
            }
            OpCode::JitDebug
            | OpCode::DebugMergePoint
            | OpCode::RecordExactClass
            | OpCode::RecordExactValueR
            | OpCode::RecordExactValueI
            | OpCode::RecordKnownResult
            | OpCode::QuasiimmutField
            | OpCode::AssertNotNone
            | OpCode::Keepalive
            | OpCode::EnterPortalFrame
            | OpCode::LeavePortalFrame
            | OpCode::IncrementDebugCounter
            | OpCode::VirtualRefFinish
            | OpCode::ForceSpill => output.push(RegAllocOp::Skip),
            _ => output.push(RegAllocOp::Skip),
        }
    }

    /// x86/regalloc.py oplist dispatch. Used only as a guard rail if the
    /// j2 plan is unexpectedly missing an operation.
    fn _dispatch_legacy(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        match op.opcode {
            // ── Integer binary (symmetric) ──
            OpCode::IntMul
            | OpCode::IntAnd
            | OpCode::IntOr
            | OpCode::IntXor
            | OpCode::IntMulOvf
            | OpCode::IntAddOvf => {
                self.consider_binop_symm(op, i, output);
            }
            // ── Integer binary (non-symmetric) ──
            OpCode::IntSubOvf => {
                self.consider_binop(op, i, output);
            }
            // x86/regalloc.py consider_int_sub — LEA when -const fits 32 bits.
            OpCode::IntSub => {
                self.consider_int_sub(op, i, output);
            }
            // ── Integer add (may use LEA) ──
            OpCode::IntAdd | OpCode::NurseryPtrIncrement => {
                self.consider_int_add(op, i, output);
            }
            // ── Shifts ──
            OpCode::IntLshift | OpCode::IntRshift | OpCode::UintRshift => {
                self.consider_int_lshift(op, i, output);
            }
            // ── Unary integer ──
            OpCode::IntNeg | OpCode::IntInvert => {
                self.consider_unary_int(op, i, output);
            }
            // ── Integer comparisons ──
            OpCode::IntLt
            | OpCode::IntLe
            | OpCode::IntGt
            | OpCode::IntGe
            | OpCode::IntEq
            | OpCode::IntNe
            | OpCode::UintLt
            | OpCode::UintLe
            | OpCode::UintGt
            | OpCode::UintGe
            | OpCode::PtrEq
            | OpCode::PtrNe
            | OpCode::InstancePtrEq
            | OpCode::InstancePtrNe => {
                self.consider_compop(op, i, output);
            }
            // ── Integer misc ──
            OpCode::IntIsTrue | OpCode::IntIsZero => {
                self.consider_unary_int(op, i, output);
            }
            OpCode::IntForceGeZero => {
                self.consider_int_force_ge_zero_j2(op.pos().get(), op.arg(0).to_opref(), i, output);
            }
            OpCode::IntFloorDiv | OpCode::IntMod => {
                self.consider_binop(op, i, output);
            }
            // x86: uses eax/edx. arm/aarch64: same shape as int_mul.
            OpCode::UintMulHigh => {
                self.consider_uint_mul_high(op, i, output);
            }
            OpCode::IntSignext => {
                self.consider_int_signext(op, i, output);
            }

            // ── Float binary ──
            OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv => {
                self.consider_float_op(op, i, output);
            }
            // ── Float unary ──
            OpCode::FloatNeg | OpCode::FloatAbs => {
                self.consider_float_unary(op, i, output);
            }
            // ── Float comparisons ──
            OpCode::FloatLt
            | OpCode::FloatLe
            | OpCode::FloatEq
            | OpCode::FloatNe
            | OpCode::FloatGt
            | OpCode::FloatGe => {
                self.consider_float_cmp(op, i, output);
            }
            // ── Casts ──
            OpCode::CastIntToFloat => {
                self.consider_cast_int_to_float(op, i, output);
            }
            OpCode::CastFloatToInt => {
                self.consider_cast_float_to_int(op, i, output);
            }
            OpCode::CastFloatToSinglefloat | OpCode::CastSinglefloatToFloat => {
                self.consider_float_unary(op, i, output);
            }
            // aarch64/regalloc.py:515-516 aliases both converts to the same
            // `prepare_unary` (:456-462) as `prepare_op_cast_float_to_int` /
            // `prepare_op_cast_int_to_float` (:502-503): the argument is read
            // through the manager owning the ARGUMENT's type and the result is
            // allocated through the manager owning the OP's type. x86 spells
            // the same split out at x86/regalloc.py consider_convert_float_bytes_to_longlong / :730-738
            // (`xrm.make_sure_var_in_reg` + `rm.force_allocate_reg`, and the
            // mirror image). These ops REINTERPRET the bits across the two
            // register files, so `consider_same_as` — which reads the argument
            // out of the RESULT's bank — reads the wrong register file.
            OpCode::ConvertFloatBytesToLonglong => {
                self.consider_cast_float_to_int(op, i, output);
            }
            OpCode::ConvertLonglongBytesToFloat => {
                self.consider_cast_int_to_float(op, i, output);
            }
            OpCode::CastPtrToInt | OpCode::CastIntToPtr | OpCode::CastOpaquePtr => {
                self.consider_same_as(op, i, output);
            }

            // ── Guards ──
            // x86/regalloc.py:440-443
            OpCode::GuardTrue
            | OpCode::VecGuardTrue
            | OpCode::GuardFalse
            | OpCode::VecGuardFalse
            | OpCode::GuardNonnull
            | OpCode::GuardIsnull => {
                self.consider_guard_cc(op, i, output);
            }
            OpCode::GuardValue => {
                self.consider_guard_value(op, i, output);
            }
            OpCode::GuardClass | OpCode::GuardNonnullClass | OpCode::GuardGcType => {
                self.consider_guard_class(op, i, output);
            }
            // x86/regalloc.py consider_guard_no_exception
            OpCode::GuardNoException
            | OpCode::GuardNoOverflow
            | OpCode::GuardOverflow
            | OpCode::GuardNotForced => {
                self.consider_guard_no_args(op, i, output);
            }
            OpCode::GuardNotForced2 => {
                self.consider_guard_not_forced_2(op, i, output);
            }
            // x86/regalloc.py consider_guard_not_invalidated
            OpCode::GuardNotInvalidated => {
                self.consider_guard_no_args(op, i, output);
            }
            // x86/regalloc.py consider_guard_exception
            OpCode::GuardException => {
                self.consider_guard_exception(op, i, output);
            }
            _ if op.opcode.is_guard() => {
                self.consider_guard_no_args(op, i, output);
            }

            // ── Same-as / identity ──
            OpCode::SameAsI
            | OpCode::SameAsR
            | OpCode::SameAsF
            | OpCode::LoadFromGcTable
            | OpCode::VirtualRefR => {
                self.consider_same_as(op, i, output);
            }

            // ── Memory loads ──
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetfieldRawI
            | OpCode::GetfieldRawR
            | OpCode::GetfieldRawF => {
                self.consider_getfield(op, i, output);
            }
            OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcR
            | OpCode::GetarrayitemGcF
            | OpCode::GetarrayitemGcPureI
            | OpCode::GetarrayitemGcPureR
            | OpCode::GetarrayitemGcPureF
            | OpCode::GetarrayitemRawI
            | OpCode::GetarrayitemRawR
            | OpCode::GetarrayitemRawF => {
                self.consider_getarrayitem(op, i, output);
            }
            OpCode::ArraylenGc => {
                self.consider_getfield(op, i, output);
            }
            OpCode::GcLoadI | OpCode::GcLoadR | OpCode::GcLoadF => {
                self.consider_gc_load(op, i, output);
            }
            OpCode::GcLoadIndexedI | OpCode::GcLoadIndexedR | OpCode::GcLoadIndexedF => {
                self.consider_gc_load_indexed(op, i, output);
            }
            OpCode::RawLoadI | OpCode::RawLoadF => {
                self.consider_gc_load(op, i, output);
            }
            OpCode::GetinteriorfieldGcI
            | OpCode::GetinteriorfieldGcR
            | OpCode::GetinteriorfieldGcF => {
                self.consider_getinteriorfield(op, i, output);
            }

            // ── Memory stores ──
            OpCode::SetfieldGc | OpCode::SetfieldRaw => {
                self.consider_setfield(op, i, output);
            }
            OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
                self.consider_setarrayitem(op, i, output);
            }
            OpCode::GcStore => {
                self.consider_gc_store(op, i, output);
            }
            OpCode::GcStoreIndexed => {
                self.consider_gc_store_indexed(op, i, output);
            }
            OpCode::RawStore => {
                self.consider_gc_store(op, i, output);
            }
            OpCode::SetinteriorfieldGc | OpCode::SetinteriorfieldRaw => {
                self.consider_setinteriorfield(op, i, output);
            }
            OpCode::ZeroArray => self.consider_zero_array(op, i, output),
            OpCode::Strsetitem | OpCode::Unicodesetitem => {
                self.consider_discard_3args(op, i, output);
            }
            OpCode::Copystrcontent | OpCode::Copyunicodecontent => {
                self.consider_discard_nargs(op, i, output);
            }

            // ── Calls ──
            OpCode::CallI
            | OpCode::CallR
            | OpCode::CallF
            | OpCode::CallN
            | OpCode::CallPureI
            | OpCode::CallPureR
            | OpCode::CallPureF
            | OpCode::CallPureN
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN
            | OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN => {
                // x86/regalloc.py `_consider_real_call`: OS_MATH_SQRT is
                // `_consider_math_sqrt`, not a real call.
                if is_math_sqrt_call(op) {
                    self.consider_math_sqrt(op, i, output);
                } else {
                    self.consider_call(op, i, output, op.opcode.is_call_may_force(), 1);
                }
            }
            OpCode::CallReleaseGilI | OpCode::CallReleaseGilF | OpCode::CallReleaseGilN => {
                self.consider_call(op, i, output, true, 2);
            }
            OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerF
            | OpCode::CallAssemblerN => {
                self.consider_call_assembler(op, i, output);
            }
            OpCode::CondCallN => {
                let args: Vec<OpRef> = op.getarglist().iter().map(|a| a.to_opref()).collect();
                self.consider_cond_call_n(&args, i, output);
            }
            OpCode::CondCallValueI | OpCode::CondCallValueR => {
                let args: Vec<OpRef> = op.getarglist().iter().map(|a| a.to_opref()).collect();
                self.consider_cond_call_value_j2(Some(op.pos().get()), &args, op, i, output);
            }

            // ── Allocation ──
            OpCode::New
            | OpCode::NewWithVtable
            | OpCode::NewArray
            | OpCode::NewArrayClear
            | OpCode::Newstr
            | OpCode::Newunicode => {
                self.consider_raw_call_like(op, i, output, SAVE_DEFAULT_REGS);
            }
            // aarch64/regalloc.py prepare_op_call_malloc_nursery
            OpCode::CallMallocNursery | OpCode::CallMallocNurseryHeaderless => {
                self.consider_call_malloc_nursery(op, i, output);
            }
            OpCode::CallMallocNurseryVarsizeFrame => {
                self.consider_call_malloc_nursery_varsize_frame(op, i, output);
            }
            OpCode::CallMallocNurseryVarsize | OpCode::CallMallocNurseryVarsizeHeaderless => {
                self.consider_call_malloc_nursery_varsize(op, i, output);
            }
            OpCode::CheckMemoryError => {
                self.consider_check_memory_error(op, i, output);
            }

            // ── Control flow ──
            OpCode::Jump => {
                self.consider_jump(op, i, output);
            }
            OpCode::Label => {
                self.consider_label(op, i, output);
            }
            OpCode::Finish => {
                self.consider_finish(op, i, output);
            }

            // ── Misc result ops ──
            OpCode::ForceToken => {
                self.consider_force_token(op, i, output);
            }
            OpCode::Strlen | OpCode::Unicodelen => {
                self.consider_getfield(op, i, output);
            }
            OpCode::Strgetitem | OpCode::Unicodegetitem => {
                self.consider_getarrayitem(op, i, output);
            }
            OpCode::LoadEffectiveAddress => {
                self.consider_load_effective_address(op, i, output);
            }
            OpCode::SaveException | OpCode::SaveExcClass => {
                self.consider_no_arg_result(op, i, output);
            }
            // x86/regalloc.py consider_restore_exception
            OpCode::RestoreException => {
                self.consider_restore_exception(op, i, output);
            }

            // ── No-ops ──
            OpCode::JitDebug
            | OpCode::DebugMergePoint
            | OpCode::RecordExactClass
            | OpCode::RecordExactValueR
            | OpCode::RecordExactValueI
            | OpCode::RecordKnownResult
            | OpCode::QuasiimmutField
            | OpCode::AssertNotNone
            | OpCode::Keepalive
            | OpCode::EnterPortalFrame
            | OpCode::LeavePortalFrame
            | OpCode::IncrementDebugCounter
            | OpCode::VirtualRefFinish
            | OpCode::ForceSpill => {
                output.push(RegAllocOp::Skip);
            }
            // aarch64/regalloc.py prepare_op_cond_call_gc_wb
            OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                let args: Vec<OpRef> =
                    op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
                let mut arglocs = Vec::new();
                for (idx, arg) in op.args_slice().iter().enumerate() {
                    let arg = arg.to_opref();
                    let tp = if idx == 0 { Type::Ref } else { Type::Int };
                    arglocs.push(self.make_sure_var_in_reg(arg, tp, &args, None, false));
                }
                self.perform_discard(i, arglocs, output);
            }

            _ => {
                // Unknown op: skip
                output.push(RegAllocOp::Skip);
            }
        }
    }

    // ── consider_* methods ──

    // Integer binop / unary `consider_*_j2` methods live in the
    // arch-specific modules (`crate::aarch64::regalloc` /
    // `crate::x86::regalloc`) because the result-register coupling
    // is encoding-dependent: x86 forces `dst = lhs` (2-operand form),
    // AAPCS64 allocates the result independently (3-operand form).
    // Mirrors the upstream `rpython/jit/backend/{aarch64,x86}/
    // regalloc.py` split.

    fn consider_compop_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let mut arglocs = [self.loc(lhs, self.tp(lhs)), self.loc(rhs, self.tp(rhs))];
        let lhs_in_reg = self.rm.reg_bindings_contains(lhs, &self.longevity);
        let rhs_in_reg = self.rm.reg_bindings_contains(rhs, &self.longevity);
        if !lhs_in_reg && !rhs_in_reg && !lhs.is_constant() && !rhs.is_constant() {
            arglocs[0] = self.make_sure_var_in_reg(lhs, Type::Int, &[], None, false);
        }
        let ops_ref: &[OpRc] = self.operations;
        let result_loc = self.force_allocate_reg_or_cc(dst, ops_ref, i);
        self.perform(i, arglocs, Some(result_loc), output);
    }

    // `consider_uint_mul_high_j2` is arch-specific: aarch64 uses the
    // 3-operand `umulh`, x86 forces EAX/EDX pinning for `MUL`. See
    // the arch-specific impl blocks.

    /// x86/regalloc.py `consider_int_force_ge_zero`.
    /// The result is forbidden from the argument's register: `genop_int_force_ge_zero`
    /// does `MOV res, 0` before `CMOVNS res, src`.
    fn consider_int_force_ge_zero_j2(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let argloc = self.make_sure_var_in_reg(arg, Type::Int, &[], None, false);
        let resloc = self.force_allocate_reg(dst, Type::Int, &[arg], None, false);
        self.perform(i, [argloc], Some(Loc::Reg(resloc)), output);
    }

    fn consider_int_signext_j2(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        numbytes: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let argloc = self.loc(arg, Type::Int);
        let numbytesloc = self.loc(numbytes, Type::Int);
        let resloc = Loc::Reg(self.force_allocate_reg(dst, Type::Int, &[], None, false));
        self.perform(i, [argloc, numbytesloc], Some(resloc), output);
    }

    /// x86/regalloc.py _consider_binop_part
    fn _consider_binop_part(&mut self, op: &Op, symm: bool) -> (Loc, Loc) {
        let mut x = op.arg(0).to_opref();
        let mut y = op.arg(1).to_opref();
        let xloc = self.loc(x, self.tp(x));
        let mut argloc = self.loc(y, self.tp(y));

        // x86/regalloc.py:536-542 symmetry optimization:
        // if x is not in a reg, but y is, and x lives longer while y dies,
        // swap the role of x and y
        if symm && !xloc.is_reg() && argloc.is_reg() {
            let x_lives_longer = !self.longevity.contains(x)
                || self.longevity.get(x).unwrap().last_usage > self.rm.position;
            let y_dies = self
                .longevity
                .get(y)
                .map(|lt| lt.last_usage == self.rm.position)
                .unwrap_or(false);
            if x_lives_longer && y_dies {
                std::mem::swap(&mut x, &mut y);
                argloc = self.loc(y, self.tp(y));
            }
        }

        let tp = self.tp(x);
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        let loc = self.rm.force_result_in_reg(
            op.pos().get(),
            x,
            tp,
            &args,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        (loc, argloc)
    }

    /// x86/regalloc.py _consider_binop
    #[allow(dead_code)] // x86/regalloc.py consider_binop
    fn consider_binop(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let (loc, argloc) = self._consider_binop_part(op, false);
        self.perform(i, [loc, argloc], Some(loc), output);
    }

    /// x86/regalloc.py _consider_binop_symm
    #[allow(dead_code)] // x86/regalloc.py consider_binop_symm
    fn consider_binop_symm(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let (loc, argloc) = self._consider_binop_part(op, true);
        self.perform(i, [loc, argloc], Some(loc), output);
    }

    /// x86/regalloc.py _consider_lea
    fn _consider_lea(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let x = op.arg(0).to_opref();
        let loc = self.make_sure_var_in_reg(x, self.tp(x), &[], None, false);
        // make it possible to have argloc be == loc if x dies
        self.possibly_free_var(x, self.tp(x));
        let y = op.arg(1).to_opref();
        let argloc = self.loc(y, self.tp(y));
        let resloc = Loc::Reg(self.force_allocate_reg(op.pos().get(), Type::Int, &[], None, false));
        self.perform(i, [loc, argloc], Some(resloc), output);
    }

    /// x86/regalloc.py consider_int_add — LEA when const fits 32 bits.
    #[allow(dead_code)] // x86/regalloc.py consider_int_add
    fn consider_int_add(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let y = op.arg(1).to_opref();
        if y.is_constant() {
            let val = self.const_value(y);
            if fits_in_32bits(val) {
                return self._consider_lea(op, i, output);
            }
        }
        self.consider_binop_symm(op, i, output);
    }

    /// x86/regalloc.py consider_int_sub
    #[allow(dead_code)] // x86/regalloc.py consider_int_sub
    fn consider_int_sub(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let y = op.arg(1).to_opref();
        if y.is_constant() {
            let val = self.const_value(y);
            if fits_in_32bits(-val) {
                return self._consider_lea(op, i, output);
            }
        }
        self.consider_binop(op, i, output);
    }

    /// x86/regalloc.py consider_int_lshift (shift operations need ecx)
    /// arm/regalloc.py prepare_op_{int,uint}_rshift/int_lshift
    /// use the regular RI path with no fixed register.
    #[allow(dead_code)] // x86/regalloc.py consider_int_lshift
    fn consider_int_lshift(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        #[cfg(target_arch = "aarch64")]
        {
            self.consider_binop(op, i, output)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let y = op.arg(1).to_opref();
            let loc2 = if y.is_constant() {
                self.rm.convert_to_imm(y, &self.constants)
            } else {
                self.make_sure_var_in_reg(y, Type::Int, &[], Some(ECX), false)
            };
            let args: Vec<OpRef> =
                op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
            let loc1 = self.rm.force_result_in_reg(
                op.pos().get(),
                op.arg(0).to_opref(),
                Type::Int,
                &args,
                &mut self.longevity,
                &mut self.fm,
                &self.constants,
                &mut self.pending_moves,
            );
            self.perform(i, [loc1, loc2], Some(loc1), output);
        }
    }

    /// x86/regalloc.py int_neg / int_invert / int_is_true / int_is_zero / int_signext
    #[allow(dead_code)] // x86/regalloc.py consider_unary_int
    fn consider_unary_int(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        let loc = self.rm.force_result_in_reg(
            op.pos().get(),
            op.arg(0).to_opref(),
            Type::Int,
            &args,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        self.perform(i, [loc], Some(loc), output);
    }

    /// x86/regalloc.py consider_uint_mul_high
    /// arm/regalloc.py prepare_op_uint_mul_high = prepare_op_int_mul
    #[allow(dead_code)] // x86/regalloc.py consider_uint_mul_high
    fn consider_uint_mul_high(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        #[cfg(target_arch = "aarch64")]
        {
            self.consider_binop_symm(op, i, output)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut arg1 = op.arg(0).to_opref();
            let mut arg2 = op.arg(1).to_opref();
            // x86/regalloc.py:594 — optimized for (box, const)
            if arg1.is_constant() {
                std::mem::swap(&mut arg1, &mut arg2);
            }
            // arg2 must be in eax
            self.make_sure_var_in_reg(arg2, Type::Int, &[], Some(EAX), false);
            let l1 = self.loc(arg1, Type::Int);
            // eax will be trash after the operation
            self.possibly_free_var(arg2, Type::Int);
            // x86/regalloc.py:605 tmpvar = TempVar() (allocated in eax,
            // trashed by MUL)
            let tmp = self.fresh_temp_var();
            self.longevity
                .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
            self.rm.force_allocate_reg(
                tmp,
                &[],
                Some(EAX),
                false,
                &mut self.longevity,
                &mut self.fm,
            );
            self.rm
                .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
            // result in edx
            self.rm.force_allocate_reg(
                op.pos().get(),
                &[],
                Some(EDX),
                false,
                &mut self.longevity,
                &mut self.fm,
            );
            self.perform(i, [l1], Some(Loc::Reg(EDX)), output);
        }
    }

    /// x86/regalloc.py consider_int_signext
    #[allow(dead_code)] // x86/regalloc.py consider_int_signext
    fn consider_int_signext(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let argloc = self.loc(op.arg(0).to_opref(), Type::Int);
        let numbytesloc = self.loc(op.arg(1).to_opref(), Type::Int);
        let resloc = Loc::Reg(self.force_allocate_reg(op.pos().get(), Type::Int, &[], None, false));
        self.perform(i, [argloc, numbytesloc], Some(resloc), output);
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
    fn next_op_can_accept_cc<T: AsRef<Op>>(&self, ops: &[T], i: usize, result: OpRef) -> bool {
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
            || next_op.arg(0).to_opref().raw() != result.raw()
        {
            return false;
        }
        if let Some(lt) = self.longevity.get(result)
            && lt.last_usage > (i as i32) + 1
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
                && fail_args.iter().any(|a| a.to_opref() == result)
            {
                return false;
            }
        } else if next_op
            .args_slice()
            .iter()
            .skip(1)
            .any(|a| a.to_opref() == result)
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
        result: OpRef,
        ops: &[T],
        i: usize,
    ) -> Loc {
        if self.next_op_can_accept_cc(ops, i, result) {
            self.rm.force_allocate_frame_reg(result);
            return Loc::Reg(arch_regalloc::frame_reg());
        }
        Loc::Reg(self.force_allocate_reg(result, Type::Int, &[], None, true))
    }

    /// x86/regalloc.py _consider_compop
    #[allow(dead_code)] // x86/regalloc.py consider_compop
    fn consider_compop(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let vx = op.arg(0).to_opref();
        let vy = op.arg(1).to_opref();
        let mut arglocs = [self.loc(vx, self.tp(vx)), self.loc(vy, self.tp(vy))];
        // x86/regalloc.py:640-644
        let vx_in_reg = self.rm.reg_bindings_contains(vx, &self.longevity);
        let vy_in_reg = self.rm.reg_bindings_contains(vy, &self.longevity);
        if !vx_in_reg && !vy_in_reg && !vx.is_constant() && !vy.is_constant() {
            arglocs[0] = self.make_sure_var_in_reg(vx, Type::Int, &[], None, false);
        }
        // x86/regalloc.py force_allocate_reg_or_cc.
        let ops_ref: &[OpRc] = self.operations;
        let result_loc = self.force_allocate_reg_or_cc(op.pos().get(), ops_ref, i);
        self.perform(i, arglocs, Some(result_loc), output);
    }

    fn consider_guard_cc_j2(
        &mut self,
        arg: OpRef,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        // x86 `_consider_guard_cc` passes `self.loc(arg)` into
        // `load_condition_into_cc` → `test_location`, which emits
        // `CMP [mem], 0` for a frame location. aarch64 still forces a register.
        #[cfg(target_arch = "x86_64")]
        let loc = self.loc(arg, self.tp(arg));
        #[cfg(not(target_arch = "x86_64"))]
        let loc = self.make_sure_var_in_reg(arg, self.tp(arg), &[], None, false);
        // aarch64/regalloc.py `_guard_impl`: one walk of `getfailargs()`.
        self.perform_guard(op, i, [loc], None, output);
    }

    /// `RegisterManager.return_constant` with `selected_reg` queues
    /// `convert_to_imm`'s `ConstFloatLoc` into that xmm/vfp register, but
    /// leaves the register in `free_regs`. A `TempVar` owns it afterwards
    /// (`VFPRegisterManager.get_scratch_reg` / `temp_boxes`) so the next
    /// ConstFloat in this instruction cannot reuse it. Binding first would
    /// make `return_constant` spill the scratch: it only queues the move
    /// while the register is still free.
    fn const_float_in_reg(&mut self, v: OpRef, forbidden: &[OpRef]) -> Loc {
        debug_assert!(v.is_constant() && self.tp(v) == Type::Float);
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.xrm.position, self.xrm.position));
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
            self.force_allocate_reg(tmp, Type::Float, forbidden, Some(reg), false)
        } else {
            let reg = self.force_allocate_reg(tmp, Type::Float, forbidden, None, false);
            let imm = self.xrm.convert_to_imm(v, &self.constants);
            self.xrm.spill_moves.push((imm, Loc::Reg(reg)));
            reg
        };
        self.xrm.temp_boxes.push(tmp);
        Loc::Reg(reg)
    }

    /// aarch64/regalloc.py `prepare_op_guard_value`: arg in a register.
    /// A ConstFloat is loaded by `ARMRegisterManager.return_constant`
    /// (`FCMP_dd` has no immediate form).
    #[cfg(target_arch = "aarch64")]
    fn prepare_guard_value_arg(&mut self, v: OpRef, forbidden: &[OpRef]) -> Loc {
        if v.is_constant() && self.tp(v) == Type::Float {
            self.const_float_in_reg(v, forbidden)
        } else {
            self.make_sure_var_in_reg(v, self.tp(v), forbidden, None, false)
        }
    }

    /// aarch64/regalloc.py `prepare_op_guard_value`.
    #[cfg(target_arch = "aarch64")]
    fn guard_value_locs(&mut self, lhs: OpRef, rhs: OpRef) -> (Loc, Loc) {
        let arg = self.prepare_guard_value_arg(lhs, &[]);
        let l1 = if arch_regalloc::check_imm_box(rhs) {
            self.loc(rhs, self.tp(rhs))
        } else {
            self.prepare_guard_value_arg(rhs, &[lhs])
        };
        (arg, l1)
    }

    /// x86/regalloc.py `consider_guard_value`: arg0 in a register, arg1
    /// stays `self.loc` (`UCOMISD` accepts that location code).
    #[cfg(not(target_arch = "aarch64"))]
    fn guard_value_locs(&mut self, lhs: OpRef, rhs: OpRef) -> (Loc, Loc) {
        let x = self.make_sure_var_in_reg(lhs, self.tp(lhs), &[], None, false);
        let y = self.loc(rhs, self.tp(rhs));
        (x, y)
    }

    fn consider_guard_value_j2(
        &mut self,
        lhs: OpRef,
        rhs: OpRef,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let (x, y) = self.guard_value_locs(lhs, rhs);
        self.perform_guard(op, i, vec![x, y], None, output);
        #[cfg(target_arch = "aarch64")]
        {
            // `get_scratch_reg` temps die with this instruction.
            self.xrm.free_temp_vars(&mut self.longevity, &mut self.fm);
        }
    }

    fn consider_guard_class_j2(
        &mut self,
        value: OpRef,
        class: OpRef,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let x = self.make_sure_var_in_reg(value, Type::Ref, &[], None, false);
        let y = self.loc(class, Type::Int);
        self.perform_guard(op, i, vec![x, y], None, output);
    }

    fn consider_guard_is_object_j2(
        &mut self,
        value: OpRef,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let x = self.make_sure_var_in_reg(value, Type::Ref, &[], None, false);
        #[cfg(target_arch = "x86_64")]
        {
            // x86/regalloc.py:514 tmp_box = TempVar()
            let tmp = self.fresh_temp_var();
            self.longevity
                .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
            let loc_tmp = Loc::Reg(self.rm.force_allocate_reg(
                tmp,
                &[value],
                None,
                false,
                &mut self.longevity,
                &mut self.fm,
            ));
            self.rm
                .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
            self.perform_guard(op, i, vec![x, loc_tmp], None, output);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.perform_guard(op, i, [x], None, output);
        }
    }

    fn consider_guard_subclass_j2(
        &mut self,
        value: OpRef,
        class: OpRef,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let x = self.make_sure_var_in_reg(value, Type::Ref, &[], None, false);
        let y = self.loc(class, Type::Int);
        #[cfg(target_arch = "x86_64")]
        {
            // x86/regalloc.py:521 tmp_box = TempVar()
            let tmp = self.fresh_temp_var();
            self.longevity
                .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
            let loc_tmp = Loc::Reg(self.rm.force_allocate_reg(
                tmp,
                &[value],
                None,
                false,
                &mut self.longevity,
                &mut self.fm,
            ));
            self.rm
                .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
            self.perform_guard(op, i, vec![x, y, loc_tmp], None, output);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.perform_guard(op, i, vec![x, y], None, output);
        }
    }

    fn consider_guard_exception_j2(
        &mut self,
        args: &[OpRef],
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let Some(&exception_class) = args.first() else {
            return self.consider_guard_no_args_j2(op, i, output);
        };
        let loc = self.make_sure_var_in_reg(exception_class, Type::Ref, &[], None, false);
        // x86/regalloc.py:470 box = TempVar()
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        let guard_args: Vec<OpRef> = args.to_vec();
        let loc1 = Loc::Reg(self.rm.force_allocate_reg(
            tmp,
            &guard_args,
            None,
            false,
            &mut self.longevity,
            &mut self.fm,
        ));
        let resloc = if self.longevity.contains(op.pos().get()) {
            let mut forbidden = guard_args;
            forbidden.push(tmp);
            Some(Loc::Reg(self.rm.force_allocate_reg(
                op.pos().get(),
                &forbidden,
                None,
                false,
                &mut self.longevity,
                &mut self.fm,
            )))
        } else {
            None
        };
        self.perform_guard(op, i, vec![loc, loc1], resloc, output);
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
    }

    fn consider_guard_no_args_j2(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        self.perform_guard(op, i, vec![], None, output);
    }

    /// x86/regalloc.py `consider_guard_not_forced_2` and
    /// aarch64/regalloc.py `prepare_op_guard_not_forced_2`.
    fn consider_guard_not_forced_2_j2(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        // regalloc.py `before_call(op.getfailargs())`. The list is the
        // guard's own failargs; `locs_for_fail` walks them again for
        // locations, the same split as `_guard_impl` after `before_call`.
        let fail_args: Vec<OpRef> = op
            .guard_fail_args()
            .map(|fa| fa.iter().map(|a| a.to_opref()).collect())
            .unwrap_or_default();
        self.rm.before_call(
            &fail_args,
            SAVE_ALL_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.xrm.before_call(
            &fail_args,
            SAVE_ALL_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.perform_guard(op, i, vec![], None, output);
    }

    /// x86/regalloc.py _consider_guard_cc
    #[allow(dead_code)] // x86/regalloc.py consider_guard_cc
    fn consider_guard_cc(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let arg = op.arg(0).to_opref();
        let loc = self.make_sure_var_in_reg(arg, self.tp(arg), &[], None, false);
        self.perform_guard(op, i, [loc], None, output);
    }

    /// x86/regalloc.py consider_guard_value
    #[allow(dead_code)] // x86/regalloc.py consider_guard_value
    fn consider_guard_value(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let arg0 = op.arg(0).to_opref();
        let arg1 = op.arg(1).to_opref();
        let x = self.make_sure_var_in_reg(arg0, self.tp(arg0), &[], None, false);
        let y = self.loc(arg1, self.tp(arg1));
        self.perform_guard(op, i, vec![x, y], None, output);
    }

    /// x86/regalloc.py consider_guard_class
    #[allow(dead_code)] // x86/regalloc.py consider_guard_class
    fn consider_guard_class(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let x = self.make_sure_var_in_reg(op.arg(0).to_opref(), Type::Ref, &[], None, false);
        let y = self.loc(op.arg(1).to_opref(), Type::Int);
        self.perform_guard(op, i, vec![x, y], None, output);
    }

    /// x86/regalloc.py consider_guard_exception
    #[allow(dead_code)] // x86/regalloc.py consider_guard_exception
    fn consider_guard_exception(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let loc = self.make_sure_var_in_reg(op.arg(0).to_opref(), Type::Ref, &[], None, false);
        // x86/regalloc.py:470 box = TempVar()
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        let loc1 = Loc::Reg(self.rm.force_allocate_reg(
            tmp,
            &args,
            None,
            false,
            &mut self.longevity,
            &mut self.fm,
        ));
        // x86/regalloc.py:473-477
        let resloc = if self.longevity.contains(op.pos().get()) {
            let mut forbidden = args.clone();
            forbidden.push(tmp);
            Some(Loc::Reg(self.rm.force_allocate_reg(
                op.pos().get(),
                &forbidden,
                None,
                false,
                &mut self.longevity,
                &mut self.fm,
            )))
        } else {
            None
        };
        self.perform_guard(op, i, vec![loc, loc1], resloc, output);
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
    }

    /// x86/regalloc.py consider_restore_exception
    #[allow(dead_code)] // x86/regalloc.py consider_restore_exception
    fn consider_restore_exception(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        let loc0 = self.make_sure_var_in_reg(op.arg(0).to_opref(), Type::Ref, &args, None, false);
        let loc1 = self.make_sure_var_in_reg(op.arg(1).to_opref(), Type::Ref, &args, None, false);
        self.perform_discard(i, vec![loc0, loc1], output);
    }

    fn consider_restore_exception_j2(
        &mut self,
        args: &[OpRef],
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let loc0 = self.make_sure_var_in_reg(args[0], Type::Ref, args, None, false);
        let loc1 = self.make_sure_var_in_reg(args[1], Type::Ref, args, None, false);
        self.perform_discard(i, vec![loc0, loc1], output);
    }

    /// Guards with no arguments (guard_no_exception, guard_not_forced, etc.)
    #[allow(dead_code)] // x86/regalloc.py consider_guard_no_args
    fn consider_guard_no_args(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        self.perform_guard(op, i, vec![], None, output);
    }

    /// x86/regalloc.py `consider_guard_not_forced_2`.
    #[allow(dead_code)] // x86/regalloc.py consider_guard_not_forced_2
    fn consider_guard_not_forced_2(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let fail_args: Vec<OpRef> = op
            .guard_fail_args()
            .map(|fa| {
                fa.iter()
                    .filter(|arg| !arg.is_none())
                    .map(|arg| arg.to_opref())
                    .collect()
            })
            .unwrap_or_default();
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        self.rm.before_call(
            &fail_args,
            SAVE_ALL_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.xrm.before_call(
            &fail_args,
            SAVE_ALL_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.perform_guard(op, i, vec![], None, output);
    }

    /// x86/regalloc.py consider_check_memory_error.
    /// Reserves a register for `args[0]` (the malloc-helper return
    /// value) so the assembler arm can TEST it against zero and route
    /// NULL into the propagate-exception path
    /// (assembler.py `genop_discard_check_memory_error` /
    /// opassembler.py:258 `emit_op_check_memory_error`).
    #[allow(dead_code)] // x86/regalloc.py consider_check_memory_error
    fn consider_check_memory_error(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let arg0 = op.arg(0).to_opref();
        let tp = self.tp(arg0);
        let loc = self.make_sure_var_in_reg(arg0, tp, &[], None, false);
        self.perform(i, [loc], None, output);
    }

    fn consider_check_memory_error_j2(
        &mut self,
        args: &[OpRef],
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let Some(&arg) = args.first() else {
            return self.perform(i, [], None, output);
        };
        let tp = self.tp(arg);
        let loc = self.make_sure_var_in_reg(arg, tp, &[], None, false);
        self.perform(i, [loc], None, output);
    }

    /// x86/regalloc.py consider_finish
    #[allow(dead_code)] // x86/regalloc.py consider_finish
    fn consider_finish(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        if op.num_args() != 0 {
            let arg0 = op.arg(0).to_opref();
            let tp = self.tp(arg0);
            let loc = self.make_sure_var_in_reg(arg0, tp, &[], None, false);
            self.perform(i, [loc], None, output);
        } else {
            self.perform(i, [], None, output);
        }
    }

    fn consider_finish_j2(&mut self, args: &[OpRef], i: usize, output: &mut Vec<RegAllocOp>) {
        if let Some(&arg) = args.first() {
            let tp = self.tp(arg);
            let loc = self.make_sure_var_in_reg(arg, tp, &[], None, false);
            self.perform(i, [loc], None, output);
        } else {
            self.perform(i, [], None, output);
        }
    }

    /// x86/regalloc.py `_consider_same_as` / aarch64 `prepare_op_same_as`.
    ///
    /// A constant is `self.loc`: FLOAT goes to
    /// `X86XMMRegisterManager.convert_to_imm` (`ConstFloatLoc`). The GPR
    /// manager's `convert_to_imm` would leave those bits in `ImmedFloat`.
    /// A non-constant is forced into the result's register bank.
    #[allow(dead_code)] // x86/regalloc.py consider_same_as
    fn consider_same_as(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let tp = op.opcode.result_type();
        let arg = op.arg(0).to_opref();
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        let argloc = if arg.is_constant() {
            self.loc(arg, self.tp(arg))
        } else {
            self.make_sure_var_in_reg(arg, tp, &args, None, false)
        };
        // aarch64/regalloc.py:885-886
        self.possibly_free_vars_for_op(op);
        self.rm.free_temp_vars(&mut self.longevity, &mut self.fm);
        self.xrm.free_temp_vars(&mut self.longevity, &mut self.fm);
        // aarch64/regalloc.py:887
        let resloc = Loc::Reg(self.force_allocate_reg(op.pos().get(), tp, &[], None, false));
        self.perform(i, [argloc], Some(resloc), output);
    }

    fn consider_same_as_j2(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let tp = op.opcode.result_type();
        let args = [arg];
        // `_consider_same_as`: `argloc = self.loc(op.getarg(0))`.
        let argloc = if arg.is_constant() {
            self.loc(arg, self.tp(arg))
        } else {
            self.make_sure_var_in_reg(arg, tp, &args, None, false)
        };
        if !arg.is_constant() && !arg.is_none() {
            self.possibly_free_var(arg, tp);
        }
        self.rm.free_temp_vars(&mut self.longevity, &mut self.fm);
        self.xrm.free_temp_vars(&mut self.longevity, &mut self.fm);
        let resloc = Loc::Reg(self.force_allocate_reg(dst, tp, &[], None, false));
        self.perform(i, [argloc], Some(resloc), output);
    }

    /// x86/regalloc.py _consider_float_op
    #[allow(dead_code)] // x86/regalloc.py consider_float_op
    fn consider_float_op(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let loc1 = self.xrm.loc(
            op.arg(1).to_opref(),
            Type::Float,
            false,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
        );
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        let loc0 = self.xrm.force_result_in_reg(
            op.pos().get(),
            op.arg(0).to_opref(),
            Type::Float,
            &args,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        self.perform(i, [loc0, loc1], Some(loc0), output);
    }

    fn consider_float_op_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        // aarch64/regalloc.py `prepare_two_regs_op`: `fadd/fmul Dd, Dn, Dm`
        // writes a third register. The first `make_sure_var_in_reg` takes no
        // forbidden vars; the second receives `op.getarglist()`. x86
        // `addsd/mulsd` is two-operand, so the result has to land on the lhs
        // (`force_result_in_reg`). Using that form here copies the accumulator
        // into the destination before the arithmetic, which lengthens the
        // loop-carried float chain.
        #[cfg(target_arch = "aarch64")]
        {
            let boxes = [lhs, rhs];
            let loc0 = self.make_sure_var_in_reg(lhs, Type::Float, &[], None, false);
            let loc1 = self.make_sure_var_in_reg(rhs, Type::Float, &boxes, None, false);
            self.possibly_free_var(lhs, Type::Float);
            self.possibly_free_var(rhs, Type::Float);
            // aarch64/regalloc.py `Regalloc.free_temp_vars` / `prepare_two_regs_op`
            self.rm.free_temp_vars(&mut self.longevity, &mut self.fm);
            self.xrm.free_temp_vars(&mut self.longevity, &mut self.fm);
            let res = self.force_allocate_reg(dst, Type::Float, &[], None, false);
            self.perform(i, [loc0, loc1], Some(Loc::Reg(res)), output);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let loc1 = self.xrm.loc(
                rhs,
                Type::Float,
                false,
                &mut self.longevity,
                &mut self.fm,
                &self.constants,
            );
            let args = [lhs, rhs];
            let loc0 = self.xrm.force_result_in_reg(
                dst,
                lhs,
                Type::Float,
                &args,
                &mut self.longevity,
                &mut self.fm,
                &self.constants,
                &mut self.pending_moves,
            );
            self.perform(i, [loc0, loc1], Some(loc0), output);
        }
    }

    /// x86/regalloc.py `_consider_math_sqrt`:
    /// `loc0 = xrm.force_result_in_reg(op, op.getarg(1)); perform_math(op, [loc0], loc0)`.
    /// No `before_call`: `SQRTSD` does not spill caller-saved registers.
    #[allow(dead_code)] // x86/regalloc.py consider_math_sqrt
    fn consider_math_sqrt(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let src = op.arg(1).to_opref();
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        let loc = self.xrm.force_result_in_reg(
            op.pos().get(),
            src,
            Type::Float,
            &args,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        self.perform(i, [loc], Some(loc), output);
    }

    fn consider_math_sqrt_j2(
        &mut self,
        dst: OpRef,
        src: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let loc = self.xrm.force_result_in_reg(
            dst,
            src,
            Type::Float,
            &[],
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        self.perform(i, [loc], Some(loc), output);
    }

    /// x86/regalloc.py float_neg / float_abs
    #[allow(dead_code)] // x86/regalloc.py consider_float_unary
    fn consider_float_unary(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        let loc = self.xrm.force_result_in_reg(
            op.pos().get(),
            op.arg(0).to_opref(),
            Type::Float,
            &args,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        self.perform(i, [loc], Some(loc), output);
    }

    fn consider_float_unary_j2(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let loc = self.xrm.force_result_in_reg(
            dst,
            arg,
            Type::Float,
            &[],
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        self.perform(i, [loc], Some(loc), output);
    }

    /// `_consider_float_cmp`: UCOMISD's first operand is a register. When
    /// neither `loc` is a `RegLoc`, force the non-constant side through
    /// `xrm.make_sure_var_in_reg`.
    fn float_cmp_ensure_reg(&mut self, vx: OpRef, vy: OpRef, arglocs: &mut [Loc; 2]) {
        if arglocs[0].is_reg() || arglocs[1].is_reg() {
            return;
        }
        if vx.is_constant() {
            arglocs[1] = self.make_sure_var_in_reg(vy, Type::Float, &[], None, false);
        } else {
            arglocs[0] = self.make_sure_var_in_reg(vx, Type::Float, &[], None, false);
        }
    }

    /// x86/regalloc.py _consider_float_cmp
    #[allow(dead_code)] // x86/regalloc.py consider_float_cmp
    fn consider_float_cmp(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let vx = op.arg(0).to_opref();
        let vy = op.arg(1).to_opref();
        let mut arglocs = [self.loc(vx, Type::Float), self.loc(vy, Type::Float)];
        self.float_cmp_ensure_reg(vx, vy, &mut arglocs);
        // `_consider_float_cmp`: a comparison whose only consumer is the next
        // guard leaves its answer in the flags.
        let ops_ref: &[OpRc] = self.operations;
        let result_loc = self.force_allocate_reg_or_cc(op.pos().get(), ops_ref, i);
        self.perform(i, arglocs, Some(result_loc), output);
    }

    fn consider_float_cmp_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let mut arglocs = [self.loc(lhs, Type::Float), self.loc(rhs, Type::Float)];
        self.float_cmp_ensure_reg(lhs, rhs, &mut arglocs);
        let ops_ref: &[OpRc] = self.operations;
        let result_loc = self.force_allocate_reg_or_cc(dst, ops_ref, i);
        self.perform(i, arglocs, Some(result_loc), output);
    }

    /// x86/regalloc.py cast_int_to_float
    #[allow(dead_code)] // x86/regalloc.py consider_cast_int_to_float
    fn consider_cast_int_to_float(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let loc0 = self.make_sure_var_in_reg(op.arg(0).to_opref(), Type::Int, &[], None, false);
        let result_loc =
            Loc::Reg(self.force_allocate_reg(op.pos().get(), Type::Float, &[], None, false));
        self.perform(i, [loc0], Some(result_loc), output);
    }

    fn consider_cast_int_to_float_j2(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let loc0 = self.make_sure_var_in_reg(arg, Type::Int, &[], None, false);
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, Type::Float, &[], None, false));
        self.perform(i, [loc0], Some(result_loc), output);
    }

    /// x86/regalloc.py cast_float_to_int
    #[allow(dead_code)] // x86/regalloc.py consider_cast_float_to_int
    fn consider_cast_float_to_int(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let loc0 = self.make_sure_var_in_reg(op.arg(0).to_opref(), Type::Float, &[], None, false);
        let result_loc =
            Loc::Reg(self.force_allocate_reg(op.pos().get(), Type::Int, &[], None, false));
        self.perform(i, [loc0], Some(result_loc), output);
    }

    /// x86/regalloc.py `consider_cast_float_to_int` calls
    /// `xrm.make_sure_var_in_reg`. On x86, `RegAlloc::make_sure_var_in_reg`
    /// returns `FloatImmedLoc` for a ConstFloat before that call, and
    /// `CVTTSD2SI` / `FCVTZS` only encode a register source.
    fn cast_float_src_in_reg(&mut self, arg: OpRef) -> Loc {
        if arg.is_constant() {
            self.const_float_in_reg(arg, &[])
        } else {
            self.make_sure_var_in_reg(arg, Type::Float, &[], None, false)
        }
    }

    fn consider_cast_float_to_int_j2(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        // execute_cast_const lets a non-finite or out-of-range ConstFloat reach here.
        let loc0 = self.cast_float_src_in_reg(arg);
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, Type::Int, &[], None, false));
        self.perform(i, [loc0], Some(result_loc), output);
        self.xrm.free_temp_vars(&mut self.longevity, &mut self.fm);
    }

    /// x86/regalloc.py `consider_cast_float_to_singlefloat`.
    ///
    /// The arg stays in an xmm, the result is a GPR (the f32 bits in the
    /// low 32), and an xmm temp is allocated after the result. The temp
    /// may alias the arg, so the forbidden list is empty.
    #[cfg(target_arch = "x86_64")]
    fn consider_cast_float_to_singlefloat_j2(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        // execute_cast_const lets a non-finite or out-of-range ConstFloat reach here.
        let loc0 = self.cast_float_src_in_reg(arg);
        let loc1 = Loc::Reg(self.force_allocate_reg(dst, Type::Int, &[], None, false));
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        let loctmp = Loc::Reg(self.xrm.force_allocate_reg(
            tmp,
            &[],
            None,
            false,
            &mut self.longevity,
            &mut self.fm,
        ));
        self.xrm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Float);
        self.perform(i, [loc0, loctmp], Some(loc1), output);
        self.xrm.free_temp_vars(&mut self.longevity, &mut self.fm);
    }

    /// Memory load: getfield pattern (1 arg → result)
    #[allow(dead_code)] // x86/regalloc.py consider_getfield
    fn consider_getfield(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let base_loc = self.make_sure_var_in_reg(op.arg(0).to_opref(), Type::Ref, &[], None, false);
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos().get(), tp, &[], None, false));
        self.perform(i, [base_loc], Some(result_loc), output);
    }

    fn consider_getfield_j2(
        &mut self,
        dst: OpRef,
        base: OpRef,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &[], None, false);
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, tp, &[], None, false));
        self.perform(i, [base_loc], Some(result_loc), output);
    }

    /// Memory load: getarrayitem pattern (2 args → result)
    #[allow(dead_code)] // x86/regalloc.py consider_getarrayitem
    fn consider_getarrayitem(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        let base_loc =
            self.make_sure_var_in_reg(op.arg(0).to_opref(), Type::Ref, &args, None, false);
        let index_loc =
            self.make_sure_var_in_reg(op.arg(1).to_opref(), Type::Int, &args, None, false);
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos().get(), tp, &[], None, false));
        self.perform(i, [base_loc, index_loc], Some(result_loc), output);
    }

    fn consider_getarrayitem_j2(
        &mut self,
        dst: OpRef,
        base: OpRef,
        index: OpRef,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let args = [base, index];
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        let index_loc = self.make_sure_var_in_reg(index, Type::Int, &args, None, false);
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, tp, &[], None, false));
        self.perform(i, [base_loc, index_loc], Some(result_loc), output);
    }

    /// Memory load: getinteriorfield (3 args → result)
    #[allow(dead_code)] // x86/regalloc.py consider_getinteriorfield
    fn consider_getinteriorfield(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        let base_loc =
            self.make_sure_var_in_reg(op.arg(0).to_opref(), Type::Ref, &args, None, false);
        let index_loc =
            self.make_sure_var_in_reg(op.arg(1).to_opref(), Type::Int, &args, None, false);
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos().get(), tp, &[], None, false));
        self.perform(i, [base_loc, index_loc], Some(result_loc), output);
    }

    fn gc_load_nsize(&self, op: &Op, size: Option<OpRef>) -> i64 {
        if let Some(size) = size {
            return self.const_value(size);
        }
        op.getdescr()
            .and_then(|d| {
                d.as_field_descr().map(|fd| {
                    let s = fd.field_size() as i64;
                    if fd.is_field_signed() { -s } else { s }
                })
            })
            .or_else(|| {
                op.getdescr().and_then(|d| {
                    d.as_array_descr().map(|ad| {
                        let s = ad.item_size() as i64;
                        if ad.is_item_signed() { -s } else { s }
                    })
                })
            })
            .unwrap_or(8)
    }

    fn gc_offset_loc(&mut self, ofs: i64) -> Loc {
        // x86 `AddressLoc` keeps any displacement that `fits_in_32bits`,
        // including negatives and values past `DEFAULT_IMM_SIZE`. aarch64
        // stays on `check_imm_arg` (`0..DEFAULT_IMM_SIZE`).
        let fits = if cfg!(target_arch = "x86_64") {
            fits_in_32bits(ofs)
        } else {
            check_imm_arg(ofs)
        };
        if fits {
            Loc::immed(ofs)
        } else {
            self.pending_moves
                .push((Loc::immed(ofs), Loc::Reg(LARGE_IMM_SCRATCH)));
            Loc::Reg(LARGE_IMM_SCRATCH)
        }
    }

    fn consider_gc_load_j2(
        &mut self,
        dst: OpRef,
        base: OpRef,
        offset: OpRef,
        size: Option<OpRef>,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let args = [base, offset];
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        let ofs_loc = self.gc_offset_loc(self.const_value(offset));
        let nsize = self.gc_load_nsize(op, size);
        let tp = op.opcode.result_type();
        let res_loc = Loc::Reg(self.force_allocate_reg(dst, tp, &[], None, false));
        self.perform_gc_load(i, base_loc, ofs_loc, res_loc, nsize, output);
    }

    /// aarch64/regalloc.py _prepare_op_gc_load_indexed parity, j2plan path.
    #[cfg(target_arch = "aarch64")]
    fn consider_gc_load_indexed_j2(
        &mut self,
        dst: OpRef,
        base: OpRef,
        index: OpRef,
        scale: OpRef,
        offset: OpRef,
        size: OpRef,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let args = [base, index, scale, offset, size];
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        let index_loc = self.make_sure_var_in_reg(index, Type::Int, &args, None, false);
        let scale_value = self.const_value(scale);
        assert_eq!(
            scale_value, 1,
            "aarch64 GcLoadIndexed requires factor == 1 (got {scale_value})"
        );
        let ofs = self.const_value(offset);
        let nsize = self.const_value(size);
        let tp = op.opcode.result_type();
        let res_loc = Loc::Reg(self.force_allocate_reg(dst, tp, &[], None, false));
        self.perform(
            i,
            vec![
                res_loc,
                base_loc,
                index_loc,
                Loc::immed(nsize),
                Loc::immed(ofs),
            ],
            Some(res_loc),
            output,
        );
    }

    /// x86/regalloc.py _consider_gc_load_indexed parity, j2plan path.
    /// Emits the same locs ordering as the regular path so the assembler
    /// dispatch can be shared.
    #[cfg(target_arch = "x86_64")]
    fn consider_gc_load_indexed_j2(
        &mut self,
        dst: OpRef,
        base: OpRef,
        index: OpRef,
        scale: OpRef,
        offset: OpRef,
        size: OpRef,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let args = [base, index, scale, offset, size];
        // x86/regalloc.py:1175
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        // x86/regalloc.py:1176
        let ofs_loc = self.make_sure_var_in_reg(index, Type::Int, &args, None, false);
        // x86/regalloc.py:1177
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, tp, &[], None, false));
        // x86/regalloc.py:1178-1186
        let scale_value = self.const_value(scale);
        let offset_value = self.const_value(offset);
        let nsize = self.const_value(size);
        // x86/regalloc.py:1187 `size_loc = imm(abs(nsize))`
        let size_loc = Loc::immed(nsize.unsigned_abs() as i64);
        // x86/regalloc.py:1188-1191 sign_loc = imm1 if nsize < 0 else imm0
        let sign_loc = Loc::immed(if nsize < 0 { 1 } else { 0 });
        // x86/regalloc.py:1192-1193
        self.perform(
            i,
            vec![
                base_loc,
                ofs_loc,
                Loc::immed(scale_value),
                Loc::immed(offset_value),
                size_loc,
                sign_loc,
            ],
            Some(result_loc),
            output,
        );
    }

    /// x86/regalloc.py _consider_gc_load
    /// aarch64/regalloc.py _prepare_op_gc_load parity.
    /// Returns [base_loc, ofs_loc, res_loc, imm(nsize)].
    #[allow(dead_code)] // x86/regalloc.py consider_gc_load
    fn consider_gc_load(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        // aarch64/regalloc.py:537
        let base_loc = self.make_sure_var_in_reg(op.arg(0).to_opref(), Type::Ref, &[], None, false);
        // aarch64/regalloc.py: ofs = op.getarg(1).getint(); check_imm_arg
        let ofs = self.const_value(op.arg(1).to_opref());
        let ofs_loc = if check_imm_arg(ofs) {
            Loc::immed(ofs)
        } else {
            self.pending_moves
                .push((Loc::immed(ofs), Loc::Reg(LARGE_IMM_SCRATCH)));
            Loc::Reg(LARGE_IMM_SCRATCH)
        };
        // aarch64/regalloc.py:536: nsize = op.getarg(2).getint()
        let nsize = if op.num_args() > 2 {
            self.const_value(op.arg(2).to_opref())
        } else {
            op.getdescr()
                .and_then(|d| {
                    d.as_field_descr().map(|fd| {
                        let s = fd.field_size() as i64;
                        if fd.is_field_signed() { -s } else { s }
                    })
                })
                .or_else(|| {
                    op.getdescr().and_then(|d| {
                        d.as_array_descr().map(|ad| {
                            let s = ad.item_size() as i64;
                            if ad.is_item_signed() { -s } else { s }
                        })
                    })
                })
                .unwrap_or(8)
        };
        // aarch64/regalloc.py:545
        let tp = op.opcode.result_type();
        let res_loc = Loc::Reg(self.force_allocate_reg(op.pos().get(), tp, &[], None, false));
        // aarch64/regalloc.py:546: return [base_loc, ofs_loc, res_loc, imm(nsize)]
        self.perform_gc_load(i, base_loc, ofs_loc, res_loc, nsize, output);
    }

    /// aarch64/regalloc.py _prepare_op_gc_load_indexed parity.
    /// Returns [res_loc, base_loc, index_loc, imm(nsize), imm(ofs)].
    #[cfg(target_arch = "aarch64")]
    #[allow(dead_code)] // x86/regalloc.py consider_gc_load_indexed
    fn consider_gc_load_indexed(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        // aarch64/regalloc.py:564
        let base_loc =
            self.make_sure_var_in_reg(op.arg(0).to_opref(), Type::Ref, &args, None, false);
        // aarch64/regalloc.py:565
        let index_loc =
            self.make_sure_var_in_reg(op.arg(1).to_opref(), Type::Int, &args, None, false);
        // aarch64/regalloc.py `assert boxes[2].getint() == 1` — aarch64 load
        // has no scaled addressing form, so the rewriter must have already
        // pre-scaled the index (load_supported_factors = (1,) per
        // `gc_store_supported_factors()` in runner.rs).
        let scale = self.const_value(op.arg(2).to_opref());
        assert_eq!(
            scale, 1,
            "aarch64 GcLoadIndexed requires factor == 1 (got {scale})"
        );
        // aarch64/regalloc.py:567-568: ofs/nsize getint
        let ofs = if op.num_args() > 3 {
            self.const_value(op.arg(3).to_opref())
        } else {
            0
        };
        let nsize = if op.num_args() > 4 {
            self.const_value(op.arg(4).to_opref())
        } else {
            8
        };
        // aarch64/regalloc.py:571
        let tp = op.opcode.result_type();
        let res_loc = Loc::Reg(self.force_allocate_reg(op.pos().get(), tp, &[], None, false));
        // aarch64/regalloc.py:572: return [res_loc, base_loc, index_loc, imm(nsize), imm(ofs)]
        self.perform(
            i,
            vec![
                res_loc,
                base_loc,
                index_loc,
                Loc::immed(nsize),
                Loc::immed(ofs),
            ],
            Some(res_loc),
            output,
        );
    }

    /// x86/regalloc.py _consider_gc_load_indexed parity.
    /// Returns [base_loc, ofs_loc, imm(scale), imm(offset), size_loc, sign_loc]
    /// + result_loc.
    #[cfg(target_arch = "x86_64")]
    fn consider_gc_load_indexed(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        // x86/regalloc.py:1175
        let base_loc =
            self.make_sure_var_in_reg(op.arg(0).to_opref(), Type::Ref, &args, None, false);
        // x86/regalloc.py:1176
        let ofs_loc =
            self.make_sure_var_in_reg(op.arg(1).to_opref(), Type::Int, &args, None, false);
        // x86/regalloc.py:1177
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos().get(), tp, &[], None, false));
        // x86/regalloc.py:1178-1186 — scale/offset/size are ConstInt boxes.
        // `op.getarg(3)` / `op.getarg(4)` raise IndexError if missing in
        // RPython; `gc/rewrite.rs::emit_gc_load_indexed` lowers every
        // GC_LOAD_INDEXED with all 5 operands, so the arity check is a
        // contract assert here.
        assert_eq!(
            op.num_args(),
            5,
            "GC_LOAD_INDEXED must have 5 operands (ptr, idx, scale, offset, size)"
        );
        let scale = self.const_value(op.arg(2).to_opref());
        let offset = self.const_value(op.arg(3).to_opref());
        // x86/regalloc.py:1186 `nsize = size_box.value  # negative for "signed"`
        let nsize = self.const_value(op.arg(4).to_opref());
        // x86/regalloc.py:1187 `size_loc = imm(abs(nsize))`
        let size_loc = Loc::immed(nsize.unsigned_abs() as i64);
        // x86/regalloc.py:1188-1191 sign_loc = imm1 if nsize < 0 else imm0
        let sign_loc = Loc::immed(if nsize < 0 { 1 } else { 0 });
        // x86/regalloc.py:1192-1193
        self.perform(
            i,
            vec![
                base_loc,
                ofs_loc,
                Loc::immed(scale),
                Loc::immed(offset),
                size_loc,
                sign_loc,
            ],
            Some(result_loc),
            output,
        );
    }

    /// Memory store: setfield pattern (2 args: base, value)
    #[allow(dead_code)] // x86/regalloc.py consider_setfield
    fn consider_setfield(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        let base_loc =
            self.make_sure_var_in_reg(op.arg(0).to_opref(), Type::Ref, &args, None, false);
        let tp_val = self.tp(op.arg(1).to_opref());
        let val_loc = self.make_sure_var_in_reg(op.arg(1).to_opref(), tp_val, &args, None, false);
        self.perform_discard(i, vec![base_loc, val_loc], output);
    }

    fn consider_setfield_j2(
        &mut self,
        base: OpRef,
        value: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let args = [base, value];
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        let tp_val = self.tp(value);
        let val_loc = self.make_sure_var_in_reg(value, tp_val, &args, None, false);
        self.perform_discard(i, vec![base_loc, val_loc], output);
    }

    /// Memory store: setarrayitem pattern (3 args: base, index, value)
    #[allow(dead_code)] // x86/regalloc.py consider_setarrayitem
    fn consider_setarrayitem(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        let base_loc =
            self.make_sure_var_in_reg(op.arg(0).to_opref(), Type::Ref, &args, None, false);
        let index_loc =
            self.make_sure_var_in_reg(op.arg(1).to_opref(), Type::Int, &args, None, false);
        let tp_val = self.tp(op.arg(2).to_opref());
        let val_loc = self.make_sure_var_in_reg(op.arg(2).to_opref(), tp_val, &args, None, false);
        self.perform_discard(i, vec![base_loc, index_loc, val_loc], output);
    }

    fn consider_setarrayitem_j2(
        &mut self,
        base: OpRef,
        index: OpRef,
        value: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let args = [base, index, value];
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        let index_loc = self.make_sure_var_in_reg(index, Type::Int, &args, None, false);
        let tp_val = self.tp(value);
        let val_loc = self.make_sure_var_in_reg(value, tp_val, &args, None, false);
        self.perform_discard(i, vec![base_loc, index_loc, val_loc], output);
    }

    /// Memory store: setinteriorfield
    #[allow(dead_code)] // x86/regalloc.py consider_setinteriorfield
    fn consider_setinteriorfield(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        self.consider_setarrayitem(op, i, output);
    }

    fn consider_gc_store_j2(
        &mut self,
        base: OpRef,
        offset: OpRef,
        value: OpRef,
        size: Option<OpRef>,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let args = [base, offset, value];
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        let tp_val = self.tp(value);
        let value_loc = self.make_sure_var_in_reg(value, tp_val, &args, None, false);
        let size = size.map(|arg| self.const_value(arg)).unwrap_or(8);
        let ofs = self.const_value(offset);
        // pyre's GC header sits at `obj - HDR_SIZE`, so `gen_initialize_tid`
        // stores at a small negative offset. Both store emitters take that
        // window as an immediate (STUR's signed 9-bit range on aarch64,
        // disp8/disp32 on x86). Wider x86 displacements, including other
        // negatives and offsets >= 4096, stay immediate via `gc_offset_loc`
        // when `fits_in_32bits` (`AddressLoc`).
        let ofs_loc = if (-256..0).contains(&ofs) {
            Loc::immed(ofs)
        } else {
            self.gc_offset_loc(ofs)
        };
        self.perform_discard_gc_store(i, value_loc, base_loc, ofs_loc, size, output);
    }

    /// aarch64/regalloc.py prepare_op_gc_store_indexed parity.
    #[cfg(target_arch = "aarch64")]
    fn consider_gc_store_indexed_j2(
        &mut self,
        base: OpRef,
        index: OpRef,
        value: OpRef,
        scale: OpRef,
        offset: OpRef,
        size: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let args = [base, index, value, scale, offset, size];
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        let tp_val = self.tp(value);
        let value_loc = self.make_sure_var_in_reg(value, tp_val, &args, None, false);
        let index_loc = self.make_sure_var_in_reg(index, Type::Int, &args, None, false);
        let scale_value = self.const_value(scale);
        assert_eq!(
            scale_value, 1,
            "aarch64 GcStoreIndexed requires factor == 1 (got {scale_value})"
        );
        let ofs = self.const_value(offset);
        let size = self.const_value(size);
        self.perform_discard(
            i,
            vec![
                value_loc,
                base_loc,
                index_loc,
                Loc::immed(size),
                Loc::immed(ofs),
            ],
            output,
        );
    }

    /// x86/regalloc.py consider_gc_store_indexed parity.
    #[cfg(target_arch = "x86_64")]
    fn consider_gc_store_indexed_j2(
        &mut self,
        base: OpRef,
        index: OpRef,
        value: OpRef,
        scale: OpRef,
        offset: OpRef,
        size: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let args = [base, index, value, scale, offset, size];
        let base_loc = self.make_sure_var_in_reg(base, Type::Ref, &args, None, false);
        let factor = self.const_value(scale);
        let offset = self.const_value(offset);
        let size = self.const_value(size);
        assert!(
            size >= 1,
            "x86 GcStoreIndexed size must be >= 1 (got {size})"
        );
        let need_lower_byte = size == 1;
        let tp_val = self.tp(value);
        let value_loc = self.make_sure_var_in_reg(value, tp_val, &args, None, need_lower_byte);
        let ofs_loc = self.make_sure_var_in_reg(index, Type::Int, &args, None, false);
        self.perform_discard(
            i,
            vec![
                base_loc,
                ofs_loc,
                value_loc,
                Loc::immed(factor),
                Loc::immed(offset),
                Loc::immed(size),
            ],
            output,
        );
    }

    /// aarch64/regalloc.py prepare_op_gc_store parity.
    /// Returns [value_loc, base_loc, ofs_loc, imm(size)].
    #[allow(dead_code)] // x86/regalloc.py consider_gc_store
    fn consider_gc_store(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        // aarch64/regalloc.py:522
        let base_loc =
            self.make_sure_var_in_reg(op.arg(0).to_opref(), Type::Ref, &args, None, false);
        // aarch64/regalloc.py:523: ofs = boxes[1].getint()
        let ofs = self.const_value(op.arg(1).to_opref());
        // aarch64/regalloc.py:524
        let tp_val = self.tp(op.arg(2).to_opref());
        let value_loc = self.make_sure_var_in_reg(op.arg(2).to_opref(), tp_val, &args, None, false);
        // aarch64/regalloc.py:525: size = boxes[3].getint()
        let size = if op.num_args() > 3 {
            self.const_value(op.arg(3).to_opref())
        } else {
            8
        };
        // aarch64/regalloc.py: check_imm_arg(ofs)
        let ofs_loc = if check_imm_arg(ofs) {
            Loc::immed(ofs)
        } else {
            // aarch64/regalloc.py:529-530: ofs_loc = r.ip1; self.assembler.load(ofs_loc, imm(ofs))
            self.pending_moves
                .push((Loc::immed(ofs), Loc::Reg(LARGE_IMM_SCRATCH)));
            Loc::Reg(LARGE_IMM_SCRATCH)
        };
        // aarch64/regalloc.py:531: return [value_loc, base_loc, ofs_loc, imm(size)]
        self.perform_discard_gc_store(i, value_loc, base_loc, ofs_loc, size, output);
    }

    /// aarch64/regalloc.py prepare_op_gc_store_indexed parity.
    /// Returns [value_loc, base_loc, index_loc, imm(size), imm(ofs)].
    #[cfg(target_arch = "aarch64")]
    #[allow(dead_code)] // x86/regalloc.py consider_gc_store_indexed
    fn consider_gc_store_indexed(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        // aarch64/regalloc.py:554
        let base_loc =
            self.make_sure_var_in_reg(op.arg(0).to_opref(), Type::Ref, &args, None, false);
        // aarch64/regalloc.py:555
        let tp_val = self.tp(op.arg(2).to_opref());
        let value_loc = self.make_sure_var_in_reg(op.arg(2).to_opref(), tp_val, &args, None, false);
        // aarch64/regalloc.py:556
        let index_loc =
            self.make_sure_var_in_reg(op.arg(1).to_opref(), Type::Int, &args, None, false);
        // aarch64/regalloc.py `assert boxes[3].getint() == 1` — the
        // aarch64 store has no scaled addressing form so the rewriter must
        // have pre-scaled the index (load_supported_factors = (1,)).
        let scale = self.const_value(op.arg(3).to_opref());
        assert_eq!(
            scale, 1,
            "aarch64 GcStoreIndexed requires factor == 1 (got {scale})"
        );
        // aarch64/regalloc.py:558-559: ofs/size getint
        let ofs = if op.num_args() > 4 {
            self.const_value(op.arg(4).to_opref())
        } else {
            0
        };
        let size = if op.num_args() > 5 {
            self.const_value(op.arg(5).to_opref())
        } else {
            8
        };
        // aarch64/regalloc.py:560: return [value_loc, base_loc, index_loc, imm(size), imm(ofs)]
        self.perform_discard(
            i,
            vec![
                value_loc,
                base_loc,
                index_loc,
                Loc::immed(size),
                Loc::immed(ofs),
            ],
            output,
        );
    }

    /// x86/regalloc.py consider_gc_store_indexed parity.
    /// Returns [base_loc, ofs_loc, value_loc, imm(factor), imm(offset), imm(size)].
    #[cfg(target_arch = "x86_64")]
    fn consider_gc_store_indexed(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        // x86/regalloc.py:1129
        let base_loc =
            self.make_sure_var_in_reg(op.arg(0).to_opref(), Type::Ref, &args, None, false);
        // x86/regalloc.py:1130-1138: scale/offset/size are ConstInt
        let factor = if op.num_args() > 3 {
            self.const_value(op.arg(3).to_opref())
        } else {
            1
        };
        let offset = if op.num_args() > 4 {
            self.const_value(op.arg(4).to_opref())
        } else {
            0
        };
        let size = if op.num_args() > 5 {
            self.const_value(op.arg(5).to_opref())
        } else {
            8
        };
        // x86/regalloc.py:1139-1143 `assert size >= 1; need_lower_byte = (size == 1)`.
        // On x86_64 `no_lower_byte_regs` is empty (regalloc.py:71) so the flag
        // is a no-op today, but the contract mirrors the byte-store constraint
        // that `mov r/m8, Rb(val)` in the x86 assembler emitter requires.
        assert!(
            size >= 1,
            "x86 GcStoreIndexed size must be >= 1 (got {size})"
        );
        let need_lower_byte = size == 1;
        // x86/regalloc.py:1144-1145
        let tp_val = self.tp(op.arg(2).to_opref());
        let value_loc =
            self.make_sure_var_in_reg(op.arg(2).to_opref(), tp_val, &args, None, need_lower_byte);
        // x86/regalloc.py:1146
        let ofs_loc =
            self.make_sure_var_in_reg(op.arg(1).to_opref(), Type::Int, &args, None, false);
        // x86/regalloc.py:1147-1148
        self.perform_discard(
            i,
            vec![
                base_loc,
                ofs_loc,
                value_loc,
                Loc::immed(factor),
                Loc::immed(offset),
                Loc::immed(size),
            ],
            output,
        );
    }

    /// aarch64/regalloc.py _prepare_call.
    ///
    /// RPython critical ordering: `loc()` for every arg must run BEFORE
    /// `before_call`. Otherwise variables that die at this call position
    /// (max_age == self.position) are dropped from reg_bindings by the
    /// "dies" branch in spill_or_move_registers_before_call without being
    /// spilled, and the subsequent `loc()` returns a fresh frame slot whose
    /// contents are unrelated to the argument value. See llsupport/
    /// regalloc.py:724-734 doc: arglocs "stay valid" only if they were
    /// captured before before_call.
    #[allow(dead_code)] // x86/regalloc.py consider_call
    fn consider_call(
        &mut self,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
        save_all_regs: bool,
        first_arg_index: usize,
    ) {
        let descr_arc = op.getdescr().expect("call op without CallDescr");
        let calldescr = descr_arc
            .as_call_descr()
            .expect("call op without CallDescr");
        assert_eq!(calldescr.arg_types().len(), op.num_args() - first_arg_index);

        // aarch64/regalloc.py — capture arglocs before before_call.
        let mut arglocs = Vec::with_capacity(op.num_args() + 3);
        arglocs.push(Loc::immed(0)); // placeholder for result_loc
        arglocs.push(Loc::immed(calldescr.result_size() as i64));
        arglocs.push(Loc::immed(if calldescr.is_result_signed() { 1 } else { 0 }));
        let mut force_store_refs = Vec::new();
        for (arg_index, arg) in op.args_slice().iter().enumerate() {
            let arg = arg.to_opref();
            let tp = if arg_index >= first_arg_index {
                calldescr.arg_types()[arg_index - first_arg_index]
            } else {
                self.tp(arg)
            };
            arglocs.push(self.loc(arg, tp));
            // callbuilder.py keeps every GC argument alive for the duration
            // of a collecting call, even when this call is its last SSA use.
            // Without force_store, before_call's `max_age <= position` arm
            // drops that binding and get_gcmap omits it; a temporary such as
            // `Counter(dict(generator))` can then be swept inside the callee.
            if tp == Type::Ref && !arg.is_constant() {
                force_store_refs.push(arg);
            }
        }

        // aarch64/regalloc.py — then _call → before_call → after_call.
        let save_regs = if save_all_regs {
            SAVE_ALL_REGS
        } else if calldescr.get_extra_info().check_can_collect() {
            SAVE_GCREF_REGS
        } else {
            SAVE_DEFAULT_REGS
        };
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        self.rm.before_call(
            &force_store_refs,
            save_regs,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.xrm.before_call(
            &[],
            save_regs,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );

        // aarch64/regalloc.py:625-628 + llsupport/gcmap.py parity:
        // a collecting residual call needs a gcmap for the live Ref slots
        // that before_call just synced to the jitframe.  In particular,
        // call_assembler frame helpers allocate PyFrames and can collect
        // before the following vable_token materialization reads inputarg 0.
        //
        // x86/callbuilder.py push_gcmap + aarch64/callbuilder.py push_gcmap exclude the
        // call_result_gpr from the gcmap so its post-call payload is not
        // mistaken for a live Ref root before the result-extension step
        // narrows it to the declared result type.
        let gcmap = if calldescr.get_extra_info().check_can_collect() {
            Some(self.get_gcmap(&[call_result_gpr()], false) as usize)
        } else {
            None
        };

        let result_tp = op.opcode.result_type();
        let result_loc = if result_tp != Type::Void {
            let r = if result_tp == Type::Float {
                self.xrm.after_call(op.pos().get(), &mut self.longevity)
            } else {
                self.rm.after_call(op.pos().get(), &mut self.longevity)
            };
            Some(Loc::Reg(r))
        } else {
            None
        };
        arglocs[0] = result_loc.unwrap_or(Loc::immed(0));

        if let Some(gcmap) = gcmap {
            self.perform_with_gcmap_ptr(i, arglocs, result_loc, gcmap, output);
        } else {
            self.perform(i, arglocs, result_loc, output);
        }
    }

    fn consider_call_j2(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
        save_all_regs: bool,
        first_arg_index: usize,
    ) {
        let descr_arc = op.getdescr().expect("call op without CallDescr");
        let calldescr = descr_arc
            .as_call_descr()
            .expect("call op without CallDescr");
        assert_eq!(calldescr.arg_types().len(), args.len() - first_arg_index);

        let mut arglocs = Vec::with_capacity(args.len() + 3);
        arglocs.push(Loc::immed(0));
        arglocs.push(Loc::immed(calldescr.result_size() as i64));
        arglocs.push(Loc::immed(if calldescr.is_result_signed() { 1 } else { 0 }));
        let mut force_store_refs = Vec::new();
        for (arg_index, &arg) in args.iter().enumerate() {
            let tp = if arg_index >= first_arg_index {
                calldescr.arg_types()[arg_index - first_arg_index]
            } else {
                self.tp(arg)
            };
            arglocs.push(self.loc(arg, tp));
            if tp == Type::Ref && !arg.is_constant() {
                force_store_refs.push(arg);
            }
        }

        let can_collect = calldescr.get_extra_info().check_can_collect();
        let save_regs = if save_all_regs {
            SAVE_ALL_REGS
        } else if can_collect {
            SAVE_GCREF_REGS
        } else {
            SAVE_DEFAULT_REGS
        };
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        self.rm.before_call(
            &force_store_refs,
            save_regs,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.xrm.before_call(
            &[],
            save_regs,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );

        // Same gcmap snapshot as `consider_call` above. The j2 dispatch
        // handles the same Call*/CallMayForce*/CallReleaseGil* opcodes,
        // so a collecting callee must record the live Ref slots that
        // before_call just synced to the jitframe.  call_result_gpr is
        // excluded for the same reason as `consider_call`
        // (x86/callbuilder.py push_gcmap, aarch64/callbuilder.py push_gcmap).
        let gcmap = if can_collect {
            Some(self.get_gcmap(&[call_result_gpr()], false) as usize)
        } else {
            None
        };

        let result_tp = op.opcode.result_type();
        let result_loc = if result_tp != Type::Void {
            let dst = dst.unwrap_or(op.pos().get());
            let r = if result_tp == Type::Float {
                self.xrm.after_call(dst, &mut self.longevity)
            } else {
                self.rm.after_call(dst, &mut self.longevity)
            };
            Some(Loc::Reg(r))
        } else {
            None
        };
        arglocs[0] = result_loc.unwrap_or(Loc::immed(0));

        if let Some(gcmap) = gcmap {
            self.perform_with_gcmap_ptr(i, arglocs, result_loc, gcmap, output);
        } else {
            self.perform(i, arglocs, result_loc, output);
        }
    }

    /// llsupport/regalloc.py `locs_for_call_assembler` parity.
    ///
    /// RPython syncs only argument 1 (the virtualizable) to the frame. The
    /// callee jitframe in argument 0 stays in its current location, captured
    /// before `before_call`.
    #[allow(dead_code)] // x86/regalloc.py consider_call_assembler
    fn consider_call_assembler(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        assert!(matches!(op.num_args(), 1 | 2));
        if op.num_args() == 2 {
            let vable = op.arg(1).to_opref();
            let tp = self.tp(vable);
            if tp == Type::Float {
                self.xrm
                    ._sync_var_to_stack(vable, tp, &mut self.longevity, &mut self.fm);
            } else {
                self.rm
                    ._sync_var_to_stack(vable, tp, &mut self.longevity, &mut self.fm);
            }
        }

        let mut arglocs = Vec::with_capacity(op.num_args());
        for arg in op.args_slice().iter() {
            let arg = arg.to_opref();
            arglocs.push(self.loc(arg, self.tp(arg)));
        }

        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        self.rm.before_call(
            &[],
            SAVE_ALL_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.xrm.before_call(
            &[],
            SAVE_ALL_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );

        let result_tp = op.opcode.result_type();
        let result_loc = if result_tp != Type::Void {
            let r = if result_tp == Type::Float {
                self.xrm.after_call(op.pos().get(), &mut self.longevity)
            } else {
                self.rm.after_call(op.pos().get(), &mut self.longevity)
            };
            Some(Loc::Reg(r))
        } else {
            None
        };
        let gcmap = self.get_gcmap(&[call_result_gpr()], false) as usize;
        self.perform_with_gcmap_ptr(i, arglocs, result_loc, gcmap, output);
    }

    fn consider_call_assembler_j2(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        // llsupport/regalloc.py `locs_for_call_assembler`.
        let arglocs = if args.len() == 2 {
            let vable = args[1];
            let tp = self.tp(vable);
            self.rm
                ._sync_var_to_stack(vable, tp, &mut self.longevity, &mut self.fm);
            vec![
                self.loc(args[0], self.tp(args[0])),
                Loc::Frame(self.fm.loc(vable, tp, false, &mut self.longevity)),
            ]
        } else {
            assert!(args.len() == 1);
            vec![self.loc(args[0], self.tp(args[0]))]
        };

        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        self.rm.before_call(
            &[],
            SAVE_ALL_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.xrm.before_call(
            &[],
            SAVE_ALL_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );

        let result_tp = op.opcode.result_type();
        let result_loc = if result_tp != Type::Void {
            let dst = dst.unwrap_or(op.pos().get());
            let r = if result_tp == Type::Float {
                self.xrm.after_call(dst, &mut self.longevity)
            } else {
                self.rm.after_call(dst, &mut self.longevity)
            };
            Some(Loc::Reg(r))
        } else {
            None
        };
        let gcmap = self.get_gcmap(&[call_result_gpr()], false) as usize;
        self.perform_with_gcmap_ptr(i, arglocs, result_loc, gcmap, output);
    }

    /// `emit_call_from_arglocs` silently falls back to all-Int argument types
    /// when the descr's arity disagrees with the call, which would place a
    /// float argument in a GPR. Every cond-call-value producer builds the arg
    /// list and `arg_types` from the same source, so check it here rather than
    /// letting the emitter guess — `consider_call_j2` carries the same check
    /// for plain calls. `num_args` counts the leading value and function
    /// operands, which the descr does not describe.
    fn check_cond_call_value_descr_arity(op: &Op, num_args: usize) {
        if !op.opcode.is_cond_call_value() {
            return;
        }
        let descr_arc = op.getdescr().expect("cond_call_value without CallDescr");
        let calldescr = descr_arc
            .as_call_descr()
            .expect("cond_call_value without CallDescr");
        assert_eq!(calldescr.arg_types().len(), num_args - 2);
    }

    #[allow(dead_code)] // x86/regalloc.py consider_raw_call_like
    fn consider_raw_call_like(
        &mut self,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
        save_regs: u8,
    ) {
        Self::check_cond_call_value_descr_arity(op, op.num_args());
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        self.rm.before_call(
            &[],
            save_regs,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.xrm.before_call(
            &[],
            save_regs,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );

        let mut arglocs = Vec::new();
        for arg in op.args_slice().iter() {
            let arg = arg.to_opref();
            let tp = self.tp(arg);
            arglocs.push(self.loc(arg, tp));
        }

        let result_tp = op.opcode.result_type();
        let result_loc = if result_tp != Type::Void {
            let r = if result_tp == Type::Float {
                self.xrm.after_call(op.pos().get(), &mut self.longevity)
            } else {
                self.rm.after_call(op.pos().get(), &mut self.longevity)
            };
            Some(Loc::Reg(r))
        } else {
            None
        };
        self.perform(i, arglocs, result_loc, output);
    }

    fn consider_raw_call_like_j2(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
        save_regs: u8,
    ) {
        Self::check_cond_call_value_descr_arity(op, args.len());
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        self.rm.before_call(
            &[],
            save_regs,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.xrm.before_call(
            &[],
            save_regs,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );

        let mut arglocs = Vec::new();
        for &arg in args {
            let tp = self.tp(arg);
            arglocs.push(self.loc(arg, tp));
        }

        let result_tp = op.opcode.result_type();
        let result_loc = if result_tp != Type::Void {
            let dst = dst.unwrap_or(op.pos().get());
            let r = if result_tp == Type::Float {
                self.xrm.after_call(dst, &mut self.longevity)
            } else {
                self.rm.after_call(dst, &mut self.longevity)
            };
            Some(Loc::Reg(r))
        } else {
            None
        };
        self.perform(i, arglocs, result_loc, output);
    }

    /// aarch64/regalloc.py prepare_op_call_malloc_nursery / x86/regalloc.py consider_call_malloc_nursery
    /// prepare/consider_call_malloc_nursery parity. Move or spill only the
    /// nursery bump registers. The AArch64 emitter uses x0/x1 plus reserved IP
    /// scratch registers; collecting slow paths save/restore the other
    /// managed registers on both native architectures.
    #[allow(dead_code)] // x86/regalloc.py consider_call_malloc_nursery
    fn consider_call_malloc_nursery(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        // aarch64/regalloc.py:962: spill_or_move_registers_before_call([r.x0, r.x1])
        self.rm.spill_or_move_registers_before_call(
            &MALLOC_NURSERY_CLOBBER,
            &[],
            SAVE_DEFAULT_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        // aarch64/regalloc.py: force_allocate_reg(op, selected_reg=r.x0)
        // x86/regalloc.py: force_allocate_reg(op, selected_reg=ecx)
        let result_reg = self.rm.force_allocate_reg(
            op.pos().get(),
            &[],
            Some(MALLOC_NURSERY_RESULT),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        // aarch64/regalloc.py: t = TempInt();
        //   force_allocate_reg(t, selected_reg=r.x1)
        // x86/regalloc.py: tmp_box = TempVar();
        //   force_allocate_reg(tmp_box, selected_reg=edx)
        // The nursery bump path needs the second register
        // (`MALLOC_NURSERY_CLOBBER[1]`) reserved before `get_gcmap` so that
        // the gcmap excludes it as a managed slot. The reservation also
        // aligns the regalloc lifetime contract with RPython.
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        self.rm.force_allocate_reg(
            tmp,
            &[],
            Some(MALLOC_NURSERY_CLOBBER[1]),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
        // aarch64/regalloc.py:968: sizeloc = size_box.getint()
        let size_val = self.const_value(op.arg(0).to_opref());
        self.perform_with_gcmap(
            i,
            [Loc::immed(size_val)],
            Some(Loc::Reg(result_reg)),
            output,
        );
    }

    fn consider_call_malloc_nursery_j2(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        self.rm.spill_or_move_registers_before_call(
            &MALLOC_NURSERY_CLOBBER,
            &[],
            SAVE_DEFAULT_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        let result_reg = self.rm.force_allocate_reg(
            dst.unwrap_or(OpRef::NONE),
            &[],
            Some(MALLOC_NURSERY_RESULT),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        // aarch64/regalloc.py:965-966 / x86/regalloc.py:1025-1026:
        //   t = TempInt() / TempVar(); reserve the second clobber register.
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        self.rm.force_allocate_reg(
            tmp,
            &[],
            Some(MALLOC_NURSERY_CLOBBER[1]),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
        let size_val = args.first().map(|&arg| self.const_value(arg)).unwrap_or(0);
        self.perform_with_gcmap(
            i,
            [Loc::immed(size_val)],
            Some(Loc::Reg(result_reg)),
            output,
        );
    }

    /// aarch64/regalloc.py prepare_op_call_malloc_nursery_varsize_frame parity.
    #[allow(dead_code)] // x86/regalloc.py consider_call_malloc_nursery_varsize_frame
    fn consider_call_malloc_nursery_varsize_frame(
        &mut self,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        // aarch64/regalloc.py: sizeloc = make_sure_var_in_reg(size_box)
        let size_box = op.arg(0).to_opref();
        let sizeloc = self.make_sure_var_in_reg(size_box, Type::Int, &[], None, false);
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        // aarch64/regalloc.py:985: only move values away from x0/x1.
        // The slow path saves/restores all managed registers except
        // x0/x1, so spilling every register here is both unnecessary and
        // diverges from the upstream calling convention. The x86 emit
        // mirrors the same contract by wrapping the slowpath with
        // `push_all_regs_to_jitframe` / `pop_all_regs_from_jitframe`.
        self.rm.spill_or_move_registers_before_call(
            &MALLOC_NURSERY_CLOBBER,
            &[],
            SAVE_DEFAULT_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        // aarch64/regalloc.py:986
        self.possibly_free_var(size_box, Type::Int);
        // aarch64/regalloc.py:988
        let result_reg = self.rm.force_allocate_reg(
            op.pos().get(),
            &[],
            Some(MALLOC_NURSERY_RESULT),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        // aarch64/regalloc.py t = TempInt() — reserve x1 as the
        // malloc temp and exclude it from the gcmap before emitting
        // malloc_cond_varsize_frame.
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        self.rm.force_allocate_reg(
            tmp,
            &[],
            Some(MALLOC_NURSERY_CLOBBER[1]),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
        self.perform_with_gcmap(i, [sizeloc], Some(Loc::Reg(result_reg)), output);
    }

    fn consider_call_malloc_nursery_varsize_frame_j2(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let Some(&size_arg) = args.first() else {
            output.push(RegAllocOp::Skip);
            return;
        };
        let sizeloc = self.make_sure_var_in_reg(size_arg, Type::Int, &[], None, false);
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        self.rm.spill_or_move_registers_before_call(
            &MALLOC_NURSERY_CLOBBER,
            &[],
            SAVE_DEFAULT_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.possibly_free_var(size_arg, Type::Int);
        let result_reg = self.rm.force_allocate_reg(
            dst.unwrap_or(OpRef::NONE),
            &[],
            Some(MALLOC_NURSERY_RESULT),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        // aarch64/regalloc.py t = TempInt()
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        self.rm.force_allocate_reg(
            tmp,
            &[],
            Some(MALLOC_NURSERY_CLOBBER[1]),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
        self.perform_with_gcmap(i, [sizeloc], Some(Loc::Reg(result_reg)), output);
    }

    /// aarch64/regalloc.py prepare_op_call_malloc_nursery_varsize parity.
    #[allow(dead_code)] // x86/regalloc.py consider_call_malloc_nursery_varsize
    fn consider_call_malloc_nursery_varsize(
        &mut self,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        // `consider_call_malloc_nursery_varsize` /
        // `prepare_op_call_malloc_nursery_varsize`: spill only when
        // `stays_alive(length_box)`. A length that dies here is unbound
        // with no stack sync, and `loc` would then allocate an empty slot.
        let length_box = op.arg(2).to_opref();
        if self.rm.stays_alive(length_box, &self.longevity) {
            let type_index = OpTypeIndex::from_parts(
                self.inputargs,
                self.operations,
                &self.inputarg_pos,
                &self.op_pos,
            );
            self.rm.spill_or_move_registers_before_call(
                &MALLOC_NURSERY_CLOBBER,
                &[],
                VARSIZE_MALLOC_SAVE_REGS,
                &mut self.longevity,
                &mut self.fm,
                &mut self.pending_moves,
                &type_index,
            );
        }
        // aarch64/regalloc.py:1018
        let result_reg = self.rm.force_allocate_reg(
            op.pos().get(),
            &[],
            Some(MALLOC_NURSERY_RESULT),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        // aarch64/regalloc.py `prepare_op_call_malloc_nursery_varsize`: tmp_box = TempVar();
        //   force_allocate_reg(tmp_box, selected_reg=r.x1)
        //   gcmap = self.get_gcmap([r.x0, r.x1])
        //   possibly_free_var(tmp_box)
        // The bump-pointer fast path writes the new nursery_free into x1.
        // Without this reservation a live Ref can stay bound to x1, so a
        // later guard's faillocs / the collecting slowpath gcmap would
        // publish that derived address as a GCREF (`jf_frame[1]`).
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        self.rm.force_allocate_reg(
            tmp,
            &[],
            Some(MALLOC_NURSERY_CLOBBER[1]),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
        // aarch64/regalloc.py: lengthloc = self.rm.loc(length_box)
        let lengthloc = self.loc(length_box, Type::Int);
        // aarch64/regalloc.py:1030: itemsize = op.getarg(1).getint()
        let itemsize = self.const_value(op.arg(1).to_opref());
        // aarch64/regalloc.py:1033: kind = op.getarg(0).getint()
        let kind = self.const_value(op.arg(0).to_opref());
        self.perform_with_gcmap(
            i,
            vec![lengthloc, Loc::immed(itemsize), Loc::immed(kind)],
            Some(Loc::Reg(result_reg)),
            output,
        );
    }

    fn consider_call_malloc_nursery_varsize_j2(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        // `consider_call_malloc_nursery_varsize` /
        // `prepare_op_call_malloc_nursery_varsize`: spill only when
        // `stays_alive(length_box)`. See `consider_call_malloc_nursery_varsize`.
        if let Some(&length_box) = args.get(2)
            && self.rm.stays_alive(length_box, &self.longevity)
        {
            let type_index = OpTypeIndex::from_parts(
                self.inputargs,
                self.operations,
                &self.inputarg_pos,
                &self.op_pos,
            );
            self.rm.spill_or_move_registers_before_call(
                &MALLOC_NURSERY_CLOBBER,
                &[],
                VARSIZE_MALLOC_SAVE_REGS,
                &mut self.longevity,
                &mut self.fm,
                &mut self.pending_moves,
                &type_index,
            );
        }
        let result_reg = self.rm.force_allocate_reg(
            dst.unwrap_or(OpRef::NONE),
            &[],
            Some(MALLOC_NURSERY_RESULT),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        // aarch64/regalloc.py `prepare_op_call_malloc_nursery_varsize` TempVar in x1 — see
        // `consider_call_malloc_nursery_varsize`.
        let tmp = self.fresh_temp_var();
        self.longevity
            .set(tmp, Lifetime::new(self.rm.position, self.rm.position));
        self.rm.force_allocate_reg(
            tmp,
            &[],
            Some(MALLOC_NURSERY_CLOBBER[1]),
            false,
            &mut self.longevity,
            &mut self.fm,
        );
        self.rm
            .possibly_free_var(tmp, &mut self.longevity, &mut self.fm, Type::Int);
        let lengthloc = args
            .get(2)
            .map(|&arg| self.loc(arg, Type::Int))
            .unwrap_or_else(|| Loc::immed(0));
        let itemsize = args.get(1).map(|&arg| self.const_value(arg)).unwrap_or(0);
        let kind = args.first().map(|&arg| self.const_value(arg)).unwrap_or(0);
        self.perform_with_gcmap(
            i,
            vec![lengthloc, Loc::immed(itemsize), Loc::immed(kind)],
            Some(Loc::Reg(result_reg)),
            output,
        );
    }

    /// x86/regalloc.py consider_jump
    #[allow(dead_code)] // x86/regalloc.py consider_jump
    fn consider_jump(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        // x86/regalloc.py:1306-1309: descr = op.getdescr(); self.jump_target_descr = descr
        let descr_id = op.getdescr().as_ref().map(descr_identity);
        self.final_jump_args = descr_id.map(|id| {
            (
                id,
                op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect()),
            )
        });
        self.jump_target_descr = descr_id;
        let mut locs = Vec::new();
        for arg in op.args_slice().iter() {
            let arg = arg.to_opref();
            let tp = self.tp(arg);
            locs.push(self.loc_must_exist(arg, tp));
        }
        self.perform(i, locs, None, output);
    }

    fn consider_jump_j2(
        &mut self,
        args: &[OpRef],
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let descr_id = op.getdescr().as_ref().map(descr_identity);
        self.final_jump_args = descr_id.map(|id| (id, args.to_vec()));
        self.jump_target_descr = descr_id;
        let mut locs = Vec::new();
        for &arg in args {
            let tp = self.tp(arg);
            locs.push(self.loc_must_exist(arg, tp));
        }
        self.perform(i, locs, None, output);
    }

    /// x86/regalloc.py consider_label
    #[allow(dead_code)] // x86/regalloc.py consider_label
    fn consider_label(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let position = self.rm.position;
        for arg in op.args_slice().iter() {
            let arg = arg.to_opref();
            let tp = self.tp(arg);
            if self
                .longevity
                .get(arg)
                .is_some_and(|lt| lt.is_last_real_use_before(position))
            {
                self.force_spill_var(arg, tp);
            }
        }

        let mut locs = Vec::new();
        for arg in op.args_slice().iter() {
            let arg = arg.to_opref();
            let tp = self.tp(arg);
            let loc = self.loc(arg, tp);
            if let Loc::Reg(_) = loc {
                self.fm.mark_as_free(arg, tp, &mut self.longevity)
            }
            locs.push(loc);
        }

        // x86/regalloc.py:1407-1409 / aarch64/regalloc.py:712-715
        //   if final_jump_op.getdescr() is descr →
        //     _compute_hint_{frame_,}locations_from_descr(descr)
        //
        // Same-loop case: the closing JUMP targets this very Label.
        // Hints come from the Label's target_arglocs (just built into
        // `locs`) — what each inputarg's *desired* location at the
        // back-edge is.
        //
        // The two upstream backends differ on what is hinted:
        //   x86/regalloc.py _compute_hint_locations_from_descr
        //     hints BOTH FrameLoc → add_frame_pos_hint AND
        //     RegLoc → longevity.fixed_register
        //   aarch64/regalloc.py _compute_hint_frame_locations_from_descr
        //     hints ONLY FrameLoc (`if loc.is_stack()`); reg hints are
        //     intentionally not emitted on aarch64
        //
        // Mirror the per-arch difference exactly via #[cfg].
        if let Some(label_id) = op.getdescr().as_ref().map(descr_identity)
            && self
                .final_jump_args
                .as_ref()
                .is_some_and(|(jump_id, _)| *jump_id == label_id)
        {
            let jump_args = self.final_jump_args.as_ref().unwrap().1.clone();
            #[cfg(target_arch = "x86_64")]
            let position = self.final_jump_op_position;
            #[cfg(target_arch = "x86_64")]
            let mut hinted: Vec<OpRef> = Vec::new();
            for (iarg, target_loc) in jump_args.iter().zip(locs.iter()) {
                if iarg.is_constant() {
                    continue;
                }
                match *target_loc {
                    Loc::Frame(frame_loc) => {
                        self.fm
                            .add_frame_pos_hint(*iarg, frame_loc, &mut self.longevity);
                    }
                    #[cfg(target_arch = "x86_64")]
                    Loc::Reg(r) => {
                        if !hinted.contains(iarg) {
                            hinted.push(*iarg);
                            self.longevity.fixed_register(position, r, Some(*iarg));
                        }
                    }
                    _ => {}
                }
            }
        }
        self.perform(i, locs, None, output);
    }

    fn consider_label_j2(
        &mut self,
        args: &[OpRef],
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let position = self.rm.position;
        for &arg in args {
            let tp = self.tp(arg);
            if self
                .longevity
                .get(arg)
                .is_some_and(|lt| lt.is_last_real_use_before(position))
            {
                self.force_spill_var(arg, tp);
            }
        }

        // x86/regalloc.py `consider_label`: we need to make sure that no
        // variable is stored in ebp
        #[cfg(target_arch = "x86_64")]
        {
            let ebp = self.rm.frame_reg;
            for &arg in args {
                let tp = self.tp(arg);
                if matches!(self.loc(arg, tp), Loc::Reg(r) if r == ebp) {
                    let loc2 = self.fm.loc(arg, tp, false, &mut self.longevity);
                    self.rm.spill_moves.push((Loc::Reg(ebp), Loc::Frame(loc2)));
                }
            }
            self.rm.box_currently_in_frame_reg = None;
        }

        let mut locs = Vec::new();
        for &arg in args {
            let tp = self.tp(arg);
            let loc = self.loc(arg, tp);
            #[cfg(target_arch = "x86_64")]
            debug_assert!(!matches!(loc, Loc::Reg(r) if r == self.rm.frame_reg));
            if let Loc::Reg(_) = loc {
                self.fm.mark_as_free(arg, tp, &mut self.longevity)
            }
            locs.push(loc);
        }

        if let Some(label_id) = op.getdescr().as_ref().map(descr_identity)
            && self
                .final_jump_args
                .as_ref()
                .is_some_and(|(jump_id, _)| *jump_id == label_id)
        {
            let jump_args = self.final_jump_args.as_ref().unwrap().1.clone();
            #[cfg(target_arch = "x86_64")]
            let position = self.final_jump_op_position;
            #[cfg(target_arch = "x86_64")]
            let mut hinted: Vec<OpRef> = Vec::new();
            for (iarg, target_loc) in jump_args.iter().zip(locs.iter()) {
                if iarg.is_constant() {
                    continue;
                }
                match *target_loc {
                    Loc::Frame(frame_loc) => {
                        self.fm
                            .add_frame_pos_hint(*iarg, frame_loc, &mut self.longevity);
                    }
                    #[cfg(target_arch = "x86_64")]
                    Loc::Reg(r) => {
                        if !hinted.contains(iarg) {
                            hinted.push(*iarg);
                            self.longevity.fixed_register(position, r, Some(*iarg));
                        }
                    }
                    _ => {}
                }
            }
        }
        self.perform(i, locs, None, output);
    }

    /// force_token: result = frame pointer (EBP)
    #[allow(dead_code)] // x86/regalloc.py consider_force_token
    fn consider_force_token(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let result_loc =
            Loc::Reg(self.force_allocate_reg(op.pos().get(), Type::Ref, &[], None, false));
        self.perform(i, [] as [Loc; 0], Some(result_loc), output);
    }

    fn consider_force_token_j2(&mut self, dst: OpRef, i: usize, output: &mut Vec<RegAllocOp>) {
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, Type::Ref, &[], None, false));
        self.perform(i, [] as [Loc; 0], Some(result_loc), output);
    }

    /// x86/regalloc.py `consider_load_effective_address` /
    /// aarch64/regalloc.py `prepare_op_load_effective_address`.
    /// Keep `p0` and `i0` in registers and forbid the result from
    /// reusing either, so LEA cannot overwrite the index while
    /// materializing a framed base.
    #[allow(dead_code)] // x86/regalloc.py consider_load_effective_address
    fn consider_load_effective_address(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.args_slice().iter().map(|a| a.to_opref()).collect();
        self.consider_load_effective_address_j2(op.pos().get(), &args, i, output);
    }

    fn consider_load_effective_address_j2(
        &mut self,
        dst: OpRef,
        args: &[OpRef],
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let [p0, i0, baseofs, shift] = match args {
            [a, b, c, d, ..] => [*a, *b, *c, *d],
            other => panic!("LoadEffectiveAddress expects 4 args, got {other:?}"),
        };
        let ploc = self.make_sure_var_in_reg(p0, Type::Int, &[i0], None, false);
        let iloc = self.make_sure_var_in_reg(i0, Type::Int, &[p0], None, false);
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, Type::Int, &[p0, i0], None, false));
        let locs = vec![
            ploc,
            iloc,
            self.loc(baseofs, Type::Int),
            self.loc(shift, Type::Int),
        ];
        self.perform(i, locs, Some(result_loc), output);
    }

    /// No-arg result (save_exception, save_exc_class)
    #[allow(dead_code)] // x86/regalloc.py consider_no_arg_result
    fn consider_no_arg_result(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(op.pos().get(), tp, &[], None, false));
        self.perform(i, [] as [Loc; 0], Some(result_loc), output);
    }

    fn consider_no_arg_result_j2(
        &mut self,
        dst: OpRef,
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let tp = op.opcode.result_type();
        let result_loc = Loc::Reg(self.force_allocate_reg(dst, tp, &[], None, false));
        self.perform(i, [] as [Loc; 0], Some(result_loc), output);
    }

    /// Discard op with 3 args (zero_array, strsetitem, etc.)
    #[allow(dead_code)] // x86/regalloc.py consider_discard_3args
    fn consider_discard_3args(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        let mut locs = Vec::new();
        for &arg in &args {
            let tp = self.tp(arg);
            locs.push(self.make_sure_var_in_reg(arg, tp, &args, None, false));
        }
        self.perform_discard(i, locs, output);
    }

    fn consider_discard_3args_j2(
        &mut self,
        args: &[OpRef],
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let mut locs = Vec::new();
        for &arg in args {
            let tp = self.tp(arg);
            locs.push(self.make_sure_var_in_reg(arg, tp, args, None, false));
        }
        self.perform_discard(i, locs, output);
    }

    /// Select the inline-store or residual-memset ZERO_ARRAY path using the
    /// same per-backend limits as upstream.  x86/regalloc.py consider_zero_array does
    /// this split in regalloc; aarch64/opassembler.py emit_op_zero_array does it while
    /// emitting, but the residual call still needs `before_call` here so only
    /// live caller-saved registers are preserved.
    fn zero_array_uses_memset(&self, op: &Op) -> bool {
        let size = op.arg(2).to_opref();
        if !size.is_constant() {
            return true;
        }
        let mut size = self.const_value(size);
        let scale = op.arg(4).to_opref();
        if !scale.is_constant() {
            return true;
        }
        size *= self.const_value(scale);

        #[cfg(target_arch = "x86_64")]
        {
            // x86/regalloc.py:1451 — 16 eight-byte chunks.
            !(0..=16 * 8).contains(&size)
        }

        #[cfg(target_arch = "aarch64")]
        {
            // aarch64/opassembler.py:796-812 — choose the natural STR width
            // from the array item size.  A constant start permits aligned
            // runs of eight-byte STRs even for byte/halfword/word arrays.
            let item_size = op.with_array_descr(|ad| ad.item_size() as i64).unwrap_or(8);
            let item_size = if item_size & 1 != 0 {
                1
            } else if item_size & 2 != 0 {
                2
            } else if item_size & 4 != 0 {
                4
            } else {
                8
            };
            let start_is_constant = op.arg(1).to_opref().is_constant();
            let limit = if item_size < 8 && start_is_constant {
                8
            } else {
                item_size
            };
            size < 0 || size > 14 * limit
        }
    }

    pub(crate) fn preserve_for_zero_array_memset(&mut self) {
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        self.rm.before_call(
            &[],
            SAVE_DEFAULT_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
        self.xrm.before_call(
            &[],
            SAVE_DEFAULT_REGS,
            &mut self.longevity,
            &mut self.fm,
            &mut self.pending_moves,
            &type_index,
        );
    }

    #[allow(dead_code)] // x86/regalloc.py consider_zero_array
    fn consider_zero_array(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let args: Vec<OpRef> = op.with_arglist(|args| args.iter().map(|a| a.to_opref()).collect());
        let mut locs = Vec::with_capacity(args.len());
        for &arg in &args {
            locs.push(self.make_sure_var_in_reg(arg, self.tp(arg), &args, None, false));
        }
        if self.zero_array_uses_memset(op) {
            self.preserve_for_zero_array_memset();
        }
        self.perform_discard(i, locs, output);
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn consider_zero_array_j2(
        &mut self,
        args: &[OpRef],
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let mut locs = Vec::with_capacity(args.len());
        for &arg in args {
            locs.push(self.make_sure_var_in_reg(arg, self.tp(arg), args, None, false));
        }
        if self.zero_array_uses_memset(op) {
            self.preserve_for_zero_array_memset();
        }
        self.perform_discard(i, locs, output);
    }

    /// aarch64/regalloc.py `_prepare_op_cond_call` / `prepare_op_cond_call_value_i`
    /// (`prepare_op_cond_call_value_r` is the same) and x86/regalloc.py
    /// `consider_cond_call` (aliased as `consider_cond_call_value_i` / `_r`).
    ///
    /// The result shares args[0]'s register via `force_result_in_reg`
    /// (`forbidden_vars=args[2:]`). The callee address stays an Immed on
    /// the op. The helper returns a plain word — no `before_call` / raw-call
    /// result-in-x0 rewrite.
    fn consider_cond_call_value_j2(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        Self::check_cond_call_value_descr_arity(op, args.len());
        assert!(
            (2..=6).contains(&args.len()),
            "COND_CALL_VALUE takes 0..=4 extra arguments (aarch64/regalloc.py _prepare_op_cond_call)"
        );
        #[cfg(target_arch = "x86_64")]
        {
            self.consider_cond_call_value_x86(dst, args, op, i, output);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.consider_cond_call_value_aarch64(dst, args, op, i, output);
        }
    }

    /// x86/regalloc.py `consider_cond_call` COND_CALL_VALUE arm.
    /// Extra-arg locations are captured before `force_result_in_reg`;
    /// `CondCallSlowPath.generate_body` remaps them into
    /// `cond_call_register_arguments` on the taken path only.
    /// `cond_call` then snapshots `rm.reg_bindings` for the save mask.
    #[cfg(target_arch = "x86_64")]
    fn consider_cond_call_value_x86(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let extra_locs: Vec<Loc> = args[2..]
            .iter()
            .map(|&arg| self.loc(arg, self.tp(arg)))
            .collect();
        let value = args[0];
        let result_v = dst.unwrap_or(op.pos().get());
        let forbidden: Vec<OpRef> = args[2..].to_vec();
        let resloc = self.rm.force_result_in_reg(
            result_v,
            value,
            self.tp(value),
            &forbidden,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        let res_reg = match resloc {
            Loc::Reg(r) => r,
            other => panic!(
                "force_result_in_reg must return a register (consider_cond_call), got {other:?}"
            ),
        };
        let gcmap = self.get_gcmap(&[res_reg], false) as usize;
        // Assembler prefix stays `[argloc, resloc]`; x86 tests `resloc`.
        // Trailing immediates are `cond_call`'s `variant_num` and
        // `should_be_saved` bitmask over GPR numbers, captured after
        // `force_result_in_reg` while the bindings are still live.
        let mut locs = Vec::with_capacity(extra_locs.len() + 4);
        locs.push(resloc);
        locs.push(resloc);
        locs.extend(extra_locs);
        locs.push(Loc::immed(self.cond_call_slowpath_variant()));
        locs.push(Loc::immed(self.cond_call_should_be_saved()));
        self.perform_with_gcmap_ptr(i, locs, Some(resloc), gcmap, output);
    }

    /// aarch64/regalloc.py `_prepare_op_cond_call`: extra args (from index 2)
    /// go into `argument_regs`. Arglocs are `[argloc, resloc]`.
    #[cfg(not(target_arch = "x86_64"))]
    fn consider_cond_call_value_aarch64(
        &mut self,
        dst: Option<OpRef>,
        args: &[OpRef],
        op: &Op,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let argument_regs = cond_call_argument_regs();
        let mut args_so_far: Vec<OpRef> = Vec::new();
        for (j, &arg) in args.iter().enumerate().skip(2) {
            let reg = argument_regs[j - 2];
            self.make_sure_var_in_reg(arg, self.tp(arg), &args_so_far, Some(reg), false);
            args_so_far.push(arg);
        }
        let value = args[0];
        let argloc = self.make_sure_var_in_reg(value, self.tp(value), &args_so_far, None, false);
        let result_v = dst.unwrap_or(op.pos().get());
        let forbidden: Vec<OpRef> = args.iter().skip(2).copied().collect();
        let resloc = self.rm.force_result_in_reg(
            result_v,
            value,
            self.tp(value),
            &forbidden,
            &mut self.longevity,
            &mut self.fm,
            &self.constants,
            &mut self.pending_moves,
        );
        let res_reg = match resloc {
            Loc::Reg(r) => r,
            other => panic!(
                "force_result_in_reg must return a register (aarch64/regalloc.py _prepare_op_cond_call), got {other:?}"
            ),
        };
        // aarch64/opassembler.py `_emit_op_cond_call`:
        // `gcmap = self._regalloc.get_gcmap([res_loc])` — exclude the result
        // register; on the miss path it is overwritten with the helper word.
        let gcmap = self.get_gcmap(&[res_reg], false) as usize;
        let locs = [argloc, resloc];
        self.perform_with_gcmap_ptr(i, locs, Some(resloc), gcmap, output);
    }

    /// `aarch64/opassembler.py _emit_op_cond_call`:
    /// `gcmap = self._regalloc.get_gcmap([res_loc])`. CondCallN has no
    /// result. aarch64 spills every register into the jitframe on the
    /// taken path and publishes this map so `force_now` allocations
    /// forward those slots. x86 `cond_call` instead calls
    /// `cond_call_slowpath[floats * 2 + callee_only]`; that index is a
    /// trailing immediate, captured while the bindings are still live,
    /// next to the `should_be_saved` GPR bitmask (`cond_call`).
    fn consider_cond_call_n(&mut self, args: &[OpRef], i: usize, output: &mut Vec<RegAllocOp>) {
        // x86/regalloc.py `consider_cond_call`: extra-arg locations first
        // (`self.loc` of args[2:]), then the gcmap. The predicate and func
        // sit in the assembler prefix; extras are remapped on the taken
        // path only (`CondCallSlowPath.generate_body`).
        let mut locs = Vec::new();
        for &arg in args {
            let tp = self.tp(arg);
            locs.push(self.loc(arg, tp));
        }
        let gcmap = self.get_gcmap(&[], false) as usize;
        #[cfg(target_arch = "x86_64")]
        {
            locs.push(Loc::immed(self.cond_call_slowpath_variant()));
            locs.push(Loc::immed(self.cond_call_should_be_saved()));
        }
        self.perform_with_gcmap_ptr(i, locs, None, gcmap, output);
    }

    /// `cond_call`: `variant_num = floats * 2 + callee_only`.
    /// The for/else over `rm.reg_bindings` sets `callee_only` when every
    /// bound core register is in `save_around_call_regs` (no bindings
    /// takes that arm). `floats` is `len(xrm.reg_bindings) != 0`.
    #[cfg(target_arch = "x86_64")]
    fn cond_call_slowpath_variant(&self) -> i64 {
        let mut callee_only = true;
        for (_, reg) in self.rm.reg_bindings_iter() {
            if !self.rm.save_around_call_regs.contains(&reg) {
                callee_only = false;
                break;
            }
        }
        let floats = self.xrm.reg_bindings_len() > 0;
        (usize::from(floats) * 2 + usize::from(callee_only)) as i64
    }

    /// `cond_call`: `should_be_saved = rm.reg_bindings.values()`, encoded
    /// as a bitmask over GPR numbers (`1 << gpr.value`). Travels as the
    /// trailing immediate next to `variant_num`.
    #[cfg(target_arch = "x86_64")]
    fn cond_call_should_be_saved(&self) -> i64 {
        let mut mask = 0i64;
        for (_, reg) in self.rm.reg_bindings_iter() {
            mask |= 1i64 << reg.value;
        }
        mask
    }

    /// Generic discard with N args
    #[allow(dead_code)] // x86/regalloc.py consider_discard_nargs
    fn consider_discard_nargs(&mut self, op: &Op, i: usize, output: &mut Vec<RegAllocOp>) {
        let mut locs = Vec::new();
        for arg in op.args_slice().iter() {
            let arg = arg.to_opref();
            let tp = self.tp(arg);
            locs.push(self.loc(arg, tp));
        }
        self.perform_discard(i, locs, output);
    }

    fn consider_discard_nargs_j2(
        &mut self,
        args: &[OpRef],
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let mut locs = Vec::new();
        for &arg in args {
            let tp = self.tp(arg);
            locs.push(self.loc(arg, tp));
        }
        self.perform_discard(i, locs, output);
    }

    fn consider_cond_call_gc_wb_j2(
        &mut self,
        args: &[OpRef],
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let mut arglocs = Vec::new();
        for (idx, &arg) in args.iter().enumerate() {
            let tp = if idx == 0 { Type::Ref } else { Type::Int };
            arglocs.push(self.make_sure_var_in_reg(arg, tp, args, None, false));
        }
        // `_write_barrier_fastpath` picks the helper that also saves the
        // float registers when `self._regalloc.xrm.reg_bindings` (x86) /
        // `vfprm.reg_bindings` (aarch64) is non-empty. The assembler runs
        // after this pass, so the answer travels as a trailing immediate.
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        arglocs.push(Loc::immed(i64::from(self.xrm.reg_bindings_len() > 0)));
        self.perform_discard(i, arglocs, output);
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
    debug_assert!(valid_addressing_size(size));
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

#[cfg(test)]
mod tests {
    use super::*;

    use majit_ir::operand::Operand;
    use majit_ir::{InputArg, Op, OpCode, OpRc, OpRef, Type};

    use majit_ir::forwarding::bound_operand_from_opref as rb;

    fn rcs(ops: Vec<Op>) -> Vec<OpRc> {
        ops.into_iter().map(OpRc::new).collect()
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

    fn has_move_from(ra_ops: &[RegAllocOp], expected: &Loc) -> bool {
        ra_ops.iter().any(|ra_op| {
            if let RegAllocOp::Move { src, .. } = ra_op {
                loc_eq(src, expected)
            } else {
                false
            }
        })
    }

    #[test]
    fn test_compute_vars_longevity_simple() {
        // i0 = inputarg
        // i1 = int_add(i0, i0)  [pos 0]
        // guard_true(i1)         [pos 1, fail_args=[i0]]
        // jump(i1)               [pos 2]
        let i0 = OpRef::int_op(100);
        let i1 = OpRef::int_op(0);
        let _i2 = OpRef::int_op(1);
        let _i3 = OpRef::int_op(2);

        let inputargs = vec![InputArg::from_type_rc(Type::Int, i0.raw())];
        let ops = vec![
            make_op(OpCode::IntAdd, 0, &[i0, i0]),
            make_guard(OpCode::GuardTrue, 1, &[i1], &[i0]),
            make_op(OpCode::Jump, 2, &[i1]),
        ];

        let longevity = compute_vars_longevity(&inputargs, &ops);

        // i0: defined at -1 (input), last used at 1 (guard failarg), real usages: [0, 0]
        // (used twice as arg to int_add at position 0)
        let lt_i0 = longevity.get(i0).unwrap();
        assert_eq!(lt_i0.last_usage, 1);
        assert_eq!(lt_i0.real_usages.as_ref().unwrap().as_slice(), &[0, 0]);

        // i1: defined at 0, last used at 2 (jump), real usages: [1]
        let lt_i1 = longevity.get(i1).unwrap();
        assert_eq!(lt_i1.definition_pos, 0);
        assert_eq!(lt_i1.last_usage, 2);
        assert_eq!(lt_i1.real_usages.as_ref().unwrap().as_slice(), &[1]);
    }

    #[test]
    fn ref_inputarg_longevity_follows_failargs() {
        let i0 = OpRef::input_arg_typed(0, Type::Int);
        let vm = OpRef::input_arg_typed(1, Type::Ref);
        let add = OpRef::int_op(0);
        let inputargs = vec![
            InputArg::from_type_rc(Type::Int, 0),
            InputArg::from_type_rc(Type::Ref, 1),
        ];
        let ops = vec![
            make_op(OpCode::IntAdd, 0, &[i0, i0]),
            make_guard(OpCode::GuardTrue, 1, &[add], &[vm]),
            make_op(OpCode::Jump, 2, &[add]),
        ];
        let longevity = compute_vars_longevity(&inputargs, &ops);
        assert_eq!(
            longevity.get(vm).unwrap().last_usage,
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
        let first = OpRef::input_arg_ref(1218);
        let second = OpRef::input_arg_ref(1220);
        let mut longevity = LifetimeManager::new();
        longevity.set(first, Lifetime::new(-1, 1));
        longevity.set(second, Lifetime::new(-1, 1));
        let shared = regs[regs.len() - 1];
        rm.reg_bindings_set(first, shared, &mut longevity);
        rm.reg_bindings_set(second, shared, &mut longevity);
        assert_eq!(rm.reg_bindings_get(first, &longevity), Some(shared));
        assert_eq!(rm.reg_bindings_get(second, &longevity), Some(shared));
    }

    #[test]
    fn test_lifetime_next_real_usage() {
        let mut lt = Lifetime::new(0, 10);
        lt.real_usages = Some(SmallVec::from_slice(&[2, 5, 8]));

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
        let v0 = OpRef::int_op(0);
        let v1 = OpRef::int_op(1);
        let v2 = OpRef::int_op(2);
        longevity.set(v0, Lifetime::new(0, 10));
        longevity.set(v1, Lifetime::new(1, 10));
        longevity.set(v2, Lifetime::new(2, 10));

        let mut fm = FrameManager::new(0);
        let loc0 = fm.get_new_loc(v0, Type::Int, &mut longevity);
        let loc1 = fm.get_new_loc(v1, Type::Int, &mut longevity);
        let loc2 = fm.get_new_loc(v2, Type::Int, &mut longevity);

        assert_eq!(loc0.position, 0);
        assert_eq!(loc1.position, 1);
        assert_eq!(loc2.position, 2);
        assert_eq!(fm.get_frame_depth(), 3);
    }

    #[test]
    fn test_register_manager_allocate() {
        let mut longevity = LifetimeManager::new();
        let v0 = OpRef::int_op(0);
        let v1 = OpRef::int_op(1);
        longevity.set(v0, Lifetime::new(0, 5));
        longevity.set(v1, Lifetime::new(1, 5));

        let mut rm = RegisterManager::new(
            vec![EAX, ECX, EDX],
            vec![],
            vec![EAX, ECX, EDX],
            EBP,
            Some(vec![Type::Int, Type::Ref]),
            EAX,
        );
        rm.position = 0;

        let reg0 = rm.try_allocate_reg(v0, None, false, &mut longevity);
        assert!(reg0.is_some());
        let reg0 = reg0.unwrap();

        let reg1 = rm.try_allocate_reg(v1, None, false, &mut longevity);
        assert!(reg1.is_some());
        let reg1 = reg1.unwrap();

        assert_ne!(reg0, reg1);
        assert_eq!(rm.reg_bindings_get(v0, &longevity), Some(reg0));
        assert_eq!(rm.reg_bindings_get(v1, &longevity), Some(reg1));
    }

    #[test]
    fn test_register_spill() {
        let mut longevity = LifetimeManager::new();
        let mut lt0 = Lifetime::new(0, 10);
        lt0.real_usages = Some(SmallVec::from_slice(&[5]));
        let mut lt1 = Lifetime::new(1, 10);
        lt1.real_usages = Some(SmallVec::from_slice(&[6]));
        let mut lt2 = Lifetime::new(2, 10);
        lt2.real_usages = Some(SmallVec::from_slice(&[7]));
        longevity.set(OpRef::int_op(0), lt0);
        longevity.set(OpRef::int_op(1), lt1);
        longevity.set(OpRef::int_op(2), lt2);

        // Only 2 registers available
        let mut rm = RegisterManager::new(vec![EAX, ECX], vec![], vec![EAX, ECX], EBP, None, EAX);
        rm.position = 3;
        let mut fm = FrameManager::new(0);

        let reg0 =
            rm.force_allocate_reg(OpRef::int_op(0), &[], None, false, &mut longevity, &mut fm);
        let reg1 =
            rm.force_allocate_reg(OpRef::int_op(1), &[], None, false, &mut longevity, &mut fm);
        assert!(reg0 != reg1);

        // Third allocation forces a spill
        let reg2 =
            rm.force_allocate_reg(OpRef::int_op(2), &[], None, false, &mut longevity, &mut fm);
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

        let ops = rcs(ops);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let unrelated = all_core_regs()
            .into_iter()
            .find(|reg| !MALLOC_NURSERY_CLOBBER.contains(reg))
            .expect("architecture must expose a managed register outside malloc clobbers");
        ra.rm.force_allocate_reg(
            live_ref,
            &[],
            Some(unrelated),
            false,
            &mut ra.longevity,
            &mut ra.fm,
        );
        ra.pending_moves.clear();
        ra.rm.spill_moves.clear();

        let mut output = Vec::new();
        ra.consider_call_malloc_nursery(&ops[0], 0, &mut output);

        assert_eq!(
            ra.rm.reg_bindings_get(live_ref, &ra.longevity),
            Some(unrelated),
            "fixed nursery allocation must preserve unrelated live bindings"
        );
        assert!(
            !has_move_from(&output, &Loc::Reg(unrelated)),
            "fixed nursery allocation must not spill an unrelated live register"
        );
    }

    #[test]
    fn loc_and_regallocop_sizes() {
        assert!(
            std::mem::size_of::<RegAllocOp>() <= 72,
            "RegAllocOp grew to {} B; PerformDiscardGcStore/PerformGcLoad must stay in the 72 B envelope",
            std::mem::size_of::<RegAllocOp>()
        );
        assert!(std::mem::size_of::<Loc>() <= 16);
    }

    #[test]
    fn test_push_perform_two_locs_uses_shared_arena() {
        // x86/regalloc.py perform(op, [loc, argloc], res) — a 2-element
        // list. A per-op `Vec<Loc>` was the 32 B class on the regex
        // and/or compile path (`push_perform` / `consider_int_ri_j2`).
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

        let ops = rcs(vec![add1, add2, finish]);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let ra_ops = ra.walk_operations().expect("walk_operations");
        let two_loc: Vec<(u32, u32)> = ra_ops
            .iter()
            .filter_map(|ra_op| match ra_op {
                RegAllocOp::Perform {
                    arglocs_start,
                    arglocs_len,
                    ..
                } if *arglocs_len == 2 => Some((*arglocs_start, *arglocs_len)),
                _ => None,
            })
            .collect();
        assert!(
            two_loc.len() >= 2,
            "two IntAdds must each be a 2-loc perform, got {ra_ops:?}"
        );
        assert_eq!(two_loc[0].0, 0);
        assert_eq!(
            two_loc[1].0,
            two_loc[0].0 + two_loc[0].1,
            "second perform is adjacent in the shared arena"
        );
        assert_eq!(ra.arglocs(two_loc[0].0, two_loc[0].1).len(), 2);
        assert_eq!(ra.arglocs(two_loc[1].0, two_loc[1].1).len(), 2);
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
        let ops = rcs(ops);
        let mut ra = RegAlloc::new(constants, &inputargs, &ops);
        ra.prepare_loop();
        let ra_ops = ra.walk_operations().expect("walk_operations");

        let guard_faillocs = ra_ops.iter().find_map(|ra_op| match ra_op {
            RegAllocOp::PerformGuard {
                op_index,
                faillocs_start,
                faillocs_len,
                ..
            }
            | RegAllocOp::PerformGuard1 {
                op_index,
                faillocs_start,
                faillocs_len,
                ..
            } if *op_index == 2 => Some(ra.faillocs(*faillocs_start, *faillocs_len)),
            _ => None,
        });

        let Some(faillocs) = guard_faillocs else {
            panic!("guard op was not lowered through RegAllocOp::PerformGuard");
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

        let ops = rcs(ops);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        ra.walk_operations().expect("walk_operations");
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

    #[test]
    fn test_walk_operations_dispatches_from_j2_lir() {
        // i0 is the typed Ref inputarg slot 0; the regalloc keys boxes
        // by typed `OpRef::input_arg_ref` (variant-aware Eq).
        let i0 = OpRef::input_arg_ref(0);
        let c0 = OpRef::const_int(0);
        let inputargs = vec![InputArg::from_type_rc(Type::Ref, i0.raw())];

        let store = Op::new(OpCode::GcStore, &[rb(i0), rb(c0), rb(i0)]);
        let ops = vec![store];

        let ops = rcs(ops);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        ra.j2_ops = vec![crate::j2plan::LirOp::Finish {
            args: vec![i0].into(),
        }];

        let ra_ops = ra.walk_operations().expect("walk_operations");
        let dispatched = ra_ops.iter().find_map(|ra_op| match ra_op {
            RegAllocOp::Perform {
                op_index,
                arglocs_len,
                result_loc,
                ..
            } if *op_index == 0 => Some((*arglocs_len as usize, result_loc.is_none())),
            RegAllocOp::Perform1 {
                op_index,
                result_loc,
                ..
            } if *op_index == 0 => Some((1, result_loc.is_none())),
            _ => None,
        });

        let Some((nlocs, result_is_none)) = dispatched else {
            panic!("j2 LIR dispatch did not produce a Perform op");
        };
        assert_eq!(
            nlocs, 1,
            "walk_operations should dispatch from j2 LIR, not raw GcStore opcode"
        );
        assert!(result_is_none);
    }

    #[test]
    fn test_j2_integer_dispatch_uses_lir_operands() {
        let i0 = OpRef::int_op(0);
        let i1 = OpRef::int_op(1);
        let i2 = OpRef::int_op(2);

        let inputargs = vec![
            InputArg::from_type_rc(Type::Int, i0.raw()),
            InputArg::from_type_rc(Type::Int, i1.raw()),
        ];

        let raw = Op::new(OpCode::IntIsTrue, &[rb(i0)]);
        raw.pos().set(i2);
        let finish = Op::new(OpCode::Finish, &[rb(i1)]);
        finish.pos().set(OpRef::int_op(3));
        let ops = vec![raw, finish];

        let ops = rcs(ops);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let expected_argloc = ra.loc(i1, Type::Int);
        ra.j2_ops = vec![crate::j2plan::LirOp::IntUnary {
            kind: crate::j2plan::IntUnaryKind::IsTrue,
            dst: i2,
            arg: i1,
        }];

        let ra_ops = ra.walk_operations().expect("walk_operations");
        let move_src = ra_ops.iter().find_map(|ra_op| {
            if let RegAllocOp::Move { src, .. } = ra_op {
                return Some(*src);
            }
            None
        });

        let Some(move_src) = move_src else {
            panic!("j2 integer dispatch did not materialize the LIR operand");
        };
        assert!(
            loc_eq(&move_src, &expected_argloc),
            "integer dispatch should materialize the LIR operand, not the raw Op arg: {:?} vs {:?}",
            move_src,
            expected_argloc
        );
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
        let i0 = OpRef::int_op(0);
        let inputargs = vec![InputArg::from_type_rc(Type::Int, i0.raw())];

        // arglocs = [lengthloc, imm(itemsize), imm(kind)] is built from
        // args [kind, itemsize, length].
        let malloc = Op::new(
            OpCode::CallMallocNurseryVarsize,
            &[rb(OpRef::const_int(0)), rb(OpRef::const_int(8)), rb(i0)],
        );
        malloc.pos().set(OpRef::ref_op(1));
        // The length outlives the allocation, which is the case upstream calls
        // typical: "it's typically also present in the next operation that will
        // copy it inside the new array".
        let finish = Op::new(OpCode::Finish, &[rb(i0)]);
        finish.pos().set(OpRef::void_op(2));
        let ops = vec![malloc, finish];

        let ops = rcs(ops);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let ra_ops = ra.walk_operations().expect("walk_operations");

        let lengthloc = ra_ops
            .iter()
            .find_map(|ra_op| match ra_op {
                RegAllocOp::Perform {
                    op_index,
                    arglocs_start,
                    arglocs_len,
                    ..
                } if ops[*op_index].opcode == OpCode::CallMallocNurseryVarsize => {
                    ra.arglocs(*arglocs_start, *arglocs_len).first().copied()
                }
                _ => None,
            })
            .expect("the varsize allocation reached the assembler");
        assert!(
            matches!(lengthloc, Loc::Frame(_) | Loc::Ebp(_)),
            "the length is spilled before it is read, so the assembler needs a \
             memory arm for it; got {lengthloc:?}"
        );
    }

    #[test]
    fn test_j2_guard_dispatch_uses_lir_operands() {
        let i0 = OpRef::int_op(0);
        let i1 = OpRef::int_op(1);

        let inputargs = vec![
            InputArg::from_type_rc(Type::Int, i0.raw()),
            InputArg::from_type_rc(Type::Int, i1.raw()),
        ];

        let raw = Op::new(OpCode::GuardTrue, &[rb(i0)]);
        raw.pos().set(OpRef::int_op(2));
        raw.setfailargs(vec![].into());
        let finish = Op::new(OpCode::Finish, &[rb(i1)]);
        finish.pos().set(OpRef::int_op(3));
        let ops = vec![raw, finish];

        let ops = rcs(ops);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let expected_argloc = ra.loc(i1, Type::Int);
        ra.j2_ops = vec![crate::j2plan::LirOp::Guard {
            kind: crate::j2plan::GuardKind::True,
            args: vec![i1].into(),
            fail_args: vec![].into(),
        }];

        let ra_ops = ra.walk_operations().expect("walk_operations");
        assert!(
            has_move_from(&ra_ops, &expected_argloc),
            "guard dispatch should materialize the LIR condition, not the raw Op arg: {:?}",
            ra_ops
        );
    }

    #[test]
    fn test_j2_load_dispatch_uses_lir_operands() {
        let i0 = OpRef::int_op(0);
        let i1 = OpRef::int_op(1);
        let i2 = OpRef::int_op(2);
        let c0 = OpRef::const_int(0);
        let c8 = OpRef::const_int(8);

        let inputargs = vec![
            InputArg::from_type_rc(Type::Ref, i0.raw()),
            InputArg::from_type_rc(Type::Ref, i1.raw()),
        ];

        let raw = Op::new(OpCode::GcLoadI, &[rb(i0), rb(c0), rb(c8)]);
        raw.pos().set(i2);
        let finish = Op::new(OpCode::Finish, &[rb(i1)]);
        finish.pos().set(OpRef::int_op(3));
        let ops = vec![raw, finish];

        let ops = rcs(ops);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let expected_argloc = ra.loc(i1, Type::Ref);
        ra.j2_ops = vec![crate::j2plan::LirOp::Load {
            kind: crate::j2plan::LoadKind::Gc,
            dst: i2,
            base: i1,
            offset: Some(c0),
            index: None,
            scale: None,
            size: Some(c8),
        }];

        let ra_ops = ra.walk_operations().expect("walk_operations");
        assert!(
            has_move_from(&ra_ops, &expected_argloc),
            "load dispatch should materialize the LIR base, not the raw Op base: {:?}",
            ra_ops
        );
    }

    #[test]
    fn test_j2_store_dispatch_uses_lir_operands() {
        // i0/i1 are typed Ref inputarg slots; i2 is typed Int. The
        // regalloc registers boxes via `inputargs.opref()` so each typed
        // OpRef variant must match (variant-aware Eq).
        let i0 = OpRef::input_arg_ref(0);
        let i1 = OpRef::input_arg_ref(1);
        let i2 = OpRef::input_arg_int(2);
        let c0 = OpRef::const_int(0);
        let c8 = OpRef::const_int(8);

        let inputargs = vec![
            InputArg::from_type_rc(Type::Ref, i0.raw()),
            InputArg::from_type_rc(Type::Ref, i1.raw()),
            InputArg::from_type_rc(Type::Int, i2.raw()),
        ];

        let raw = Op::new(OpCode::GcStore, &[rb(i0), rb(c0), rb(i2), rb(c8)]);
        let finish = Op::new(OpCode::Finish, &[rb(i1)]);
        finish.pos().set(OpRef::int_op(3));
        let ops = vec![raw, finish];

        let ops = rcs(ops);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let expected_argloc = ra.loc(i1, Type::Ref);
        ra.j2_ops = vec![crate::j2plan::LirOp::Store {
            kind: crate::j2plan::StoreKind::Gc,
            base: i1,
            offset: Some(c0),
            index: None,
            scale: None,
            value: i2,
            size: Some(c8),
        }];

        let ra_ops = ra.walk_operations().expect("walk_operations");
        assert!(
            has_move_from(&ra_ops, &expected_argloc),
            "store dispatch should materialize the LIR base, not the raw Op base: {:?}",
            ra_ops
        );
    }

    #[test]
    fn test_j2_opcode_dispatch_uses_lir_operands() {
        let i0 = OpRef::int_op(0);
        let i1 = OpRef::int_op(1);
        let i2 = OpRef::int_op(2);

        let inputargs = vec![
            InputArg::from_type_rc(Type::Int, i0.raw()),
            InputArg::from_type_rc(Type::Int, i1.raw()),
        ];

        let raw = Op::new(OpCode::SameAsI, &[rb(i0)]);
        raw.pos().set(i2);
        let finish = Op::new(OpCode::Finish, &[rb(i1)]);
        finish.pos().set(OpRef::int_op(3));
        let ops = vec![raw, finish];

        let ops = rcs(ops);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let expected_argloc = ra.loc(i1, Type::Int);
        ra.j2_ops = vec![crate::j2plan::LirOp::Opcode {
            opcode: OpCode::SameAsI,
            dst: Some(i2),
            args: vec![i1].into(),
            fail_args: vec![].into(),
        }];

        let ra_ops = ra.walk_operations().expect("walk_operations");
        assert!(
            has_move_from(&ra_ops, &expected_argloc),
            "generic j2 opcode dispatch should materialize the LIR arg, not the raw Op arg: {:?}",
            ra_ops
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
        let ops = rcs(vec![
            make_op(OpCode::IntAdd, 0, &[i0, OpRef::const_int(1)]),
            make_op(OpCode::IntAdd, 1, &[i1, OpRef::const_int(1)]),
            make_guard(OpCode::GuardTrue, 2, &[i2], &[i3]),
            finish,
        ]);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let output = ra.walk_operations().expect("walk_operations");
        let (arg_loc, fail_loc) = output
            .iter()
            .find_map(|op| match op {
                RegAllocOp::PerformGuard1 {
                    loc,
                    faillocs_start,
                    faillocs_len,
                    ..
                } => Some((*loc, ra.faillocs(*faillocs_start, *faillocs_len)[0])),
                _ => None,
            })
            .expect("GUARD_TRUE must be a PerformGuard1");
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
        let ops = rcs(vec![
            label,
            make_op(OpCode::IntAdd, 1, &[i0, OpRef::const_int(1)]),
            make_op(OpCode::Finish, 2, &[i1]),
        ]);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_bridge(&[Loc::Reg(EBP)]);
        assert_eq!(ra.rm.box_currently_in_frame_reg, Some(i0));
        let output = ra.walk_operations().expect("walk_operations");
        let label_pos = output
            .iter()
            .position(|op| {
                matches!(
                    op,
                    RegAllocOp::Perform { op_index: 0, .. }
                        | RegAllocOp::Perform1 { op_index: 0, .. }
                )
            })
            .expect("LABEL perform");
        let label_loc = match &output[label_pos] {
            RegAllocOp::Perform1 { loc, .. } => *loc,
            RegAllocOp::Perform {
                arglocs_start,
                arglocs_len,
                ..
            } => ra.arglocs(*arglocs_start, *arglocs_len)[0],
            _ => unreachable!(),
        };
        let Loc::Frame(slot) = label_loc else {
            panic!("LABEL argument must be its frame slot, got {label_loc:?}");
        };
        assert!(
            output[..label_pos].iter().any(|op| matches!(
                op,
                RegAllocOp::Move { src: Loc::Reg(r), dst: Loc::Frame(f) }
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
        let ops = rcs(vec![same, call, finish]);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let output = ra.walk_operations().expect("walk_operations");
        let arglocs = output
            .iter()
            .find_map(|op| match op {
                RegAllocOp::Perform {
                    op_index: 1,
                    arglocs_start,
                    arglocs_len,
                    ..
                } => Some(ra.arglocs(*arglocs_start, *arglocs_len).to_vec()),
                _ => None,
            })
            .expect("CALL_ASSEMBLER perform");
        assert!(
            matches!(arglocs[1], Loc::Frame(_)),
            "vable must be passed as its frame slot: {arglocs:?}"
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
        let token = OpRef::ref_op(0);
        let force = Op::new(OpCode::ForceToken, &[]);
        force.pos().set(token);
        let inputargs = vec![];
        let ops = vec![force, make_guard(OpCode::GuardNotForced2, 1, &[], &[token])];

        let ops = rcs(ops);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let output = ra.walk_operations().expect("walk_operations");
        // FORCE_TOKEN produces a Ref in a register, with no pre-existing
        // inputarg frame home. SAVE_ALL_REGS must emit a real spill for it.
        let Some(RegAllocOp::Perform {
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

        match output
            .iter()
            .find(|op| {
                matches!(
                    op,
                    RegAllocOp::PerformGuard { .. } | RegAllocOp::PerformGuard1 { .. }
                )
            })
            .expect("expected PerformGuard")
        {
            RegAllocOp::PerformGuard {
                arglocs_start,
                arglocs_len,
                faillocs_start,
                faillocs_len,
                ..
            } => {
                let arglocs = ra.arglocs(*arglocs_start, *arglocs_len);
                let faillocs = ra.faillocs(*faillocs_start, *faillocs_len);
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
            RegAllocOp::PerformGuard1 {
                loc,
                faillocs_start,
                faillocs_len,
                ..
            } => {
                let faillocs = ra.faillocs(*faillocs_start, *faillocs_len);
                assert_eq!(faillocs.len(), 1);
                assert!(
                    matches!(faillocs[0], Some(Loc::Frame(_))),
                    "GuardNotForced2 must spill every failarg into the jitframe, got {faillocs:?}"
                );
                assert!(
                    !matches!(loc, Loc::Immed(_) | Loc::ImmedFloat(_)),
                    "GuardNotForced2 must not carry a frame-depth immediate argloc, got {loc:?}"
                );
            }
            other => panic!("expected PerformGuard, got {other:?}"),
        }
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
            (inputargs, rcs(vec![add, finish]))
        }

        fn same_as_float_trace() -> (Vec<majit_ir::InputArgRc>, Vec<OpRc>) {
            let same = Op::new(OpCode::SameAsF, &[rb(OpRef::const_float(1.5))]);
            same.pos().set(OpRef::float_op(1));
            let finish = Op::new(OpCode::Finish, &[rb(OpRef::float_op(1))]);
            finish.pos().set(OpRef::void_op(2));
            finish.setfailargs(vec![].into());
            finish.set_fail_arg_types(vec![Type::Float]);
            (vec![], rcs(vec![same, finish]))
        }

        fn const_float_addrs(ra: &RegAlloc<'_>, ops: &[RegAllocOp]) -> Vec<usize> {
            let mut addrs = Vec::new();
            let mut note = |loc: &Loc| {
                if let Loc::ConstFloat(loc) = loc {
                    addrs.push(loc.value);
                }
            };
            for op in ops {
                match op {
                    RegAllocOp::Move { src, dst } => {
                        note(src);
                        note(dst);
                    }
                    RegAllocOp::Perform1 {
                        loc, result_loc, ..
                    } => {
                        note(loc);
                        if let Some(result) = result_loc {
                            note(result);
                        }
                    }
                    RegAllocOp::Perform {
                        arglocs_start,
                        arglocs_len,
                        result_loc,
                        ..
                    } => {
                        for loc in ra.arglocs(*arglocs_start, *arglocs_len) {
                            note(loc);
                        }
                        if let Some(result) = result_loc {
                            note(result);
                        }
                    }
                    _ => {}
                }
            }
            addrs
        }

        fn contains_immed_float(ra: &RegAlloc<'_>, ops: &[RegAllocOp]) -> bool {
            let mut hit = false;
            let mut note = |loc: &Loc| {
                if matches!(loc, Loc::ImmedFloat(_)) {
                    hit = true;
                }
            };
            for op in ops {
                match op {
                    RegAllocOp::Move { src, dst } => {
                        note(src);
                        note(dst);
                    }
                    RegAllocOp::Perform1 {
                        loc, result_loc, ..
                    } => {
                        note(loc);
                        if let Some(result) = result_loc {
                            note(result);
                        }
                    }
                    RegAllocOp::Perform {
                        arglocs_start,
                        arglocs_len,
                        result_loc,
                        ..
                    } => {
                        for loc in ra.arglocs(*arglocs_start, *arglocs_len) {
                            note(loc);
                        }
                        if let Some(result) = result_loc {
                            note(result);
                        }
                    }
                    _ => {}
                }
            }
            hit
        }

        let bits = 1.5f64.to_bits();
        let (inputargs, ops) = float_add_trace();
        let stats = std::sync::Arc::new(majit_backend::AsmMemoryManagerStats::default());
        let manager = majit_backend::AsmMemoryManager::new_isolated(std::sync::Arc::clone(&stats));
        let mut wrapper = majit_backend::MachineDataBlockWrapper::new(manager);
        let _writing = majit_backend::AssemblerWriting::enter();
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        ra.xrm.set_datablockwrapper(&mut wrapper);
        let ra_ops = ra.walk_operations().expect("walk_operations");
        let addrs = const_float_addrs(&ra, &ra_ops);
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
        ra.xrm.set_datablockwrapper(&mut wrapper);
        let ra_ops = ra.walk_operations().expect("walk_operations");
        let addrs = const_float_addrs(&ra, &ra_ops);
        assert!(
            !addrs.is_empty(),
            "SameAsF float constant must be a ConstFloatLoc, got {ra_ops:?}"
        );
        assert!(
            !contains_immed_float(&ra, &ra_ops),
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
            ra.xrm.set_datablockwrapper(&mut wrapper);
            ra.walk_operations()
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
