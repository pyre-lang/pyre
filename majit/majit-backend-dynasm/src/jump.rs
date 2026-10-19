//! `jump.py` — parallel assignment between location sets.
//!
//! Moving a set of values into a set of destinations is not a sequence of
//! independent moves: a destination may still be some other move's source, so
//! an order has to be found, and when the dependency graph has a cycle no
//! order exists and one value must be parked. `remap_frame_layout` is that
//! algorithm (`jump.py:4-64`), and `remap_frame_layout_mixed` (`jump.py:67-97`)
//! is the two-register-class variant used when integer and float arguments are
//! remapped together.
//!
//! ⚠ `dst_locations` and `src_locations` are swapped relative to upstream,
//! which spells the call `remap_frame_layout(assembler, dst_locations,
//! src_locations, tmpreg)`. Both are `&[Loc]`, so nothing catches a call
//! written from upstream's order — read the signature, not the memory of the
//! Python one.
//!
//! The algorithm is identical for every backend; only the three primitive
//! emitters it drives are not, which is why they are the trait and this is a
//! free function over it.

use crate::arch::WORD;
use crate::regloc::Loc;
use indexmap::IndexMap;

/// The three emitters `remap_frame_layout` drives, named as `jump.py` calls
/// them on the `assembler` it is handed.
///
/// **Operand contract.** A destination is a register or a frame-pointer
/// location in either spelling — `Loc::Frame`, which knows its stack position,
/// or the bare `Loc::Ebp`, which does not (`regloc.py class
/// FrameLoc(RawEbpLoc)`); an x86 call argument may also go to a
/// `Loc::Esp` (`RawEspLoc`). A source is one of those or an immediate. The one
/// location left out is `Loc::Addr`, which upstream cannot key either. An
/// implementation must fault on anything outside this rather than emit
/// nothing: a move that is counted in `pending_dests` and then silently
/// dropped leaves the destination holding a stale value, and a dropped
/// `regalloc_pop` leaves the machine stack pointer shifted as well.
///
/// `regalloc_push`/`regalloc_pop` exist only to serve the cycle-breaking arm
/// below — parking one value of a cycle on the machine stack is the whole
/// reason a cycle can be resolved at all. They are only ever handed a
/// destination, so an immediate never reaches them.
pub(crate) trait RegallocMoves {
    /// `assembler.py:1145 regalloc_mov(from_loc, to_loc)`.
    fn regalloc_mov(&mut self, src: &Loc, dst: &Loc);
    fn regalloc_push(&mut self, loc: &Loc);
    fn regalloc_pop(&mut self, loc: &Loc);
    /// `jump.py _move`, which each backend's `jump.py` defines for itself:
    /// x86 stages a pair of memory references through `tmpreg` and stores a
    /// `ConstFloatLoc` with `regalloc_immedmem2mem`, aarch64 stages a pair
    /// of stack locations.
    fn _move(&mut self, src: &Loc, dst: &Loc, tmpreg: &Loc);
}

/// A location's identity for the dependency bookkeeping.
///
/// Two locations must get the same key exactly when they are the same storage.
/// Handing two distinct destinations one key collapses them to a single
/// `IndexMap` entry, and the move for whichever one loses is never emitted.
///
/// Registers and stack slots are told apart **by sign**: a register key is
/// positive, a stack key is `!offset` and so negative. Upstream instead keys a
/// register on its bare number and a frame slot on its byte offset, and argues
/// the two cannot meet because offsets start above the register file —
/// `regloc.py:117-120` says so in as many words and asserts `ebp_offset >= 8 +
/// 8 * IS_X86_64` rather than trusting it. That argument does not survive the
/// per-class bias below: an offset of 4096 is an ordinary frame for a large
/// trace and is exactly the general-register base.
///
/// The bias is still worth keeping. Upstream gives `r0` and `xmm0` the same key
/// — harmless there because the two files are remapped in separate calls, but
/// `remap_frame_layout_mixed` compares one call's destination keys against the
/// other's sources, which is precisely across the two files.
pub(crate) fn loc_as_key(loc: &Loc) -> i32 {
    /// Keeps the two register files apart; both stay positive.
    const XMM_KEY_BASE: i32 = 0x2000;
    const GPR_KEY_BASE: i32 = 0x1000;
    /// `RawEspLoc._getregkey` is `~value`, the same space upstream gives a
    /// frame slot's `value`; here stack slots already own the negative keys,
    /// so rsp-relative words take a positive range far above both register
    /// files.
    const ESP_KEY_BASE: i32 = 0x4000_0000;

    match loc {
        Loc::Reg(r) if r.is_xmm => XMM_KEY_BASE + i32::from(r.value),
        Loc::Reg(r) => GPR_KEY_BASE + i32::from(r.value),
        // `!offset` for a non-negative offset is negative, so no stack slot can
        // ever land on a register key however deep the frame gets.
        Loc::Frame(f) => stack_key(f.ebp_loc.value),
        Loc::Ebp(e) => stack_key(e.value),
        Loc::Esp(e) => ESP_KEY_BASE + e.value,
        // Never a destination and re-materialisable at will, so it needs no
        // identity — only a value no real location can take.
        Loc::Immed(_) | Loc::ImmedFloat(_) | Loc::ConstFloat(_) => i32::MIN,
        // `AddressLoc` is the one location class upstream leaves without a key:
        // it overrides neither `_getregkey` nor, for its `'a'`/`'m'` codes, the
        // `value` the inherited one reads (`regloc.py`), so a parallel
        // move handed one raises there as well. Minting a key here instead
        // would put an entry in `pending_dests` that no emitter can retire.
        Loc::Addr(a) => panic!(
            "parallel move over an address location (offset {}), which no \
             regalloc_mov can emit",
            a.offset,
        ),
    }
}

/// The key for a stack slot at `offset` bytes from the frame pointer.
fn stack_key(offset: i32) -> i32 {
    debug_assert!(
        offset >= 0,
        "a negative frame offset ({offset}) inverts back into the positive \
         register key space",
    );
    !offset
}

/// The key of the slot one machine word past `key`.
///
/// Stack keys run backwards against offsets — `!(offset + WORD)` is
/// `!offset - WORD` — so the neighbour is found by subtracting, and writing
/// `key + WORD` here would silently read the slot on the wrong side.
fn stack_key_next_word(key: i32) -> i32 {
    key - WORD as i32
}

pub(crate) fn loc_width(loc: &Loc) -> usize {
    match loc {
        Loc::Reg(r) => r.get_width(),
        Loc::Frame(f) => f.ebp_loc.get_width(),
        Loc::Ebp(e) => e.get_width(),
        Loc::Esp(e) => e.get_width(),
        // `ConstFloatLoc.get_width` is 8.
        Loc::ConstFloat(_) => 8,
        _ => WORD,
    }
}

/// `jump.py:4 remap_frame_layout` — emit the moves that put `src_locations`
/// into `dst_locations`, in an order no move invalidates.
///
/// `tmpreg` is needed for a stack-to-stack pair, which no machine here can
/// move in one instruction.
pub(crate) fn remap_frame_layout<A: RegallocMoves + ?Sized>(
    asm: &mut A,
    src_locations: &[Loc],
    dst_locations: &[Loc],
    tmpreg: Loc,
) {
    let mut pending_dests = dst_locations.len() as i32;
    let mut srccount: IndexMap<i32, i32, rustc_hash::FxBuildHasher> =
        IndexMap::with_hasher(rustc_hash::FxBuildHasher);
    for dst in dst_locations {
        // `jump.py:7 assert key not in srccount`. A repeated destination shares
        // one entry while `pending_dests` counts both, so the second one can
        // never be retired: every key reaches -1, the loop stops making
        // progress, and the cycle-breaking arm finds no key left at or above
        // zero to park — the whole call spins. `insert` returns the displaced
        // value, so the check costs nothing beyond the store already made.
        assert!(
            srccount.insert(loc_as_key(dst), 0).is_none(),
            "duplicate value in dst_locations!",
        );
    }
    for i in 0..dst_locations.len() {
        let src = src_locations[i];
        if src.is_immed() {
            continue;
        }
        let key = loc_as_key(&src);
        if let Some(cnt) = srccount.get_mut(&key) {
            if key == loc_as_key(&dst_locations[i]) {
                *cnt = -(dst_locations.len() as i32) - 1;
                pending_dests -= 1;
            } else {
                *cnt += 1;
            }
        }
    }

    while pending_dests > 0 {
        let mut progress = false;
        for i in 0..dst_locations.len() {
            let dst = dst_locations[i];
            let key = loc_as_key(&dst);
            if srccount.get(&key).copied().unwrap_or(-1) == 0 {
                srccount.insert(key, -1);
                pending_dests -= 1;
                let src = src_locations[i];
                if !src.is_immed() {
                    let src_key = loc_as_key(&src);
                    if let Some(cnt) = srccount.get_mut(&src_key) {
                        *cnt -= 1;
                    }
                }
                asm._move(&src, &dst, &tmpreg);
                progress = true;
            }
        }
        if !progress {
            let mut sources: IndexMap<i32, Loc, rustc_hash::FxBuildHasher> =
                IndexMap::with_hasher(rustc_hash::FxBuildHasher);
            for i in 0..dst_locations.len() {
                sources.insert(loc_as_key(&dst_locations[i]), src_locations[i]);
            }
            for dst in dst_locations {
                let originalkey = loc_as_key(dst);
                if srccount.get(&originalkey).copied().unwrap_or(-1) >= 0 {
                    asm.regalloc_push(dst);
                    let mut cur_dst = *dst;
                    loop {
                        let key = loc_as_key(&cur_dst);
                        srccount.insert(key, -1);
                        pending_dests -= 1;
                        let src = sources[&key];
                        if loc_as_key(&src) == originalkey {
                            break;
                        }
                        asm._move(&src, &cur_dst, &tmpreg);
                        cur_dst = src;
                    }
                    asm.regalloc_pop(&cur_dst);
                }
            }
        }
    }
}

/// `jump.py:67 remap_frame_layout_mixed` — two location sets remapped with a
/// temporary each, as integer and float arguments need.
///
/// The sets are not independent: a set-2 stack source may be a set-1
/// destination, and set 1 runs first. Those sources are pushed before either
/// remap and popped into place after, which is why they are dropped from the
/// set-2 lists rather than reordered.
pub(crate) fn remap_frame_layout_mixed<A: RegallocMoves + ?Sized>(
    asm: &mut A,
    src_locations1: &[Loc],
    dst_locations1: &[Loc],
    tmpreg1: Loc,
    src_locations2: &[Loc],
    dst_locations2: &[Loc],
    tmpreg2: Loc,
) {
    let mut extrapushes = Vec::new();
    let mut dst_keys = IndexMap::with_hasher(rustc_hash::FxBuildHasher);
    for loc in dst_locations1 {
        dst_keys.insert(loc_as_key(loc), ());
    }
    let mut src_locations2red = Vec::new();
    let mut dst_locations2red = Vec::new();
    for i in 0..src_locations2.len() {
        let loc = src_locations2[i];
        let dstloc = dst_locations2[i];
        if loc.is_stack() {
            let key = loc_as_key(&loc);
            if dst_keys.contains_key(&key)
                || (loc_width(&loc) > WORD && dst_keys.contains_key(&stack_key_next_word(key)))
            {
                asm.regalloc_push(&loc);
                extrapushes.push(dstloc);
                continue;
            }
        }
        src_locations2red.push(loc);
        dst_locations2red.push(dstloc);
    }
    remap_frame_layout(asm, src_locations1, dst_locations1, tmpreg1);
    remap_frame_layout(asm, &src_locations2red, &dst_locations2red, tmpreg2);
    while let Some(loc) = extrapushes.pop() {
        asm.regalloc_pop(&loc);
    }
}
