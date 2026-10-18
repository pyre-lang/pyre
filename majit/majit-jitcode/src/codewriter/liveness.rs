//! Liveness encoding read by the JIT runtime.
//!
//! RPython equivalent: the helper half of `rpython/jit/codewriter/liveness.py`.
//! `compute_liveness` stays in `majit-translate`.

// ____________________________________________________________
// helper functions for compactly encoding and decoding liveness info
//
// liveness is encoded as a 2 byte offset into the single string all_liveness
// (which is stored on the metainterp_sd)

/// RPython liveness.py `OFFSET_SIZE`.
pub const OFFSET_SIZE: usize = 2;

/// RPython liveness.py `encode_offset(pos, code)`.
pub fn encode_offset(pos: usize, code: &mut Vec<u8>) {
    assert_eq!(OFFSET_SIZE, 2);
    code.push((pos & 0xff) as u8);
    code.push(((pos >> 8) & 0xff) as u8);
    assert_eq!(pos >> 16, 0);
}

/// RPython liveness.py `decode_offset(jitcode, pc)`.
#[inline(always)]
pub fn decode_offset(jitcode: &[u8], pc: usize) -> usize {
    debug_assert_eq!(OFFSET_SIZE, 2);
    (jitcode[pc] as usize) | ((jitcode[pc + 1] as usize) << 8)
}

/// Translated `decode_offset`: `ord(code[pc]) | (ord(code[pc+1]) << 8)`.
///
/// # Safety
/// `pc + 1 < jitcode.len()`. The caller has a valid `-live-` pc in an
/// assembled jitcode body (`JitCode.get_live_vars_info`).
#[inline(always)]
pub unsafe fn decode_offset_unchecked(jitcode: &[u8], pc: usize) -> usize {
    debug_assert_eq!(OFFSET_SIZE, 2);
    debug_assert!(pc + 1 < jitcode.len());
    unsafe {
        *jitcode.get_unchecked(pc) as usize | ((*jitcode.get_unchecked(pc + 1) as usize) << 8)
    }
}

// within the string of all_liveness, we encode the bitsets of which of the 256
// registers are live as follows: first three byte with the number of set bits
// for each of the categories ints, refs, floats followed by the necessary
// number of bytes to store them (this number of bytes is implicit), for each of
// the categories
// | len live_i | len live_r | len live_f
// | bytes for live_i | bytes for live_r | bytes for live_f

/// RPython liveness.py `encode_liveness(live)`.
///
/// Encodes a single register-kind bitset: `live` is a list of register
/// indices (each `< 256`). Returns the packed bitset bytes (no length
/// header — the caller is responsible for emitting the three
/// `len_i/len_r/len_f` header bytes).
///
/// Mirrors RPython's `live = sorted(live)` (liveness.py): the input
/// is sorted internally so callers can pass arbitrary-order or
/// duplicated slices without normalization.
pub fn encode_liveness(live: &[u8]) -> Vec<u8> {
    // RPython liveness.py `live = sorted(live)`.
    let mut sorted: Vec<u8> = live.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut liveness: Vec<u8> = Vec::new();
    let mut offset: u32 = 0;
    let mut char_: u32 = 0;
    let mut i = 0;
    while i < sorted.len() {
        let x = sorted[i] as u32;
        let x = x.wrapping_sub(offset);
        if x >= 8 {
            liveness.push(char_ as u8);
            char_ = 0;
            offset += 8;
            continue;
        }
        char_ |= 1 << x;
        assert!(char_ < 256);
        i += 1;
    }
    if char_ != 0 {
        liveness.push(char_ as u8);
    }
    liveness
}

/// Walk an `all_liveness` buffer back into the `(live_i, live_r, live_f)`
/// records [`encode_liveness`] wrote into it, each with its own offset.
///
/// `assembler.py _encode_liveness` appends every record at the
/// buffer's current end — three count bytes then the three bitsets — so the
/// buffer is a contiguous run of records starting at 0 and this walk recovers
/// them exactly. Upstream never needs it: one `Assembler` holds one buffer and
/// one `all_liveness_positions` dict for the whole program. pyre assembles the
/// extracted interpreter graphs in `build.rs` and the Python-bytecode graphs at
/// runtime, and the runtime side resumes the build-time bytes; without this the
/// resumed side starts with an empty dedup dict and re-appends a record the
/// prefix already holds, spending the 2-byte `-live-` operand's 64 KiB budget
/// on duplicates.
///
/// Bytes `LivenessIterator` consumes for a bank of `length` set bits.
/// `None` if the slice ends before the iterator would finish
/// (`ord(all_liveness[self.offset])` in `liveness.py` `LivenessIterator.next`).
fn encoded_bank_len(all_liveness: &[u8], offset: usize, length: u32) -> Option<usize> {
    if length == 0 {
        return Some(0);
    }
    let mut remaining = length;
    let mut pos = offset;
    let mut count = 0u32;
    let mut curr_byte = 0u32;
    while remaining > 0 {
        if (count & 7) == 0 {
            curr_byte = *all_liveness.get(pos)? as u32;
            pos += 1;
        }
        let tail = curr_byte >> (count & 7);
        if tail != 0 {
            count += tail.trailing_zeros();
            remaining -= 1;
            count += 1;
        } else {
            count = (count & !7) + 8;
        }
    }
    Some(pos - offset)
}

/// True when `all_liveness[offset..]` is a complete assembler record:
/// three count bytes then three `encode_liveness` banks of those lengths
/// (`jitcode.py` `enumerate_vars`).
pub fn liveness_record_is_complete(all_liveness: &[u8], offset: usize) -> bool {
    if offset + 2 >= all_liveness.len() {
        return false;
    }
    let mut bank_end = offset + 3;
    for i in 0..3 {
        match encoded_bank_len(all_liveness, bank_end, u32::from(all_liveness[offset + i])) {
            Some(len) => bank_end += len,
            None => return false,
        }
    }
    true
}

/// Each returned tuple is `(live_i, live_r, live_f, offset)`, with the sets in
/// the sorted/deduped form `encode_liveness` canonicalises to — the same shape
/// upstream's `frozenset` key has.
///
/// Incomplete trailing bytes (a header that names more set bits than the
/// remaining slice holds) are dropped: this is a safe assembler-seed walk,
/// not the resume hot loop.
#[expect(
    clippy::type_complexity,
    reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
)]
pub fn decode_liveness_records(all_liveness: &[u8]) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>, usize)> {
    let mut records = Vec::new();
    let mut pos = 0usize;
    while pos + 3 <= all_liveness.len() {
        let offset = pos;
        let counts = [
            all_liveness[pos],
            all_liveness[pos + 1],
            all_liveness[pos + 2],
        ];
        let mut bank_end = pos + 3;
        let mut complete = true;
        for &count in &counts {
            match encoded_bank_len(all_liveness, bank_end, u32::from(count)) {
                Some(len) => bank_end += len,
                None => {
                    complete = false;
                    break;
                }
            }
        }
        if !complete {
            break;
        }
        pos += 3;
        let mut sets: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (bank, &count) in counts.iter().enumerate() {
            if count == 0 {
                // `encode_liveness(&[])` emits no bytes, so an empty bank
                // consumes none either.
                continue;
            }
            // SAFETY: `encoded_bank_len` just proved this bank is a complete
            // `encode_liveness` bitset (`LivenessIterator.__init__`).
            let mut it = unsafe { LivenessIterator::new(pos, u32::from(count), all_liveness) };
            for index in it.by_ref() {
                sets[bank].push(index as u8);
            }
            pos = it.offset();
        }
        debug_assert_eq!(pos, bank_end);
        let [live_i, live_r, live_f] = sets;
        records.push((live_i, live_r, live_f, offset));
    }
    records
}

/// RPython liveness.py `LivenessIterator`.
///
/// Iterates set bit positions from a bitset stored in `all_liveness`
/// starting at `offset`, producing `length` indices total.
#[derive(Debug, Clone)]
pub struct LivenessIterator<'a> {
    all_liveness: &'a [u8],
    offset: usize,
    length: u32,
    curr_byte: u32,
    count: u32,
}

impl<'a> LivenessIterator<'a> {
    /// RPython liveness.py `__init__(self, offset, length, all_liveness)`.
    ///
    /// # Safety
    /// `all_liveness[offset..]` holds a complete encoded bank of `length`
    /// set bits (`encode_liveness` / assembler `all_liveness`).
    /// `LivenessIterator.next` does `ord(all_liveness[self.offset])`.
    /// A truncated buffer is a producer bug.
    pub unsafe fn new(offset: usize, length: u32, all_liveness: &'a [u8]) -> Self {
        assert!(length != 0);
        assert!(
            offset < all_liveness.len(),
            "liveness bank offset past all_liveness"
        );
        LivenessIterator {
            all_liveness,
            offset,
            length,
            curr_byte: 0,
            count: 0,
        }
    }

    /// `LivenessIterator.offset` after a bank walk (`enumerate_vars`).
    #[inline(always)]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Remaining set bits (`self.length` after `if not self.length`).
    #[inline(always)]
    pub fn remaining(&self) -> u32 {
        self.length
    }

    /// liveness.py `next` after `if not self.length: raise StopIteration`.
    ///
    /// `enumerate_vars` drives `for index in it:` with that check as the
    /// loop end. The `Option` wrapper on `Iterator::next` is the StopIteration
    /// translation; this returns the index the way the inlined `next`
    /// does (`@objectmodel.always_inline`).
    ///
    /// `self.length != 0` is the caller's proof: `enumerate_vars` only
    /// constructs the iterator when `length_*` is nonzero, and this is
    /// called once per remaining live register.
    ///
    /// # Safety
    /// `all_liveness` holds a complete encoded record from `offset` for
    /// `length` set bits (`ord(all_liveness[self.offset])` in
    /// `liveness.py` `LivenessIterator.next`). A truncated buffer is a
    /// producer bug.
    #[inline(always)]
    pub unsafe fn next_index(&mut self) -> u32 {
        debug_assert!(self.length != 0);
        self.length -= 1;
        let mut count = self.count;
        let all_liveness = self.all_liveness;
        let mut curr_byte = self.curr_byte;
        loop {
            if (count & 7) == 0 {
                debug_assert!(self.offset < all_liveness.len());
                curr_byte = unsafe { *all_liveness.get_unchecked(self.offset) } as u32;
                self.curr_byte = curr_byte;
                self.offset += 1;
            }
            let remaining = curr_byte >> (count & 7);
            if remaining != 0 {
                count += remaining.trailing_zeros();
                self.count = count + 1;
                return count;
            }
            count = (count & !7) + 8;
        }
    }
}

impl<'a> Iterator for LivenessIterator<'a> {
    type Item = u32;

    /// RPython liveness.py `next(self)`.
    ///
    /// `@objectmodel.always_inline`. Same scan as the bit-at-a-time
    /// loop: skip a zero tail with `trailing_zeros` and return the next
    /// set index. The produced sequence is identical. The byte load is
    /// `ord(all_liveness[self.offset])` (`rffi`); a truncated buffer is
    /// a producer bug.
    #[inline(always)]
    fn next(&mut self) -> Option<u32> {
        if self.length == 0 {
            return None;
        }
        // SAFETY: [`Self::new`] requires a complete encoded bank, and
        // `length` counts remaining set bits (`LivenessIterator.next`).
        Some(unsafe { self.next_index() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decode_liveness_records_round_trips_encode() {
        // Same layout `assembler.py:241-247` writes: three count bytes then
        // the three encoded banks, appended back to back.
        let banks: [(Vec<u8>, Vec<u8>, Vec<u8>); 4] = [
            (vec![0, 3, 7], vec![1], vec![]),
            (vec![], vec![], vec![]),
            (vec![2], vec![], vec![9, 10]),
            (vec![0, 1, 2, 3, 4, 5, 6, 7, 8], vec![255], vec![4]),
        ];
        let mut all = Vec::new();
        let mut offsets = Vec::new();
        for (i, r, f) in &banks {
            offsets.push(all.len());
            all.push(i.len() as u8);
            all.push(r.len() as u8);
            all.push(f.len() as u8);
            for live in [i.as_slice(), r.as_slice(), f.as_slice()] {
                all.extend_from_slice(&encode_liveness(live));
            }
        }
        let decoded = decode_liveness_records(&all);
        assert_eq!(decoded.len(), banks.len());
        for (idx, (i, r, f, off)) in decoded.iter().enumerate() {
            assert_eq!((i, r, f), (&banks[idx].0, &banks[idx].1, &banks[idx].2));
            assert_eq!(*off, offsets[idx]);
        }
    }

    #[test]
    fn decode_liveness_records_drops_a_truncated_bank() {
        // Header claims two int bits; one bank byte holds one set bit.
        // `LivenessIterator.next_index` would load past the slice.
        assert!(decode_liveness_records(&[2, 0, 0, 0x01]).is_empty());
        assert_eq!(
            decode_liveness_records(&[1, 0, 0, 0x01]),
            vec![(vec![0], Vec::new(), Vec::new(), 0)]
        );
    }

    #[test]
    fn liveness_record_is_complete_rejects_a_header_with_missing_bank_bytes() {
        assert!(!liveness_record_is_complete(&[2, 0, 0, 0x01], 0));
        assert!(liveness_record_is_complete(&[1, 0, 0, 0x01], 0));
        assert!(liveness_record_is_complete(&[0, 0, 0], 0));
        assert!(!liveness_record_is_complete(&[0, 0], 0));
    }

    #[test]
    fn encode_decode_offset_roundtrip() {
        let mut code: Vec<u8> = Vec::new();
        encode_offset(0x1234, &mut code);
        assert_eq!(code, vec![0x34, 0x12]);
        assert_eq!(decode_offset(&code, 0), 0x1234);
    }

    #[test]
    fn encode_liveness_bitmaps() {
        let cases: &[(&str, &[u8], &[u8])] = &[
            ("empty", &[], &[]),
            ("small", &[0, 1, 7], &[0x83]),
            ("multi_byte", &[0, 8, 15], &[0x01, 0x81]),
        ];
        for &(name, live, want) in cases {
            assert_eq!(encode_liveness(live), want, "case {name}");
        }
    }

    #[test]
    fn liveness_iterator_roundtrip() {
        let live = [0u8, 3, 5, 9, 12, 17];
        let encoded = encode_liveness(&live);
        // SAFETY: `encoded` is `encode_liveness(&live)` (`LivenessIterator.__init__`).
        let mut it = unsafe { LivenessIterator::new(0, live.len() as u32, &encoded) };
        let decoded: Vec<u32> = (&mut it).collect();
        assert_eq!(decoded, live.iter().map(|&i| i as u32).collect::<Vec<_>>());
    }

    #[test]
    fn liveness_iterator_skips_zero_bytes() {
        // live = [0, 16, 23] spans three bytes with a zero middle byte.
        let live = [0u8, 16, 23];
        let encoded = encode_liveness(&live);
        assert_eq!(encoded, vec![0x01, 0x00, 0x81]);
        // SAFETY: `encoded` is `encode_liveness(&live)` (`LivenessIterator.__init__`).
        let decoded: Vec<u32> =
            unsafe { LivenessIterator::new(0, live.len() as u32, &encoded) }.collect();
        assert_eq!(decoded, vec![0, 16, 23]);
    }
}
