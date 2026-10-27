//! RPython's `rpython.rlib.rutf8`, ported structurally to Rust.
//!
//! # Relationship to `rustpython-wtf8`
//!
//! RPython has no string *type*: `W_UnicodeObject._utf8` is a plain `str`
//! (immutable bytes) and every operation on it is a free function in `rutf8`
//! taking `(code, pos)`.  pyre stores the same buffer as a `Wtf8Buf`, a type
//! that already carries the well-formedness invariant, so the two overlap and
//! the split has to be deliberate:
//!
//! * **`rustpython-wtf8` owns the representation and sequential access.** An
//!   `rutf8` function whose whole content is "encode, decode or validate,
//!   scanning forward" has a checked counterpart there and is *not* re-ported:
//!   a second, unchecked implementation beside a checked one would be two
//!   sources of truth for one invariant.  `unichr_as_utf8*` is
//!   `CodePoint::encode_wtf8` / `Wtf8Buf::push`; `get_utf8_length` is
//!   `codepoints_in_utf8`, which upstream also says to prefer; `check_ascii`
//!   and `first_non_ascii_char` are `Wtf8::is_ascii` and a byte scan;
//!   `has_surrogates` and `surrogate_in_utf8` are `Wtf8::as_str().is_err()`;
//!   `islinebreak`, `isspace` and `utf8_in_chars` are predicates over a
//!   decoded `CodePoint`; `char_escape_helper`, `make_utf8_escape_function`
//!   and `decode_latin_1` belong to `repr` and `_codecs`.
//!
//! * **This module owns random access**, which the crate deliberately has no
//!   counterpart for — its iterators are sequential, so resolving the n-th code
//!   point through them costs O(n).  `UTF8_INDEX_STORAGE` is PyPy's index over
//!   an *already valid* buffer, and everything written against it lives here:
//!   the table and its build, the three lookups that read it, and the byte
//!   offset stepping (`next_codepoint_pos`, `prev_codepoint_pos`) they are
//!   defined in terms of, which is pure arithmetic the crate exposes no
//!   equivalent for.
//!
//! That is why the `utf8` parameter is `&Wtf8` rather than `&[u8]`: `&Wtf8` is
//! what RPython's `str` payload becomes once the invariant lives in the type,
//! and holding it lets `codepoint_at_pos` decode through the crate instead of
//! carrying a second copy of its decoder.
//!
//! `check_utf8` is the one validator that *is* re-ported, because
//! `Wtf8::from_bytes` is not equivalent to it: it accepts sequences upstream
//! rejects (see that function), and it has no way to spell
//! `allow_surrogates=False`.  Its three predicates come with it, so the
//! decoders that already share them read them from here.
//!
//! Two members of the family are deliberately absent.  `null_storage` has no
//! counterpart: an absent table is a null pointer in the `W_UnicodeObject`
//! slot.  `_pos_at_index`, the "Slow!" linear fallback, has no pyre caller —
//! upstream reaches it from `unicodehelper` and `formatting`, neither of
//! which pyre routes this way.
//!
//! Names, entry layout, group sizes and the build loop follow
//! `rpython/rlib/rutf8.py`.  The `_is_64bit` branch of
//! `next_codepoint_pos` (a hand-tuned x86-64 sequence guarded by
//! `not jit.we_are_jitted()`) is left out: it computes the same value as the
//! comparison ladder it sits in front of.

use rustpython_wtf8::Wtf8;

/// `UTF8_INDEX_STORAGE`'s element (`rutf8.py`) — `baseindex` is the
/// byte offset the 64-code-point group starts at, and `ofs[i]` the byte delta
/// from it to the code point *after* the group's `4 * i`-th.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Utf8LocElem {
    pub baseindex: isize,
    pub ofs: [u8; 16],
}

/// `UTF8_INDEX_STORAGE` (`rutf8.py`).  A leaf (no inner `PyObjectRef`), so
/// a mortal string's table lives in a storage box whose drop glue reclaims it,
/// as its WTF-8 buffer does.
pub type Utf8IndexStorage = Vec<Utf8LocElem>;

/// `next_codepoint_pos` (`rutf8.py`) — the position of the code point after
/// `pos`.  Assumes valid WTF-8 and `pos` before the end.
#[inline]
pub fn next_codepoint_pos(code: &Wtf8, pos: usize) -> usize {
    let chr1 = code.as_bytes()[pos];
    if chr1 <= 0x7F {
        return pos + 1;
    }
    if chr1 <= 0xDF {
        return pos + 2;
    }
    if chr1 <= 0xEF {
        return pos + 3;
    }
    pos + 4
}

/// `prev_codepoint_pos` (`rutf8.py`) — the position of the code point
/// before `pos`, which must not be zero.  A `pos` one past the end reads as the
/// extra `'\x00'` the build loop pretends is there.
#[inline]
pub fn prev_codepoint_pos(code: &Wtf8, pos: usize) -> usize {
    let code = code.as_bytes();
    let mut pos = pos - 1;
    if pos >= code.len() {
        return pos;
    }
    if code[pos] <= 0x7F {
        return pos;
    }
    pos -= 1;
    if code[pos] >= 0xC0 {
        return pos;
    }
    pos -= 1;
    if code[pos] >= 0xC0 {
        return pos;
    }
    pos - 1
}

/// `codepoint_at_pos` (`rutf8.py`) — the code point starting at byte `pos`.
///
/// Assumes valid WTF-8 with `pos` on a boundary before the end, as upstream
/// does ("no checking!"). Returns RPython's signed integer code point and
/// keeps the upstream bit arithmetic literal so the JIT sees the same graph.
#[inline]
pub fn codepoint_at_pos(code: &Wtf8, pos: usize) -> i64 {
    let code = code.as_bytes();
    let len = code.len();
    let ordch1 = code[pos] as i64;
    if ordch1 <= 0x7f || pos + 1 >= len {
        return ordch1;
    }
    let ordch2 = code[pos + 1] as i64;
    if ordch1 <= 0xdf || pos + 2 >= len {
        return (ordch1 << 6) + ordch2 - ((0xc0 << 6) + 0x80);
    }
    let ordch3 = code[pos + 2] as i64;
    if ordch1 <= 0xef || pos + 3 >= len {
        return (ordch1 << 12) + (ordch2 << 6) + ordch3 - ((0xe0 << 12) + (0x80 << 6) + 0x80);
    }
    let ordch4 = code[pos + 3] as i64;
    (ordch1 << 18) + (ordch2 << 12) + (ordch3 << 6) + ordch4
        - ((0xf0 << 18) + (0x80 << 12) + (0x80 << 6) + 0x80)
}

/// `codepoint_before_pos` (`rutf8.py`) — the code point immediately before
/// `pos`, which must not be zero.
///
/// Upstream walks the continuation bytes backwards inline; composing the
/// already-ported `prev_codepoint_pos` with `codepoint_at_pos` yields the same
/// code point by construction, and inherits `prev_codepoint_pos`'s handling of
/// a `pos` one past the end.
#[inline]
pub fn codepoint_before_pos(code: &Wtf8, pos: usize) -> i64 {
    codepoint_at_pos(code, prev_codepoint_pos(code, pos))
}

/// `CheckError` (`rutf8.py`) — the byte position `check_utf8` stopped at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckError {
    /// RPython `CheckError.pos` is a normal Signed integer. Keep the stored
    /// field signed; convert only at Rust slice-index API boundaries.
    pub pos: i64,
}

/// `_invalid_cont_byte` (`rutf8.py`) — a byte outside `0x80..=0xBF`, which
/// upstream spells as a signed-char comparison against `-0x40`.
#[inline]
pub fn invalid_cont_byte(b: u8) -> bool {
    (b as i8) >= -0x40
}

/// `_surrogate_bytes` (`rutf8.py`) — a three-byte lead pair that encodes a
/// surrogate, which is the one pair `invalid_byte_2_of_3` admits only when
/// the caller allows surrogates.
#[inline]
pub fn surrogate_bytes(ch1: u8, ch2: u8) -> bool {
    ch1 == 0xED && ch2 > 0x9F
}

/// `_invalid_byte_2_of_3` (`rutf8.py`) — the second byte of a three-byte
/// sequence, rejecting surrogate encodings unless the caller selected the
/// `surrogatepass` path.
#[inline]
pub fn invalid_byte_2_of_3(ch1: u8, ch2: u8, allow_surrogates: bool) -> bool {
    invalid_cont_byte(ch2)
        || (ch1 == 0xE0 && ch2 < 0xA0)
        || (ch1 == 0xED && ch2 > 0x9F && !allow_surrogates)
}

/// `_invalid_byte_2_of_4` (`rutf8.py`) — the second byte of a four-byte
/// sequence, which also carries the range bound the leading byte leaves open.
#[inline]
pub fn invalid_byte_2_of_4(ch1: u8, ch2: u8) -> bool {
    invalid_cont_byte(ch2) || (ch1 == 0xF0 && ch2 < 0x90) || (ch1 == 0xF4 && ch2 > 0x8F)
}

/// `check_utf8` (`rutf8.py`) — the code point count of `s`, or the byte
/// position where it stops being well formed.
///
/// `_check_utf8`'s ones'-complement return and the `CheckError` its caller
/// raises from it are one `Result` here.  Upstream's `start`/`stop` window is
/// left out: every pyre caller validates a whole buffer.
#[majit_macros::elidable]
pub fn check_utf8(s: &[u8], allow_surrogates: bool) -> Result<i64, CheckError> {
    // RPython's `start`, `stop`, `pos`, and return value are all Signed.
    // Convert only where Rust's slice API requires an unsigned index.
    let end = s.len() as i64;
    let mut pos = 0i64;
    let mut continuation_bytes = 0i64;
    while pos < end {
        let ordch1 = s[pos as usize];
        pos += 1;
        // fast path for ASCII
        if ordch1 <= 0x7F {
            continue;
        }
        if ordch1 <= 0xC1 {
            return Err(CheckError { pos: pos - 1 });
        }
        if ordch1 <= 0xDF {
            if pos >= end {
                return Err(CheckError { pos: pos - 1 });
            }
            let ordch2 = s[pos as usize];
            pos += 1;
            if invalid_cont_byte(ordch2) {
                return Err(CheckError { pos: pos - 2 });
            }
            continuation_bytes += 1;
            continue;
        }
        if ordch1 <= 0xEF {
            if pos + 2 > end {
                return Err(CheckError { pos: pos - 1 });
            }
            let ordch2 = s[pos as usize];
            let ordch3 = s[(pos + 1) as usize];
            pos += 2;
            if invalid_byte_2_of_3(ordch1, ordch2, allow_surrogates) || invalid_cont_byte(ordch3) {
                return Err(CheckError { pos: pos - 3 });
            }
            continuation_bytes += 2;
            continue;
        }
        if ordch1 <= 0xF4 {
            if pos + 3 > end {
                return Err(CheckError { pos: pos - 1 });
            }
            let ordch2 = s[pos as usize];
            let ordch3 = s[(pos + 1) as usize];
            let ordch4 = s[(pos + 2) as usize];
            pos += 3;
            if invalid_byte_2_of_4(ordch1, ordch2)
                || invalid_cont_byte(ordch3)
                || invalid_cont_byte(ordch4)
            {
                return Err(CheckError { pos: pos - 4 });
            }
            continuation_bytes += 3;
            continue;
        }
        return Err(CheckError { pos: pos - 1 });
    }
    Ok(pos - continuation_bytes)
}

/// The `Wtf8` view of a buffer `check_utf8` accepts — `str_decode_utf8`'s
/// "fast version first" arm, and the check every untrusted-bytes boundary
/// performs before the buffer becomes a string.
///
/// `Wtf8::from_bytes` does not stand in for this.  Its surrogate arm matches
/// `[0xed, 0xa0.., b3, ..]`, leaving the second byte unbounded above and the
/// third unconstrained, so it accepts `ED C0 80` — three bytes encoding no code
/// point.  A buffer admitted that way reaches `W_UnicodeObject` with a code
/// point count that disagrees with what `next_codepoint_pos` steps over, and
/// the two disagree first inside `create_utf8_index_storage`, which reads past
/// the buffer.
///
/// The scan is [`check_utf8`] (`rutf8.py` `_check_utf8`, `@jit.elidable`);
/// `check_utf8_and_wtf8_from_bytes_agree` holds the two to one answer.
pub fn wtf8_from_bytes(s: &[u8], allow_surrogates: bool) -> Result<&Wtf8, CheckError> {
    check_utf8(s, allow_surrogates)?;
    // SAFETY: `check_utf8` admitted every sequence in `s`; with
    // `allow_surrogates` that is WTF-8, otherwise UTF-8, a subset of it.
    Ok(unsafe { Wtf8::from_bytes_unchecked(s) })
}

/// `codepoints_in_utf8` (`rutf8.py`) — the number of code points in
/// `value[start..end]`.
///
/// Counts the bytes that are *not* continuation bytes, which upstream spells as
/// a signed-char comparison against `-0x40`.
#[majit_macros::elidable]
pub fn codepoints_in_utf8(value: &Wtf8, start: usize, end: usize) -> usize {
    let value = value.as_bytes();
    let end = end.min(value.len());
    debug_assert!(start <= end);
    value[start..end]
        .iter()
        .filter(|&&ch| invalid_cont_byte(ch))
        .count()
}

/// `create_utf8_index_storage` (`rutf8.py`) — the table for `utf8`, whose
/// code point count is `utf8len`.
pub fn create_utf8_index_storage(utf8: &Wtf8, utf8len: usize) -> Utf8IndexStorage {
    let arraysize = utf8len / 64 + 1;
    let mut storage = vec![
        Utf8LocElem {
            baseindex: 0,
            ofs: [0u8; 16],
        };
        arraysize
    ];
    // Signed: the count overshoots the last group by design and the loop stops
    // on the first negative value.
    let mut utf8len = utf8len as isize;
    let mut baseindex = 0usize;
    let mut current = 0usize;
    loop {
        storage[current].baseindex = baseindex as isize;
        let mut next = baseindex;
        // Stands in for the `for ... else` upstream breaks out of.
        let mut group_filled = true;
        for i in 0..16 {
            if utf8len == 0 {
                next += 1; // assume there is an extra '\x00' character
            } else {
                next = next_codepoint_pos(utf8, next);
            }
            storage[current].ofs[i] = (next - baseindex) as u8;
            utf8len -= 4;
            if utf8len < 0 {
                debug_assert_eq!(current + 1, storage.len());
                group_filled = false;
                break;
            }
            next = next_codepoint_pos(utf8, next);
            next = next_codepoint_pos(utf8, next);
            next = next_codepoint_pos(utf8, next);
        }
        if !group_filled {
            break;
        }
        current += 1;
        baseindex = next;
    }
    storage
}

/// `codepoint_position_at_index` (`rutf8.py`) — the byte offset of code
/// point `index`, which must not exceed the string's code point count.
#[inline]
#[majit_macros::elidable]
pub fn codepoint_position_at_index(utf8: &Wtf8, storage: &[Utf8LocElem], index: usize) -> usize {
    let elem = &storage[index >> 6];
    let bytepos = elem.baseindex as usize + elem.ofs[(index >> 2) & 0x0F] as usize;
    match index & 0x3 {
        0 => prev_codepoint_pos(utf8, bytepos),
        1 => bytepos,
        2 => next_codepoint_pos(utf8, bytepos),
        _ => next_codepoint_pos(utf8, next_codepoint_pos(utf8, bytepos)),
    }
}

/// `codepoint_at_index` (`rutf8.py`) — the code point at code point
/// `index`, which must be below the string's code point count.
///
/// Upstream fuses the position lookup with the decode so the JIT sees one
/// elidable call; both spellings walk to the same byte offset, so this is the
/// composition of the two.
#[inline]
#[majit_macros::elidable]
pub fn codepoint_at_index(utf8: &Wtf8, storage: &[Utf8LocElem], index: usize) -> i64 {
    codepoint_at_pos(utf8, codepoint_position_at_index(utf8, storage, index))
}

/// `codepoint_index_at_byte_position` (`rutf8.py`) — the code point index
/// for which `codepoint_position_at_index` is `bytepos`.
///
/// Logarithmic in the string length, plus a constant that is not tiny either.
/// Upstream's leading `if bytepos < 0: return bytepos` guard carries `str.find`
/// misses through; pyre's search returns `Option`, so the caller handles a miss
/// and this takes an offset that is always real.
#[majit_macros::elidable]
pub fn codepoint_index_at_byte_position(
    utf8: &Wtf8,
    storage: &[Utf8LocElem],
    bytepos: usize,
    num_codepoints: usize,
) -> usize {
    let bytes = utf8.as_bytes();
    // binary search on elements of storage
    let bytes_remaining = bytes.len() - bytepos;
    // the fact that one codepoint is encoded in 1-4 bytes constrains the
    // result. pick good min and max indexes based on this observation. saves a
    // few bisection steps.  Both `max` candidates are clamped at zero, which
    // the `max`/`min` against a non-negative bound would have discarded anyway.
    let mut index_min = std::cmp::max(
        bytepos / 4,
        num_codepoints.saturating_sub(bytes_remaining + 1),
    ) >> 6;
    let mut index_max =
        std::cmp::min(bytepos, num_codepoints.saturating_sub(bytes_remaining / 4)) >> 6;
    while index_min < index_max {
        // this addition can't overflow because storage has a length that is
        // 1/64 of the length of a string
        let index_middle = (index_min + index_max).div_ceil(2);
        let base_bytepos = storage[index_middle].baseindex as usize;
        if bytepos < base_bytepos {
            index_max = index_middle - 1;
        } else {
            index_min = index_middle;
        }
    }

    let baseindex = storage[index_min].baseindex as usize;
    if baseindex == bytepos {
        return index_min << 6;
    }

    // use ofs to get closer to the correct character index.  Reaching here
    // means `bytepos` is past a group start, so the string is not empty and
    // `num_codepoints - 1` cannot underflow.
    let mut result = index_min << 6;
    let mut bytepos1 = baseindex;
    let maxindex = if index_min == storage.len() - 1 {
        ((num_codepoints - 1) >> 2) & 0x0F
    } else {
        16
    };
    for i in 0..maxindex {
        let x = baseindex + storage[index_min].ofs[i] as usize;
        if x >= bytepos {
            break;
        }
        bytepos1 = x;
        result = (index_min << 6) + (i << 2) + 1;
    }

    // this loop should run at most four times
    while bytepos1 < bytepos {
        bytepos1 = next_codepoint_pos(utf8, bytepos1);
        result += 1;
    }
    result
}

/// `rutf8.Utf8StringBuilder` (`rutf8.py`).
///
/// `@always_inline` methods wrap a [`crate::rstring::StringBuilder`] (`_s`)
/// and a code-point counter (`_lgt`).  `pyopcode.py BUILD_STRING` is
/// `append_utf8` + `space.newutf8(builder.build(), builder.getlength())`.
/// The rbuilder helpers residualise when jitted (`rbuilder.py ll_append`
/// / `ll_build`); these wrappers stay look-inside like upstream.
pub struct Utf8StringBuilder {
    _s: crate::rstring::StringBuilder,
    _lgt: i64,
}

impl Utf8StringBuilder {
    /// `Utf8StringBuilder.__init__(self, size=0)`.
    #[inline]
    pub fn new(size: i64) -> Self {
        Self {
            _s: crate::rstring::StringBuilder::new(size),
            _lgt: 0,
        }
    }

    /// `Utf8StringBuilder.append_utf8(self, utf8, length)`.
    #[inline]
    pub fn append_utf8(
        &mut self,
        utf8: *mut crate::unicodeobject::UnicodeValueStorage,
        length: i64,
    ) {
        self._s.append(utf8);
        self._lgt += length;
    }

    /// `Utf8StringBuilder.build(self)` — the rstr `STR` payload.
    #[inline]
    pub fn build(&mut self) -> *mut crate::unicodeobject::UnicodeValueStorage {
        self._s.build()
    }

    /// `Utf8StringBuilder.getlength(self)` — code-point count.
    #[inline]
    pub fn getlength(&self) -> i64 {
        self._lgt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustpython_wtf8::{CodePoint, Wtf8Buf};

    #[test]
    fn check_utf8_counts_code_points() {
        assert_eq!(check_utf8(b"", true), Ok(0));
        assert_eq!(check_utf8(b"abc", true), Ok(3));
        assert_eq!(check_utf8("é中\u{10000}".as_bytes(), false), Ok(3));
    }

    #[test]
    fn check_utf8_rejects_what_wtf8_from_bytes_admits() {
        // `Wtf8::from_bytes`'s surrogate arm leaves the second byte unbounded
        // above and the third unconstrained, so both of these reach a string.
        for bad in [b"\xed\xc0\x80".as_slice(), b"\xed\xa0\x41".as_slice()] {
            assert!(Wtf8::from_bytes(bad).is_some());
            assert_eq!(check_utf8(bad, true), Err(CheckError { pos: 0 }));
            assert_eq!(wtf8_from_bytes(bad, true), Err(CheckError { pos: 0 }));
        }
    }

    #[test]
    fn surrogate_bytes_names_exactly_what_the_allowance_admits() {
        // The decoder's span arms read `surrogate_bytes` after
        // `invalid_byte_2_of_3` has already passed, and take it to mean "the
        // allowance is why this pair got through". That holds only if the
        // predicate names precisely the pairs the two `allow_surrogates`
        // answers disagree on.
        for ch1 in 0xE0..=0xEFu8 {
            for ch2 in 0..=0xFFu8 {
                let admitted_by_the_allowance =
                    invalid_byte_2_of_3(ch1, ch2, false) && !invalid_byte_2_of_3(ch1, ch2, true);
                assert_eq!(
                    admitted_by_the_allowance,
                    surrogate_bytes(ch1, ch2) && !invalid_cont_byte(ch2),
                    "{ch1:#04x} {ch2:#04x}"
                );
            }
        }
    }

    #[test]
    fn check_utf8_and_wtf8_from_bytes_agree() {
        // Every two-byte buffer, every three-byte buffer whose lead byte can
        // begin a three-byte sequence, and the four-byte leads around the two
        // range bounds.  `wtf8_from_bytes` reports where the strict scan
        // stopped, which is the sequence start `check_utf8` names.
        let mut buf = Vec::new();
        let probe = |buf: &[u8]| {
            for allow in [true, false] {
                let want = check_utf8(buf, allow);
                let got = wtf8_from_bytes(buf, allow).map(|_| ());
                assert_eq!(want.is_ok(), got.is_ok(), "{buf:02x?} allow={allow}");
                if let (Err(a), Err(b)) = (want, got) {
                    assert_eq!(a, b, "{buf:02x?} allow={allow}");
                }
            }
        };
        for b1 in 0u16..=0xFF {
            for b2 in 0u16..=0xFF {
                buf.clear();
                buf.extend_from_slice(&[b1 as u8, b2 as u8]);
                probe(&buf);
            }
        }
        for b1 in 0xE0u16..=0xEF {
            for b2 in 0u16..=0xFF {
                for b3 in 0u16..=0xFF {
                    buf.clear();
                    buf.extend_from_slice(&[b1 as u8, b2 as u8, b3 as u8]);
                    probe(&buf);
                }
            }
        }
        for b1 in [0xF0u8, 0xF3, 0xF4, 0xF5] {
            for b2 in 0u16..=0xFF {
                buf.clear();
                buf.extend_from_slice(&[b1, b2 as u8, 0x80, 0x80]);
                probe(&buf);
                buf.clear();
                buf.extend_from_slice(&[b1, 0x90, b2 as u8, 0x80]);
                probe(&buf);
            }
        }
        // A surrogate before the offending sequence: the scan must resume
        // past it rather than report its own position.
        probe(b"\xed\xa0\x80\xed\xc0\x80");
        probe(b"a\xed\xa0\x80\xed\xb0\x80z");
    }

    #[test]
    fn check_utf8_admits_surrogates_only_when_asked() {
        let lone = b"\xed\xa0\x80";
        assert_eq!(check_utf8(lone, true), Ok(1));
        assert_eq!(check_utf8(lone, false), Err(CheckError { pos: 0 }));
        // A pair stays two code points, which is what `surrogatepass` decodes
        // `'\\ud800\\udc00'.encode('utf-8', 'surrogatepass')` back to.
        assert_eq!(check_utf8(b"\xed\xa0\x80\xed\xb0\x80", true), Ok(2));
    }

    #[test]
    fn check_utf8_reports_the_offending_sequences_position() {
        assert_eq!(check_utf8(b"ab\x80", true), Err(CheckError { pos: 2 }));
        assert_eq!(check_utf8(b"ab\xc2", true), Err(CheckError { pos: 2 }));
        assert_eq!(
            check_utf8(b"ab\xe0\x80\x80", true),
            Err(CheckError { pos: 2 })
        );
        assert_eq!(
            check_utf8(b"ab\xf4\x90\x80\x80", true),
            Err(CheckError { pos: 2 })
        );
        assert_eq!(check_utf8(b"ab\xf5", true), Err(CheckError { pos: 2 }));
        assert_eq!(check_utf8(b"ab\xc1\x81", true), Err(CheckError { pos: 2 }));
    }

    #[test]
    fn a_checked_buffer_agrees_with_the_index_table() {
        // The count `check_utf8` returns is the one `create_utf8_index_storage`
        // walks to, which is the disagreement the whole check exists to stop.
        let buf = Wtf8Buf::from_string("a\u{e9}\u{4e2d}\u{10000}".repeat(40));
        let len = check_utf8(buf.as_bytes(), true).unwrap();
        assert_eq!(len, buf.code_points().count() as i64);
        let len = len as usize;
        let storage = create_utf8_index_storage(&buf, len);
        assert_eq!(
            codepoint_at_index(&buf, &storage, len - 1),
            buf.code_points().last().unwrap().to_u32() as i64
        );
    }

    fn sample(kind: &str, repeat: usize) -> Wtf8Buf {
        let mut buf = Wtf8Buf::new();
        for i in 0..repeat {
            match kind {
                "ascii" => buf.push_str("abcd"),
                "latin1" => buf.push_str("éèêë"),
                "wide" => buf.push_str("一二三四"),
                "astral" => buf.push_str("\u{1f600}\u{1f601}\u{1f602}\u{1f603}"),
                "mixed" => buf.push_str("a\u{e9}\u{4e00}\u{1f600}"),
                "surrogate" => {
                    buf.push_str("a");
                    buf.push(CodePoint::from_u32(0xD800 + (i % 0x400) as u32).unwrap());
                    buf.push_str("\u{4e00}");
                    buf.push(CodePoint::from_u32(0xDC80).unwrap());
                }
                other => panic!("unknown sample kind {other}"),
            }
        }
        buf
    }

    const KINDS: [&str; 6] = ["ascii", "latin1", "wide", "astral", "mixed", "surrogate"];
    // Lengths straddling the 4-code-point delta stride and the 64-code point
    // group, in both directions.
    const REPEATS: [usize; 11] = [0, 1, 2, 15, 16, 17, 32, 63, 64, 65, 100];

    /// The table must answer exactly what a walk from the start would.
    fn assert_matches_walk(buf: &Wtf8Buf) {
        let positions: Vec<usize> = buf.code_point_indices().map(|(pos, _)| pos).collect();
        let storage = create_utf8_index_storage(buf, positions.len());
        for (index, &expected) in positions.iter().enumerate() {
            assert_eq!(
                codepoint_position_at_index(buf, &storage, index),
                expected,
                "index {index} of a {} byte / {} code point payload",
                buf.len(),
                positions.len(),
            );
        }
        // "smaller than or equal to the utf8 length" — the build loop's
        // padding slot exists so the count itself resolves to the end, which
        // is what a `start`/`end` bound equal to the length asks for.
        assert_eq!(
            codepoint_position_at_index(buf, &storage, positions.len()),
            buf.len(),
            "the code point count of a {} byte payload",
            buf.len(),
        );
    }

    #[test]
    fn test_index_storage_matches_a_walk() {
        for repeat in REPEATS {
            for kind in KINDS {
                assert_matches_walk(&sample(kind, repeat));
            }
        }
    }

    #[test]
    fn test_index_storage_handles_a_partial_final_group() {
        // A code point count that is neither a multiple of 4 nor of 64 leaves
        // the last group half-written; every filled entry must still answer.
        let mut buf = sample("mixed", 3);
        buf.push_str("x\u{4e00}");
        assert_eq!(buf.code_points().count(), 14);
        assert_matches_walk(&buf);
    }

    #[test]
    fn test_index_storage_size_is_one_entry_per_64_code_points() {
        assert_eq!(create_utf8_index_storage(Wtf8::new(""), 0).len(), 1);
        let ascii = "a".repeat(128);
        assert_eq!(
            create_utf8_index_storage(Wtf8::new(&ascii[..64]), 64).len(),
            2
        );
        assert_eq!(
            create_utf8_index_storage(Wtf8::new(&ascii[..127]), 127).len(),
            2
        );
        assert_eq!(
            create_utf8_index_storage(Wtf8::new(&ascii[..]), 128).len(),
            3
        );
    }

    /// `codepoint_at_index` must yield what iteration yields, at every index.
    #[test]
    fn test_codepoint_at_index_matches_iteration() {
        for repeat in REPEATS {
            for kind in KINDS {
                let buf = sample(kind, repeat);
                let expected: Vec<CodePoint> = buf.code_points().collect();
                let storage = create_utf8_index_storage(&buf, expected.len());
                for (index, &cp) in expected.iter().enumerate() {
                    assert_eq!(
                        codepoint_at_index(&buf, &storage, index),
                        cp.to_u32() as i64,
                        "{kind} * {repeat}, index {index}",
                    );
                }
            }
        }
    }

    /// `codepoint_index_at_byte_position` must invert
    /// `codepoint_position_at_index` at every code point boundary.
    #[test]
    fn test_byte_position_inverts_index_lookup() {
        for repeat in REPEATS {
            for kind in KINDS {
                let buf = sample(kind, repeat);
                let positions: Vec<usize> = buf.code_point_indices().map(|(pos, _)| pos).collect();
                let n = positions.len();
                let storage = create_utf8_index_storage(&buf, n);
                for (index, &bytepos) in positions.iter().enumerate() {
                    assert_eq!(
                        codepoint_index_at_byte_position(&buf, &storage, bytepos, n),
                        index,
                        "{kind} * {repeat}, byte {bytepos}",
                    );
                }
                // The end of the string maps to the code point count.
                assert_eq!(
                    codepoint_index_at_byte_position(&buf, &storage, buf.len(), n),
                    n,
                    "{kind} * {repeat}, end",
                );
            }
        }
    }

    #[test]
    fn test_codepoints_in_utf8_counts_a_byte_window() {
        for repeat in REPEATS {
            for kind in KINDS {
                let buf = sample(kind, repeat);
                let positions: Vec<usize> = buf.code_point_indices().map(|(pos, _)| pos).collect();
                let n = positions.len();
                assert_eq!(codepoints_in_utf8(&buf, 0, buf.len()), n);
                // Every code point-aligned window counts its own span, and an
                // `end` past the buffer clamps.
                for (index, &start) in positions.iter().enumerate() {
                    assert_eq!(codepoints_in_utf8(&buf, start, buf.len()), n - index);
                    assert_eq!(codepoints_in_utf8(&buf, 0, start), index);
                }
                assert_eq!(codepoints_in_utf8(&buf, 0, buf.len() + 10), n);
            }
        }
    }
}
