//! `rpython/rlib/rstring.py` — substring search used by
//! `LLHelpers.ll_search` (`rtyper/lltypesystem/rstr.py`).
//!
//! `SEARCH_COUNT`, `SEARCH_FIND`, `SEARCH_RFIND` live here, matching
//! the upstream module. The residual `_search_elidable` reads `Ptr(STR)`
//! through the shared layout in [`crate::lltypesystem::rstr`]. UNICODE
//! `_search_normal` reads `Ptr(UNICODE)` through the same header with
//! UCS-4 `UniChar` cells.

use crate::lltypesystem::rstr::{str_as_bytes, unicode_as_unichars};

/// `rstring.py` `SEARCH_COUNT`.
pub const SEARCH_COUNT: i64 = 0;
/// `rstring.py` `SEARCH_FIND`.
pub const SEARCH_FIND: i64 = 1;
/// `rstring.py` `SEARCH_RFIND`.
pub const SEARCH_RFIND: i64 = 2;

/// `rpython/rlib/rarithmetic.py` `LONG_BIT`, used as `BLOOM_WIDTH`.
const BLOOM_WIDTH: u32 = usize::BITS;

const TWOWAY_MAX_SHIFT: usize = 255;
const TWOWAY_TABLE_SIZE: usize = 64;
const TWOWAY_TABLE_MASK: usize = TWOWAY_TABLE_SIZE - 1;

/// `STR` `Char` (`u8`) or `UNICODE` `UniChar` (`u32`, UCS-4).
pub trait SearchChar: Copy + Eq + Ord {
    fn as_ord(self) -> u32;
    const NUL: Self;
}

impl SearchChar for u8 {
    fn as_ord(self) -> u32 {
        u32::from(self)
    }
    const NUL: Self = 0;
}

impl SearchChar for u32 {
    fn as_ord(self) -> u32 {
        self
    }
    const NUL: Self = 0;
}

/// `rstring.py` `bloom_add`.
#[inline]
fn bloom_add<C: SearchChar>(mask: u64, c: C) -> u64 {
    mask | (1u64 << (c.as_ord() & (BLOOM_WIDTH - 1)))
}

/// `rstring.py` `bloom`.
#[inline]
fn bloom<C: SearchChar>(mask: u64, c: C) -> bool {
    (mask & (1u64 << (c.as_ord() & (BLOOM_WIDTH - 1)))) != 0
}

/// `rstring.py` `_search_elidable`. `@jit.elidable` wrapper around
/// `_search_normal` for the STR arm of `_search` (`isinstance(value, str)`
/// is a translation-time constant).
#[majit_macros::elidable]
pub fn _search_elidable(s1: *const u8, s2: *const u8, start: i64, end: i64, mode: i64) -> i64 {
    // SAFETY: residual `Ptr(STR)` callee; the caller supplies live STRs.
    let value = unsafe { str_as_bytes(s1) };
    let other = unsafe { str_as_bytes(s2) };
    _search_normal(value, other, start, end, mode)
}

/// UNICODE `@specialize.argtype(0, 1)` of `_search_normal` (`rstring.py`).
/// `_search` sends STR through `_search_elidable` and UNICODE through
/// `_search_normal`; this residual takes two `Ptr(UNICODE)` and is
/// published on `HELPER_FNADDRS` as `_search_normal`, the way the STR
/// residual is published as `_search_elidable`.
pub extern "C" fn _search_normal_unicode(
    s1: *const u8,
    s2: *const u8,
    start: i64,
    end: i64,
    mode: i64,
) -> i64 {
    // SAFETY: residual `Ptr(UNICODE)` callee; the caller supplies live UNICODEs.
    let value = unsafe { unicode_as_unichars(s1) };
    let other = unsafe { unicode_as_unichars(s2) };
    _search_normal(value, other, start, end, mode)
}

#[cfg(not(target_arch = "wasm32"))]
#[majit_ir::linkme::distributed_slice(majit_ir::helper_fnaddr::HELPER_FNADDRS)]
#[linkme(crate = majit_ir::linkme)]
#[allow(non_upper_case_globals)]
static __MAJIT_HELPER_FNADDR__search_normal: majit_ir::helper_fnaddr::HelperFnAddr =
    majit_ir::helper_fnaddr::HelperFnAddr::new(
        ::core::concat!(::core::module_path!(), "::", stringify!(_search_normal)),
        _search_normal_unicode as *const (),
        5,
    );

#[cfg(target_arch = "wasm32")]
#[ctor::ctor(unsafe)]
fn __majit_register_helper_fnaddr__search_normal() {
    majit_ir::helper_fnaddr::register(
        ::core::concat!(::core::module_path!(), "::", stringify!(_search_normal)),
        _search_normal_unicode as *const (),
        5,
    );
}

/// `rstring.py` `_search_normal`.
pub fn _search_normal<C: SearchChar>(
    value: &[C],
    other: &[C],
    mut start: i64,
    mut end: i64,
    mode: i64,
) -> i64 {
    if start < 0 {
        start = 0;
    }
    let n_value = value.len() as i64;
    if end > n_value {
        end = n_value;
    }
    if start > end {
        if mode == SEARCH_COUNT {
            return 0;
        }
        return -1;
    }

    let start_u = start as usize;
    let end_u = end as usize;
    let n = end_u - start_u;
    let m = other.len();

    if m == 0 {
        if mode == SEARCH_COUNT {
            return end - start + 1;
        } else if mode == SEARCH_RFIND {
            return end;
        } else {
            return start;
        }
    }

    let Some(w) = n.checked_sub(m) else {
        if mode == SEARCH_COUNT {
            return 0;
        }
        return -1;
    };

    if mode != SEARCH_RFIND {
        let res = if n < 2500 || (m < 100 && n < 30000) || m < 6 {
            default_find(value, start_u, n, other, m, mode)
        } else if (m >> 2) * 3 < (n >> 2) {
            let (cut, period, gap, is_periodic, table) = twoway_preprocess(other, m);
            if mode == SEARCH_COUNT {
                return two_way_count(
                    value,
                    start_u,
                    n,
                    other,
                    m,
                    cut,
                    period,
                    gap,
                    is_periodic,
                    &table,
                );
            }
            two_way(
                value,
                start_u,
                n,
                other,
                m,
                cut,
                period,
                gap,
                is_periodic,
                &table,
            )
        } else {
            adaptive_find(value, start_u, n, other, m, mode)
        };
        if mode == SEARCH_COUNT {
            return res;
        }
        return if res == -1 { -1 } else { start + res };
    }

    // Reverse find (rfind): Boyer-Moore-Horspool, unchanged.
    let mlast = m - 1;
    let mut skip = mlast;
    let mut mask = 0u64;
    mask = bloom_add(mask, other[0]);
    let mut i = mlast;
    while i > 0 {
        mask = bloom_add(mask, other[i]);
        if other[i] == other[0] {
            skip = i - 1;
        }
        i -= 1;
    }

    let mut i = start_u + w + 1;
    while i > start_u {
        i -= 1;
        if value[i] == other[0] {
            let mut j = mlast;
            let mut matched = true;
            while j > 0 {
                if value[i + j] != other[j] {
                    matched = false;
                    break;
                }
                j -= 1;
            }
            if matched {
                return i as i64;
            }
            if i > 0 && !bloom(mask, value[i - 1]) {
                i = i.saturating_sub(m);
            } else {
                i = i.saturating_sub(skip);
            }
        } else if i > 0 && !bloom(mask, value[i - 1]) {
            i = i.saturating_sub(m);
        }
    }
    -1
}

/// `rstring.py` `_lex_search`.
fn lex_search<C: SearchChar>(
    needle: &[C],
    len_needle: usize,
    invert_alphabet: bool,
) -> (usize, usize) {
    let mut max_suffix = 0usize;
    let mut candidate = 1usize;
    let mut k = 0usize;
    let mut period = 1usize;
    while candidate + k < len_needle {
        let a = needle[candidate + k];
        let b = needle[max_suffix + k];
        if if invert_alphabet { b < a } else { a < b } {
            candidate += k + 1;
            k = 0;
            period = candidate - max_suffix;
        } else if a == b {
            if k + 1 != period {
                k += 1;
            } else {
                candidate += period;
                k = 0;
            }
        } else {
            max_suffix = candidate;
            candidate += 1;
            k = 0;
            period = 1;
        }
    }
    (max_suffix, period)
}

/// `rstring.py` `_factorize`.
fn factorize<C: SearchChar>(needle: &[C], len_needle: usize) -> (usize, usize) {
    let (cut1, period1) = lex_search(needle, len_needle, false);
    let (cut2, period2) = lex_search(needle, len_needle, true);
    if cut1 > cut2 {
        (cut1, period1)
    } else {
        (cut2, period2)
    }
}

/// `rstring.py` `_twoway_preprocess`.
fn twoway_preprocess<C: SearchChar>(
    needle: &[C],
    len_needle: usize,
) -> (usize, usize, usize, bool, Vec<isize>) {
    let (cut, mut period) = factorize(needle, len_needle);
    let mut is_periodic = true;
    let mut i = 0usize;
    while i < cut {
        if needle[i] != needle[period + i] {
            is_periodic = false;
            break;
        }
        i += 1;
    }
    let gap = if is_periodic {
        0
    } else {
        period = cut.max(len_needle - cut) + 1;
        let mut gap = len_needle;
        let last = (needle[len_needle - 1].as_ord() as usize) & TWOWAY_TABLE_MASK;
        let mut i = len_needle - 1;
        while i > 0 {
            i -= 1;
            if ((needle[i].as_ord() as usize) & TWOWAY_TABLE_MASK) == last {
                gap = len_needle - 1 - i;
                break;
            }
        }
        gap
    };
    // compressed Boyer-Moore "bad character" table. Build it with an
    // explicit append loop rather than repeating `not_found_shift`
    // TWOWAY_TABLE_SIZE times: `_twoway_preprocess` is reached as a
    // low-level helper (`rstr.py` `ll_search` via hlstr), and the
    // post-annotation `transform_allocate` pass rewrites list-multiply
    // into an `alloc_and_set` SpaceOperation that low-level re-annotation
    // cannot reflow.
    let not_found_shift = len_needle.min(TWOWAY_MAX_SHIFT) as isize;
    let mut table = Vec::new();
    let mut _i = 0;
    while _i < TWOWAY_TABLE_SIZE {
        table.push(not_found_shift);
        _i += 1;
    }
    let mut i = len_needle - not_found_shift as usize;
    while i < len_needle {
        table[(needle[i].as_ord() as usize) & TWOWAY_TABLE_MASK] = (len_needle - 1 - i) as isize;
        i += 1;
    }
    (cut, period, gap, is_periodic, table)
}

/// `rstring.py` `_two_way`.
fn two_way<C: SearchChar>(
    value: &[C],
    base: usize,
    n: usize,
    needle: &[C],
    m: usize,
    cut: usize,
    mut period: usize,
    gap: usize,
    is_periodic: bool,
    table: &[isize],
) -> i64 {
    let haystack_end = base + n;
    let mut window_last = base + m - 1;
    if is_periodic {
        let mut memory = 0usize;
        let mut skip_horspool = false;
        while window_last < haystack_end {
            if !skip_horspool {
                loop {
                    let shift =
                        table[(value[window_last].as_ord() as usize) & TWOWAY_TABLE_MASK] as usize;
                    window_last += shift;
                    if shift == 0 {
                        break;
                    }
                    if window_last >= haystack_end {
                        return -1;
                    }
                }
            }
            skip_horspool = false;
            let window = window_last + 1 - m;
            let mut i = cut.max(memory);
            let mut mismatch = false;
            while i < m {
                if needle[i] != value[window + i] {
                    window_last += i - cut + 1;
                    memory = 0;
                    mismatch = true;
                    break;
                }
                i += 1;
            }
            if mismatch {
                continue;
            }
            i = memory;
            while i < cut {
                if needle[i] != value[window + i] {
                    window_last += period;
                    memory = m - period;
                    if window_last >= haystack_end {
                        return -1;
                    }
                    let shift =
                        table[(value[window_last].as_ord() as usize) & TWOWAY_TABLE_MASK] as usize;
                    if shift != 0 {
                        let mem_jump = cut.max(memory) - cut + 1;
                        memory = 0;
                        window_last += shift.max(mem_jump);
                    } else {
                        skip_horspool = true;
                    }
                    mismatch = true;
                    break;
                }
                i += 1;
            }
            if mismatch {
                continue;
            }
            return (window - base) as i64;
        }
        -1
    } else {
        if period < gap {
            period = gap;
        }
        let gap_jump_end = (cut + gap).min(m);
        while window_last < haystack_end {
            loop {
                let shift =
                    table[(value[window_last].as_ord() as usize) & TWOWAY_TABLE_MASK] as usize;
                window_last += shift;
                if shift == 0 {
                    break;
                }
                if window_last >= haystack_end {
                    return -1;
                }
            }
            let window = window_last + 1 - m;
            let mut mismatch = false;
            let mut i = cut;
            while i < gap_jump_end {
                if needle[i] != value[window + i] {
                    window_last += gap;
                    mismatch = true;
                    break;
                }
                i += 1;
            }
            if mismatch {
                continue;
            }
            i = gap_jump_end;
            while i < m {
                if needle[i] != value[window + i] {
                    window_last += i - cut + 1;
                    mismatch = true;
                    break;
                }
                i += 1;
            }
            if mismatch {
                continue;
            }
            i = 0;
            while i < cut {
                if needle[i] != value[window + i] {
                    window_last += period;
                    mismatch = true;
                    break;
                }
                i += 1;
            }
            if mismatch {
                continue;
            }
            return (window - base) as i64;
        }
        -1
    }
}

/// `rstring.py` `_two_way_count`.
fn two_way_count<C: SearchChar>(
    value: &[C],
    base: usize,
    n: usize,
    needle: &[C],
    m: usize,
    cut: usize,
    period: usize,
    gap: usize,
    is_periodic: bool,
    table: &[isize],
) -> i64 {
    let mut index = 0usize;
    let mut count = 0i64;
    loop {
        let result = two_way(
            value,
            base + index,
            n - index,
            needle,
            m,
            cut,
            period,
            gap,
            is_periodic,
            table,
        );
        if result == -1 {
            return count;
        }
        count += 1;
        index += result as usize + m;
    }
}

/// `rstring.py` `_default_find`.
fn default_find<C: SearchChar>(
    value: &[C],
    base: usize,
    n: usize,
    needle: &[C],
    m: usize,
    mode: i64,
) -> i64 {
    let w = n - m;
    let mlast = m - 1;
    let mut count = 0i64;
    let mut gap = mlast;
    let last = needle[mlast];
    let mut mask = 0u64;
    let mut j = 0usize;
    while j < mlast {
        mask = bloom_add(mask, needle[j]);
        if needle[j] == last {
            gap = mlast - j - 1;
        }
        j += 1;
    }
    mask = bloom_add(mask, last);
    let mut i = 0usize;
    while i <= w {
        if value[base + mlast + i] == last {
            j = 0;
            while j < mlast {
                if value[base + i + j] != needle[j] {
                    break;
                }
                j += 1;
            }
            if j == mlast {
                if mode != SEARCH_COUNT {
                    return i as i64;
                }
                count += 1;
                i += mlast;
            } else {
                let la = base + mlast + i + 1;
                let c = if la < value.len() { value[la] } else { C::NUL };
                if !bloom(mask, c) {
                    i += m;
                } else {
                    i += gap;
                }
            }
        } else {
            let la = base + mlast + i + 1;
            let c = if la < value.len() { value[la] } else { C::NUL };
            if !bloom(mask, c) {
                i += m;
            }
        }
        i += 1;
    }
    if mode != SEARCH_COUNT { -1 } else { count }
}

/// `rstring.py` `_adaptive_find`.
fn adaptive_find<C: SearchChar>(
    value: &[C],
    base: usize,
    n: usize,
    needle: &[C],
    m: usize,
    mode: i64,
) -> i64 {
    let w = n - m;
    let mlast = m - 1;
    let mut count = 0i64;
    let mut gap = mlast;
    let mut hits = 0usize;
    let last = needle[mlast];
    let mut mask = 0u64;
    let mut j = 0usize;
    while j < mlast {
        mask = bloom_add(mask, needle[j]);
        if needle[j] == last {
            gap = mlast - j - 1;
        }
        j += 1;
    }
    mask = bloom_add(mask, last);
    let mut i = 0usize;
    while i <= w {
        if value[base + mlast + i] == last {
            j = 0;
            while j < mlast {
                if value[base + i + j] != needle[j] {
                    break;
                }
                j += 1;
            }
            if j == mlast {
                if mode != SEARCH_COUNT {
                    return i as i64;
                }
                count += 1;
                i += mlast;
            } else {
                hits += j + 1;
                if hits > m / 4 && w - i > 2000 {
                    let (cut, period, gap, is_periodic, table) = twoway_preprocess(needle, m);
                    if mode != SEARCH_COUNT {
                        let res = two_way(
                            value,
                            base + i,
                            n - i,
                            needle,
                            m,
                            cut,
                            period,
                            gap,
                            is_periodic,
                            &table,
                        );
                        return if res == -1 { -1 } else { res + i as i64 };
                    }
                    let res = two_way_count(
                        value,
                        base + i,
                        n - i,
                        needle,
                        m,
                        cut,
                        period,
                        gap,
                        is_periodic,
                        &table,
                    );
                    return res + count;
                }
                let la = base + mlast + i + 1;
                let c = if la < value.len() { value[la] } else { C::NUL };
                if !bloom(mask, c) {
                    i += m;
                } else {
                    i += gap;
                }
            }
        } else {
            let la = base + mlast + i + 1;
            let c = if la < value.len() { value[la] } else { C::NUL };
            if !bloom(mask, c) {
                i += m;
            }
        }
        i += 1;
    }
    if mode != SEARCH_COUNT { -1 } else { count }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lltypesystem::rstr::{
        STR_CHARS_OFFSET, STR_HASH_OFFSET, STR_LEN_OFFSET, UNICODE_CHARS_OFFSET,
        UNICODE_HASH_OFFSET, UNICODE_LEN_OFFSET,
    };

    fn naive<C: SearchChar>(hay: &[C], needle: &[C], lo: i64, hi: i64, mode: i64) -> i64 {
        let mut start = lo;
        let mut end = hi;
        if start < 0 {
            start = 0;
        }
        let n_value = hay.len() as i64;
        if end > n_value {
            end = n_value;
        }
        if start > end {
            return if mode == SEARCH_COUNT { 0 } else { -1 };
        }
        let lo = start as usize;
        let hi = end as usize;
        let n = hi - lo;
        let m = needle.len();
        if m == 0 {
            return if mode == SEARCH_COUNT {
                end - start + 1
            } else if mode == SEARCH_RFIND {
                end
            } else {
                start
            };
        }
        if n < m {
            return if mode == SEARCH_COUNT { 0 } else { -1 };
        }
        let window = &hay[lo..hi];
        match mode {
            SEARCH_FIND => window
                .windows(m)
                .position(|w| w == needle)
                .map(|p| (lo + p) as i64)
                .unwrap_or(-1),
            SEARCH_RFIND => window
                .windows(m)
                .rposition(|w| w == needle)
                .map(|p| (lo + p) as i64)
                .unwrap_or(-1),
            SEARCH_COUNT => {
                let mut count = 0i64;
                let mut i = 0usize;
                while i + m <= window.len() {
                    if &window[i..i + m] == needle {
                        count += 1;
                        i += m;
                    } else {
                        i += 1;
                    }
                }
                count
            }
            other => panic!("unknown SEARCH_* mode {other}"),
        }
    }

    fn assert_agree<C: SearchChar>(hay: &[C], needle: &[C], lo: i64, hi: i64) {
        for mode in [SEARCH_FIND, SEARCH_RFIND, SEARCH_COUNT] {
            let got = _search_normal(hay, needle, lo, hi, mode);
            let expect = naive(hay, needle, lo, hi, mode);
            assert_eq!(got, expect, "lo={lo} hi={hi} mode={mode}");
        }
    }

    #[test]
    fn search_normal_find_rfind_count() {
        let hay = b"abracadabra";
        assert_agree(hay, b"bra", 0, hay.len() as i64);
        assert_agree(hay, b"a", 0, hay.len() as i64);
        assert_agree(hay, b"cad", 1, 6);
        assert_eq!(
            _search_normal(hay, b"zzz", 0, hay.len() as i64, SEARCH_FIND),
            -1
        );
        assert_eq!(
            _search_normal(hay, b"bra", 0, hay.len() as i64, SEARCH_RFIND),
            8
        );
        assert_eq!(
            _search_normal(hay, b"a", 0, hay.len() as i64, SEARCH_COUNT),
            5
        );
    }

    #[test]
    fn search_normal_repeated_prefix_needle() {
        let hay = b"aaaabaaa";
        assert_agree(hay, b"aaa", 0, hay.len() as i64);
        assert_eq!(
            _search_normal(hay, b"aaa", 0, hay.len() as i64, SEARCH_FIND),
            0
        );
        assert_eq!(
            _search_normal(hay, b"aaa", 0, hay.len() as i64, SEARCH_RFIND),
            5
        );
        assert_eq!(
            _search_normal(hay, b"aaa", 0, hay.len() as i64, SEARCH_COUNT),
            2
        );
        assert_agree(b"aaaaaa", b"aa", 0, 6);
    }

    #[test]
    fn search_normal_empty_range() {
        let hay = b"hello";
        assert_eq!(_search_normal(hay, b"el", 3, 1, SEARCH_FIND), -1);
        assert_eq!(_search_normal(hay, b"el", 3, 1, SEARCH_RFIND), -1);
        assert_eq!(_search_normal(hay, b"el", 3, 1, SEARCH_COUNT), 0);
        assert_eq!(_search_normal(hay, b"", 3, 1, SEARCH_FIND), -1);
        assert_eq!(_search_normal(hay, b"", 3, 1, SEARCH_COUNT), 0);
        assert_eq!(_search_normal(hay, b"el", 2, 2, SEARCH_FIND), -1);
        assert_eq!(_search_normal(hay, b"", 2, 2, SEARCH_FIND), 2);
        assert_eq!(_search_normal(hay, b"", 2, 2, SEARCH_COUNT), 1);
    }

    fn str_buf(bytes: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; STR_CHARS_OFFSET + bytes.len() + 1];
        unsafe {
            buf.as_mut_ptr()
                .add(STR_HASH_OFFSET)
                .cast::<isize>()
                .write_unaligned(0);
            buf.as_mut_ptr()
                .add(STR_LEN_OFFSET)
                .cast::<usize>()
                .write_unaligned(bytes.len());
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buf.as_mut_ptr().add(STR_CHARS_OFFSET),
                bytes.len(),
            );
        }
        buf
    }

    #[test]
    fn search_elidable_reads_str_layout() {
        let hay = str_buf(b"abracadabra");
        let needle = str_buf(b"cad");
        assert_eq!(
            _search_elidable(hay.as_ptr(), needle.as_ptr(), 0, 11, SEARCH_FIND),
            4
        );
    }

    fn unicode_buf(chars: &[u32]) -> Vec<u8> {
        let mut buf = vec![0u8; UNICODE_CHARS_OFFSET + chars.len() * 4];
        unsafe {
            buf.as_mut_ptr()
                .add(UNICODE_HASH_OFFSET)
                .cast::<isize>()
                .write_unaligned(0);
            buf.as_mut_ptr()
                .add(UNICODE_LEN_OFFSET)
                .cast::<usize>()
                .write_unaligned(chars.len());
            std::ptr::copy_nonoverlapping(
                chars.as_ptr(),
                buf.as_mut_ptr().add(UNICODE_CHARS_OFFSET).cast::<u32>(),
                chars.len(),
            );
        }
        buf
    }

    fn latin1(bytes: &[u8]) -> Vec<u32> {
        bytes.iter().map(|&b| u32::from(b)).collect()
    }

    #[test]
    fn search_normal_unicode_find_rfind_count() {
        let hay = latin1(b"abracadabra");
        let bra = latin1(b"bra");
        let a = latin1(b"a");
        let cad = latin1(b"cad");
        assert_agree(&hay, &bra, 0, hay.len() as i64);
        assert_agree(&hay, &a, 0, hay.len() as i64);
        assert_agree(&hay, &cad, 1, 6);
        let grin = 0x1F600u32;
        let hay_u = [b'a' as u32, grin, b'b' as u32, grin, b'c' as u32];
        let needle_u = [grin];
        assert_agree(&hay_u, &needle_u, 0, hay_u.len() as i64);
        assert_eq!(
            _search_normal(&hay_u, &needle_u, 0, hay_u.len() as i64, SEARCH_FIND),
            1
        );
        assert_eq!(
            _search_normal(&hay_u, &needle_u, 0, hay_u.len() as i64, SEARCH_RFIND),
            3
        );
        assert_eq!(
            _search_normal(&hay_u, &needle_u, 0, hay_u.len() as i64, SEARCH_COUNT),
            2
        );
        let two = [grin, b'b' as u32];
        assert_eq!(
            _search_normal(&hay_u, &two, 0, hay_u.len() as i64, SEARCH_FIND),
            1
        );
    }

    #[test]
    fn search_normal_unicode_repeated_prefix() {
        let hay = latin1(b"aaaabaaa");
        let aaa = latin1(b"aaa");
        assert_agree(&hay, &aaa, 0, hay.len() as i64);
        assert_eq!(
            _search_normal(&hay, &aaa, 0, hay.len() as i64, SEARCH_FIND),
            0
        );
        assert_eq!(
            _search_normal(&hay, &aaa, 0, hay.len() as i64, SEARCH_RFIND),
            5
        );
        assert_eq!(
            _search_normal(&hay, &aaa, 0, hay.len() as i64, SEARCH_COUNT),
            2
        );
        let grin = 0x1F600u32;
        let hay_u = [grin, grin, grin, b'x' as u32, grin, grin, grin];
        let needle_u = [grin, grin, grin];
        assert_agree(&hay_u, &needle_u, 0, hay_u.len() as i64);
        assert_eq!(
            _search_normal(&hay_u, &needle_u, 0, hay_u.len() as i64, SEARCH_RFIND),
            4
        );
    }

    #[test]
    fn search_normal_unicode_reads_unicode_layout() {
        let hay = unicode_buf(&[b'a' as u32, 0x1F600, b'b' as u32, 0x1F600]);
        let needle = unicode_buf(&[0x1F600]);
        assert_eq!(
            _search_normal_unicode(hay.as_ptr(), needle.as_ptr(), 0, 4, SEARCH_FIND),
            1
        );
        assert_eq!(
            _search_normal_unicode(hay.as_ptr(), needle.as_ptr(), 0, 4, SEARCH_RFIND),
            3
        );
        assert_eq!(
            _search_normal_unicode(hay.as_ptr(), needle.as_ptr(), 0, 4, SEARCH_COUNT),
            2
        );
    }

    #[test]
    fn search_normal_unicode_is_published_on_helper_fnaddrs() {
        let mut found = false;
        majit_ir::helper_fnaddr::for_each_helper_fnaddr(|row| {
            if row.path.ends_with("rstring::_search_normal") {
                found = true;
                assert_ne!(row.get(), 0);
                assert_eq!(row.arity, 5);
            }
        });
        assert!(found, "HELPER_FNADDRS must publish _search_normal");
    }

    #[test]
    fn search_elidable_is_published_on_helper_fnaddrs() {
        let mut found = false;
        majit_ir::helper_fnaddr::for_each_helper_fnaddr(|row| {
            if row.path.ends_with("rstring::_search_elidable") {
                found = true;
                assert_ne!(row.get(), 0);
                assert_eq!(row.arity, 5);
            }
        });
        assert!(found, "HELPER_FNADDRS must publish _search_elidable");
    }
}
