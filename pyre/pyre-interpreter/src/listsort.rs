//! `rpython/rlib/listsort.py` — the adaptive, stable, natural mergesort
//! (TimSort with the powersort merge policy) behind `list.sort` and `sorted`.
//!
//! Two properties of the upstream algorithm are the reason for the port:
//!
//! * `lt` is THE comparison primitive (`listsort.py`: `le` is
//!   `not self.lt(b, a)`), so a user `__lt__` runs **exactly once** per
//!   comparison.  A three-way `Ordering` comparator cannot express that — it
//!   needs both directions — so the sort driver itself has to be the one that
//!   only ever asks "is a < b".
//! * The merge steps hold elements in a scratch copy and reinstate them in a
//!   `finally` (`TimSort.merge_lo` / `merge_hi`), so a comparison that raises
//!   still leaves a permutation of the input behind rather than duplicates.
//!
//! `make_timsort_class`'s `getitem`/`setitem`/`length` become [`SortStorage`];
//! `lt` becomes [`SortLt`].  `IntegerListStrategy.sort` / `IntSort` and
//! `FloatListStrategy.sort` / `FloatSort` pass the unerased storage list so
//! every `self.list[i] = x` is a `setarrayitem` on the GC array.  Object and
//! key sorts still permute indices into a caller-owned rooted area
//! (`descr_sort`'s snapshot), so a moving collection during a comparison
//! cannot invalidate anything the sorter holds.

use crate::PyError;

/// `listsort.py` — `lt` is the single comparison primitive and `le` is
/// derived from it.  Implementors must not override `le`.
///
/// `T` is what `make_timsort_class`'s `lt` receives: the list's element type.
/// For an object list that is an index into a rooted area; for the unwrapped
/// strategies (`listobject.py` `IntegerListStrategy.sort`,
/// `FloatListStrategy.sort`) it is the scalar itself.
pub(crate) trait SortLt<T> {
    fn lt(&mut self, a: T, b: T) -> Result<bool, PyError>;

    #[inline]
    fn le(&mut self, a: T, b: T) -> Result<bool, PyError> {
        Ok(!self.lt(b, a)?)
    }
}

/// `make_timsort_class`'s `getitem` / `setitem` / `length`.
///
/// `TimSort.list` is the list being sorted.  Integer and float strategy sorts
/// implement this with `ll_list_int_getitem_fast` / `ll_list_int_setitem_fast`
/// (and the float twins) so `IntSort` / `FloatSort` write the `GcArray` in
/// place.  `scratch_list` stays a fresh `Vec`, matching upstream's fresh
/// RPython list.
pub(crate) trait SortStorage<T: Copy> {
    fn len(&self) -> usize;
    fn getitem(&self, index: usize) -> T;
    fn setitem(&mut self, index: usize, value: T);
}

impl<T: Copy> SortStorage<T> for [T] {
    #[inline(always)]
    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    #[inline(always)]
    fn getitem(&self, index: usize) -> T {
        self[index]
    }

    #[inline(always)]
    fn setitem(&mut self, index: usize, value: T) {
        self[index] = value;
    }
}

/// `ListSlice.reverse` over the whole list — `IntegerListStrategy.sort`'s
/// `l.reverse()` / `FloatListStrategy.sort`'s `l.reverse()`.
pub(crate) fn reverse_storage<T: Copy, S: SortStorage<T> + ?Sized>(list: &mut S) {
    let n = list.len();
    if n < 2 {
        return;
    }
    let mut lo = 0;
    let mut hi = n - 1;
    while lo < hi {
        let a = list.getitem(lo);
        let b = list.getitem(hi);
        list.setitem(lo, b);
        list.setitem(hi, a);
        lo += 1;
        hi -= 1;
    }
}

/// `listsort.py` `merge_compute_minrun`.
fn merge_compute_minrun(mut n: usize) -> usize {
    // Becomes 1 if any 1 bits are shifted off.
    let mut r = 0;
    while n >= 64 {
        r |= n & 1;
        n >>= 1;
    }
    n + r
}

/// `listsort.py` `powerloop` — the "power" (depth in the conceptual binary
/// merge tree) of the run of length `n1` starting at `s1`, followed by a run of
/// length `n2`, in a list of length `n`.
fn powerloop(s1: usize, n1: usize, n2: usize, n: usize) -> usize {
    // a = 2 * (s1 + n1/2), b = 2 * (s1 + n1 + n2/2), kept doubled so both
    // midpoints stay integral.
    let mut a = 2 * s1 + n1;
    let mut b = a + n1 + n2;
    let mut result = 0;
    loop {
        result += 1;
        if a >= n {
            a -= n;
            b -= n;
        } else if b >= n {
            break;
        }
        a <<= 1;
        b <<= 1;
    }
    result
}

/// `listsort.py` `ListSlice` — a sublist of the array being sorted.
///
/// `ListSlice.list` is either the array itself or, after `copyitems`, the
/// sorter's scratch copy; `scratch` is that discriminator.
#[derive(Clone, Copy)]
struct ListSlice {
    scratch: bool,
    base: usize,
    len: usize,
    power: usize,
}

impl ListSlice {
    fn of_list(base: usize, len: usize) -> Self {
        Self {
            scratch: false,
            base,
            len,
            power: 0,
        }
    }
}

/// `listsort.py` `MIN_GALLOP` — galloping mode is entered after this many
/// consecutive wins by the same run.
const MIN_GALLOP: usize = 7;

struct TimSort<'a, T: Copy, S: SortStorage<T> + ?Sized, L: SortLt<T>> {
    list: &'a mut S,
    listlength: usize,
    scratch_list: Vec<T>,
    min_gallop: usize,
    pending: Vec<ListSlice>,
    cmp: &'a mut L,
}

impl<T: Copy, S: SortStorage<T> + ?Sized, L: SortLt<T>> TimSort<'_, T, S, L> {
    #[inline(always)]
    fn slice_get(&self, slice: &ListSlice, index: usize) -> T {
        if slice.scratch {
            self.scratch_list[index]
        } else {
            self.list.getitem(index)
        }
    }

    /// `listsort.py` — `gallop`'s `lower`: `le` searches for the
    /// largest `k` with `a[k] <= key`, `lt` for the largest with `a[k] < key`.
    fn lower(&mut self, a: T, b: T, rightmost: bool) -> Result<bool, PyError> {
        if rightmost {
            self.cmp.le(a, b)
        } else {
            self.cmp.lt(a, b)
        }
    }

    /// `listsort.py` `ListSlice.reverse`.
    fn reverse_slice(&mut self, slice: &ListSlice) {
        debug_assert!(!slice.scratch);
        if slice.len == 0 {
            return;
        }
        let mut lo = slice.base;
        let mut hi = lo + slice.len - 1;
        while lo < hi {
            let a = self.list.getitem(lo);
            let b = self.list.getitem(hi);
            self.list.setitem(lo, b);
            self.list.setitem(hi, a);
            lo += 1;
            hi -= 1;
        }
    }

    /// `listsort.py` `ListSlice.popleft`.
    fn popleft(&mut self, slice: &mut ListSlice) -> T {
        let result = self.slice_get(slice, slice.base);
        slice.base += 1;
        slice.len -= 1;
        result
    }

    /// `listsort.py` `ListSlice.popright`.
    fn popright(&mut self, slice: &mut ListSlice) -> T {
        slice.len -= 1;
        self.slice_get(slice, slice.base + slice.len)
    }

    /// `listsort.py` `ListSlice.copyitems` — move the slice into the
    /// sorter's scratch store, growing it when the run does not fit.
    fn copyitems(&mut self, slice: &mut ListSlice) {
        debug_assert!(!slice.scratch);
        if slice.len > self.scratch_list.len() {
            let listlength = self.listlength;
            let mut scratchsize = std::cmp::min(listlength.div_ceil(2), 256);
            if slice.len > scratchsize {
                scratchsize = slice.len;
            }
            let start = slice.base;
            let stop = std::cmp::min(slice.base + scratchsize, listlength);
            self.scratch_list.clear();
            self.scratch_list.reserve(stop - start);
            for index in start..stop {
                self.scratch_list.push(self.list.getitem(index));
            }
        } else {
            let base = slice.base;
            for index in 0..slice.len {
                self.scratch_list[index] = self.list.getitem(base + index);
            }
        }
        slice.scratch = true;
        slice.base = 0;
    }

    /// `listsort.py` `binarysort` — binary insertion sort of the slice,
    /// whose first `sorted` elements are already in order.  Stable.
    fn binarysort(&mut self, a: &ListSlice, sorted: usize) -> Result<(), PyError> {
        debug_assert!(!a.scratch);
        let abase = a.base;
        for start in (a.base + sorted)..(a.base + a.len) {
            let mut l = abase;
            let mut r = start;
            let pivot = self.list.getitem(r);
            // pivot >= all in [base, l) and pivot < all in [r, start).
            while l < r {
                let p = l + ((r - l) >> 1);
                let candidate = self.list.getitem(p);
                if self.cmp.lt(pivot, candidate)? {
                    r = p;
                } else {
                    l = p + 1;
                }
            }
            // Elements equal to the pivot leave `l` past them, which is what
            // makes the insertion stable.  Slide over to make room.
            let mut p = start;
            while p > l {
                let prev = self.list.getitem(p - 1);
                self.list.setitem(p, prev);
                p -= 1;
            }
            self.list.setitem(l, pivot);
        }
        Ok(())
    }

    /// `listsort.py` `count_run` — length of the longest ascending
    /// (`a[0] <= a[1] <= …`) or strictly descending (`a[0] > a[1] > …`) prefix
    /// of `a`, written to `run.len`.  Returns whether it was descending; the
    /// strictness of "descending" is what lets the caller reverse it without
    /// breaking stability.
    fn count_run(&mut self, a: &ListSlice, run: &mut ListSlice) -> Result<bool, PyError> {
        let descending;
        let mut n;
        if a.len <= 1 {
            n = a.len;
            descending = false;
        } else {
            n = 2;
            let first = self.slice_get(a, a.base + 1);
            let zeroth = self.slice_get(a, a.base);
            if self.cmp.lt(first, zeroth)? {
                descending = true;
                for p in (a.base + 2)..(a.base + a.len) {
                    let this = self.slice_get(a, p);
                    let prev = self.slice_get(a, p - 1);
                    if self.cmp.lt(this, prev)? {
                        n += 1;
                    } else {
                        break;
                    }
                }
            } else {
                descending = false;
                for p in (a.base + 2)..(a.base + a.len) {
                    let this = self.slice_get(a, p);
                    let prev = self.slice_get(a, p - 1);
                    if self.cmp.lt(this, prev)? {
                        break;
                    }
                    n += 1;
                }
            }
        }
        run.len = n;
        Ok(descending)
    }

    /// `listsort.py` `gallop` — locate where `key` belongs in the sorted
    /// slice `a`, starting the search at `hint`.  With `rightmost`, return the
    /// index right of the rightmost equal element, otherwise left of the
    /// leftmost one.
    ///
    /// The offsets are signed because the left-gallop's `hint - ofs` reaches
    /// `-1` before the closing binary search bumps it back to `0`
    /// (`listsort.py` asserts `-1 <= lastofs`).
    fn gallop(
        &mut self,
        key: T,
        a: &ListSlice,
        hint: usize,
        rightmost: bool,
    ) -> Result<usize, PyError> {
        debug_assert!(hint < a.len);
        let p = a.base + hint;
        let hint = hint as isize;
        let mut lastofs: isize = 0;
        let mut ofs: isize = 1;
        let at_hint = self.slice_get(a, p);
        if self.lower(at_hint, key, rightmost)? {
            // a[hint] < key — gallop right until a[hint+lastofs] < key <= a[hint+ofs].
            let maxofs = a.len as isize - hint; // a[a.len-1] is highest
            while ofs < maxofs {
                let value = self.slice_get(a, p + ofs as usize);
                if self.lower(value, key, rightmost)? {
                    lastofs = ofs;
                    ofs = ofs
                        .checked_mul(2)
                        .and_then(|doubled| doubled.checked_add(1))
                        .unwrap_or(maxofs);
                } else {
                    break;
                }
            }
            if ofs > maxofs {
                ofs = maxofs;
            }
            lastofs += hint;
            ofs += hint;
        } else {
            // key <= a[hint] — gallop left until a[hint-ofs] < key <= a[hint-lastofs].
            let maxofs = hint + 1; // a[0] is lowest
            while ofs < maxofs {
                let value = self.slice_get(a, p - ofs as usize);
                if self.lower(value, key, rightmost)? {
                    break;
                }
                lastofs = ofs;
                ofs = ofs
                    .checked_mul(2)
                    .and_then(|doubled| doubled.checked_add(1))
                    .unwrap_or(maxofs);
            }
            if ofs > maxofs {
                ofs = maxofs;
            }
            (lastofs, ofs) = (hint - ofs, hint - lastofs);
        }

        // a[lastofs] < key <= a[ofs]; binary-search the gap with the invariant
        // a[lastofs-1] < key <= a[ofs].
        lastofs += 1;
        while lastofs < ofs {
            let m = lastofs + ((ofs - lastofs) >> 1);
            let value = self.slice_get(a, a.base + m as usize);
            if self.lower(value, key, rightmost)? {
                lastofs = m + 1;
            } else {
                ofs = m;
            }
        }
        Ok(ofs as usize)
    }

    /// `listsort.py` `merge_lo` — stable in-place merge of adjacent slices
    /// with `a.len <= b.len`, holding `a` in the scratch store.
    fn merge_lo(&mut self, a: &mut ListSlice, b: &mut ListSlice) -> Result<(), PyError> {
        debug_assert!(a.len > 0 && b.len > 0 && a.base + a.len == b.base);
        let mut dest = a.base;
        self.copyitems(a);
        let result = self.merge_lo_inner(a, b, &mut dest);
        // `TimSort.merge_lo` finally: the last element of `a` belongs at the end
        // of the merge, so what remains of `b` is reinstated before what
        // remains of `a` — on the error path too, keeping the array a
        // permutation of its input.
        for p in b.base..(b.base + b.len) {
            let value = self.slice_get(b, p);
            self.list.setitem(dest, value);
            dest += 1;
        }
        for p in a.base..(a.base + a.len) {
            let value = self.slice_get(a, p);
            self.list.setitem(dest, value);
            dest += 1;
        }
        result
    }

    fn merge_lo_inner(
        &mut self,
        a: &mut ListSlice,
        b: &mut ListSlice,
        dest: &mut usize,
    ) -> Result<(), PyError> {
        let mut min_gallop = self.min_gallop;
        let value = self.popleft(b);
        self.list.setitem(*dest, value);
        *dest += 1;
        if a.len == 1 || b.len == 0 {
            return Ok(());
        }
        loop {
            let mut acount = 0; // number of times A won in a row
            let mut bcount = 0; // number of times B won in a row

            // Do the straightforward thing until (if ever) one run appears to
            // win consistently.
            loop {
                let bhead = self.slice_get(b, b.base);
                let ahead = self.slice_get(a, a.base);
                if self.cmp.lt(bhead, ahead)? {
                    let value = self.popleft(b);
                    self.list.setitem(*dest, value);
                    *dest += 1;
                    if b.len == 0 {
                        return Ok(());
                    }
                    bcount += 1;
                    acount = 0;
                    if bcount >= min_gallop {
                        break;
                    }
                } else {
                    let value = self.popleft(a);
                    self.list.setitem(*dest, value);
                    *dest += 1;
                    if a.len == 1 {
                        return Ok(());
                    }
                    acount += 1;
                    bcount = 0;
                    if acount >= min_gallop {
                        break;
                    }
                }
            }

            // One run is winning so consistently that galloping may be a huge
            // win.  Stay there until neither run wins consistently anymore.
            min_gallop += 1;
            loop {
                if min_gallop > 1 {
                    min_gallop -= 1;
                }
                self.min_gallop = min_gallop;

                let bhead = self.slice_get(b, b.base);
                acount = self.gallop(bhead, a, 0, true)?;
                for p in a.base..(a.base + acount) {
                    let value = self.slice_get(a, p);
                    self.list.setitem(*dest, value);
                    *dest += 1;
                }
                a.base += acount;
                a.len -= acount;
                // a.len == 0 is impossible here for a consistent comparison
                // function, which cannot be assumed.
                if a.len <= 1 {
                    return Ok(());
                }

                let value = self.popleft(b);
                self.list.setitem(*dest, value);
                *dest += 1;
                if b.len == 0 {
                    return Ok(());
                }

                let ahead = self.slice_get(a, a.base);
                bcount = self.gallop(ahead, b, 0, false)?;
                for p in b.base..(b.base + bcount) {
                    let value = self.slice_get(b, p);
                    self.list.setitem(*dest, value);
                    *dest += 1;
                }
                b.base += bcount;
                b.len -= bcount;
                if b.len == 0 {
                    return Ok(());
                }

                let value = self.popleft(a);
                self.list.setitem(*dest, value);
                *dest += 1;
                if a.len == 1 {
                    return Ok(());
                }

                if acount < MIN_GALLOP && bcount < MIN_GALLOP {
                    break;
                }
            }
            min_gallop += 1; // penalize it for leaving galloping mode
            self.min_gallop = min_gallop;
        }
    }

    /// `listsort.py` `merge_hi` — the `a.len >= b.len` mirror of
    /// [`Self::merge_lo`], holding `b` in the scratch store and filling from
    /// the right.
    fn merge_hi(&mut self, a: &mut ListSlice, b: &mut ListSlice) -> Result<(), PyError> {
        debug_assert!(a.len > 0 && b.len > 0 && a.base + a.len == b.base);
        let mut dest = b.base + b.len;
        self.copyitems(b);
        let result = self.merge_hi_inner(a, b, &mut dest);
        // `TimSort.merge_hi` finally, mirrored: what remains of `a` first.
        for p in (a.base..(a.base + a.len)).rev() {
            let value = self.slice_get(a, p);
            dest -= 1;
            self.list.setitem(dest, value);
        }
        for p in (b.base..(b.base + b.len)).rev() {
            let value = self.slice_get(b, p);
            dest -= 1;
            self.list.setitem(dest, value);
        }
        result
    }

    fn merge_hi_inner(
        &mut self,
        a: &mut ListSlice,
        b: &mut ListSlice,
        dest: &mut usize,
    ) -> Result<(), PyError> {
        let mut min_gallop = self.min_gallop;
        *dest -= 1;
        let value = self.popright(a);
        self.list.setitem(*dest, value);
        if a.len == 0 || b.len == 1 {
            return Ok(());
        }
        loop {
            let mut acount = 0;
            let mut bcount = 0;
            loop {
                let nexta = self.slice_get(a, a.base + a.len - 1);
                let nextb = self.slice_get(b, b.base + b.len - 1);
                if self.cmp.lt(nextb, nexta)? {
                    *dest -= 1;
                    self.list.setitem(*dest, nexta);
                    a.len -= 1;
                    if a.len == 0 {
                        return Ok(());
                    }
                    acount += 1;
                    bcount = 0;
                    if acount >= min_gallop {
                        break;
                    }
                } else {
                    *dest -= 1;
                    self.list.setitem(*dest, nextb);
                    b.len -= 1;
                    if b.len == 1 {
                        return Ok(());
                    }
                    bcount += 1;
                    acount = 0;
                    if bcount >= min_gallop {
                        break;
                    }
                }
            }

            min_gallop += 1;
            loop {
                if min_gallop > 1 {
                    min_gallop -= 1;
                }
                self.min_gallop = min_gallop;

                let nextb = self.slice_get(b, b.base + b.len - 1);
                let k = self.gallop(nextb, a, a.len - 1, true)?;
                acount = a.len - k;
                for p in ((a.base + k)..(a.base + a.len)).rev() {
                    let value = self.slice_get(a, p);
                    *dest -= 1;
                    self.list.setitem(*dest, value);
                }
                a.len -= acount;
                if a.len == 0 {
                    return Ok(());
                }

                *dest -= 1;
                let value = self.popright(b);
                self.list.setitem(*dest, value);
                if b.len == 1 {
                    return Ok(());
                }

                let nexta = self.slice_get(a, a.base + a.len - 1);
                let k = self.gallop(nexta, b, b.len - 1, false)?;
                bcount = b.len - k;
                for p in ((b.base + k)..(b.base + b.len)).rev() {
                    let value = self.slice_get(b, p);
                    *dest -= 1;
                    self.list.setitem(*dest, value);
                }
                b.len -= bcount;
                // b.len == 0 is impossible here for a consistent comparison
                // function, which cannot be assumed.
                if b.len <= 1 {
                    return Ok(());
                }

                *dest -= 1;
                let value = self.popright(a);
                self.list.setitem(*dest, value);
                if a.len == 0 {
                    return Ok(());
                }

                if acount < MIN_GALLOP && bcount < MIN_GALLOP {
                    break;
                }
            }
            min_gallop += 1;
            self.min_gallop = min_gallop;
        }
    }

    /// `listsort.py` `merge_at` — merge the two runs at pending indices
    /// `i` and `i+1`.
    fn merge_at(&mut self, i: usize) -> Result<(), PyError> {
        let mut a = self.pending[i];
        let mut b = self.pending[i + 1];
        debug_assert!(a.len > 0 && b.len > 0 && a.base + a.len == b.base);

        // Record the length of the combined runs and remove run b.
        self.pending[i] = ListSlice::of_list(a.base, a.len + b.len);
        self.pending.remove(i + 1);

        // Where does b start in a?  Elements of a before that are in place.
        let bhead = self.slice_get(&b, b.base);
        let k = self.gallop(bhead, &a, 0, true)?;
        a.base += k;
        a.len -= k;
        if a.len == 0 {
            return Ok(());
        }

        // Where does a end in b?  Elements of b after that are in place.
        let atail = self.slice_get(&a, a.base + a.len - 1);
        b.len = self.gallop(atail, &b, b.len - 1, false)?;
        if b.len == 0 {
            return Ok(());
        }

        // Direction chosen to minimize the scratch storage needed.
        if a.len <= b.len {
            self.merge_lo(&mut a, &mut b)
        } else {
            self.merge_hi(&mut a, &mut b)
        }
    }

    /// `listsort.py` `found_new_run` — the powersort merge policy: merge
    /// the pending runs whose power exceeds the newly identified run's.
    fn found_new_run(&mut self, run: &ListSlice) -> Result<(), PyError> {
        let Some(last) = self.pending.last() else {
            return Ok(());
        };
        let power = powerloop(last.base, last.len, run.len, self.listlength);
        while self.pending.len() > 1 && self.pending[self.pending.len() - 2].power > power {
            self.merge_at(self.pending.len() - 2)?;
        }
        let top = self.pending.len() - 1;
        self.pending[top].power = power;
        Ok(())
    }

    /// `listsort.py` `merge_force_collapse`.
    fn merge_force_collapse(&mut self) -> Result<(), PyError> {
        while self.pending.len() > 1 {
            let n = self.pending.len();
            let at = if n >= 3 && self.pending[n - 3].len < self.pending[n - 1].len {
                n - 3
            } else {
                n - 2
            };
            self.merge_at(at)?;
        }
        Ok(())
    }

    /// `listsort.py` `sort` — march over the array once left to right
    /// finding natural runs, extending short ones to `minrun`, merging as the
    /// powersort policy dictates.
    fn sort(&mut self) -> Result<(), PyError> {
        if self.listlength < 2 {
            return Ok(());
        }
        let mut remaining = ListSlice::of_list(0, self.listlength);
        self.min_gallop = MIN_GALLOP;
        self.pending.clear();
        let minrun = merge_compute_minrun(remaining.len);

        while remaining.len > 0 {
            let mut run = ListSlice::of_list(remaining.base, remaining.len);
            if self.count_run(&remaining, &mut run)? {
                self.reverse_slice(&run);
            }
            if run.len < minrun {
                let sorted = run.len;
                run.len = std::cmp::min(minrun, remaining.len);
                self.binarysort(&run, sorted)?;
            }
            // Maybe merge, but never the newest run.
            self.found_new_run(&run)?;
            self.pending.push(run);
            remaining.base += run.len;
            remaining.len -= run.len;
        }
        debug_assert_eq!(remaining.base, self.listlength);

        self.merge_force_collapse()
    }
}

/// Sort `list` in place with `cmp` as the only comparison primitive.
///
/// On a raising comparison the error propagates and `list` is left holding
/// some permutation of its input (`listsort.py`'s merge `finally` blocks).
pub(crate) fn sort_storage<T, S, L>(list: &mut S, cmp: &mut L) -> Result<(), PyError>
where
    T: Copy,
    S: SortStorage<T> + ?Sized,
    L: SortLt<T>,
{
    let listlength = list.len();
    let mut sorter = TimSort {
        list,
        listlength,
        scratch_list: Vec::new(),
        min_gallop: MIN_GALLOP,
        pending: Vec::new(),
        cmp,
    };
    sorter.sort()
}

/// Sort a contiguous slice.  Object/key `descr_sort` permutes rooted indices
/// this way; integer and float strategy sorts use [`sort_storage`] over the
/// unerased GC storage instead.
pub(crate) fn sort_with<T: Copy, L: SortLt<T>>(list: &mut [T], cmp: &mut L) -> Result<(), PyError> {
    sort_storage(list, cmp)
}
