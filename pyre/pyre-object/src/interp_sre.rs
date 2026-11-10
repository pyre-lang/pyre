//! `pypy/module/_sre/interp_sre.py W_SRE_Pattern` /
//! `:675 W_SRE_Match` — typed layouts for compiled patterns and match
//! results.  Engine state lives in interp-level fields, not in a user
//! attribute store.

use crate::pyobject::*;
use pyre_macros::pyre_class;

/// Compiled regular expression object (interp_sre.py W_SRE_Pattern).
///
/// `code`/`code_len` stand in for `srepat.code =
/// rsre_core.CompiledPattern(code, flags)` (interp_sre.py:635): pyre
/// runs the sre-engine crate's u32 opcode buffer, leaked once at
/// compile time and immutable for the pattern's lifetime
/// (`_immutable_fields_ = ["code", ...]`, interp_sre.py:148).
#[pyre_class("re.Pattern", static_name = "SRE_PATTERN", weakrefable)]
pub struct W_SRE_Pattern {
    /// interp_sre.py `srepat.w_pattern` — original uncompiled pattern.
    pub w_pattern: PyObjectRef,
    /// interp_sre.py `srepat.flags`.
    pub flags: i64,
    /// interp_sre.py `srepat.code` (see type doc).
    pub code: *const u32,
    pub code_len: usize,
    /// interp_sre.py SRE_Pattern__new__ `srepat.num_groups`.
    pub num_groups: i64,
    /// interp_sre.py:638 `srepat.w_groupindex`.
    pub w_groupindex: PyObjectRef,
    /// interp_sre.py:639 `srepat.w_indexgroup`.
    pub w_indexgroup: PyObjectRef,
}

/// Allocate a `W_SRE_Pattern` — `SRE_Pattern__new__` field stamping
/// (interp_sre.py).
pub fn w_sre_pattern_new(
    w_pattern: PyObjectRef,
    flags: i64,
    code: &'static [u32],
    num_groups: i64,
    w_groupindex: PyObjectRef,
    w_indexgroup: PyObjectRef,
) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`).
    let _roots = crate::gc_roots::push_roots();
    let w_pattern = crate::gc_roots::pin_root(w_pattern);
    let w_groupindex = crate::gc_roots::pin_root(w_groupindex);
    let w_indexgroup = crate::gc_roots::pin_root(w_indexgroup);
    // interp_sre.py `SRE_Pattern__new__` leaves ownership to the app-level
    // cache and ordinary references. The generated class descriptor traces
    // this managed object's fields; an address-only global list would outlive
    // a collected pattern and later trace reclaimed storage as its fields.
    W_SRE_Pattern::allocate(W_SRE_Pattern {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_pattern,
        flags,
        code: code.as_ptr(),
        code_len: code.len(),
        num_groups,
        w_groupindex,
        w_indexgroup,
        lifeline: PY_NULL,
    })
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_sre_pattern(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &SRE_PATTERN_TYPE) }
}

/// Match result object (interp_sre.py:675).
///
/// `W_SRE_Match` keeps `self.ctx` and flattens marks lazily
/// (`flatten_marks`). The engine surfaces marks eagerly, so the span
/// table (group 0 = whole match, `(-1, -1)` = unmatched) is flattened at
/// construction into `flatten_cache`, the `GcArray(Signed)` of unboxed
/// RPython-level integers `do_flatten_marks` returns.
#[pyre_class("re.Match", static_name = "SRE_MATCH")]
pub struct W_SRE_Match {
    /// `self.srepat`.
    pub w_srepat: PyObjectRef,
    /// `self.w_string`.
    pub w_string: PyObjectRef,
    /// The buffer captured at match time for slicing — `self.ctx._buffer`
    /// (`BufMatchContext`). The match holds that buffer so group slices
    /// never re-read the original object. `PY_NULL` when `w_string` is
    /// itself the subject (a `str`/`bytes`/`bytearray`).
    pub w_buffer: PyObjectRef,
    /// `ctx.original_pos` (`fget_pos`).
    pub pos: i64,
    /// `ctx.end` (`fget_endpos`).
    pub endpos: i64,
    /// `_last_index()`; `-1` plays None.
    pub lastindex: i64,
    /// `self.flatten_cache`: a `GcArray(Signed)` (`TypedItemsBlock`) of
    /// flat `(start, end)` pairs, two words per group, group 0 first.
    pub flatten_cache: PyObjectRef,
    pub spans_len: usize,
}

/// Allocate a `W_SRE_Match` — `W_SRE_Match.__init__` field stamping
/// (interp_sre.py) plus the eager span flattening described on
/// the type.
pub fn w_sre_match_new(
    w_srepat: PyObjectRef,
    w_string: PyObjectRef,
    w_buffer: PyObjectRef,
    pos: i64,
    endpos: i64,
    lastindex: i64,
    spans: &[(i64, i64)],
) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`).
    let _roots = crate::gc_roots::push_roots();
    let base = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_srepat);
    let _ = crate::gc_roots::pin_root(w_string);
    let _ = crate::gc_roots::pin_root(w_buffer);
    // `do_flatten_marks`: `[0] * (num_groups * 2)` of unboxed integers, one
    // word per span edge. The block holds no GC pointer, so nothing is rooted
    // across filling it.
    // W_SRE_Match.__init__ / do_flatten_marks: a refused span table is not filled.
    let flatten_cache = unsafe {
        crate::object_array::alloc_typed_items_block_nursery(
            spans.len() * 2,
            crate::object_array::gc_int_array_gc_type_id(),
        )
    };
    if crate::gc_hook::gc_alloc_failed_pending() && flatten_cache.is_null() {
        return PY_NULL;
    }
    unsafe {
        let items = crate::object_array::typed_items_block_items_base(flatten_cache) as *mut i64;
        for (group, &(start, end)) in spans.iter().enumerate() {
            items.add(group * 2).write(start);
            items.add(group * 2 + 1).write(end);
        }
    }
    let flatten_cache_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(flatten_cache as PyObjectRef);
    W_SRE_Match::allocate_stable(W_SRE_Match {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_srepat: crate::gc_roots::shadow_stack_get(base),
        w_string: crate::gc_roots::shadow_stack_get(base + 1),
        w_buffer: crate::gc_roots::shadow_stack_get(base + 2),
        pos,
        endpos,
        lastindex,
        flatten_cache: crate::gc_roots::shadow_stack_get(flatten_cache_slot),
        spans_len: spans.len(),
    })
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_sre_match(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &SRE_MATCH_TYPE) }
}

/// The flattened span for group `groupnum` (0 = whole match), or
/// `None` past the table — `do_span`'s table read (interp_sre.py).
///
/// # Safety
/// `obj` must point to a valid `W_SRE_Match`.
#[inline]
pub unsafe fn w_sre_match_get_span(obj: PyObjectRef, groupnum: usize) -> Option<(i64, i64)> {
    let m = obj as *const W_SRE_Match;
    unsafe {
        if groupnum >= (*m).spans_len {
            return None;
        }
        let items = crate::object_array::typed_items_block_items_base(
            (*m).flatten_cache as *mut crate::object_array::TypedItemsBlock,
        ) as *const i64;
        Some((*items.add(groupnum * 2), *items.add(groupnum * 2 + 1)))
    }
}

/// `_sre.SRE_Scanner` (interp_sre.py) — the stateful iterator behind
/// `Pattern.finditer` (and the undocumented `scanner()`), yielding a
/// `W_SRE_Match` per non-overlapping match.
///
/// Upstream keeps the live `rsre_core` context (`self.ctx`); pyre's
/// sre-engine context borrows the subject string and code, so it cannot
/// be parked in a GC object.  Instead the resumable cursor is reduced to
/// the character position `pos` and the `must_advance` flag — exactly
/// the two fields `SearchIter` threads across calls
/// — and a fresh `Request`/`State` is rebuilt from the pattern + subject
/// on each step. The subject is re-read from the GC string; the pattern
/// code is immutable for the pattern's lifetime.
/// `pos == -1` plays upstream's `self.ctx is None` exhausted state.
#[pyre_class("_sre.SRE_Scanner", static_name = "SRE_SCANNER")]
pub struct W_SRE_Scanner {
    /// interp_sre.py:907 `self.srepat`.
    pub w_srepat: PyObjectRef,
    /// interp_sre.py:910 `self.w_string`.
    pub w_string: PyObjectRef,
    /// The buffer captured at scanner creation — `self.ctx._buffer`.  Threaded
    /// into each produced `W_SRE_Match` so the matches slice the same validated
    /// buffer; `PY_NULL` for a `str`/`bytes`/`bytearray` subject.
    pub w_buffer: PyObjectRef,
    /// Original search position (`ctx.original_pos`) exposed by each match.
    pub original_pos: i64,
    /// Character position of the next search (`ctx.match_start`); `-1` once
    /// the iterator is exhausted (`self.ctx is None`).
    pub pos: i64,
    /// Character end position (`ctx.end`) — the `endpos` argument of finditer.
    pub endpos: i64,
    /// `req.must_advance` (engine.rs) — set after a zero-width match so
    /// the next search refuses to re-match at the same position.
    pub must_advance: i64,
    /// Whether this lazy scanner owns one buffer export on `w_string`.
    pub export_active: bool,
}

/// Allocate a `W_SRE_Scanner` — `W_SRE_Scanner.__init__` (interp_sre.py).
pub fn w_sre_scanner_new(
    w_srepat: PyObjectRef,
    w_string: PyObjectRef,
    w_buffer: PyObjectRef,
    pos: i64,
    endpos: i64,
    export_active: bool,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let w_srepat = crate::gc_roots::pin_root(w_srepat);
    let w_string = crate::gc_roots::pin_root(w_string);
    let w_buffer = crate::gc_roots::pin_root(w_buffer);
    W_SRE_Scanner::allocate_stable(W_SRE_Scanner {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_srepat,
        w_string,
        w_buffer,
        original_pos: pos,
        pos,
        endpos,
        must_advance: 0,
        export_active,
    })
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_sre_scanner(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &SRE_SCANNER_TYPE) }
}

/// `_sre.SRE_Template` (`sre.h TemplateObject`). `literal` is the first
/// chunk; `indexes` and `literals` are parallel tuples, one pair per group
/// reference. A `None` literal is the empty literal `expand_template` skips.
#[pyre_class("_sre.SRE_Template", static_name = "SRE_TEMPLATE")]
pub struct W_SRE_Template {
    pub literal: PyObjectRef,
    /// `TemplateObject.chunks`.
    pub chunks: i64,
    /// Group indexes, each a non-negative int.
    pub indexes: PyObjectRef,
    /// Literal after each group, or `None` when that literal was empty.
    pub literals: PyObjectRef,
}

/// Allocate a `W_SRE_Template`.
pub fn w_sre_template_new(
    literal: PyObjectRef,
    chunks: i64,
    indexes: PyObjectRef,
    literals: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let base = crate::gc_roots::pin_roots(&[literal, indexes, literals]);
    W_SRE_Template::allocate(W_SRE_Template {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        literal: crate::gc_roots::shadow_stack_get(base),
        chunks,
        indexes: crate::gc_roots::shadow_stack_get(base + 1),
        literals: crate::gc_roots::shadow_stack_get(base + 2),
    })
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_sre_template(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &SRE_TEMPLATE_TYPE) }
}
