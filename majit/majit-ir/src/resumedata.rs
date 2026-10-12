//! Resume data decoding primitives.
//!
//! resume.py tag layout + `rebuild_from_numbering` (decode-side) shared by
//! `majit-metainterp` (guard failure) and `pyre-jit-trace` (trace-side
//! recovery). The encoder-side (`Snapshot`, `NumberingState`,
//! `ResumeDataLoopMemo`) lives in `majit-metainterp/src/resume.rs` to stay
//! colocated with the metainterp context that drives it.

use crate::resumecode;
use crate::{Const, Type};

use std::sync::atomic::{AtomicUsize, Ordering};

/// Global frame_value_count callback for RPython-parity multi-frame decode.
/// resume.py: consume_boxes(f.get_current_position_info(), ...) uses
/// per-jitcode liveness at the decode site. pyre registers this callback
/// from pyre-jit-trace::state so that all callers of rebuild_from_numbering
/// (including those in majit-metainterp and majit-backend-cranelift) can
/// split multi-frame sections correctly.
static FRAME_VALUE_COUNT_FN: AtomicUsize = AtomicUsize::new(0);

/// Register the global frame_value_count callback.
///
/// Signature: `(jitcode_index, pc) -> count`.
pub fn set_frame_value_count_fn(f: fn(i32, i32) -> usize) {
    FRAME_VALUE_COUNT_FN.store(f as usize, Ordering::Relaxed);
}

/// Get the registered frame_value_count callback (if any).
pub fn get_frame_value_count_fn() -> Option<fn(i32, i32) -> usize> {
    let p = FRAME_VALUE_COUNT_FN.load(Ordering::Relaxed);
    if p == 0 {
        None
    } else {
        Some(unsafe { std::mem::transmute::<usize, fn(i32, i32) -> usize>(p) })
    }
}

// resume.py — tag constants
pub const TAGCONST: u8 = 0;
pub const TAGINT: u8 = 1;
pub const TAGBOX: u8 = 2;
pub const TAGVIRTUAL: u8 = 3;

/// Snapshot frame pc words now carry the published resume coordinate directly.
///
/// RPython stores the JitCode pc in `frame.pc`
/// (`pyjitpl.py capture_resumedata(resumepc=-1)`).  Pyre now matches
/// that shape for published resume data: the pc word is a genuine JitCode byte
/// offset, and after-residual-call guards carry their post-call
/// `-live-`/`catch_exception` coordinate through the companion `jitcode_pc`
/// field where needed.  No high-bit marker is reserved in the pc word; offsets
/// whose natural value has bit 14 set must remain unchanged.
///
/// Sentinel for a snapshot frame's `jitcode_pc` word: no direct JitCode
/// resume coordinate. Such a frame is unrepresentable and must decline before
/// resume data is published.
///
/// Each per-frame header carries a `jitcode_pc` word after `pc`. A branch guard whose
/// not-taken arm keeps operand-stack temps the JitCode virtualized cannot
/// be described by the merge-target Python pc: its kept temps are only
/// live at the *guard* JitCode position, not at the resume Python opcode
/// boundary.  For such guards the capture stores the guard's JitCode byte
/// offset here (a non-negative value), and both the box-count liveness
/// query and the blackhole `setposition` resolve directly off it,
/// bypassing `pc_map` — the direct `setposition(jitcode, pc)` shape
/// (`pyjitpl.py`) without the pc round-trip.  The Python `pc`
/// word stays populated for interpreter re-entry / `last_instr`.
pub const NO_JITCODE_PC: i32 = -1;

/// #73: a depth-0 branch guard's resume word may carry the walk's
/// arm-independent `-live-` BEFORE anchor (`orgpc`) plus the GuardTrue/False
/// flavor, tagged into the free negative space of the `jitcode_pc` word
/// (offsets are >=0, NO_JITCODE_PC is -1). Decode expands it to the block-head
/// resume marker. `orgpc` must be <= BRANCH_ORGPC_MAX to stay in i16 range.
pub const BRANCH_ORGPC_MAX: usize = 16382;

/// Tag `orgpc` (+ 1 flavor bit) into the free negative space `[-32767, -2]` of
/// the `jitcode_pc` word. `orgpc` must be `<= BRANCH_ORGPC_MAX`.
pub const fn encode_branch_orgpc(orgpc: usize, flavor_guard_true: bool) -> i32 {
    // orgpc in [0, BRANCH_ORGPC_MAX], 1 flavor bit -> [-32767, -2]
    -2 - (((orgpc as i32) << 1) | (flavor_guard_true as i32))
}

/// Inverse of [`encode_branch_orgpc`]. Returns `Some((orgpc, flavor))` for a
/// tagged word (`<= -2`), `None` for offsets (`>= 0`) and NO_JITCODE_PC (`-1`).
pub const fn decode_branch_orgpc(word: i32) -> Option<(usize, bool)> {
    if word <= -2 {
        let v = (-2 - word) as usize;
        Some((v >> 1, (v & 1) == 1))
    } else {
        None
    }
}

/// `resume.py` tag discriminator used by `ResumeValueSource` /
/// `ResumeValueLayoutSummary` to record where a guard-time value
/// comes from.  Moved here from `majit-metainterp::resume` so it can
/// be referenced from `majit-backend` along with the rest of the
/// Phase C-1 unified-descr type migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeValueKind {
    FailArg,
    Constant,
    Virtual,
    Uninitialized,
    Unavailable,
}

/// Summary of a `ResumeValueSource` in serialization-friendly form.
///
/// Moved here from `majit-metainterp::resume` (Phase C-1 cascade)
/// alongside `ResumeValueKind` so backend-side resume readers can
/// build summaries without depending on the metainterp crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeValueLayoutSummary {
    pub kind: ResumeValueKind,
    pub fail_arg_index: usize,
    pub raw_fail_arg_position: Option<usize>,
    /// RPython parity: `Const.value` raw bits (getint/getref_base/getfloatstorage
    /// all project to `i64`).
    pub constant: Option<i64>,
    /// RPython parity: `Const.type` — paired with `constant` so the summary
    /// round-trips back into a typed `ResumeValueSource::Constant(Const)`.
    pub constant_type: Option<Type>,
    pub virtual_index: Option<usize>,
}

/// `resume.py` virtual-info class discriminator used by `VirtualInfo`
/// to record the variant kind without unfolding the enum.  Moved here
/// from `majit-metainterp::resume` (Phase C-1 cascade) so the
/// `ResumeData` chain can live in a backend-accessible crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeVirtualKind {
    Object,
    Struct,
    Array,
    ArrayStruct,
    RawBuffer,
    /// resume.py VStrPlainInfo — virtual plain string.
    StrPlain,
    /// resume.py VStrConcatInfo — virtual concatenated string.
    StrConcat,
    /// resume.py VStrSliceInfo — virtual string slice.
    StrSlice,
    /// resume.py VUniPlainInfo — virtual plain unicode string.
    UniPlain,
    /// resume.py VUniConcatInfo — virtual concatenated unicode.
    UniConcat,
    /// resume.py VUniSliceInfo — virtual unicode slice.
    UniSlice,
    /// resume.py `rd_virtuals[i] is None` hole.
    Hole,
}

/// Per-frame layout summary used by `ResumeLayoutSummary`.
///
/// Moved here from `majit-metainterp::resume` (Phase C-1 cascade) —
/// all field types (`Type`, `ResumeValueKind`, `ResumeValueLayoutSummary`)
/// live in `majit-ir`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeFrameLayoutSummary {
    pub trace_id: Option<u64>,
    pub header_pc: Option<u64>,
    pub source_guard: Option<(u64, u32)>,
    /// resume.py:250 jitcode_index — index into metainterp_sd.jitcodes[].
    pub jitcode_index: i32,
    pub pc: u64,
    pub slot_sources: Vec<ResumeValueKind>,
    pub slot_layouts: Vec<ResumeValueLayoutSummary>,
    pub slot_types: Option<Vec<Type>>,
}

/// Serialization-friendly summary of a `PendingFieldInfo`.
///
/// Moved here from `majit-metainterp::resume` (Phase C-1 cascade);
/// dependencies (`DescrRef`, `ResumeValueKind`, `ResumeValueLayoutSummary`)
/// are all in `majit-ir`.
#[derive(Debug, Clone)]
pub struct PendingFieldLayoutSummary {
    /// `resume.py:88 lldescr` — identity-compared via `Arc::ptr_eq`
    /// (`history.py:125`).
    pub descr: Option<crate::DescrRef>,
    pub item_index: Option<usize>,
    pub is_array_item: bool,
    pub target_kind: ResumeValueKind,
    pub value_kind: ResumeValueKind,
    pub target: ResumeValueLayoutSummary,
    pub value: ResumeValueLayoutSummary,
}

impl PartialEq for PendingFieldLayoutSummary {
    fn eq(&self, other: &Self) -> bool {
        opt_descr_arc_ptr_eq(&self.descr, &other.descr)
            && self.item_index == other.item_index
            && self.is_array_item == other.is_array_item
            && self.target_kind == other.target_kind
            && self.value_kind == other.value_kind
            && self.target == other.target
            && self.value == other.value
    }
}
impl Eq for PendingFieldLayoutSummary {}

/// `history.py:125 id(descr)` parity: `Option<DescrRef>` identity
/// comparison via `Arc::ptr_eq`.
#[inline]
pub fn opt_descr_arc_ptr_eq(a: &Option<crate::DescrRef>, b: &Option<crate::DescrRef>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
        _ => false,
    }
}

/// Serialization-friendly summary of a `VirtualInfo`.
///
/// Moved here from `majit-metainterp::resume` (Phase C-1 cascade).
/// Carries the same shape as `VirtualInfo` but with
/// `ResumeValueLayoutSummary` instead of `ResumeValueSource` field
/// values — enables round-trip through backend-side resume layout
/// builders without depending on `majit-metainterp`.
#[derive(Debug, Clone)]
pub enum ResumeVirtualLayoutSummary {
    Object {
        /// resume.py:615 self.descr — live SizeDescr, preserved across summary round-trip.
        descr: Option<crate::DescrRef>,
        type_id: u32,
        /// info.py:318 _known_class — vtable pointer.
        known_class: Option<i64>,
        fields: Vec<(u32, ResumeValueLayoutSummary)>,
        fielddescrs: Vec<crate::FieldDescrInfo>,
        descr_size: usize,
    },
    Struct {
        /// resume.py:631 self.typedescr — live SizeDescr, preserved across summary round-trip.
        typedescr: Option<crate::DescrRef>,
        type_id: u32,
        fields: Vec<(u32, ResumeValueLayoutSummary)>,
        fielddescrs: Vec<crate::FieldDescrInfo>,
        descr_size: usize,
    },
    /// resume.py AbstractVArrayInfo
    Array {
        /// resume.py:646: self.arraydescr
        arraydescr: Option<crate::DescrRef>,
        /// resume.py: VArrayInfoClear.clear=True / VArrayInfoNotClear.clear=False
        clear: bool,
        items: Vec<ResumeValueLayoutSummary>,
    },
    /// resume.py VArrayStructInfo(arraydescr, size, fielddescrs)
    ArrayStruct {
        /// resume.py:739: self.arraydescr
        arraydescr: Option<crate::DescrRef>,
        /// resume.py:740: self.fielddescrs
        fielddescrs: Vec<crate::DescrRef>,
        element_fields: Vec<Vec<(u32, ResumeValueLayoutSummary)>>,
    },
    RawBuffer {
        /// resume.py __init__: self.func
        func: i64,
        size: usize,
        /// resume.py: self.offsets — signed (rawbuffer.py __init__).
        offsets: Vec<i64>,
        /// resume.py:697: self.descrs
        descrs: Vec<crate::ArrayDescrInfo>,
        /// resume.py:693: fieldnums (decoded)
        values: Vec<ResumeValueLayoutSummary>,
    },
    /// `resume.py: VRawSliceInfo` — a slice into a virtual raw buffer.
    RawSlice {
        offset: i64,
        parent: ResumeValueLayoutSummary,
    },
    /// `resume.py VStrPlainInfo` — virtual string (known characters).
    StrPlain {
        chars: Vec<ResumeValueLayoutSummary>,
    },
    /// `resume.py VStrConcatInfo` — virtual string concat. OS_STR_CONCAT
    /// funcptr is resolved at materialization via
    /// `callinfocollection.funcptr_for_oopspec(...)` (resume.py),
    /// not stored on the summary.
    StrConcat {
        left: ResumeValueLayoutSummary,
        right: ResumeValueLayoutSummary,
    },
    /// `resume.py VStrSliceInfo` — virtual slice of a larger string.
    /// OS_STR_SLICE funcptr resolved via callinfocollection at
    /// materialization (resume.py).
    StrSlice {
        source: ResumeValueLayoutSummary,
        start: ResumeValueLayoutSummary,
        length: ResumeValueLayoutSummary,
    },
    /// `resume.py VUniPlainInfo` — unicode counterpart.
    UniPlain {
        chars: Vec<ResumeValueLayoutSummary>,
    },
    /// `resume.py VUniConcatInfo` — unicode counterpart.
    /// OS_UNI_CONCAT funcptr resolved via callinfocollection
    /// (resume.py).
    UniConcat {
        left: ResumeValueLayoutSummary,
        right: ResumeValueLayoutSummary,
    },
    /// `resume.py VUniSliceInfo` — unicode counterpart.
    /// OS_UNI_SLICE funcptr resolved via callinfocollection
    /// (resume.py).
    UniSlice {
        source: ResumeValueLayoutSummary,
        start: ResumeValueLayoutSummary,
        length: ResumeValueLayoutSummary,
    },
    /// resume.py `rd_virtuals[i] is None` hole.
    Hole,
}
const TAGMASK: u8 = 3;

pub const UNASSIGNED: i16 = ((-1i32 << 13) << 2 | TAGBOX as i32) as i16;
pub const NULLREF: i16 = ((-1i32 << 2) | TAGCONST as i32) as i16;
pub const UNINITIALIZED_TAG: i16 = ((-2i32 << 2) | TAGCONST as i32) as i16;
pub const TAG_CONST_OFFSET: i32 = 0;

/// resume.py untag
pub fn untag(value: i16) -> (i32, u8) {
    let widened = value as i32;
    let tagbits = (widened & TAGMASK as i32) as u8;
    (widened >> 2, tagbits)
}

// ── Decoding (for guard failure recovery) ──

/// Decoded value from rd_numb.
#[derive(Debug, Clone, PartialEq)]
pub enum RebuiltValue {
    /// TAGBOX(n, kind): value from deadframe slot n with the kind that
    /// the parent guard's `fail_arg_types[n]` recorded at numbering time.
    /// resume.py `decode_box(num, kind)` parity — RPython resolves
    /// the kind at decode time from the callback (i/r/f) that the encoder
    /// dispatched through `enumerate_vars(info, liveness_info, ...)`. The
    /// box's `.type` matches that kind. majit stores the kind on the
    /// variant so consumers don't need a parallel `fail_arg_types` lookup.
    Box(usize, Type),
    /// TAGCONST / TAGINT: compile-time constant. Carries a `Const`
    /// (Int/Float/Ref), matching RPython's `decode_box` which returns a
    /// `ConstInt`/`ConstFloat`/`ConstPtr` regardless of whether the value
    /// came from the inline TAGINT encoding or the TAGCONST pool
    /// (resume.py).
    Const(crate::Const),
    /// TAGVIRTUAL(n): virtual object index n.
    Virtual(usize),
    /// Uninitialized/unassigned slot.
    Unassigned,
}

/// Decoded frame from rd_numb.
#[derive(Debug, Clone)]
pub struct RebuiltFrame {
    pub jitcode_index: i32,
    /// resume.py:250 `pc` — the JitCode byte offset.
    pub pc: i32,
    pub values: Vec<RebuiltValue>,
}

/// Decode one tagged resume value outside the numbered frame stream.
///
/// Pending fields carry their target and value through the same
/// TAGBOX/TAGCONST/TAGVIRTUAL encoding as frame slots
/// (`resume.py:_add_pending_fields`).  Bridge setup uses this entry point so
/// its consume-side view is identical to `rebuild_from_numbering`.
pub fn decode_tagged_value(
    tagged: i16,
    num_failargs: i32,
    rd_consts: &[Const],
    fail_arg_types: &[Type],
    num_virtuals: usize,
) -> RebuiltValue {
    let (val, tagbits) = untag(tagged);
    match tagbits {
        // resume.py ConstInt(num) — TAGINT always produces an int box.
        TAGINT => RebuiltValue::Const(crate::Const::Int(val as i64)),
        TAGCONST => {
            if tagged == NULLREF {
                // history.py CONST_NULL = ConstPtr(null).
                RebuiltValue::Const(crate::Const::from_gcref(crate::GcRef::NULL))
            } else if tagged == UNINITIALIZED_TAG {
                RebuiltValue::Unassigned
            } else {
                let idx = (val - TAG_CONST_OFFSET) as usize;
                // resume.py `ResumeDataDirectReader.decode_int` /
                // `decode_ref` / `decode_float` each read `self.consts[num -
                // TAG_CONST_OFFSET]` — a plain index, so an out-of-range
                // number is an `IndexError` there.  The Const carries its type
                // with it, which is why there is no defaulting answer to give:
                // `Const::Int(0)` is a well-formed integer zero, so a bridge
                // rebuilt from a truncated `rd_consts` would run on it instead
                // of failing, while `ResumeDataDirectReader`'s `decode_int` /
                // `decode_ref` panic on the very same index.
                let Some(c) = rd_consts.get(idx).copied() else {
                    panic!(
                        "decode_tagged_value: TAGCONST index {idx} is past the \
                         guard's {} consts (tagged {tagged})",
                        rd_consts.len(),
                    );
                };
                RebuiltValue::Const(c)
            }
        }
        TAGBOX => {
            let index = if val < 0 {
                (val + num_failargs) as usize
            } else {
                val as usize
            };
            // resume.py decode_box(num, kind) parity — pull the
            // box's `.type` from the parent guard's fail_arg_types,
            // which `_number_boxes` populated when encoding via
            // `env.get_type(opref)`. resume.py asserts
            // `box.type == kind`; a missing entry means the
            // fail_arg_types vector is shorter than the encoded box
            // count (encoder/decoder disagreement), which must fail
            // loudly rather than silently recover as Int.
            let kind = *fail_arg_types.get(index).unwrap_or_else(|| {
                panic!(
                    "resume decode_box: fail_arg_types[{index}] missing (len {}); \
                     encoder/decoder failarg-count disagreement (resume.py:1245)",
                    fail_arg_types.len(),
                )
            });
            RebuiltValue::Box(index, kind)
        }
        TAGVIRTUAL => {
            // resume.py assign_number_to_virtual numbers nested
            // virtuals negatively; resume.py getvirtual_ptr resolves
            // them via Python negative list indexing into rd_virtuals.
            let index = if val < 0 {
                (num_virtuals as i32 + val) as usize
            } else {
                val as usize
            };
            RebuiltValue::Virtual(index)
        }
        _ => RebuiltValue::Unassigned,
    }
}

/// Decode rd_numb back into vable/vref values and per-frame tagged values.
///
/// resume.py `ResumeDataLoopMemo.number` / `ResumeDataDirectReader.read_jitcode_pos_pc`:
/// frames are `jitcode_index, pc, [tagged_values...]`. Liveness
/// (`get_current_position_info`) at the decode site says how many values
/// each frame has.
///
/// `frame_value_count`: when `Some(f)`, `f(jitcode_index, pc)`
/// returns the number of tagged values for that frame (RPython parity:
/// liveness-driven decode). When `None`, all remaining items after
/// `(jitcode_index, pc)` are consumed as a single frame (callers that only
/// ever see single-frame data).
///
/// `fail_arg_types`: parent guard's per-failarg type vector. resume.py
/// `decode_box(num, kind)` parity — TAGBOX values use this to fill in their
/// kind so the resulting `RebuiltValue::Box` carries its own type and
/// downstream consumers don't need a parallel side channel.
pub fn rebuild_from_numbering(
    rd_numb: &[u8],
    rd_consts: &[Const],
    fail_arg_types: &[Type],
    frame_value_count: Option<&dyn Fn(i32, i32) -> usize>,
    num_virtuals: usize,
) -> (i32, Vec<RebuiltValue>, Vec<RebuiltValue>, Vec<RebuiltFrame>) {
    let mut reader = resumecode::Reader::new(rd_numb);

    let total_size = reader.next_item();
    let num_failargs = reader.next_item();

    // resume.py: consume_vref_and_vable_boxes — virtualizable array.
    let vable_len = reader.next_item();
    let mut vable_values = Vec::new();
    for _ in 0..vable_len {
        if !reader.has_more() {
            break;
        }
        let tagged = reader.next_item() as i16;
        vable_values.push(decode_tagged_value(
            tagged,
            num_failargs,
            rd_consts,
            fail_arg_types,
            num_virtuals,
        ));
    }

    // resume.py:1045: virtualref array (pairs).
    let vref_len = reader.next_item();
    let mut vref_values = Vec::new();
    for _ in 0..(vref_len * 2) {
        if !reader.has_more() {
            break;
        }
        let tagged = reader.next_item() as i16;
        vref_values.push(decode_tagged_value(
            tagged,
            num_failargs,
            rd_consts,
            fail_arg_types,
            num_virtuals,
        ));
    }

    // resume.py number: jitcode_index, pc, [tagged_values...].
    // consume_one_section → enumerate_vars(liveness) splits frames.
    let mut frames = Vec::new();
    while reader.items_read < total_size as usize && reader.has_more() {
        let jitcode_index = reader.next_item();
        let pc = if reader.has_more() && reader.items_read < total_size as usize {
            reader.next_item()
        } else {
            0
        };
        let box_count = if let Some(f) = &frame_value_count {
            // RPython parity: liveness-driven frame boundary.
            f(jitcode_index, pc)
        } else {
            // Single-frame fallback: consume all remaining items.
            (total_size as usize).saturating_sub(reader.items_read)
        };
        let mut values = Vec::with_capacity(box_count);
        for _ in 0..box_count {
            if !reader.has_more() {
                break;
            }
            let tagged = reader.next_item() as i16;
            values.push(decode_tagged_value(
                tagged,
                num_failargs,
                rd_consts,
                fail_arg_types,
                num_virtuals,
            ));
        }
        frames.push(RebuiltFrame {
            jitcode_index,
            pc,
            values,
        });
    }
    (num_failargs, vable_values, vref_values, frames)
}

#[cfg(test)]
mod branch_orgpc_tag_tests {
    use super::{BRANCH_ORGPC_MAX, NO_JITCODE_PC, decode_branch_orgpc, encode_branch_orgpc};

    #[test]
    fn roundtrips_across_orgpc_and_flavor() {
        // Every (orgpc, flavor) in range tags into the negative space
        // and decodes back exactly, including the two endpoints.
        for orgpc in [
            0usize,
            1,
            2,
            7,
            100,
            8191,
            BRANCH_ORGPC_MAX - 1,
            BRANCH_ORGPC_MAX,
        ] {
            for flavor in [false, true] {
                let word = encode_branch_orgpc(orgpc, flavor);
                assert!(word <= -2, "tagged word {word} must land in negative space");
                assert_eq!(decode_branch_orgpc(word), Some((orgpc, flavor)));
            }
        }
    }

    #[test]
    fn tagged_words_stay_in_i16_range() {
        // The word is serialized by `append_int`, which asserts i16 range.
        for orgpc in [0usize, BRANCH_ORGPC_MAX] {
            for flavor in [false, true] {
                let word = encode_branch_orgpc(orgpc, flavor);
                assert!(
                    word >= i16::MIN as i32 && word <= i16::MAX as i32,
                    "tagged word {word} out of i16 range"
                );
            }
        }
    }

    #[test]
    fn non_tagged_words_decode_none() {
        // Offsets (>=0) and NO_JITCODE_PC (-1) are NOT tagged: decode returns
        // None so the expand at decode is a pure no-op for them.
        for word in [0, 5, 32767, NO_JITCODE_PC] {
            assert_eq!(decode_branch_orgpc(word), None);
        }
    }
}
