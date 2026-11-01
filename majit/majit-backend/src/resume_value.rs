//! Resume-value tagged sources used by `ResumeData` (and ultimately
//! `ResumeGuardDescr`) — moved here from `majit-metainterp::resume` as
//! part of the Phase C-1 cascade toward backend struct deletion.
//!
//! `compile.py AbstractResumeGuardDescr._attrs_` resume payload
//! references these tags through `rd_virtuals` / pending field
//! sources; placing them in a backend-accessible crate lets the
//! eventual `ResumeGuardDescr` definition live where the backend
//! codegen can instantiate it directly.

use majit_ir::resumedata::{
    PendingFieldLayoutSummary, ResumeValueKind, ResumeValueLayoutSummary, ResumeVirtualKind,
    ResumeVirtualLayoutSummary, opt_descr_arc_ptr_eq,
};
use majit_ir::{ArrayDescrInfo, Const, DescrRef, FieldDescrInfo};

/// Tagged source for a value that must be reconstructed on resume.
///
/// This is the majit equivalent of the tagged numbering used by
/// `rpython/jit/metainterp/resume.py`. Each `Constant` entry carries a
/// full `majit_ir::Const` (Int/Float/Ref) so the encoder's `getconst`
/// dispatch (resume.py) can route through the shared pool
/// (`ResumeDataLoopMemo.consts`) without losing type information.
#[derive(Debug, Clone, PartialEq)]
pub enum ResumeValueSource {
    /// Value comes from the deadframe fail-args array.
    FailArg(usize),
    /// Value is a compile-time constant — carries the Const so that the
    /// type survives encoding, matching RPython's `Const` object.
    Constant(Const),
    /// Value is a virtual object that must be materialized on resume.
    Virtual(usize),
    /// Raw resume.py field number, retained until the direct reader decodes
    /// it. RPython stores `AbstractVirtualInfo.fieldnums` in this form and
    /// resolves TAGCONST through the live `rd_consts` list only after any
    /// allocation-triggered collection has forwarded that list.
    ///
    /// This variant is runtime-only: layout/export paths construct the
    /// structured variants above.
    Tagged(i16),
    /// Value exists conceptually but is still uninitialized.
    ///
    /// Mirrors RPython's `UNINITIALIZED` tag used for string/unicode content.
    Uninitialized,
    /// Slot is not live at this guard.
    Unavailable,
}

impl ResumeValueSource {
    pub fn kind(&self) -> ResumeValueKind {
        match self {
            ResumeValueSource::FailArg(_) => ResumeValueKind::FailArg,
            ResumeValueSource::Constant(_) => ResumeValueKind::Constant,
            ResumeValueSource::Virtual(_) => ResumeValueKind::Virtual,
            ResumeValueSource::Tagged(_) => {
                panic!("runtime tagged resume source has no exported ResumeValueKind")
            }
            ResumeValueSource::Uninitialized => ResumeValueKind::Uninitialized,
            ResumeValueSource::Unavailable => ResumeValueKind::Unavailable,
        }
    }

    pub fn layout_summary(&self) -> ResumeValueLayoutSummary {
        match self {
            ResumeValueSource::FailArg(index) => ResumeValueLayoutSummary {
                kind: ResumeValueKind::FailArg,
                fail_arg_index: *index,
                raw_fail_arg_position: Some(*index),
                constant: None,
                constant_type: None,
                virtual_index: None,
            },
            ResumeValueSource::Constant(c) => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Constant,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: Some(c.as_raw_i64()),
                constant_type: Some(c.get_type()),
                virtual_index: None,
            },
            ResumeValueSource::Virtual(index) => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Virtual,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                constant_type: None,
                virtual_index: Some(*index),
            },
            ResumeValueSource::Tagged(_) => {
                panic!("runtime tagged resume source cannot be exported as a layout summary")
            }
            ResumeValueSource::Uninitialized => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Uninitialized,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                constant_type: None,
                virtual_index: None,
            },
            ResumeValueSource::Unavailable => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Unavailable,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                constant_type: None,
                virtual_index: None,
            },
        }
    }
}

/// Cross-crate impl methods for `ResumeValueLayoutSummary`.
pub trait ResumeValueLayoutSummaryExt {
    /// `resume.py:226` raw fail-arg position lookup — falls back to
    /// `fail_arg_index` when the explicit `raw_fail_arg_position`
    /// override is absent.
    fn raw_fail_arg_position_or_fallback(&self) -> usize;
    fn to_resume_source(&self) -> ResumeValueSource;
}

impl ResumeValueLayoutSummaryExt for ResumeValueLayoutSummary {
    fn raw_fail_arg_position_or_fallback(&self) -> usize {
        self.raw_fail_arg_position.unwrap_or(self.fail_arg_index)
    }

    fn to_resume_source(&self) -> ResumeValueSource {
        match self.kind {
            ResumeValueKind::FailArg => {
                ResumeValueSource::FailArg(self.raw_fail_arg_position_or_fallback())
            }
            ResumeValueKind::Constant => {
                let raw = self.constant.expect("missing constant value");
                let tp = self.constant_type.expect("missing constant type");
                ResumeValueSource::Constant(Const::from_raw_i64(raw, tp))
            }
            ResumeValueKind::Virtual => {
                ResumeValueSource::Virtual(self.virtual_index.expect("missing virtual index"))
            }
            ResumeValueKind::Uninitialized => ResumeValueSource::Uninitialized,
            ResumeValueKind::Unavailable => ResumeValueSource::Unavailable,
        }
    }
}

/// Source for a resumed frame slot.  Type alias kept for caller
/// compatibility with `resume.py` naming.
pub type FrameSlotSource = ResumeValueSource;

/// Source of a virtual object's field value (alias for the
/// resume.py-style tagged source, used by `VirtualInfo` variants).
pub type VirtualFieldSource = ResumeValueSource;

/// Describes how to reconstruct a single frame in the interpreter's call stack.
///
/// Each frame has a bytecode position (pc) and a set of named/indexed slots
/// that map to tagged resume sources.  Moved here from
/// `majit-metainterp::resume` (Phase C-1 cascade) as the next hop
/// toward the ResumeData migration.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameInfo {
    /// resume.py:250 jitcode_index — index into metainterp_sd.jitcodes[].
    pub jitcode_index: i32,
    /// JitCode byte offset (`resume.py` `ResumeDataLoopMemo.number`).
    pub pc: u64,
    /// Mapping from slot index to a tagged resume source.
    pub slot_map: Vec<FrameSlotSource>,
}

/// Description of a virtual object that needs materialization on resume.
///
/// Mirrors RPython's AbstractVirtualInfo hierarchy:
/// - VirtualInfo (NEW_WITH_VTABLE)
/// - VStructInfo (NEW / plain struct)
/// - VArrayInfoClear / VArrayInfoNotClear (NEW_ARRAY)
/// - VArrayStructInfo (array of structs with interior fields)
/// - VRawBufferInfo (raw memory buffer)
#[derive(Debug, Clone)]
pub enum VirtualInfo {
    /// resume.py VirtualInfo(descr, fielddescrs).
    VirtualObj {
        /// resume.py:615 self.descr — live SizeDescr.
        descr: Option<DescrRef>,
        type_id: u32,
        /// info.py:318 _known_class — vtable pointer.
        known_class: Option<i64>,
        fields: Vec<(u32, VirtualFieldSource)>,
        fielddescrs: Vec<FieldDescrInfo>,
        descr_size: usize,
    },
    /// resume.py VStructInfo(typedescr, fielddescrs).
    VStruct {
        /// resume.py:631 self.typedescr — the full SizeDescr.
        typedescr: Option<DescrRef>,
        type_id: u32,
        fields: Vec<(u32, VirtualFieldSource)>,
        fielddescrs: Vec<FieldDescrInfo>,
        descr_size: usize,
    },
    /// resume.py AbstractVArrayInfo (from NEW_ARRAY).
    VArray {
        /// resume.py:646: self.arraydescr
        arraydescr: Option<DescrRef>,
        /// resume.py: VArrayInfoClear.clear / VArrayInfoNotClear.clear
        clear: bool,
        /// Element values.
        items: Vec<VirtualFieldSource>,
    },
    /// resume.py VArrayStructInfo (from arrays with interior field access).
    VArrayStruct {
        /// resume.py:739: self.arraydescr
        arraydescr: Option<DescrRef>,
        /// resume.py:740: self.fielddescrs — live InteriorFieldDescr objects
        /// for setinteriorfield dispatch.
        fielddescrs: Vec<DescrRef>,
        /// Per-element fields: outer Vec = elements, inner Vec = (field_index, source).
        element_fields: Vec<Vec<(u32, VirtualFieldSource)>>,
    },
    /// resume.py VRawBufferInfo(func, size, offsets, descrs).
    VRawBuffer {
        /// resume.py __init__: self.func — raw malloc function pointer.
        func: i64,
        /// Size of the buffer in bytes.
        size: usize,
        /// resume.py:695: self.offsets — byte offsets for each stored
        /// value. Signed to match rawbuffer.py __init__.
        offsets: Vec<i64>,
        /// resume.py:697: self.descrs — per-entry ArrayDescr snapshots.
        descrs: Vec<ArrayDescrInfo>,
        /// resume.py:693: fieldnums — per-entry source (decoded from tagged fieldnums).
        values: Vec<VirtualFieldSource>,
    },
    /// resume.py: VRawSliceInfo — a slice into a virtual raw buffer.
    VRawSlice {
        /// Offset from the parent raw buffer.
        offset: i64,
        /// Source of the parent buffer.
        parent: VirtualFieldSource,
    },
    /// resume.py VStrPlainInfo — virtual string (known characters).
    VStrPlain {
        /// Character values (as OpRef sources).
        chars: Vec<VirtualFieldSource>,
    },
    /// resume.py VStrConcatInfo — virtual string concat (left + right).
    /// OS_STR_CONCAT funcptr is resolved at materialization via
    /// `callinfocollection.funcptr_for_oopspec(OS_STR_CONCAT)`
    /// (resume.py); the layout carries no funcptr.
    VStrConcat {
        left: Box<VirtualFieldSource>,
        right: Box<VirtualFieldSource>,
    },
    /// resume.py VStrSliceInfo — virtual string slice. OS_STR_SLICE
    /// funcptr resolved via callinfocollection at materialization
    /// (resume.py).
    VStrSlice {
        source: Box<VirtualFieldSource>,
        start: Box<VirtualFieldSource>,
        length: Box<VirtualFieldSource>,
    },
    /// resume.py VUniPlainInfo — virtual unicode string.
    VUniPlain { chars: Vec<VirtualFieldSource> },
    /// resume.py VUniConcatInfo — virtual unicode concat.
    /// OS_UNI_CONCAT funcptr resolved via callinfocollection
    /// (resume.py).
    VUniConcat {
        left: Box<VirtualFieldSource>,
        right: Box<VirtualFieldSource>,
    },
    /// resume.py VUniSliceInfo — virtual unicode slice.
    /// OS_UNI_SLICE funcptr resolved via callinfocollection
    /// (resume.py).
    VUniSlice {
        source: Box<VirtualFieldSource>,
        start: Box<VirtualFieldSource>,
        length: Box<VirtualFieldSource>,
    },
    /// resume.py `rd_virtuals = [None] * length` hole. Not a virtual;
    /// `force_all_virtuals` skips it (`if rd_virtual is not None`).
    Empty,
}

// `history.py:125 id(descr)` parity — descr identity via Arc::ptr_eq;
// fields with no descr (None) compare equal only when both are None.
impl PartialEq for VirtualInfo {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                VirtualInfo::VirtualObj {
                    descr: a_descr,
                    type_id: a1,
                    fields: a3,
                    fielddescrs: a4,
                    descr_size: a5,
                    ..
                },
                VirtualInfo::VirtualObj {
                    descr: b_descr,
                    type_id: b1,
                    fields: b3,
                    fielddescrs: b4,
                    descr_size: b5,
                    ..
                },
            ) => {
                opt_descr_arc_ptr_eq(a_descr, b_descr)
                    && a1 == b1
                    && a3 == b3
                    && a4 == b4
                    && a5 == b5
            }
            (
                VirtualInfo::VStruct {
                    typedescr: a_descr,
                    type_id: a1,
                    fields: a3,
                    fielddescrs: a4,
                    descr_size: a5,
                    ..
                },
                VirtualInfo::VStruct {
                    typedescr: b_descr,
                    type_id: b1,
                    fields: b3,
                    fielddescrs: b4,
                    descr_size: b5,
                    ..
                },
            ) => {
                opt_descr_arc_ptr_eq(a_descr, b_descr)
                    && a1 == b1
                    && a3 == b3
                    && a4 == b4
                    && a5 == b5
            }
            (
                VirtualInfo::VArray {
                    arraydescr: a_descr,
                    clear: a_clear,
                    items: a2,
                },
                VirtualInfo::VArray {
                    arraydescr: b_descr,
                    clear: b_clear,
                    items: b2,
                },
            ) => opt_descr_arc_ptr_eq(a_descr, b_descr) && a_clear == b_clear && a2 == b2,
            (VirtualInfo::Empty, VirtualInfo::Empty) => true,
            _ => false,
        }
    }
}
impl Eq for VirtualInfo {}

impl VirtualInfo {
    /// Iterate over all field sources in this virtual.
    /// resume.py: visitor_walk_recursive walks all box references in a virtual.
    pub fn field_sources(&self) -> Vec<&VirtualFieldSource> {
        match self {
            VirtualInfo::VirtualObj { fields, .. } | VirtualInfo::VStruct { fields, .. } => {
                fields.iter().map(|(_, src)| src).collect()
            }
            VirtualInfo::VArray { items, .. } => items.iter().collect(),
            VirtualInfo::VArrayStruct { element_fields, .. } => element_fields
                .iter()
                .flat_map(|el| el.iter().map(|(_, src)| src))
                .collect(),
            VirtualInfo::VRawBuffer { values, .. } => values.iter().collect(),
            VirtualInfo::VRawSlice { parent, .. } => vec![parent],
            VirtualInfo::VStrPlain { chars } | VirtualInfo::VUniPlain { chars } => {
                chars.iter().collect()
            }
            VirtualInfo::VStrConcat { left, right, .. }
            | VirtualInfo::VUniConcat { left, right, .. } => {
                vec![left.as_ref(), right.as_ref()]
            }
            VirtualInfo::VStrSlice {
                source,
                start,
                length,
                ..
            }
            | VirtualInfo::VUniSlice {
                source,
                start,
                length,
                ..
            } => vec![source.as_ref(), start.as_ref(), length.as_ref()],
            VirtualInfo::Empty => Vec::new(),
        }
    }

    pub fn kind(&self) -> ResumeVirtualKind {
        match self {
            VirtualInfo::VirtualObj { .. } => ResumeVirtualKind::Object,
            VirtualInfo::VStruct { .. } => ResumeVirtualKind::Struct,
            VirtualInfo::VArray { .. } => ResumeVirtualKind::Array,
            VirtualInfo::VArrayStruct { .. } => ResumeVirtualKind::ArrayStruct,
            VirtualInfo::VRawBuffer { .. } | VirtualInfo::VRawSlice { .. } => {
                ResumeVirtualKind::RawBuffer
            }
            VirtualInfo::VStrPlain { .. } => ResumeVirtualKind::StrPlain,
            VirtualInfo::VStrConcat { .. } => ResumeVirtualKind::StrConcat,
            VirtualInfo::VStrSlice { .. } => ResumeVirtualKind::StrSlice,
            VirtualInfo::VUniPlain { .. } => ResumeVirtualKind::UniPlain,
            VirtualInfo::VUniConcat { .. } => ResumeVirtualKind::UniConcat,
            VirtualInfo::VUniSlice { .. } => ResumeVirtualKind::UniSlice,
            VirtualInfo::Empty => ResumeVirtualKind::Hole,
        }
    }

    pub fn layout_summary(&self) -> ResumeVirtualLayoutSummary {
        match self {
            VirtualInfo::VirtualObj {
                descr,
                type_id,
                known_class,
                fields,
                fielddescrs,
                descr_size,
            } => ResumeVirtualLayoutSummary::Object {
                descr: descr.clone(),
                type_id: *type_id,
                known_class: *known_class,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.layout_summary()))
                    .collect(),
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            VirtualInfo::VStruct {
                typedescr,
                type_id,
                fields,
                fielddescrs,
                descr_size,
            } => ResumeVirtualLayoutSummary::Struct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.layout_summary()))
                    .collect(),
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            VirtualInfo::VArray {
                arraydescr,
                clear,
                items,
            } => ResumeVirtualLayoutSummary::Array {
                arraydescr: arraydescr.clone(),
                clear: *clear,
                items: items.iter().map(|source| source.layout_summary()).collect(),
            },
            VirtualInfo::VArrayStruct {
                arraydescr,
                fielddescrs,
                element_fields,
            } => ResumeVirtualLayoutSummary::ArrayStruct {
                arraydescr: arraydescr.clone(),
                fielddescrs: fielddescrs.clone(),
                element_fields: element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_descr, source)| (*field_descr, source.layout_summary()))
                            .collect()
                    })
                    .collect(),
            },
            VirtualInfo::VRawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => ResumeVirtualLayoutSummary::RawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: values.iter().map(|src| src.layout_summary()).collect(),
            },
            VirtualInfo::VRawSlice { offset, parent } => ResumeVirtualLayoutSummary::RawSlice {
                offset: *offset,
                parent: parent.layout_summary(),
            },
            VirtualInfo::VStrPlain { chars } => ResumeVirtualLayoutSummary::StrPlain {
                chars: chars.iter().map(|src| src.layout_summary()).collect(),
            },
            VirtualInfo::VStrConcat { left, right } => ResumeVirtualLayoutSummary::StrConcat {
                left: left.layout_summary(),
                right: right.layout_summary(),
            },
            VirtualInfo::VStrSlice {
                source,
                start,
                length,
            } => ResumeVirtualLayoutSummary::StrSlice {
                source: source.layout_summary(),
                start: start.layout_summary(),
                length: length.layout_summary(),
            },
            VirtualInfo::VUniPlain { chars } => ResumeVirtualLayoutSummary::UniPlain {
                chars: chars.iter().map(|src| src.layout_summary()).collect(),
            },
            VirtualInfo::VUniConcat { left, right } => ResumeVirtualLayoutSummary::UniConcat {
                left: left.layout_summary(),
                right: right.layout_summary(),
            },
            VirtualInfo::VUniSlice {
                source,
                start,
                length,
            } => ResumeVirtualLayoutSummary::UniSlice {
                source: source.layout_summary(),
                start: start.layout_summary(),
                length: length.layout_summary(),
            },
            VirtualInfo::Empty => ResumeVirtualLayoutSummary::Hole,
        }
    }
}

/// Complete resume data for a guard exit point.
///
/// Moved here from `majit-metainterp::resume` (Phase C-1 cascade).
/// All field types (`ResumeValueSource`, `FrameInfo`, `VirtualInfo`,
/// `PendingFieldInfo`) live in `majit-backend`.  Inherent impl
/// methods are provided by the `ResumeDataExt` trait in
/// `majit-metainterp::resume` (cross-crate orphan rule prevents
/// inherent impl outside this crate).
#[derive(Debug, Clone, PartialEq)]
pub struct ResumeData {
    /// resume.py: snapshot_iter.vable_array / virtualizable_boxes
    pub vable_array: Vec<ResumeValueSource>,
    /// resume.py: snapshot_iter.vref_array / virtualref_boxes
    pub vref_array: Vec<ResumeValueSource>,
    /// Stack of frames, outermost first.
    /// For a simple non-inlined trace, this has exactly one entry.
    pub frames: Vec<FrameInfo>,
    /// Virtual object descriptions for virtualized state.
    /// Each entry maps a fail_arg position to a virtual object that needs
    /// to be materialized when resuming.
    pub virtuals: Vec<VirtualInfo>,
    /// Deferred heap writes that must be replayed when resuming.
    ///
    /// Mirrors RPython's `rd_pendingfields`, which applies writes after
    /// virtuals and boxes have been reconstructed.
    pub pending_fields: Vec<PendingFieldInfo>,
}

/// Deferred heap write to replay during resume.
///
/// `resume.py PENDINGFIELDSTRUCT` parity — carries the live
/// `lldescr` Arc.  Identity via `Arc::ptr_eq` (`history.py:125`).
#[derive(Debug, Clone)]
pub struct PendingFieldInfo {
    /// `resume.py:88 lldescr` — the field/array descriptor itself.
    pub descr: Option<DescrRef>,
    /// Source of the object/array pointer to update.
    pub target: ResumeValueSource,
    /// Source of the value to write.
    pub value: ResumeValueSource,
    /// Array item index. `None` means a plain field write.
    pub item_index: Option<usize>,
}

impl PartialEq for PendingFieldInfo {
    fn eq(&self, other: &Self) -> bool {
        // `history.py:125 id(descr)` parity: descr identity via Arc::ptr_eq.
        opt_descr_arc_ptr_eq(&self.descr, &other.descr)
            && self.target == other.target
            && self.value == other.value
            && self.item_index == other.item_index
    }
}

impl PendingFieldInfo {
    pub fn layout_summary(&self) -> PendingFieldLayoutSummary {
        PendingFieldLayoutSummary {
            descr: self.descr.clone(),
            item_index: self.item_index,
            is_array_item: self.item_index.is_some(),
            target_kind: self.target.kind(),
            value_kind: self.value.kind(),
            target: self.target.layout_summary(),
            value: self.value.layout_summary(),
        }
    }
}
