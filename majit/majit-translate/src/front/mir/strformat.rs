//! `fmt::Arguments` chain collapse (`rstr.py` string formatting).

use super::*;

/// Decode the packed format-string template that `format_args!` lowers
/// its `&[&str]` pieces into.  The pieces argument to `fmt::Arguments::new`
/// is a `[u8; N]` byte buffer (charon lowers it as an `Array` of `U8`
/// constant elements); this reconstructs the literal `pieces` and counts
/// the argument placeholders.
///
/// Grammar (verified against the real LLBC for several handler graphs):
/// - literal segment: a length byte `L` with `L < 0x80`, then `L` bytes
///   of UTF-8 text appended to the current piece;
/// - long literal segment: the byte `0x80` followed by a little-endian
///   `u16` length `L`, then `L` bytes of UTF-8 text appended to the
///   current piece — the encoding a literal of 128+ bytes takes (a single
///   length byte tops out at `0x7F`), e.g. the 130-byte deprecation tail
///   in `space_int` / `space_index`;
/// - placeholder: any byte `>= 0xC0`, whose low bits select which optional
///   fields follow it, each little-endian: `0x01` a `u32` flag word, `0x02`
///   a `u16` width, `0x04` a `u16` precision, `0x08` a `u16` argument index.
///   Without `0x08` the placeholder is sequential and takes the next
///   argument in order.  A bare `0xC0` is therefore the plain `{}` and
///   `0xC8` + `u16` the explicit `{0}`, but they are two points in one
///   grammar, not two tokens: every field length is decoded so an embedded
///   zero byte is never mistaken for the terminator.  Bits `0x10` / `0x20`
///   mark an indirect width / precision (`{:1$}`), whose encoding this
///   decoder does not know, so it bails on them.  The Display-vs-Debug
///   choice lives in the parallel args array, not in this template;
/// - terminator: a `0x00` byte (only at a segment boundary; `0`/`0xC0`
///   bytes inside a literal are consumed by its length prefix).
///
/// Returns `(pieces, placeholders)` where each placeholder records its
/// argument index plus the static formatting fields encoded in the template.
/// `FmtPlaceholder::is_default()` is the old `0xC0`/`0xC8` subset; retaining
/// the fields lets a later parity rewrite recognize the one non-default shape
/// used by `bytearray_repr_string` without treating arbitrary Rust formatting
/// as RPython string operations.
///
/// `placeholders[*].arg_index` holds the argument
/// index each placeholder renders (one per placeholder, so `pieces.len() ==
/// placeholders.len() + 1`).  A sequential `0xC0` auto-increments; a reused
/// `0xC8` index points back at an already-seen argument, so the distinct
/// argument count is `max(arg_indices) + 1`, which can be less than the
/// placeholder count.  Returns `None` (bail, leaving the graph untouched)
/// on `0x81..=0xBF` (neither a literal length nor a placeholder), on an
/// indirect width/precision bit, on a truncated field, on a missing `0x00`
/// terminator, and on non-UTF-8 literal bytes.  A placeholder that carries
/// flags/width/precision is decoded rather than refused; refusing it is the
/// job of `FmtPlaceholder::is_default()`, which every collapse consults.
pub(in crate::front::mir) fn decode_packed_format_pieces(
    bytes: &[u8],
) -> Option<(Vec<String>, Vec<FmtPlaceholder>)> {
    let mut pieces: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut placeholders: Vec<FmtPlaceholder> = Vec::new();
    let mut auto = 0usize; // next sequential argument index
    let mut i = 0;
    let mut terminated = false;
    while i < bytes.len() {
        let b = bytes[i];
        if b == 0x00 {
            // terminator — must be the final byte
            if i + 1 != bytes.len() {
                return None;
            }
            terminated = true;
            break;
        } else if b >= 0xC0 {
            // `core::fmt::Arguments`' packed placeholder grammar (the pinned
            // Rust core `fmt/mod.rs`): low bits select optional u32 flags,
            // u16 width, u16 precision, and u16 argument-index fields; bits
            // 4/5 mark indirect width/precision.  Decode all field lengths so
            // embedded zero bytes are never mistaken for the terminator.
            pieces.push(std::mem::take(&mut current));
            i += 1;
            let read_u16 = |at: &mut usize| -> Option<u16> {
                let lo = *bytes.get(*at)?;
                let hi = *bytes.get(*at + 1)?;
                *at += 2;
                Some(u16::from_le_bytes([lo, hi]))
            };
            let read_u32 = |at: &mut usize| -> Option<u32> {
                let b0 = *bytes.get(*at)?;
                let b1 = *bytes.get(*at + 1)?;
                let b2 = *bytes.get(*at + 2)?;
                let b3 = *bytes.get(*at + 3)?;
                *at += 4;
                Some(u32::from_le_bytes([b0, b1, b2, b3]))
            };
            let flags = if b & 0x01 != 0 {
                Some(read_u32(&mut i)?)
            } else {
                None
            };
            let width = if b & 0x02 != 0 {
                Some(read_u16(&mut i)?)
            } else {
                None
            };
            let precision = if b & 0x04 != 0 {
                Some(read_u16(&mut i)?)
            } else {
                None
            };
            let arg_index = if b & 0x08 != 0 {
                usize::from(read_u16(&mut i)?)
            } else {
                let index = auto;
                auto += 1;
                index
            };
            // An indirect width/precision (`{:1$}`) names an argument for
            // the field instead of inlining it, and this decoder does not
            // know how many bytes that costs.  Guessing would desync the
            // byte stream, and a desync that happened to land on one default
            // placeholder would be accepted by the collapses downstream —
            // so refuse the template outright.  No consumer accepts a
            // non-default placeholder anyway.
            if b & 0x30 != 0 {
                return None;
            }
            placeholders.push(FmtPlaceholder {
                arg_index,
                flags,
                width,
                precision,
                width_indirect: false,
                precision_indirect: false,
            });
        } else if b == 0x80 {
            // long literal segment: `0x80` + little-endian u16 length,
            // then that many UTF-8 bytes (a 128+ byte literal that the
            // single length byte below cannot express).
            let lo = *bytes.get(i + 1)? as usize;
            let hi = *bytes.get(i + 2)? as usize;
            let len = lo | (hi << 8);
            let start = i + 3;
            let end = start.checked_add(len)?;
            let seg = bytes.get(start..end)?;
            current.push_str(std::str::from_utf8(seg).ok()?);
            i = end;
        } else if b < 0x80 {
            // literal segment of length `b`
            let start = i + 1;
            let end = start.checked_add(b as usize)?;
            let seg = bytes.get(start..end)?;
            current.push_str(std::str::from_utf8(seg).ok()?);
            i = end;
        } else {
            // 0x81..0xBF are neither literal lengths nor placeholders.
            return None;
        }
    }
    // The grammar requires the `0x00` terminator at a segment boundary;
    // running out of bytes without it is a truncated or wrong array, not
    // a valid template — bail rather than accept it once wired.
    if !terminated {
        return None;
    }
    pieces.push(current);
    Some((pieces, placeholders))
}

/// Static formatting fields carried by one packed `fmt::Arguments`
/// placeholder.  This mirrors the storage shape documented in the pinned
/// Rust core; it is a Vec entry in template order, not a side-table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::front::mir) struct FmtPlaceholder {
    pub(in crate::front::mir) arg_index: usize,
    pub(in crate::front::mir) flags: Option<u32>,
    pub(in crate::front::mir) width: Option<u16>,
    pub(in crate::front::mir) precision: Option<u16>,
    pub(in crate::front::mir) width_indirect: bool,
    pub(in crate::front::mir) precision_indirect: bool,
}

impl FmtPlaceholder {
    pub(in crate::front::mir) fn is_default(self) -> bool {
        self.flags.is_none()
            && self.width.is_none()
            && self.precision.is_none()
            && !self.width_indirect
            && !self.precision_indirect
    }

    /// Rust `{:02x}` as emitted for the `u8` in
    /// `display::bytearray_repr_string`: space fill + unknown alignment,
    /// sign-aware zero padding, width present and equal to two.
    pub(in crate::front::mir) fn is_lower_hex_byte_02(self) -> bool {
        const FILL_SPACE: u32 = b' ' as u32;
        const ZERO_PAD: u32 = 1 << 24;
        const WIDTH_PRESENT: u32 = 1 << 27;
        const ALIGN_UNKNOWN: u32 = 3 << 29;
        self.flags == Some(FILL_SPACE | ZERO_PAD | WIDTH_PRESENT | ALIGN_UNKNOWN)
            && self.width == Some(2)
            && self.precision.is_none()
            && !self.width_indirect
            && !self.precision_indirect
    }
}

/// Read an `Array` aggregate literal: given the Variable holding a
/// `SyntheticTransparentCtor { name: "Array<T;N>" }` result, collect the
/// values written to its `__pos_0..__pos_{n-1}` fields in index order.
/// The ctor and its element `FieldWrite`s are emitted into one block by
/// the `Rvalue::Aggregate` array lowering, so the search is block-local
/// once the ctor is found. Returns `None` if `array_var` is not produced
/// by an `Array` ctor, or its `__pos_i` writes are not the contiguous
/// range `0..n`.
pub(in crate::front::mir) fn read_array_literal_elements(
    graph: &FunctionGraph,
    array_var: &crate::flowspace::model::Variable,
) -> Option<Vec<crate::flowspace::model::Variable>> {
    use crate::model::{CallTarget, OpKind};
    let (block_id, ctor_idx) = resolve_to_producer_op(graph, array_var)?;
    let block = graph.blocks.iter().find(|b| b.id == block_id)?;
    match &block.operations.get(ctor_idx)?.kind {
        OpKind::Call {
            target: CallTarget::SyntheticTransparentCtor { name, .. },
            ..
        } if name == "Array" || majit_ir::descr::is_shaped_array_name(name) => {}
        _ => return None,
    }
    let mut by_index: Vec<(usize, crate::flowspace::model::Variable)> = Vec::new();
    for op in &block.operations {
        if let OpKind::FieldWrite {
            base, field, value, ..
        } = &op.kind
            && base.id() == array_var.id()
        {
            let idx = field.name.strip_prefix("__pos_")?.parse::<usize>().ok()?;
            // A `setfield_gc` inline `Const` element carries no SSA
            // Variable; the recognizer needs a register operand per slot.
            let write_var = value.as_variable()?;
            by_index.push((idx, write_var.clone()));
        }
    }
    by_index.sort_by_key(|(i, _)| *i);
    for (expected, (idx, _)) in by_index.iter().enumerate() {
        if *idx != expected {
            return None;
        }
    }
    Some(by_index.into_iter().map(|(_, v)| v).collect())
}

/// Resolve a Variable to the `i64` value of the integer constant op that
/// produces it, following cross-block links via
/// [`resolve_to_producer_op`].  Packed-format pieces are `[u8; N]`, so the
/// ordinary lowering now preserves each byte as `ConstUInt`; older/synthetic
/// fixtures may still spell the same in-range byte as `ConstInt`.  Accept both
/// at this byte-decoding boundary without changing either SSA value's unsigned
/// annotation.
pub(in crate::front::mir) fn resolve_const_int(
    graph: &FunctionGraph,
    var: &crate::flowspace::model::Variable,
) -> Option<i64> {
    use crate::model::OpKind;
    let (block_id, idx) = resolve_to_producer_op(graph, var)?;
    let block = graph.blocks.iter().find(|b| b.id == block_id)?;
    match &block.operations.get(idx)?.kind {
        OpKind::ConstInt(n) => Some(*n),
        OpKind::ConstUInt(n) => i64::try_from(*n).ok(),
        _ => None,
    }
}

/// Whether a `Display` (`{}`) or `Debug` (`{:?}`) placeholder rendered an
/// argument. Carried by the `fmt::rt::Argument::new_display` /
/// `new_debug` constructor in the parallel args array (the packed pieces
/// template does not encode the choice).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::front::mir) enum FmtArgKind {
    Display,
    Debug,
    LowerHex,
}

/// One placeholder argument recovered from a `format_args!` chain: the
/// value Variable the placeholder renders and its Display/Debug flavour.
#[derive(Debug, Clone)]
pub(in crate::front::mir) struct FmtArg {
    pub(in crate::front::mir) value: crate::flowspace::model::Variable,
    pub(in crate::front::mir) kind: FmtArgKind,
}

/// The decoded contents of a `format_args!` chain — the literal string
/// pieces interleaved with the placeholder arguments.  `args` holds one
/// entry per *distinct* argument (the args-array elements), while
/// `arg_indices` holds the argument index each *placeholder* renders in
/// template order (`pieces.len() == arg_indices.len() + 1`).  A `{0}…{0}`
/// template that reuses one argument has `arg_indices == [0, …, 0]` with a
/// single `args` entry, so `arg_indices.len()` (placeholders) can exceed
/// `args.len()` (distinct arguments).
#[derive(Debug, Clone)]
pub(in crate::front::mir) struct FmtChain {
    pub(in crate::front::mir) pieces: Vec<String>,
    pub(in crate::front::mir) args: Vec<FmtArg>,
    pub(in crate::front::mir) placeholders: Vec<FmtPlaceholder>,
}

/// Match a `FunctionPath`'s trailing segments against `tail`, so a
/// crate-qualified spelling (`core::fmt::Arguments::new`) and the
/// crate-stripped front-end spelling (`fmt::Arguments::new`) both
/// resolve.  The leaf is compared through its `__spec_` marker: a
/// specialized copy is the same function.  A trait-impl identity
/// segment ([`is_trait_impl_identity_segment`]) is skipped, so
/// `["vec","Vec","<impl Index<…>>","index"]` still matches
/// `["vec","Vec","index"]` (`Bookkeeper.getdesc`).
pub(in crate::front::mir) fn fmt_path_ends_with(segments: &[String], tail: &[&str]) -> bool {
    let mut si = segments.len();
    let mut ti = tail.len();
    while ti > 0 {
        if si == 0 {
            return false;
        }
        si -= 1;
        if is_trait_impl_identity_segment(&segments[si]) {
            continue;
        }
        let s = segments[si].as_str();
        let t = tail[ti - 1];
        let matched = if ti == tail.len() {
            crate::front::clause_spec::unspecialized_leaf(s) == t
        } else {
            s == t
        };
        if !matched {
            return false;
        }
        ti -= 1;
    }
    true
}

pub(in crate::front::mir) fn is_trait_impl_path_segment(segment: &str) -> bool {
    segment == "<Impl>" || segment.starts_with("<impl ") || segment.starts_with("<Impl#")
}

/// True when `segment` is `<impl {T}>` and `{T}` with its generic argument
/// list removed (cut at the first `<`) split on `::` equals `trait_path`.
/// `CallControl.graphs_from` follows the function object, so a suffix match
/// would treat `<impl application::cmp::PartialEq<…>>` as `core::cmp::PartialEq`.
pub(in crate::front::mir) fn trait_impl_identity_names_trait(
    segment: &str,
    trait_path: &[&str],
) -> bool {
    let Some(inner) = segment
        .strip_prefix("<impl ")
        .and_then(|s| s.strip_suffix('>'))
    else {
        return false;
    };
    let trait_name = inner.split_once('<').map(|(head, _)| head).unwrap_or(inner);
    trait_name.split("::").eq(trait_path.iter().copied())
}

/// `for_impl_method` (`[..., Owner, leaf]`) or a trait-impl identity
/// (`[..., Owner, <impl {T}>, leaf]`) whose identity segment names
/// `trait_path`. `spec_segments` emits the latter for an ADT-owned trait
/// method (`CallControl.graphs_from` keys on the function object).
pub(in crate::front::mir) fn fmt_path_ends_with_owner_method(
    segments: &[String],
    owner: &[&str],
    trait_path: &[&str],
    leaf: &str,
) -> bool {
    let n = segments.len();
    if n == 0 || crate::front::clause_spec::unspecialized_leaf(&segments[n - 1]) != leaf {
        return false;
    }
    let owner_end = if n >= 2 && is_trait_impl_path_segment(&segments[n - 2]) {
        if !trait_impl_identity_names_trait(&segments[n - 2], trait_path) {
            return false;
        }
        n - 2
    } else {
        n - 1
    };
    if owner_end < owner.len() {
        return false;
    }
    segments[owner_end - owner.len()..owner_end]
        .iter()
        .map(String::as_str)
        .eq(owner.iter().copied())
}

/// The `fmt::Arguments::new(pieces, args)` constructor that `format_args!`
/// builds from the on-stack pieces+args arrays.
pub(in crate::front::mir) fn is_arguments_new_path(segments: &[String]) -> bool {
    fmt_path_ends_with(segments, &["Arguments", "new"])
}

/// Classify a `fmt::rt::Argument::new_display` / `new_debug` constructor
/// path. Any other path (positional/named/width-bearing argument ctor) is
/// not in the recognized subset.
pub(in crate::front::mir) fn fmt_argument_ctor_kind(segments: &[String]) -> Option<FmtArgKind> {
    if fmt_path_ends_with(segments, &["Argument", "new_display"]) {
        Some(FmtArgKind::Display)
    } else if fmt_path_ends_with(segments, &["Argument", "new_debug"]) {
        Some(FmtArgKind::Debug)
    } else if fmt_path_ends_with(segments, &["Argument", "new_lower_hex"]) {
        Some(FmtArgKind::LowerHex)
    } else {
        None
    }
}

/// Unwrap the `format_args!` argument tuple-ref. Each Display/Debug
/// argument reaches `Argument::new_display(&v)` as a `FieldRead` off the
/// on-stack argument `Tuple` aggregate (`&(v,).0`); follow it back to the
/// value written into that tuple field. Returns `var` unchanged when it
/// is not produced by a `FieldRead` (value passed directly), and `None`
/// when the tuple field has conflicting writers.
pub(in crate::front::mir) fn unwrap_fmt_arg_tuple_ref(
    graph: &FunctionGraph,
    var: &crate::flowspace::model::Variable,
) -> Option<crate::flowspace::model::Variable> {
    use crate::model::OpKind;
    let (block_id, idx) = resolve_to_producer_op(graph, var)?;
    let block = graph.blocks.iter().find(|b| b.id == block_id)?;
    let (base, field_name) = match &block.operations.get(idx)?.kind {
        OpKind::FieldRead { base, field, .. } => (base.clone(), field.name.clone()),
        _ => return Some(var.clone()),
    };
    // The aggregate `FieldWrite`s live in the block that built the tuple,
    // but a `FieldRead` for the 2nd+ placeholder reads a *threaded copy*
    // of the tuple ref (forwarded across a block boundary as an inputarg),
    // so its `base` Variable id differs from the write `base`'s id even
    // though both denote the same tuple ctor.  Resolve the read base to
    // its producing op so the write base can be matched by producer
    // identity rather than raw Variable id.
    let base_producer = resolve_to_producer_op(graph, &base);
    let mut found: Option<crate::flowspace::model::Variable> = None;
    for b in &graph.blocks {
        for op in &b.operations {
            if let OpKind::FieldWrite {
                base: write_base,
                field,
                value,
                ..
            } = &op.kind
            {
                let base_matches = write_base.id() == base.id()
                    || (base_producer.is_some()
                        && resolve_to_producer_op(graph, write_base) == base_producer);
                if base_matches && field.name == field_name {
                    // A `setfield_gc` inline `Const` carries no SSA Variable;
                    // the back-trace needs a register operand to follow.
                    let write_var = value.as_variable()?;
                    if found.as_ref().is_some_and(|f| f.id() != write_var.id()) {
                        return None; // ambiguous: distinct values written
                    }
                    found = Some(write_var.clone());
                }
            }
        }
    }
    found
}

/// Recover one placeholder argument from an args-array element: the
/// element is the result of a `fmt::rt::Argument::new_display` /
/// `new_debug` constructor, whose sole argument back-traces (through the
/// tuple-ref wrap) to the rendered value.
pub(in crate::front::mir) fn extract_fmt_arg(
    graph: &FunctionGraph,
    arg_elem: &crate::flowspace::model::Variable,
) -> Option<FmtArg> {
    use crate::model::{CallTarget, OpKind};
    let (block_id, idx) = resolve_to_producer_op(graph, arg_elem)?;
    let block = graph.blocks.iter().find(|b| b.id == block_id)?;
    let (kind, inner) = match &block.operations.get(idx)?.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } => (fmt_argument_ctor_kind(segments)?, args.first()?.clone()),
        _ => return None,
    };
    let value = unwrap_fmt_arg_tuple_ref(graph, &inner)?;
    Some(FmtArg { value, kind })
}

/// Back-trace a recognized `format_args!` chain from the Variable passed
/// to `alloc::fmt::format(args)` (the String producer) to the literal
/// pieces and the Display/Debug argument values. Composes the staged
/// primitives:
///   fmt-args → [`resolve_to_producer_op`] → `fmt::Arguments::new(pieces, args)`
///     · pieces: [`read_array_literal_elements`] → [`resolve_const_int`] per
///       byte → [`decode_packed_format_pieces`]
///     · args: [`read_array_literal_elements`] → per element
///       [`extract_fmt_arg`]
/// Returns `None` (the recognizer leaves the graph untouched) on any
/// shape outside the recognized subset, so it never fires on an
/// unsupported chain.
pub(in crate::front::mir) fn extract_fmt_chain(
    graph: &FunctionGraph,
    fmt_args_var: &crate::flowspace::model::Variable,
) -> Option<FmtChain> {
    use crate::model::{CallTarget, OpKind};
    // The format(args) argument is the result of `Arguments::new(pieces, args)`.
    let (block_id, idx) = resolve_to_producer_op(graph, fmt_args_var)?;
    let block = graph.blocks.iter().find(|b| b.id == block_id)?;
    let (pieces_var, args_var) = match &block.operations.get(idx)?.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if is_arguments_new_path(segments) => (args.first()?.clone(), args.get(1)?.clone()),
        _ => return None,
    };
    // Pieces: an `Array` of `ConstInt` bytes → decode the packed template.
    let piece_byte_vars = read_array_literal_elements(graph, &pieces_var)?;
    let mut bytes = Vec::with_capacity(piece_byte_vars.len());
    for v in &piece_byte_vars {
        bytes.push(u8::try_from(resolve_const_int(graph, v)?).ok()?);
    }
    let (pieces, placeholders) = decode_packed_format_pieces(&bytes)?;
    // Args: an `Array` of `Argument::new_display|new_debug(&v)` ctors, one
    // per *distinct* argument.  The args array holds the distinct arguments;
    // reused placeholders (`0xC8`) point back at an earlier index, so the
    // distinct count is `max(arg_indices) + 1` — bail unless it matches the
    // args array, and every placeholder index must be in range.
    let arg_elems = read_array_literal_elements(graph, &args_var)?;
    let distinct = placeholders
        .iter()
        .map(|placeholder| placeholder.arg_index + 1)
        .max()
        .unwrap_or(0);
    if arg_elems.len() != distinct {
        return None;
    }
    if placeholders
        .iter()
        .any(|placeholder| placeholder.arg_index >= arg_elems.len())
    {
        return None;
    }
    let mut args = Vec::with_capacity(arg_elems.len());
    for elem in &arg_elems {
        args.push(extract_fmt_arg(graph, elem)?);
    }
    Some(FmtChain {
        pieces,
        args,
        placeholders,
    })
}

/// Mark `var` as rstr `STR` (`Ptr(STR)` / `rpy_string`). `__str_const`,
/// `str` (`ll_str`) and `add` (`ll_strconcat`) produce that struct, not a
/// Rust-heap `String`.
pub(in crate::front::mir) fn stamp_as_rstr_str(var: &Variable) {
    use crate::translator::rtyper::lltypesystem::rstr::STRPTR;
    FunctionGraph::set_concretetype_of_inline(var, crate::model::ConcreteType::GcRef);
    var.set_concretetype(Some(STRPTR.clone()));
}

/// Shim / window data word is `CCHARP` (`rffi.py` `CCHARP`). The result
/// bank stays `ValueType::Int` (`SomePtr` is not a `ValueType`).
pub(in crate::front::mir) fn stamp_as_ccharp(var: &Variable) {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    var.set_concretetype(Some(lltype_for_direct_ptradd_pointer(
        None,
        &LowLevelType::Char,
    )));
}

/// `Variable.copy` (`flowspace/model.py`) keeps `concretetype`.
pub(in crate::front::mir) fn copy_concretetype(from: &Variable, to: &Variable) {
    if let Some(ct) = from.concretetype() {
        to.set_concretetype(Some(ct));
    }
}

/// `rutf8.unichr_as_utf8(code, allow_surrogates=True)` for a Unicode
/// scalar constant.  Surrogates are left to the runtime helper (Rust
/// `String` cannot hold them).
pub(in crate::front::mir) fn unichr_as_utf8_text(code: i64) -> Option<String> {
    let code = u32::try_from(code).ok()?;
    if code > 0x10FFFF {
        return None;
    }
    char::from_u32(code).map(String::from)
}

/// Emit a `__str_const` constant of `text` into `bb_id` and return its
/// Variable — the same synthetic `Call(["__str_const", text])` shape the
/// constant lowering uses (see [`Lowering::emit_constant`]); the flowspace
/// adapter pre-folds it to the upstream string `Constant` and types it as
/// a String.
pub(in crate::front::mir) fn emit_str_const(
    graph: &mut FunctionGraph,
    bb_id: BlockId,
    text: &str,
) -> Variable {
    use crate::model::{CallTarget, OpKind, ValueType};
    let var = graph.alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
    stamp_as_rstr_str(&var);
    graph.block_mut(bb_id).operations.push(SpaceOperation {
        result: Some(var.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["__str_const".to_string(), text.to_string()],
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Str,
        },
    });
    var
}

/// Emit a string concatenation `lhs + rhs` into `bb_id` and return its
/// Variable. The `"add"` opname passes through the flowspace adapter
/// unchanged; when both operands type as String the rtyper routes it to
/// `pair(StringRepr, StringRepr).rtype_add` → `direct_call(ll_strconcat)`.
pub(in crate::front::mir) fn emit_str_add(
    graph: &mut FunctionGraph,
    bb_id: BlockId,
    lhs: &Variable,
    rhs: &Variable,
) -> Variable {
    use crate::model::{OpKind, ValueType};
    let var = graph.alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
    stamp_as_rstr_str(&var);
    graph.block_mut(bb_id).operations.push(SpaceOperation {
        result: Some(var.clone()),
        kind: OpKind::BinOp {
            op: "add".to_string(),
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            result_ty: ValueType::Str,
        },
    });
    var
}

/// Emit the StringRepr-add left fold that materializes a recognized
/// `format_args!` chain as a runtime String:
///
/// ```text
/// acc = strconst(pieces[0])
/// for each arg i:
///     acc = acc + args[i].value          // render(&str) == identity
///     acc = acc + strconst(pieces[i+1])
/// ```
///
/// Returns the final String Variable. Each literal piece becomes a
/// `__str_const` call and each concatenation a `BinOp("add")` the rtyper
/// lowers through `ll_strconcat`. The caller must have verified every
/// argument renders by identity (a `&str` `Display`); `Debug` rendering
/// and non-`&str` `Display` are outside the recognized subset and are
/// rejected before emission.
pub(in crate::front::mir) fn emit_fmt_concat(
    graph: &mut FunctionGraph,
    bb_id: BlockId,
    chain: &FmtChain,
) -> Variable {
    let mut acc = emit_str_const(graph, bb_id, &chain.pieces[0]);
    // Walk placeholders in template order via `arg_indices`; a reused
    // argument re-renders the same recovered value (`str(&str)` is pure, so
    // referencing the value var twice is sound).
    for (i, placeholder) in chain.placeholders.iter().enumerate() {
        let ai = placeholder.arg_index;
        acc = emit_str_add(graph, bb_id, &acc, &chain.args[ai].value);
        let next_piece = emit_str_const(graph, bb_id, &chain.pieces[i + 1]);
        acc = emit_str_add(graph, bb_id, &acc, &next_piece);
    }
    acc
}

pub(in crate::front::mir) fn value_type_is_rust_heap_string(ty: &crate::model::ValueType) -> bool {
    matches!(ty, crate::model::ValueType::Ref(Some(root)) if is_rust_heap_string_root(root))
}

pub(in crate::front::mir) fn producer_is_rust_heap_string(
    graph: &FunctionGraph,
    var: &crate::flowspace::model::Variable,
) -> bool {
    use crate::model::OpKind;
    let Some((bid, idx)) = resolve_to_producer_op(graph, var) else {
        return false;
    };
    let Some(block) = graph.blocks.iter().find(|b| b.id == bid) else {
        return false;
    };
    let Some(op) = block.operations.get(idx) else {
        return false;
    };
    match &op.kind {
        OpKind::Input { ty, class_root, .. } => {
            value_type_is_rust_heap_string(ty)
                || class_root.as_deref().is_some_and(is_rust_heap_string_root)
        }
        OpKind::UnaryOp { result_ty, .. }
        | OpKind::BinOp { result_ty, .. }
        | OpKind::Call { result_ty, .. } => value_type_is_rust_heap_string(result_ty),
        OpKind::FieldRead { ty, .. } => value_type_is_rust_heap_string(ty),
        _ => false,
    }
}

/// Whether `var` is a rust-heap `String` (`{cap, ptr, len}`), including a
/// linear SSA chain and a phi whose incoming values are that header.
pub(crate) fn interpolant_is_rust_heap_string(
    graph: &FunctionGraph,
    var: &crate::flowspace::model::Variable,
) -> bool {
    use crate::model::LinkArg;
    if producer_is_rust_heap_string(graph, var) {
        return true;
    }
    let Some((owner, pos)) = graph.blocks.iter().find_map(|block| {
        block
            .inputargs
            .iter()
            .position(|a| a.id() == var.id())
            .map(|pos| (block.id, pos))
    }) else {
        return false;
    };
    graph
        .blocks
        .iter()
        .flat_map(|b| b.exits.iter())
        .filter(|link| link.target == owner)
        .any(|link| match link.args.get(pos) {
            Some(LinkArg::Value(v)) => producer_is_rust_heap_string(graph, v),
            _ => false,
        })
}

/// True when a `ToString::to_string` receiver is a float kind, looking
/// through `&`/`&mut`. `tyref_to_value_type_with` yields Ref for `&f64`;
/// [`tyref_deref_value_type_with`] peels that borrow. `FloatRepr.ll_str`
/// (`rfloat.py`) / `float_repr` (`floatobject.py`) residualizes through
/// [`crate::front::rfloat_call::repr_residual_path`].
pub(in crate::front::mir) fn to_string_receiver_is_float(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> bool {
    matches!(
        tyref_deref_value_type_with(
            ty,
            llbc,
            tombstoned,
            gc_struct_ids,
            raw_alloc_flavor_structs
        ),
        crate::model::ValueType::Float | crate::model::ValueType::SingleFloat
    )
}

pub(in crate::front::mir) fn str_render_kind(
    operand: &crate::flowspace::model::Variable,
) -> crate::model::OpKind {
    use crate::model::{OpKind, ValueType};
    // `Repr.rtype_str`: int → `ll_int2dec`, rstr `STR` → identity,
    // rust-heap `String` → copy into `STR`. The front end emits one
    // `UnaryOp str`; jtransform residualizes the rust-heap case through
    // `GraphTransformConfig::rust_string_str_helper`, the same host-helper
    // contract as `int_str_helper`.
    OpKind::UnaryOp {
        op: "str".to_string(),
        operand: operand.clone(),
        result_ty: ValueType::Str,
    }
}

pub(in crate::front::mir) fn str_render_op(
    operand: &crate::flowspace::model::Variable,
    result: crate::flowspace::model::Variable,
) -> SpaceOperation {
    SpaceOperation {
        result: Some(result),
        kind: str_render_kind(operand),
    }
}

/// Build the orthodox string-build expansion for a recognized
/// single-argument `format!("{pre}{}{post}", value)` chain: the literal
/// pieces become `__str_const`s, the `{}` placeholder renders its value
/// with `str(value)` (`OpKind::UnaryOp { op: "str" }` → the rtyper's
/// `ll_str`), and the parts fold left through `BinOp("add")` (→
/// `ll_strconcat`).  A default-spec `{:x}` interpolant (`hex`) is
/// `rstr.py do_stringformat` over `'%x' % r_uint(n)`: `BinOp mod` of
/// `__str_const("%x")` and `r_uint(n)`, skipping the cast when the
/// value is already Unsigned.  A signed interpolant whose tuple width is
/// under 64 bits is then `and`-masked to that width (`int_to_int_cast`
/// `And`).  Empty literal pieces are skipped.  The
/// final op reuses `result` (the displaced `alloc::fmt::format` result
/// var) so the downstream consumer is untouched.
pub(in crate::front::mir) fn emit_fmt_expansion_ops(
    graph: &mut FunctionGraph,
    pieces: &[String],
    value: &Variable,
    result: Variable,
    hex: bool,
    hex_signed_bits: Option<u32>,
) -> Vec<SpaceOperation> {
    use crate::model::{CallTarget, OpKind, ValueType};
    let already_unsigned = hex && fmt_value_is_unsigned(graph, value);
    let str_const = |graph: &mut FunctionGraph, ops: &mut Vec<SpaceOperation>, text: &str| {
        let v = graph.alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
        stamp_as_rstr_str(&v);
        ops.push(SpaceOperation {
            result: Some(v.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__str_const".to_string(), text.to_string()],
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![]),
                result_ty: ValueType::Str,
            },
        });
        v
    };
    let mut ops: Vec<SpaceOperation> = Vec::new();
    let mut parts: Vec<Variable> = Vec::new();
    if !pieces[0].is_empty() {
        parts.push(str_const(graph, &mut ops, &pieces[0]));
    }
    let rendered = graph.alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
    stamp_as_rstr_str(&rendered);
    if hex {
        // `rstr.py do_stringformat` `'x'` → `ll_str.py ll_int2hex(i, False)`
        // over `'%x' % r_uint(n)`.
        let fmt = str_const(graph, &mut ops, "%x");
        let rhs = if already_unsigned {
            value.clone()
        } else {
            let masked = graph.alloc_value_var_with_type(crate::model::ConcreteType::Signed);
            ops.push(SpaceOperation {
                result: Some(masked.clone()),
                kind: rarithmetic_mask("r_uint", value.clone(), ValueType::Unsigned),
            });
            if let Some(bits) = hex_signed_bits.filter(|&b| b < 64) {
                let mask_var = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                ops.push(SpaceOperation {
                    result: Some(mask_var.clone()),
                    kind: OpKind::ConstUInt((1u64 << bits) - 1),
                });
                let truncated =
                    graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                ops.push(SpaceOperation {
                    result: Some(truncated.clone()),
                    kind: OpKind::BinOp {
                        op: "and".to_string(),
                        lhs: masked,
                        rhs: mask_var,
                        result_ty: ValueType::Unsigned,
                    },
                });
                truncated
            } else {
                masked
            }
        };
        ops.push(SpaceOperation {
            result: Some(rendered.clone()),
            kind: OpKind::BinOp {
                op: "mod".to_string(),
                lhs: fmt,
                rhs,
                result_ty: ValueType::Str,
            },
        });
    } else {
        // `str(value)` — render the single Display placeholder as `UnaryOp
        // str` (`Repr.rtype_str`: `ll_str` / `ll_int2dec` / rust-heap
        // `String` copy). jtransform residualizes the rust-heap case.
        ops.push(str_render_op(value, rendered.clone()));
    }
    parts.push(rendered);
    if pieces.len() > 1 && !pieces[1].is_empty() {
        parts.push(str_const(graph, &mut ops, &pieces[1]));
    }
    // Left-fold the parts through `add` (`ll_strconcat`).
    let mut acc = parts[0].clone();
    for part in &parts[1..] {
        let sum = graph.alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
        stamp_as_rstr_str(&sum);
        ops.push(SpaceOperation {
            result: Some(sum.clone()),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: acc.clone(),
                rhs: part.clone(),
                result_ty: ValueType::Str,
            },
        });
        acc = sum;
    }
    // Reuse the displaced format result var on the final op so the
    // downstream link still forwards the rendered String.
    stamp_as_rstr_str(&result);
    if let Some(last) = ops.last_mut() {
        last.result = Some(result);
    }
    ops
}

/// Expand a byte-only `{:02x}` placeholder to the nibble lookup PyPy uses in
/// `bytearrayobject.py::descr_repr` (`bytearrayobject.py`):
///
/// ```python
///             elif not '\x20' <= c < '\x7f':
///                 n = ord(c)
///                 buf.append('\\x')
///                 buf.append("0123456789abcdef"[n >> 4])
///                 buf.append("0123456789abcdef"[n & 0xF])
/// ```
///
/// Upstream appends three values into a `StringBuilder`; the Rust shell this
/// replaces is a `format!` that *returns* a `String`, so the same three values
/// are folded left through `add` (`ll_strconcat`) into the displaced format
/// result instead.  `pieces` carries the template's literal halves — for the
/// `format!("\\x{c:02x}")` shell that is `["\\x", ""]`, and the leading `\x`
/// is upstream's first `buf.append`, so it must be emitted, not merely
/// matched.
///
/// The caller proves the formatted value came from a `(u8,)` argument tuple,
/// so two digits are complete (never truncating a wider integer).  Each
/// getitem annotates as `SomeChar`; `str(char)` follows
/// `AbstractCharRepr.ll_str -> ll_chr2str` (`rstr.py`).  No Rust fmt
/// runtime or new helper survives.
pub(in crate::front::mir) fn emit_lower_hex_byte_02_ops(
    graph: &mut FunctionGraph,
    pieces: &[String],
    value: &Variable,
    result: Variable,
) -> Vec<SpaceOperation> {
    use crate::model::{CallTarget, OpKind, ValueType};
    let alloc = |graph: &mut FunctionGraph| {
        graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown)
    };
    let mut ops = Vec::new();

    let digits = alloc(graph);
    stamp_as_rstr_str(&digits);
    ops.push(SpaceOperation {
        result: Some(digits.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["__str_const".to_string(), "0123456789abcdef".to_string()],
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Str,
        },
    });
    let four = alloc(graph);
    ops.push(SpaceOperation {
        result: Some(four.clone()),
        kind: OpKind::ConstInt(4),
    });
    let fifteen = alloc(graph);
    ops.push(SpaceOperation {
        result: Some(fifteen.clone()),
        kind: OpKind::ConstInt(15),
    });
    let hi_index = alloc(graph);
    ops.push(SpaceOperation {
        result: Some(hi_index.clone()),
        kind: OpKind::BinOp {
            op: "rshift".to_string(),
            lhs: value.clone(),
            rhs: four,
            result_ty: ValueType::Int,
        },
    });
    let lo_index = alloc(graph);
    ops.push(SpaceOperation {
        result: Some(lo_index.clone()),
        kind: OpKind::BinOp {
            op: "and".to_string(),
            lhs: value.clone(),
            rhs: fifteen,
            result_ty: ValueType::Int,
        },
    });
    let hi_char = alloc(graph);
    ops.push(SpaceOperation {
        result: Some(hi_char.clone()),
        kind: OpKind::ArrayRead {
            base: digits.clone(),
            index: hi_index,
            item_ty: ValueType::Int,
            array_type_id: None,
            nolength: false,
            gckind: crate::model::ArrayGcKind::Gc,
            pure: true,
        },
    });
    let lo_char = alloc(graph);
    ops.push(SpaceOperation {
        result: Some(lo_char.clone()),
        kind: OpKind::ArrayRead {
            base: digits,
            index: lo_index,
            item_ty: ValueType::Int,
            array_type_id: None,
            nolength: false,
            gckind: crate::model::ArrayGcKind::Gc,
            pure: true,
        },
    });
    let hi = alloc(graph);
    stamp_as_rstr_str(&hi);
    ops.push(SpaceOperation {
        result: Some(hi.clone()),
        kind: OpKind::UnaryOp {
            op: "str".to_string(),
            operand: hi_char,
            result_ty: ValueType::Str,
        },
    });
    let lo = alloc(graph);
    stamp_as_rstr_str(&lo);
    ops.push(SpaceOperation {
        result: Some(lo.clone()),
        kind: OpKind::UnaryOp {
            op: "str".to_string(),
            operand: lo_char,
            result_ty: ValueType::Str,
        },
    });
    // Upstream's `buf.append('\\x')` — the template's leading literal — comes
    // first, then the two nibble characters.  A non-empty trailing piece is
    // appended last; the `["\\x", ""]` shell has none.
    let mut parts: Vec<Variable> = Vec::new();
    if !pieces[0].is_empty() {
        let literal = alloc(graph);
        stamp_as_rstr_str(&literal);
        ops.push(SpaceOperation {
            result: Some(literal.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__str_const".to_string(), pieces[0].clone()],
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![]),
                result_ty: ValueType::Str,
            },
        });
        parts.push(literal);
    }
    parts.push(hi);
    parts.push(lo);
    if pieces.len() > 1 && !pieces[1].is_empty() {
        let literal = alloc(graph);
        stamp_as_rstr_str(&literal);
        ops.push(SpaceOperation {
            result: Some(literal.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__str_const".to_string(), pieces[1].clone()],
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![]),
                result_ty: ValueType::Str,
            },
        });
        parts.push(literal);
    }
    // Left-fold the parts through `add` (`ll_strconcat`).
    let mut acc = parts[0].clone();
    for part in &parts[1..] {
        let sum = alloc(graph);
        stamp_as_rstr_str(&sum);
        ops.push(SpaceOperation {
            result: Some(sum.clone()),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: acc.clone(),
                rhs: part.clone(),
                result_ty: ValueType::Str,
            },
        });
        acc = sum;
    }
    // Reuse the displaced format result var on the final op so the downstream
    // link still forwards the rendered String.
    stamp_as_rstr_str(&result);
    if let Some(last) = ops.last_mut() {
        last.result = Some(result);
    }
    ops
}

/// The unique `Link` feeding `target`'s inputarg at `pos`: the source
/// block id, its exit index, and the threaded value Variable. Returns
/// `None` when `target` has more than one predecessor (a phi merge, so
/// no single thread to rewrite) or the position carries a `Const`.
pub(in crate::front::mir) fn single_incoming_link(
    graph: &FunctionGraph,
    target: BlockId,
    pos: usize,
) -> Option<(BlockId, usize, Variable)> {
    let mut found: Option<(BlockId, usize, Variable)> = None;
    for src in &graph.blocks {
        for (ei, link) in src.exits.iter().enumerate() {
            if link.target == target {
                if found.is_some() {
                    return None; // phi merge: ambiguous predecessor
                }
                let v = link.args.get(pos)?.as_variable()?.clone();
                found = Some((src.id, ei, v));
            }
        }
    }
    found
}

/// Position of `var` in `block`'s inputargs.
pub(in crate::front::mir) fn inputarg_pos(
    graph: &FunctionGraph,
    block: BlockId,
    var: &Variable,
) -> Option<usize> {
    graph
        .blocks
        .iter()
        .find(|b| b.id == block)?
        .inputargs
        .iter()
        .position(|a| a.id() == var.id())
}

/// A recognized single-argument `format!` chain ready to collapse: the
/// terminal `alloc::fmt::format` op, the literal pieces, the forwarding
/// links to rewrite (so the rendered value threads straight to the format
/// block in place of the `new_display`/`Arguments` values), and the
/// now-dead chain ops to delete.
pub(in crate::front::mir) struct FmtCollapse {
    pub(in crate::front::mir) format_block: BlockId,
    pub(in crate::front::mir) format_result: u64,
    pub(in crate::front::mir) pieces: Vec<String>,
    /// Default-spec `FmtArgKind::LowerHex` over an integer interpolant:
    /// the placeholder is `'%x' % r_uint(n)` (`rstr.py do_stringformat`).
    pub(in crate::front::mir) hex: bool,
    /// Signed `{:x}` interpolant width from the argument tuple ctor
    /// (`Tuple<i8>` → 8). `None` when the interpolant is already unsigned,
    /// a bool, or 64-bit (no extra mask).
    pub(in crate::front::mir) hex_signed_bits: Option<u32>,
    /// `(block, exit_index, arg_pos, replacement)` — replace the chain
    /// value the link forwarded with the threaded rendered value.
    pub(in crate::front::mir) link_rewrites: Vec<(BlockId, usize, usize, Variable)>,
    /// Op result var ids to delete (chain ctors / `new_display` / pieces
    /// bytes / `Arguments::new`).
    pub(in crate::front::mir) dead_results: Vec<u64>,
    /// Aggregate base var ids whose `FieldWrite`s are deleted.
    pub(in crate::front::mir) dead_bases: Vec<u64>,
}

/// The arg-kind-agnostic navigation of a single-argument `format!` chain
/// terminating at the `alloc::fmt::format` op `(bf, fi)`.  Both the
/// `Display` collapse ([`collect_fmt_collapse`]) and the fieldless-enum
/// `Debug` collapse ([`collect_debug_enum_fmt_collapse`]) walk the identical
/// three-block topology and delete the identical residual chain — they
/// differ only in the render gate (`Display` vs `Debug`) and how the format
/// op is replaced (native `str`+concat vs discriminant switch).
pub(in crate::front::mir) struct SingleArgFmtChainNav {
    /// The render kind of the single argument (`Display` / `Debug`).
    pub(in crate::front::mir) kind: FmtArgKind,
    /// Static options and argument index decoded from the packed template.
    pub(in crate::front::mir) placeholder: FmtPlaceholder,
    /// The literal template pieces around the single placeholder.
    pub(in crate::front::mir) pieces: Vec<String>,
    /// The `alloc::fmt::format` result var.
    pub(in crate::front::mir) format_result: Variable,
    /// The single rendered value (the tuple field source).
    pub(in crate::front::mir) context: Variable,
    /// The argument tuple is exactly Rust `(u8,)`; used by the LowerHex
    /// parity collapse to prove that two hexadecimal digits cannot truncate.
    pub(in crate::front::mir) context_is_u8: bool,
    /// Signed interpolant width from the argument tuple ctor name
    /// (`Tuple<i8>` → 8, `Tuple<i16>` → 16, `Tuple<i32>` → 32,
    /// `Tuple<i64>` / `Tuple<isize>` → 64). `None` when the ctor is not
    /// one of those shapes.
    pub(in crate::front::mir) context_signed_bits: Option<u32>,
    /// Argument tuple ctor is `Tuple<u8>` / `Tuple<u16>` / `Tuple<u32>` / `Tuple<u64>` / `Tuple<usize>`.
    pub(in crate::front::mir) context_unsigned_int: bool,
    /// `(block, exit_index, arg_pos, replacement)` — re-thread the deleted
    /// chain value the link forwarded onto a still-live value so no link
    /// references a deleted result var after the chain ops are removed.
    pub(in crate::front::mir) link_rewrites: Vec<(BlockId, usize, usize, Variable)>,
    /// Op result var ids to delete (chain ctors / pieces bytes /
    /// `Arguments::new`).
    pub(in crate::front::mir) dead_results: Vec<u64>,
    /// Aggregate base var ids whose `FieldWrite`s are deleted.
    pub(in crate::front::mir) dead_bases: Vec<u64>,
}

/// Back-navigate the single-argument `format!` chain the charon lowering
/// builds for `f(format!("…{x}"))` — block B0 builds `Argument::new_*(&x)`
/// off the argument Tuple and forwards it; block Bp builds the args + pieces
/// arrays and `Arguments::new`, forwarding the `Arguments`; block Bf calls
/// `alloc::fmt::format` — collecting the render kind, template pieces, the
/// rendered value, and the residual-chain delete / re-thread plan shared by
/// both collapses.  `None` for any shape outside the recognized subset
/// (multi-argument, non-threaded, or phi-merged), so a caller leaves the
/// graph untouched.
pub(in crate::front::mir) fn navigate_single_arg_fmt_chain(
    graph: &FunctionGraph,
    bf: BlockId,
    fi: usize,
) -> Option<SingleArgFmtChainNav> {
    use crate::model::{CallTarget, OpKind};
    let block_f = graph.blocks.iter().find(|b| b.id == bf)?;
    let format_op = block_f.operations.get(fi)?;
    let (fmt_args, format_result) = match &format_op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if fmt_path_ends_with(segments, &["fmt", "format"]) => (
            args.first()?.clone().into_variable(),
            format_op.result.as_ref()?.clone(),
        ),
        _ => return None,
    };
    let chain = extract_fmt_chain(graph, &fmt_args)?;
    if chain.args.len() != 1 || chain.placeholders.len() != 1 {
        // scope: exactly one argument rendered by exactly one placeholder.
        // A single argument reused across several placeholders (`{0}…{0}`)
        // has `placeholders.len() > 1` and stays residual here (none of the
        // census walls), rather than mis-driving the single-placeholder
        // expansion below.
        return None;
    }
    let kind = chain.args[0].kind;
    let placeholder = chain.placeholders[0];
    let pieces = chain.pieces.clone();

    // `fmt_args` reaches Bf as an inputarg threaded from Bp's
    // `Arguments::new` result.
    let pf = inputarg_pos(graph, bf, &fmt_args)?;
    let (bp, ei_bf, arguments_var) = single_incoming_link(graph, bf, pf)?;
    if bp == bf {
        return None;
    }
    let block_p = graph.blocks.iter().find(|b| b.id == bp)?;
    let (pieces_var, args_var) = block_p.operations.iter().find_map(|op| match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if op.result.as_ref().map(|r| r.id()) == Some(arguments_var.id())
            && is_arguments_new_path(segments) =>
        {
            Some((args.first()?.clone(), args.get(1)?.clone()))
        }
        _ => None,
    })?;
    let piece_byte_vars = read_array_literal_elements(graph, &pieces_var)?;
    let arg_elems = read_array_literal_elements(graph, &args_var)?;
    if arg_elems.len() != 1 {
        return None;
    }
    let arg_elem = arg_elems.into_iter().next()?;

    // `arg_elem` reaches Bp as an inputarg threaded from B0's
    // `new_*` result.
    let pe = inputarg_pos(graph, bp, &arg_elem)?;
    let (b0, ei_bp, new_arg_var) = single_incoming_link(graph, bp, pe)?;
    if b0 == bp {
        return None;
    }
    let block_0 = graph.blocks.iter().find(|b| b.id == b0)?;
    let (arg_ref, tuple_var) = block_0.operations.iter().find_map(|op| match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if op.result.as_ref().map(|r| r.id()) == Some(new_arg_var.id())
            && fmt_argument_ctor_kind(segments).is_some() =>
        {
            let arg_ref = args.first()?.clone();
            let (fr_block, fr_idx) = resolve_to_producer_op(graph, &arg_ref)?;
            let frb = graph.blocks.iter().find(|b| b.id == fr_block)?;
            match &frb.operations.get(fr_idx)?.kind {
                OpKind::FieldRead { base, .. } => Some((arg_ref.clone(), base.clone())),
                _ => None,
            }
        }
        _ => None,
    })?;
    // The rendered value written into the argument tuple field.
    let context = unwrap_fmt_arg_tuple_ref(graph, &arg_ref)?;
    let tuple_ctor_name = block_0.operations.iter().find_map(|op| match &op.kind {
        OpKind::Call {
            target: CallTarget::SyntheticTransparentCtor { name, .. },
            ..
        } if op.result.as_ref().map(|r| r.id()) == Some(tuple_var.id()) => Some(name.as_str()),
        _ => None,
    });
    let context_is_u8 = tuple_ctor_name == Some("Tuple<u8>");
    let context_signed_bits = match tuple_ctor_name {
        Some("Tuple<i8>") => Some(8),
        Some("Tuple<i16>") => Some(16),
        Some("Tuple<i32>") => Some(32),
        Some("Tuple<i64>") | Some("Tuple<isize>") => Some(64),
        _ => None,
    };
    let context_unsigned_int = matches!(
        tuple_ctor_name,
        Some("Tuple<u8>" | "Tuple<u16>" | "Tuple<u32>" | "Tuple<u64>" | "Tuple<usize>")
    );

    // Thread `context` straight through the slots the chain values used:
    // B0→Bp forwards `context` where it forwarded `new_*`, Bp→Bf forwards
    // Bp's now-`context`-bound inputarg where it forwarded `Arguments`, so
    // no link references a deleted result var once the chain ops are gone.
    let link_rewrites = vec![
        (b0, ei_bp, pe, context.clone()),
        (bp, ei_bf, pf, arg_elem.clone()),
    ];
    let mut dead_results = vec![
        new_arg_var.id(),
        arg_ref.id(),
        tuple_var.id(),
        arguments_var.id(),
        pieces_var.id(),
        args_var.id(),
    ];
    dead_results.extend(piece_byte_vars.iter().map(|v| v.id()));
    let dead_bases = vec![tuple_var.id(), pieces_var.id(), args_var.id()];

    Some(SingleArgFmtChainNav {
        kind,
        placeholder,
        pieces,
        format_result,
        context,
        context_is_u8,
        context_signed_bits,
        context_unsigned_int,
        link_rewrites,
        dead_results,
        dead_bases,
    })
}

/// Whether `var` can be rendered by the rstr `str` / `ll_int2dec` helpers
/// the format collapse emits. A rust-heap `String` is `Ref("String")` or
/// `Ref("string::String")`; collapse still emits `str(value)` (`ll_str`)
/// so the interpolant is rstr `STR` before `ll_strconcat`. An untyped
/// producer (synthetic tests, a phi with no defining op) keeps the
/// historical collapse.
pub(in crate::front::mir) fn fmt_value_renders_via_rstr(
    graph: &FunctionGraph,
    var: &crate::flowspace::model::Variable,
) -> bool {
    use crate::model::{OpKind, ValueType};
    let Some((bid, idx)) = resolve_to_producer_op(graph, var) else {
        return true;
    };
    let Some(block) = graph.blocks.iter().find(|b| b.id == bid) else {
        return true;
    };
    let Some(op) = block.operations.get(idx) else {
        return true;
    };
    let ty = match &op.kind {
        OpKind::Input { ty, .. } => ty,
        OpKind::UnaryOp { result_ty, .. }
        | OpKind::BinOp { result_ty, .. }
        | OpKind::Call { result_ty, .. } => result_ty,
        OpKind::ConstStr(_) | OpKind::ConstInternedStr(_) => return true,
        OpKind::ConstInt(_) | OpKind::ConstUInt(_) | OpKind::ConstBool(_) => return true,
        _ => return true,
    };
    matches!(
        ty,
        ValueType::Str
            | ValueType::Int
            | ValueType::Unsigned
            | ValueType::Bool
            | ValueType::Int128
            | ValueType::UInt128
    ) || matches!(ty, ValueType::Ref(Some(root)) if is_rust_heap_string_root(root))
}

/// `{name:?}` of a string-family value is Display plus quotes
/// (`Debug for str` / `Debug for String`). Integers and bools share
/// Display; fieldless enums stay with `collapse_debug_enum_fmt_chains`.
pub(in crate::front::mir) fn fmt_value_is_rstr_string(
    graph: &FunctionGraph,
    var: &crate::flowspace::model::Variable,
) -> bool {
    use crate::model::{OpKind, ValueType};
    let Some((bid, idx)) = resolve_to_producer_op(graph, var) else {
        return false;
    };
    let Some(block) = graph.blocks.iter().find(|b| b.id == bid) else {
        return false;
    };
    let Some(op) = block.operations.get(idx) else {
        return false;
    };
    let ty = match &op.kind {
        OpKind::ConstStr(_) | OpKind::ConstInternedStr(_) => return true,
        OpKind::Input { ty, .. }
        | OpKind::FieldRead { ty, .. }
        | OpKind::UnaryOp { result_ty: ty, .. }
        | OpKind::BinOp { result_ty: ty, .. }
        | OpKind::Call { result_ty: ty, .. } => ty,
        _ => return false,
    };
    matches!(ty, ValueType::Str)
        || matches!(ty, ValueType::Ref(Some(root)) if is_rust_heap_string_root(root))
}

/// Default-spec `{:x}` interpolant: a word-sized integer (Signed /
/// Unsigned / Bool). Int128 / UInt128 / string stay residual. A missing
/// producer (loop phi, e.g. `sre_pattern_flag_tail`'s `flags`) is not
/// proof; `collect_fmt_collapse` takes the argument tuple's Rust integer
/// type (`Tuple<i64>` …) as the proof instead.
pub(in crate::front::mir) fn fmt_value_is_integer_hex(
    graph: &FunctionGraph,
    var: &crate::flowspace::model::Variable,
) -> bool {
    use crate::model::{OpKind, ValueType};
    let Some((bid, idx)) = resolve_to_producer_op(graph, var) else {
        return false;
    };
    let Some(block) = graph.blocks.iter().find(|b| b.id == bid) else {
        return false;
    };
    let Some(op) = block.operations.get(idx) else {
        return false;
    };
    match &op.kind {
        OpKind::ConstInt(_) | OpKind::ConstUInt(_) | OpKind::ConstBool(_) => true,
        OpKind::Input { ty, .. }
        | OpKind::FieldRead { ty, .. }
        | OpKind::UnaryOp { result_ty: ty, .. }
        | OpKind::BinOp { result_ty: ty, .. }
        | OpKind::Call { result_ty: ty, .. } => {
            matches!(ty, ValueType::Int | ValueType::Unsigned | ValueType::Bool)
        }
        _ => false,
    }
}

/// `ConstBool` / `ValueType::Bool` interpolants have no signed width.
fn fmt_value_is_bool(graph: &FunctionGraph, var: &crate::flowspace::model::Variable) -> bool {
    use crate::model::{OpKind, ValueType};
    let Some((bid, idx)) = resolve_to_producer_op(graph, var) else {
        return false;
    };
    let Some(block) = graph.blocks.iter().find(|b| b.id == bid) else {
        return false;
    };
    let Some(op) = block.operations.get(idx) else {
        return false;
    };
    match &op.kind {
        OpKind::ConstBool(_) => true,
        OpKind::Input { ty, .. }
        | OpKind::FieldRead { ty, .. }
        | OpKind::UnaryOp { result_ty: ty, .. }
        | OpKind::BinOp { result_ty: ty, .. }
        | OpKind::Call { result_ty: ty, .. } => matches!(ty, ValueType::Bool),
        _ => false,
    }
}

/// Skip `r_uint` when the interpolant is already Unsigned (`ConstUInt` /
/// `ValueType::Unsigned` / `Input` ty Unsigned).
pub(in crate::front::mir) fn fmt_value_is_unsigned(
    graph: &FunctionGraph,
    var: &crate::flowspace::model::Variable,
) -> bool {
    use crate::model::{OpKind, ValueType};
    let Some((bid, idx)) = resolve_to_producer_op(graph, var) else {
        return false;
    };
    let Some(block) = graph.blocks.iter().find(|b| b.id == bid) else {
        return false;
    };
    let Some(op) = block.operations.get(idx) else {
        return false;
    };
    match &op.kind {
        OpKind::ConstUInt(_) => true,
        OpKind::Input { ty, .. }
        | OpKind::FieldRead { ty, .. }
        | OpKind::UnaryOp { result_ty: ty, .. }
        | OpKind::BinOp { result_ty: ty, .. }
        | OpKind::Call { result_ty: ty, .. } => matches!(ty, ValueType::Unsigned),
        _ => false,
    }
}

pub(in crate::front::mir) fn quote_debug_string_pieces(pieces: &mut Vec<String>) {
    if pieces.is_empty() {
        pieces.push(String::new());
        pieces.push(String::new());
    }
    pieces[0].push_str("\"");
    if pieces.len() == 1 {
        pieces.push("\"".to_string());
    } else {
        pieces[1].insert_str(0, "\"");
    }
}

/// Recognize the single-argument `Display` `format!` chain terminating at
/// the `alloc::fmt::format` op `(bf, fi)` and collect the rewrite plan, or
/// `None` for any shape outside the recognized subset so the collapse
/// leaves the graph untouched.
pub(in crate::front::mir) fn collect_fmt_collapse(
    graph: &FunctionGraph,
    bf: BlockId,
    fi: usize,
) -> Option<FmtCollapse> {
    let nav = navigate_single_arg_fmt_chain(graph, bf, fi)?;
    if !nav.placeholder.is_default() {
        return None;
    }
    let mut pieces = nav.pieces;
    let mut hex = false;
    let mut hex_signed_bits = None;
    match nav.kind {
        FmtArgKind::Display => {
            // `str` + `ll_strconcat` consume rstr `STR` (or an int `ll_int2dec`).
            // A rust-heap `String` interpolant is included: collapse emits
            // `str(value)` first so the concat operands are rstr `STR`.
            if !fmt_value_renders_via_rstr(graph, &nav.context) {
                return None;
            }
        }
        FmtArgKind::Debug => {
            if !fmt_value_is_rstr_string(graph, &nav.context) {
                return None;
            }
            quote_debug_string_pieces(&mut pieces);
        }
        FmtArgKind::LowerHex => {
            // Default-spec `{:x}` only (`is_default` already excluded
            // `{:02x}` / `{:0width$x}`). Integer interpolant → `'%x' % r_uint`.
            // The argument tuple's Rust integer type is also proof.
            let proven_integer = nav.context_signed_bits.is_some()
                || nav.context_unsigned_int
                || fmt_value_is_integer_hex(graph, &nav.context);
            if !proven_integer {
                return None;
            }
            if !nav.context_unsigned_int
                && !fmt_value_is_unsigned(graph, &nav.context)
                && !fmt_value_is_bool(graph, &nav.context)
            {
                hex_signed_bits = Some(nav.context_signed_bits?);
            }
            hex = true;
        }
    }
    Some(FmtCollapse {
        format_block: bf,
        format_result: nav.format_result.id(),
        pieces,
        hex,
        hex_signed_bits,
        link_rewrites: nav.link_rewrites,
        dead_results: nav.dead_results,
        dead_bases: nav.dead_bases,
    })
}

/// Expand every recognized single-argument `format!` chain into the
/// rtyper's native string-build operations: the chain's
/// `Argument::new_display` ctor, on-stack pieces/args arrays, argument
/// Tuple and `Arguments::new` are deleted, the rendered value is threaded
/// straight to the `alloc::fmt::format` block, and that op is replaced by
/// `str(value)` (`ll_str`) folded with the literal pieces through
/// `add` (`ll_strconcat`).  This is the orthodox lowering — every emitted
/// op (`__str_const`, `str`, `add`) the rtyper natively types `SomeString`
/// and the legacy walker / codewriter / runtime already handle — so the
/// graph-less `fmt::rt::Argument` / `fmt::Arguments` externs no longer
/// block the rtyper, with no opaque residual or fresh runtime helper.
pub(in crate::front::mir) fn collapse_fmt_chains(graph: &mut FunctionGraph) -> usize {
    use crate::model::{CallTarget, LinkArg, OpKind};
    let sites: Vec<FmtCollapse> = graph
        .blocks
        .iter()
        .flat_map(|block| {
            block
                .operations
                .iter()
                .enumerate()
                .filter_map(move |(fi, op)| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if fmt_path_ends_with(segments, &["fmt", "format"]) => Some((block.id, fi)),
                    _ => None,
                })
        })
        .filter_map(|(bid, fi)| collect_fmt_collapse(graph, bid, fi))
        .collect();
    if sites.is_empty() {
        return 0;
    }
    // 1. Re-thread the rendered values onto the forwarding links.
    for site in &sites {
        for (bid, ei, pos, repl) in &site.link_rewrites {
            if let Some(link) = graph.block_mut(*bid).exits.get_mut(*ei)
                && let Some(arg) = link.args.get_mut(*pos)
            {
                *arg = LinkArg::Value(repl.clone());
            }
        }
    }
    // 2. Delete the now-dead chain ops across all blocks.
    let dead_results: std::collections::HashSet<u64> = sites
        .iter()
        .flat_map(|s| s.dead_results.iter().copied())
        .collect();
    let dead_bases: std::collections::HashSet<u64> = sites
        .iter()
        .flat_map(|s| s.dead_bases.iter().copied())
        .collect();
    for block in &mut graph.blocks {
        block.operations.retain(|op| {
            if let Some(r) = &op.result
                && dead_results.contains(&r.id())
            {
                return false;
            }
            if let OpKind::FieldWrite { base, .. } = &op.kind
                && dead_bases.contains(&base.id())
            {
                return false;
            }
            true
        });
    }
    // 3. Replace each `alloc::fmt::format` op (now reading the threaded
    //    rendered value) with the orthodox `str(value)` + `ll_strconcat`
    //    expansion, reusing the format result var.
    for site in &sites {
        let Some((idx, value, result)) = graph
            .blocks
            .iter()
            .find(|b| b.id == site.format_block)
            .and_then(|b| {
                let idx = b.operations.iter().position(|op| {
                    op.result.as_ref().map(|r| r.id()) == Some(site.format_result)
                })?;
                let value = match &b.operations[idx].kind {
                    OpKind::Call { args, .. } => args.first()?.clone(),
                    _ => return None,
                };
                Some((idx, value, b.operations[idx].result.clone()?))
            })
        else {
            continue;
        };
        let expansion = emit_fmt_expansion_ops(
            graph,
            &site.pieces,
            &value,
            result,
            site.hex,
            site.hex_signed_bits,
        );
        graph
            .block_mut(site.format_block)
            .operations
            .splice(idx..idx + 1, expansion);
    }
    sites.len()
}

/// A byte `{:02x}` chain whose Rust formatting shell can be replaced by
/// PyPy's two-nibble lookup while reusing the common single-argument chain
/// deletion plan.
pub(in crate::front::mir) struct LowerHexByteFmtCollapse {
    pub(in crate::front::mir) format_block: BlockId,
    pub(in crate::front::mir) format_result: u64,
    /// The template's literal halves, carried so the expansion re-emits the
    /// leading `\x` — upstream's first `buf.append` — instead of consuming it
    /// as a match condition only.
    pub(in crate::front::mir) pieces: Vec<String>,
    pub(in crate::front::mir) link_rewrites: Vec<(BlockId, usize, usize, Variable)>,
    pub(in crate::front::mir) dead_results: Vec<u64>,
    pub(in crate::front::mir) dead_bases: Vec<u64>,
}

pub(in crate::front::mir) fn collect_lower_hex_byte_fmt_collapse(
    graph: &FunctionGraph,
    bf: BlockId,
    fi: usize,
) -> Option<LowerHexByteFmtCollapse> {
    let nav = navigate_single_arg_fmt_chain(graph, bf, fi)?;
    if nav.kind != FmtArgKind::LowerHex
        || !nav.placeholder.is_lower_hex_byte_02()
        || !nav.context_is_u8
    {
        return None;
    }
    Some(LowerHexByteFmtCollapse {
        format_block: bf,
        format_result: nav.format_result.id(),
        pieces: nav.pieces,
        link_rewrites: nav.link_rewrites,
        dead_results: nav.dead_results,
        dead_bases: nav.dead_bases,
    })
}

/// Replace the exact `format!("\\x{c:02x}")` shell used by the bytearray
/// repr loop with PyPy's `"0123456789abcdef"[c >> 4]` / `[c & 0xF]`
/// implementation.  The packed template, `new_lower_hex` ctor, argument tuple
/// and `alloc::fmt::format` all disappear; only RPython-native integer,
/// string-getitem and string-concat operations remain.
pub(in crate::front::mir) fn collapse_lower_hex_byte_fmt_chains(
    graph: &mut FunctionGraph,
) -> usize {
    use crate::model::{CallTarget, LinkArg, OpKind};
    let sites: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| {
            block
                .operations
                .iter()
                .enumerate()
                .filter_map(move |(fi, op)| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if fmt_path_ends_with(segments, &["fmt", "format"]) => Some((block.id, fi)),
                    _ => None,
                })
        })
        .filter_map(|(bid, fi)| collect_lower_hex_byte_fmt_collapse(graph, bid, fi))
        .collect();
    if sites.is_empty() {
        return 0;
    }

    for site in &sites {
        for (bid, ei, pos, replacement) in &site.link_rewrites {
            if let Some(link) = graph.block_mut(*bid).exits.get_mut(*ei)
                && let Some(arg) = link.args.get_mut(*pos)
            {
                *arg = LinkArg::Value(replacement.clone());
            }
        }
    }
    let dead_results: std::collections::HashSet<u64> = sites
        .iter()
        .flat_map(|site| site.dead_results.iter().copied())
        .collect();
    let dead_bases: std::collections::HashSet<u64> = sites
        .iter()
        .flat_map(|site| site.dead_bases.iter().copied())
        .collect();
    for block in &mut graph.blocks {
        block.operations.retain(|op| {
            if op
                .result
                .as_ref()
                .is_some_and(|result| dead_results.contains(&result.id()))
            {
                return false;
            }
            !matches!(&op.kind, OpKind::FieldWrite { base, .. } if dead_bases.contains(&base.id()))
        });
    }
    for site in &sites {
        let Some((idx, value, result)) = graph
            .blocks
            .iter()
            .find(|block| block.id == site.format_block)
            .and_then(|block| {
                let idx = block.operations.iter().position(|op| {
                    op.result.as_ref().map(|result| result.id()) == Some(site.format_result)
                })?;
                let value = match &block.operations[idx].kind {
                    OpKind::Call { args, .. } => args.first()?.clone(),
                    _ => return None,
                };
                Some((idx, value, block.operations[idx].result.clone()?))
            })
        else {
            continue;
        };
        // The original context value was defined in B0.  After link rewrites,
        // the format op's own argument Variable is Bf's inputarg carrying that
        // value; use it so every emitted operand is defined in this block.
        let expansion = emit_lower_hex_byte_02_ops(graph, &site.pieces, &value, result);
        graph
            .block_mut(site.format_block)
            .operations
            .splice(idx..idx + 1, expansion);
    }
    sites.len()
}

/// Collapse a constant `format!("literal")` — a zero-placeholder format whose
/// `format_args!` charon has already folded to the fully-rendered constant
/// string, so `alloc::fmt::format`'s single argument resolves to a
/// `__str_const` rather than to an `Arguments::new(pieces, args)` chain.  The
/// format of a constant string IS that string, so replace the `fmt::format` op
/// in place with the same `__str_const`, keeping the result var; the now-unread
/// producer is reclaimed by the dead-op sweep.  No pieces/args arrays exist
/// (unlike the single/multi-arg Display collapsers), so no link re-threading is
/// needed.  Retires the graph-less `alloc::fmt::format` extern with only the
/// `__str_const` op the rtyper natively types `SomeString`.
pub(in crate::front::mir) fn collapse_const_fmt(graph: &mut FunctionGraph) -> usize {
    use crate::model::{CallTarget, OpKind, ValueType};
    // `__str_const` carries its text as the second path segment (see
    // `emit_str_const`), so a zero-arg `["__str_const", text]` producer is the
    // whole rendered message.
    let mut rewrites: Vec<(BlockId, usize, String)> = Vec::new();
    for block in &graph.blocks {
        for (fi, op) in block.operations.iter().enumerate() {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } = &op.kind
            else {
                continue;
            };
            if !fmt_path_ends_with(segments, &["fmt", "format"]) || op.result.is_none() {
                continue;
            }
            let Some(arg) = args.first() else {
                continue;
            };
            let Some((pbid, pidx)) = resolve_to_producer_op(graph, arg) else {
                continue;
            };
            let text = match graph
                .blocks
                .iter()
                .find(|b| b.id == pbid)
                .and_then(|b| b.operations.get(pidx))
                .map(|o| &o.kind)
            {
                Some(OpKind::Call {
                    target:
                        CallTarget::FunctionPath {
                            segments: pseg,
                            fun_decl_id: None,
                            specialization: crate::parse::GraphSpecialization::Regular,
                        },
                    args: pargs,
                    ..
                }) if pargs.is_empty() && pseg.len() == 2 && pseg[0] == "__str_const" => {
                    pseg[1].clone()
                }
                _ => continue,
            };
            rewrites.push((block.id, fi, text));
        }
    }
    let n = rewrites.len();
    // In-place kind replacement keeps every op index valid across the batch.
    for (bid, fi, text) in rewrites {
        if let Some(op) = graph.block_mut(bid).operations.get_mut(fi) {
            op.kind = OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__str_const".to_string(), text],
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![]),
                result_ty: ValueType::Ref(None),
            };
        }
    }
    n
}

/// A recognized multi-argument `format!` chain ready to collapse.  Unlike
/// the single-argument [`FmtCollapse`], a multi-arg chain builds an N-field
/// argument tuple and one `Argument::new_display` ctor per placeholder
/// (spread across several blocks), and the rendered values already thread
/// to the `Arguments::new` block as the args-array elements.  So the
/// collapse renders each placeholder in place (`new_display(inner)` →
/// `str(inner)`) and folds the rendered values with the literal pieces at
/// the `Arguments::new` block, threading the single resulting String to the
/// `alloc::fmt::format` block.
pub(in crate::front::mir) struct FmtCollapseMulti {
    /// Block holding `Arguments::new` — where the concat fold is emitted
    /// (every rendered value is live there, as an args-array element).
    pub(in crate::front::mir) args_block: BlockId,
    /// `Arguments::new` result var id (re-threaded to the folded String,
    /// then deleted).
    pub(in crate::front::mir) arguments_var: u64,
    /// The rendered-value vars, one per *distinct* argument (the args-array
    /// elements, live as inputargs of `args_block`).
    pub(in crate::front::mir) arg_elem_vars: Vec<Variable>,
    /// The distinct-argument index each placeholder renders, in template
    /// order (`0xC8` reuse points several placeholders at one `arg_elem_var`).
    pub(in crate::front::mir) arg_indices: Vec<usize>,
    pub(in crate::front::mir) pieces: Vec<String>,
    /// `(block, op_index, inner)` per `Argument::new_display` ctor to
    /// rewrite into `str(inner)` in place.
    pub(in crate::front::mir) new_display_ops: Vec<(BlockId, usize, Variable)>,
    pub(in crate::front::mir) format_block: BlockId,
    pub(in crate::front::mir) format_idx: usize,
    /// The var the `alloc::fmt::format` op reads (an `Arguments`-threaded
    /// inputarg that, after the link rewrite, carries the folded String).
    pub(in crate::front::mir) fmt_args: Variable,
    /// Op result var ids to delete (args/pieces array ctors, packed byte
    /// consts, `Arguments::new`, and — when the argument-Tuple round-trip is
    /// eliminated — the Tuple ctor + its `__pos_N` `FieldRead`s).
    pub(in crate::front::mir) dead_results: Vec<u64>,
    /// Aggregate base var ids whose `FieldWrite`s are deleted (the args and
    /// pieces arrays, plus the argument-Tuple base when eliminated).
    pub(in crate::front::mir) dead_bases: Vec<u64>,
    /// `(block, exit_idx, arg_pos, value)` link-arg rebinds applied before
    /// deletion: a `format_args!` Tuple ref threaded to a placeholder's block
    /// is replaced by the rendered value, so the threaded inputarg `str`
    /// reads now carries the value rather than the Tuple field. Empty unless
    /// the Tuple round-trip is eliminated ([`attempt_fmt_tuple_elimination`]).
    pub(in crate::front::mir) link_rebinds: Vec<(BlockId, usize, usize, Variable)>,
}

/// True if `op` reads `var` in any operand position.  Covers the front-end
/// operand-bearing `OpKind` variants present during `finish` — the fmt
/// collapse runs pre-`jtransform`, so the JIT-only call / vable / guard-value
/// families never appear and cannot carry the on-stack `format_args!` Tuple
/// ref; other variants default to `false`.
pub(in crate::front::mir) fn op_reads_var(
    op: &crate::model::SpaceOperation,
    var: &Variable,
) -> bool {
    use crate::model::{LinkArg, OpKind};
    let v = var.id();
    let is = |x: &Variable| x.id() == v;
    let arg_is = |a: &LinkArg| matches!(a, LinkArg::Value(x) if x.id() == v);
    match &op.kind {
        OpKind::FieldRead { base, .. } => is(base),
        OpKind::FieldWrite { base, value, .. } => is(base) || arg_is(value),
        OpKind::ArrayRead { base, index, .. } => is(base) || is(index),
        OpKind::ArrayWrite {
            base, index, value, ..
        } => is(base) || is(index) || arg_is(value),
        OpKind::InteriorFieldRead { base, index, .. } => is(base) || is(index),
        OpKind::InteriorFieldWrite {
            base, index, value, ..
        } => is(base) || is(index) || is(value),
        OpKind::Call { args, .. } => args.iter().any(|a| is(a)),
        OpKind::BinOp { lhs, rhs, .. } => is(lhs) || is(rhs),
        OpKind::UnaryOp { operand, .. } => is(operand),
        OpKind::GuardTrue { cond } | OpKind::GuardFalse { cond } => is(cond),
        _ => false,
    }
}

/// The plan to also delete the `format_args!` argument-Tuple round-trip in a
/// multi-arg chain — the deletion the single-arg [`collect_fmt_collapse`]
/// always performs but the multi-arg path historically skipped.  Without it
/// the Tuple ctor + its `__pos_N` `FieldWrite`/`FieldRead`s survive, and a
/// placeholder value (a `StringRepr`) written into the Tuple's
/// PyObject-erased `__pos_N` field walls the rtyper (`convertvar(StringRepr →
/// InstanceRepr PyObject)`, no pairtype arm) when the Tuple is pinned by a
/// field read-back.  The Tuple is a Rust `format_args!` artifact with no PyPy
/// counterpart (PyPy joins the message flat, then boxes once at the exception
/// boundary), so deleting it is the parity-correct collapse.
///
/// Resolves each placeholder to its rendered value (`chain.args[i].value`,
/// already unwrapped through the Tuple by `extract_fmt_chain`) so `str` reads
/// the value, not the Tuple field.  A placeholder whose `new_display` sits in
/// the Tuple-construction block reads the value directly; one in a successor
/// block (the common Charon call-boundary split) reads it through a slot
/// rebind on the single incoming link that forwards the Tuple ref.  Returns
/// `None` (leaving the historic in-place `str(field-read)` collapse, which
/// keeps the residual Tuple but never reintroduces the `fmt` extern) for any
/// shape outside this set, so the change is strictly additive and a hard
/// shape bails rather than half-deleting.
pub(in crate::front::mir) struct FmtTupleElim {
    /// Per placeholder (chain order): the var `str` should read.
    pub(in crate::front::mir) str_inputs: Vec<Variable>,
    /// `(block, exit_idx, arg_pos, value)` Tuple-ref slot rebinds.
    pub(in crate::front::mir) link_rebinds: Vec<(BlockId, usize, usize, Variable)>,
    /// Tuple ctor + per-placeholder `FieldRead` result ids to delete.
    pub(in crate::front::mir) dead_results: Vec<u64>,
    /// Tuple base id whose `FieldWrite`s are deleted.
    pub(in crate::front::mir) dead_bases: Vec<u64>,
}

/// Whether every `FieldWrite` storing into the argument `Tuple` `tv` lives
/// in `bc` (the ctor block).  Matches a write by Variable id or by producer
/// identity (a threaded copy of the ref carries a different id but the same
/// producing ctor op), mirroring [`unwrap_fmt_arg_tuple_ref`].  When true,
/// every field value is in scope at `bc`, hence at any block `bc`
/// dominates — so an in-place `str(value)` rewrite at a cross-block reader
/// is sound.
pub(in crate::front::mir) fn all_tuple_writes_in_block(
    graph: &FunctionGraph,
    tv: &Variable,
    bc: BlockId,
) -> bool {
    use crate::model::OpKind;
    let tv_producer = resolve_to_producer_op(graph, tv);
    for b in &graph.blocks {
        for op in &b.operations {
            if let OpKind::FieldWrite { base, .. } = &op.kind {
                let denotes_tuple = base.id() == tv.id()
                    || (tv_producer.is_some()
                        && resolve_to_producer_op(graph, base) == tv_producer);
                if denotes_tuple && b.id != bc {
                    return false;
                }
            }
        }
    }
    true
}

pub(in crate::front::mir) fn attempt_fmt_tuple_elimination(
    graph: &FunctionGraph,
    chain: &FmtChain,
    new_display_ops: &[(BlockId, usize, Variable)],
) -> Option<FmtTupleElim> {
    use crate::model::{ExitSwitch, LinkArg, OpKind};
    if chain.args.len() != new_display_ops.len() {
        return None;
    }

    // The shared argument-Tuple ctor, resolved from each placeholder's
    // `new_display` arg `FieldRead` base; all placeholders must read the same
    // Tuple, or the chain is not the single private `format_args!` Tuple.
    let mut tuple_var: Option<Variable> = None;
    let mut ctor_block: Option<BlockId> = None;

    enum Plan {
        Local {
            value: Variable,
            getattr_result: u64,
        },
        Threaded {
            carrier: Variable,
            getattr_result: u64,
            rebind: (BlockId, usize, usize, Variable),
        },
    }
    let mut plans: Vec<Plan> = Vec::with_capacity(new_display_ops.len());

    for (i, (nd_block, _nd_idx, inner)) in new_display_ops.iter().enumerate() {
        let value = chain.args.get(i)?.value.clone();
        // `inner` is a `FieldRead` of the Tuple's `__pos_i` field; co-located
        // with its `new_display` (the Charon array lowering keeps them in one
        // block).
        let (fr_block, fr_idx) = resolve_to_producer_op(graph, inner)?;
        if fr_block != *nd_block {
            return None;
        }
        let frb = graph.blocks.iter().find(|b| b.id == fr_block)?;
        let carrier = match &frb.operations.get(fr_idx)?.kind {
            OpKind::FieldRead { base, .. } => base.clone(),
            _ => return None,
        };
        let getattr_result = inner.id();

        // The Tuple ctor: the carrier resolves to it (directly for the local
        // placeholder, through the forwarding link for a threaded copy).
        let (cb, ci) = resolve_to_producer_op(graph, &carrier)?;
        let ctor_result = graph
            .blocks
            .iter()
            .find(|b| b.id == cb)?
            .operations
            .get(ci)?
            .result
            .as_ref()?
            .clone();
        match (&tuple_var, &ctor_block) {
            (Some(tv), Some(bc)) if tv.id() != ctor_result.id() || *bc != cb => return None,
            _ => {
                tuple_var = Some(ctor_result.clone());
                ctor_block = Some(cb);
            }
        }
        let tv = tuple_var.clone()?;
        let bc = ctor_block?;

        if carrier.id() == tv.id() {
            // Local: the value is in scope where the Tuple is built.  A
            // cross-block reader (`format_args!` splits the placeholder
            // field reads across a straight-line block boundary) is still
            // sound when `bc` dominates the reader and every Tuple write
            // lives in `bc`, so each field value is available at the reader.
            if *nd_block != bc
                && !(block_dominates(graph, bc, *nd_block)
                    && all_tuple_writes_in_block(graph, &tv, bc))
            {
                return None;
            }
            plans.push(Plan::Local {
                value,
                getattr_result,
            });
            continue;
        }

        // Threaded: the carrier is an inputarg at position `pos`, fed by a
        // single incoming link from the ctor block that forwards the Tuple
        // ref at that position; rebind that slot to the value.  The carrier
        // must be used in this block only by this one `FieldRead` (never
        // forwarded onward / read again / used in the exit switch) so the
        // rebind is sound.  `LinkArg`s hold the predecessor's source vars,
        // which correspond positionally to the target's inputargs.
        let nd_blk = graph.blocks.iter().find(|b| b.id == *nd_block)?;
        let pos = nd_blk
            .inputargs
            .iter()
            .position(|a| a.id() == carrier.id())?;
        let incoming: Vec<(BlockId, usize)> = graph
            .blocks
            .iter()
            .flat_map(|b| {
                b.exits
                    .iter()
                    .enumerate()
                    .filter_map(move |(ei, l)| (l.target == *nd_block).then_some((b.id, ei)))
            })
            .collect();
        if incoming.len() != 1 {
            return None; // phi merge: no single forwarding slot to rebind
        }
        let (pred, ei) = incoming[0];
        if pred != bc {
            return None;
        }
        let forwards_tuple = matches!(
            graph
                .blocks
                .iter()
                .find(|b| b.id == pred)?
                .exits
                .get(ei)
                .and_then(|l| l.args.get(pos)),
            Some(LinkArg::Value(x)) if x.id() == tv.id()
        );
        if !forwards_tuple {
            return None;
        }
        let used_elsewhere = nd_blk
            .operations
            .iter()
            .enumerate()
            .any(|(idx, op)| idx != fr_idx && op_reads_var(op, &carrier))
            || nd_blk.exits.iter().any(|l| {
                l.args
                    .iter()
                    .any(|a| matches!(a, LinkArg::Value(x) if x.id() == carrier.id()))
            })
            || matches!(&nd_blk.exitswitch, Some(ExitSwitch::Value(x)) if x.id() == carrier.id());
        if used_elsewhere {
            return None;
        }
        plans.push(Plan::Threaded {
            carrier: carrier.clone(),
            getattr_result,
            rebind: (pred, ei, pos, value.clone()),
        });
    }

    let tuple_var = tuple_var?;
    let link_rebinds: Vec<(BlockId, usize, usize, Variable)> = plans
        .iter()
        .filter_map(|p| match p {
            Plan::Threaded { rebind, .. } => Some(rebind.clone()),
            Plan::Local { .. } => None,
        })
        .collect();

    let mut dead_results = vec![tuple_var.id()];
    let mut str_inputs = Vec::with_capacity(plans.len());
    for p in &plans {
        match p {
            Plan::Local {
                value,
                getattr_result,
            } => {
                str_inputs.push(value.clone());
                dead_results.push(*getattr_result);
            }
            Plan::Threaded {
                carrier,
                getattr_result,
                ..
            } => {
                str_inputs.push(carrier.clone());
                dead_results.push(*getattr_result);
            }
        }
    }

    // Deleting the ctor, its `FieldWrite`s, and the placeholder `FieldRead`s
    // leaves the framestate forwards of the (now dead) Tuple ref for the
    // post-collapse `prune_dead_phis` in `finish` to clean up (it trims dead
    // `Link.args` + inputargs).  That is sound only when the Tuple ref has no
    // *surviving* op reader: every op that reads it must be one we delete (a
    // placeholder `FieldRead` whose result is in `dead_results`, or a
    // `FieldWrite` into it).  Threaded slots are redirected to the value by
    // `link_rebinds`; any remaining forward of a fully-dead Tuple ref is what
    // `prune_dead_phis` removes.  A surviving reader (the ref escapes the
    // chain) would dangle, so bail to the historic in-place `str(field-read)`
    // collapse (residual Tuple, no regression).
    let tuple_has_live_reader = graph.blocks.iter().any(|b| {
        b.operations.iter().any(|op| {
            let deleted = op
                .result
                .as_ref()
                .is_some_and(|r| dead_results.contains(&r.id()))
                || matches!(&op.kind, OpKind::FieldWrite { base, .. } if base.id() == tuple_var.id());
            !deleted
                && crate::inline::op_variable_refs(&op.kind)
                    .iter()
                    .any(|v| v.id() == tuple_var.id())
        }) || matches!(&b.exitswitch, Some(ExitSwitch::Value(x)) if x.id() == tuple_var.id())
    });
    if tuple_has_live_reader {
        return None;
    }

    Some(FmtTupleElim {
        str_inputs,
        link_rebinds,
        dead_results,
        dead_bases: vec![tuple_var.id()],
    })
}

/// Recognize a multi-argument `format!` chain terminating at the
/// `alloc::fmt::format` op `(bf, fi)` and collect the rewrite plan, or
/// `None` for any shape outside the recognized subset (single-argument,
/// Debug, non-threaded, phi-merged) so the collapse leaves the graph
/// untouched.
pub(in crate::front::mir) fn collect_fmt_collapse_multi(
    graph: &FunctionGraph,
    bf: BlockId,
    fi: usize,
) -> Option<FmtCollapseMulti> {
    use crate::model::{CallTarget, OpKind};
    let block_f = graph.blocks.iter().find(|b| b.id == bf)?;
    let format_op = block_f.operations.get(fi)?;
    let fmt_args = match &format_op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if fmt_path_ends_with(segments, &["fmt", "format"]) => {
            args.first()?.clone().into_variable()
        }
        _ => return None,
    };
    let chain = extract_fmt_chain(graph, &fmt_args)?;
    if chain.args.len() < 2 {
        return None; // single-argument chains handled by `collapse_fmt_chains`
    }
    if chain.args.iter().any(|a| a.kind != FmtArgKind::Display)
        || chain
            .placeholders
            .iter()
            .any(|placeholder| !placeholder.is_default())
    {
        // `{:?}` Debug over an enum has no native rstr render (no
        // `rtype_str` on enum reprs); leave Debug chains residual.
        return None;
    }
    if chain
        .args
        .iter()
        .any(|a| !fmt_value_renders_via_rstr(graph, &a.value))
    {
        return None;
    }
    let pieces = chain.pieces.clone();

    // The `Arguments::new(pieces, args)` producer block: where the concat
    // fold is emitted (all rendered values are live there as the args-array
    // elements).  Require it distinct from the format block so a forwarding
    // link exists to re-thread the folded String onto.
    let (args_block, args_idx) = resolve_to_producer_op(graph, &fmt_args)?;
    if args_block == bf {
        return None;
    }
    let block_p = graph.blocks.iter().find(|b| b.id == args_block)?;
    let args_new_op = block_p.operations.get(args_idx)?;
    let (pieces_var, args_var) = match &args_new_op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if is_arguments_new_path(segments) => (args.first()?.clone(), args.get(1)?.clone()),
        _ => return None,
    };
    let arguments_var = args_new_op.result.as_ref()?.id();
    let piece_byte_vars = read_array_literal_elements(graph, &pieces_var)?;
    let arg_elem_vars = read_array_literal_elements(graph, &args_var)?;
    if arg_elem_vars.len() != chain.args.len() {
        return None;
    }

    // Each args-array element traces back to a `new_display` ctor; record it
    // for the in-place `→ str` rewrite.
    let mut new_display_ops = Vec::with_capacity(arg_elem_vars.len());
    for elem in &arg_elem_vars {
        let (b_i, idx_i) = resolve_to_producer_op(graph, elem)?;
        let blk = graph.blocks.iter().find(|b| b.id == b_i)?;
        let inner = match &blk.operations.get(idx_i)?.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if fmt_argument_ctor_kind(segments) == Some(FmtArgKind::Display) => {
                args.first()?.clone().into_variable()
            }
            _ => return None,
        };
        new_display_ops.push((b_i, idx_i, inner));
    }

    let mut dead_results = vec![arguments_var, pieces_var.id(), args_var.id()];
    dead_results.extend(piece_byte_vars.iter().map(|v| v.id()));
    let mut dead_bases = vec![pieces_var.id(), args_var.id()];

    // Also delete the `format_args!` argument-Tuple round-trip when the shape
    // permits (mirroring the single-arg path).  On success each placeholder's
    // `str` reads the rendered value instead of the Tuple field, so the Tuple
    // ctor + its writes/reads become dead.  On `None` the historic in-place
    // `str(field-read)` collapse stands (residual Tuple, no `fmt` extern).
    let mut link_rebinds = Vec::new();
    if let Some(elim) = attempt_fmt_tuple_elimination(graph, &chain, &new_display_ops) {
        for (slot, value) in new_display_ops.iter_mut().zip(elim.str_inputs) {
            slot.2 = value;
        }
        dead_results.extend(elim.dead_results);
        dead_bases.extend(elim.dead_bases);
        link_rebinds = elim.link_rebinds;
    }

    Some(FmtCollapseMulti {
        args_block,
        arguments_var,
        arg_elem_vars,
        arg_indices: chain
            .placeholders
            .iter()
            .map(|placeholder| placeholder.arg_index)
            .collect(),
        pieces,
        new_display_ops,
        format_block: bf,
        format_idx: fi,
        fmt_args,
        dead_results,
        dead_bases,
        link_rebinds,
    })
}

/// Expand every recognized multi-argument `format!` chain into native
/// `str` + `ll_strconcat` ops: each `Argument::new_display(inner)` becomes
/// `str(inner)` in place (rendering the placeholder; `str(&str)` folds to
/// identity), the rendered values are folded with the literal pieces at the
/// `Arguments::new` block, and the resulting String is threaded to the
/// `alloc::fmt::format` block (whose op becomes `same_as`).  The on-stack
/// args/pieces arrays and `Arguments::new` are deleted.  Every emitted op
/// the rtyper / codewriter / runtime already handle, so the graph-less
/// `fmt::rt::Argument` / `fmt::Arguments` externs no longer block the
/// rtyper.
pub(in crate::front::mir) fn collapse_fmt_chains_multi(graph: &mut FunctionGraph) -> usize {
    use crate::model::{CallTarget, LinkArg, OpKind, ValueType};
    let sites: Vec<FmtCollapseMulti> = graph
        .blocks
        .iter()
        .flat_map(|block| {
            block
                .operations
                .iter()
                .enumerate()
                .filter_map(move |(fi, op)| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if fmt_path_ends_with(segments, &["fmt", "format"]) => Some((block.id, fi)),
                    _ => None,
                })
        })
        .filter_map(|(bid, fi)| collect_fmt_collapse_multi(graph, bid, fi))
        .collect();
    if sites.is_empty() {
        return 0;
    }
    for site in &sites {
        // 1. Render each placeholder in place: `new_display(inner)` →
        //    `str(inner)`.  The result var is unchanged, so the rendered
        //    &str threads to the args block exactly where the opaque
        //    `Argument` value did.
        for (b_i, idx_i, inner) in &site.new_display_ops {
            let kind = str_render_kind(inner);
            if let Some(op) = graph.block_mut(*b_i).operations.get_mut(*idx_i) {
                if let Some(result) = &op.result {
                    stamp_as_rstr_str(result);
                }
                op.kind = kind;
            }
        }
        // 1b. Rebind any `format_args!` Tuple-ref slots to the rendered value
        //     (Tuple round-trip elimination): the threaded inputarg a `str`
        //     now reads carries the value instead of the deleted Tuple field.
        for (bid, ei, pos, value) in &site.link_rebinds {
            if let Some(link) = graph.block_mut(*bid).exits.get_mut(*ei)
                && let Some(arg) = link.args.get_mut(*pos)
            {
                *arg = LinkArg::Value(value.clone());
            }
        }
        // 2. Replace the `alloc::fmt::format` op with `same_as(fmt_args)`:
        //    after the link re-thread below, `fmt_args` carries the folded
        //    String, so the format result forwards it unchanged.
        let format_result = graph
            .block(site.format_block)
            .operations
            .get(site.format_idx)
            .and_then(|op| op.result.clone());
        if let Some(format_result) = format_result
            && !forward_identity(graph, &format_result, &site.fmt_args)
        {
            // The format result is a block input. Keep the copy.
            if let Some(op) = graph
                .block_mut(site.format_block)
                .operations
                .get_mut(site.format_idx)
            {
                op.kind = OpKind::UnaryOp {
                    op: "same_as".to_string(),
                    operand: site.fmt_args.clone(),
                    result_ty: ValueType::Ref(None),
                };
            }
        }
        // 3. Fold the rendered values with the literal pieces at the args
        //    block (`piece0 ++ rendered0 ++ piece1 ++ … ++ pieceN`).  The
        //    args-array elements are already rendered (step 1), so they are
        //    concatenated directly.
        let fold_chain = FmtChain {
            pieces: site.pieces.clone(),
            args: site
                .arg_elem_vars
                .iter()
                .map(|v| FmtArg {
                    value: v.clone(),
                    kind: FmtArgKind::Display,
                })
                .collect(),
            placeholders: site
                .arg_indices
                .iter()
                .map(|&arg_index| FmtPlaceholder {
                    arg_index,
                    flags: None,
                    width: None,
                    precision: None,
                    width_indirect: false,
                    precision_indirect: false,
                })
                .collect(),
        };
        let folded = emit_fmt_concat(graph, site.args_block, &fold_chain);
        // 4. Re-thread the folded String onto the link that forwarded the
        //    `Arguments` value out of the args block.
        for link in &mut graph.block_mut(site.args_block).exits {
            for arg in &mut link.args {
                if let LinkArg::Value(v) = arg
                    && v.id() == site.arguments_var
                {
                    *arg = LinkArg::Value(folded.clone());
                }
            }
        }
    }
    // 5. Delete the now-dead chain ops across all blocks (args/pieces
    //    arrays + their `FieldWrite`s, packed byte consts, `Arguments::new`).
    let dead_results: std::collections::HashSet<u64> = sites
        .iter()
        .flat_map(|s| s.dead_results.iter().copied())
        .collect();
    let dead_bases: std::collections::HashSet<u64> = sites
        .iter()
        .flat_map(|s| s.dead_bases.iter().copied())
        .collect();
    for block in &mut graph.blocks {
        block.operations.retain(|op| {
            if let Some(r) = &op.result
                && dead_results.contains(&r.id())
            {
                return false;
            }
            if let OpKind::FieldWrite { base, .. } = &op.kind
                && dead_bases.contains(&base.id())
            {
                return false;
            }
            true
        });
    }
    sites.len()
}

/// A recognized single-argument fieldless-enum `Debug` (`{:?}`) `format!`
/// chain ready to lower to a discriminant switch of per-variant constant
/// messages.  Where [`collapse_fmt_chains`] renders a `Display` argument
/// with `str(value)` (`ll_str`), a `Debug` render of a fieldless enum has
/// no native rstr counterpart — but every variant's `#[derive(Debug)]`
/// output is the bare variant identifier, a compile-time constant, and the
/// literal template pieces are constants too, so the whole rendered
/// message is a per-variant constant selected by the enum discriminant.
/// This reproduces the `discriminant → constant string` table shape a
/// fieldless (C-like) enum's `Debug` is (an int-indexed name table), which
/// the `Display`-only fmt collapser declines.
pub(in crate::front::mir) struct DebugEnumFmtCollapse {
    /// The `alloc::fmt::format` result var — the rendered `String` the
    /// continuation consumes; each switch arm forwards its constant message
    /// through this slot.
    pub(in crate::front::mir) format_result: Variable,
    /// The enum value being `Debug`-rendered (recovered through the
    /// `format_args!` argument Tuple), live at `format_block`.  A fieldless
    /// enum is `Int`-valued, so this IS the discriminant the switch reads
    /// directly (no `__discriminant` field read).
    pub(in crate::front::mir) enum_value: Variable,
    /// The literal template pieces around the single `{:?}` placeholder
    /// (`["intrinsic function ", " not implemented"]`).
    pub(in crate::front::mir) pieces: Vec<String>,
    /// `(discriminant, variant_ident)` for every variant — the switch arms
    /// and their per-variant constant messages.
    pub(in crate::front::mir) by_discr: Vec<(i64, String)>,
    /// `(block, exit_index, arg_pos, replacement)` — re-thread the residual
    /// chain values the predecessor links forwarded onto still-live values,
    /// so no link references a deleted result var once the chain is removed.
    pub(in crate::front::mir) link_rewrites: Vec<(BlockId, usize, usize, Variable)>,
    /// Op result var ids of the residual `new_debug` / `Arguments::new` /
    /// on-stack arrays / packed pieces bytes to delete.  Without this the
    /// switch would build but the graph-less `fmt::rt::Argument::new_debug`
    /// extern would survive in a predecessor block and keep walling the
    /// rtyper (`prune_dead_phis` never reclaims a dead `Call{FunctionPath}`).
    pub(in crate::front::mir) dead_results: Vec<u64>,
    /// Aggregate base var ids whose `FieldWrite`s are deleted.
    pub(in crate::front::mir) dead_bases: Vec<u64>,
}

/// Recover the fieldless-enum identity (crate-stripped `module::Enum`
/// spelling) for `enum_value` from its origin `Input` op's `class_root`.
/// A fieldless enum is `Int`-valued (`tyref_to_value_type`), so its
/// `Discriminant` read is an identity with no `__discriminant` field read
/// left to scavenge; the enum's variant-name table is keyed by the enum
/// type carried on the param (`tyref_fieldless_enum_class_root`,
/// in this file).  Traces the value back through the single-predecessor
/// link-arg forwarding chain to its defining `Input` op so a
/// framestate-renamed alias of the param still resolves.  `None` when the
/// value does not originate at a fieldless-enum-typed input (the collapse
/// then declines, leaving the chain residual).
pub(in crate::front::mir) fn debug_enum_disc_owner(
    graph: &FunctionGraph,
    enum_value: &Variable,
) -> Option<String> {
    let (block, idx) = resolve_to_producer_op(graph, enum_value)?;
    let op = graph
        .blocks
        .iter()
        .find(|b| b.id == block)?
        .operations
        .get(idx)?;
    match &op.kind {
        OpKind::Input {
            class_root: Some(root),
            ..
        } => Some(root.clone()),
        _ => None,
    }
}

/// Build `(discriminant, variant_ident)` for the fieldless enum named
/// `owner_root`, matching the `v.name` / `discriminant_i64()` pairing the
/// program's `enum_variant_by_discriminant` table records.
/// `None` unless the resolved type is a fieldless enum carrying at least
/// one integer-discriminant variant.
pub(in crate::front::mir) fn debug_enum_variants_by_discr(
    llbc: &Llbc,
    owner_root: &str,
) -> Option<Vec<(i64, String)>> {
    // The `__discriminant` read's `owner_root` is crate-stripped
    // (`strip_crate_prefix`, mir.rs Discriminant lowering) while
    // `item_meta.name_path()` carries the full crate-qualified path, so
    // compare on the crate-stripped spelling.
    let td = llbc
        .iter_type_decls()
        .find(|td| strip_crate_prefix(&td.item_meta.name_path()) == owner_root)?;
    let TypeDeclKind::Enum(variants) = &td.kind else {
        return None;
    };
    if !type_decl_is_fieldless_enum(td, llbc) {
        return None; // not fieldless
    }
    let by_discr: Vec<(i64, String)> = variants
        .iter()
        .filter_map(|v| v.discriminant_i64().map(|d| (d, v.name.clone())))
        .collect();
    (by_discr.len() == variants.len()).then_some(by_discr)
}

/// Recognize a single-argument fieldless-enum `Debug` `format!` chain
/// terminating at the `alloc::fmt::format` op `(bf, fi)`, or `None` for any
/// shape outside the subset (multi-argument, `Display`, non-fieldless-enum
/// argument, or a format op not closing its block with a single goto) so
/// the collapse leaves the graph untouched.
pub(in crate::front::mir) fn collect_debug_enum_fmt_collapse(
    graph: &FunctionGraph,
    llbc: &Llbc,
    bf: BlockId,
    fi: usize,
) -> Option<DebugEnumFmtCollapse> {
    let block_f = graph.blocks.iter().find(|b| b.id == bf)?;
    // The format op must be the block's last op (so the switch replaces the
    // block tail) closed by a single plain goto to the continuation.
    if fi + 1 != block_f.operations.len() {
        return None;
    }
    let [exit] = block_f.exits.as_slice() else {
        return None;
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return None;
    }
    // Navigate the chain (shared with the `Display` collapse); the `Debug`
    // gate below selects only the `{:?}` renders, and the navigation also
    // supplies the residual-chain delete / re-thread plan the switch needs.
    let nav = navigate_single_arg_fmt_chain(graph, bf, fi)?;
    if nav.kind != FmtArgKind::Debug || !nav.placeholder.is_default() {
        return None; // Display is handled by `collapse_fmt_chains`
    }
    // The rendered value is the enum being `Debug`-formatted.
    let enum_value = nav.context;
    // Recover the enum identity from the value's origin `Input` op
    // `class_root`; without one the value is not a modeled fieldless enum
    // here (a fieldless enum is `Int`-valued, so no `__discriminant` field
    // read survives to identify it).
    let owner_root = debug_enum_disc_owner(graph, &enum_value)?;
    let by_discr = debug_enum_variants_by_discr(llbc, &owner_root)?;
    Some(DebugEnumFmtCollapse {
        format_result: nav.format_result,
        enum_value,
        pieces: nav.pieces,
        by_discr,
        link_rewrites: nav.link_rewrites,
        dead_results: nav.dead_results,
        dead_bases: nav.dead_bases,
    })
}

/// Lower every recognized fieldless-enum `Debug` (`{:?}`) `format!` chain to
/// a discriminant switch of per-variant constant messages.  The
/// `alloc::fmt::format` block is split: its tail becomes an N-way switch on
/// the enum's `__discriminant`, one arm per variant emitting the constant
/// `"{pre}{variant}{post}"` message (`emit_str_const`, natively `SomeString`)
/// and forwarding it to the continuation through the format result slot; the
/// default arm raises the implicit `AssertionError` (a closed enum's
/// out-of-range tag can't occur, the shape `remove_assertion_errors`
/// prunes).  The residual `new_debug` / `Arguments::new` / on-stack arrays
/// in the predecessor blocks are deleted explicitly here (their cross-block
/// forwards re-threaded onto live values first) — no sweep reclaims a dead
/// `Call{FunctionPath}`, so leaving them in place would let the graph-less
/// `fmt::rt::Argument::new_debug` extern keep walling the rtyper even after
/// the switch is built.  Every emitted op the rtyper / codewriter / runtime
/// already handle.  Returns the number of chains lowered.
///
/// Fail-safe: any structural mismatch leaves the chain residual (census
/// Skip), never a regression.
pub(in crate::front::mir) fn collapse_debug_enum_fmt_chains(
    graph: &mut FunctionGraph,
    llbc: &Llbc,
) -> usize {
    let targets: Vec<(BlockId, usize)> = graph
        .blocks
        .iter()
        .flat_map(|block| {
            block
                .operations
                .iter()
                .enumerate()
                .filter_map(move |(fi, op)| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if fmt_path_ends_with(segments, &["fmt", "format"]) => Some((block.id, fi)),
                    _ => None,
                })
        })
        .collect();
    let sites: Vec<DebugEnumFmtCollapse> = targets
        .into_iter()
        .filter_map(|(bid, fi)| collect_debug_enum_fmt_collapse(graph, llbc, bid, fi))
        .collect();
    if sites.is_empty() {
        return 0;
    }
    // 1. Re-thread the residual chain's cross-block forwards onto live
    //    values so no link references a chain result var about to be deleted
    //    (B0→Bp forwarded `new_debug`, Bp→Bf forwarded `Arguments`).
    for site in &sites {
        for (bid, ei, pos, repl) in &site.link_rewrites {
            if let Some(link) = graph.block_mut(*bid).exits.get_mut(*ei)
                && let Some(arg) = link.args.get_mut(*pos)
            {
                *arg = LinkArg::Value(repl.clone());
            }
        }
    }
    // 2. Delete the now-dead chain ops across all blocks.  Essential: the
    //    switch replaces only the `alloc::fmt::format` op, so without this
    //    the graph-less `fmt::rt::Argument::new_debug` extern survives in a
    //    predecessor block and keeps walling the rtyper (no sweep reclaims a
    //    dead `Call{FunctionPath}` — `prune_dead_phis` leaves it).
    let dead_results: std::collections::HashSet<u64> = sites
        .iter()
        .flat_map(|s| s.dead_results.iter().copied())
        .collect();
    let dead_bases: std::collections::HashSet<u64> = sites
        .iter()
        .flat_map(|s| s.dead_bases.iter().copied())
        .collect();
    for block in &mut graph.blocks {
        block.operations.retain(|op| {
            if let Some(r) = &op.result
                && dead_results.contains(&r.id())
            {
                return false;
            }
            if let OpKind::FieldWrite { base, .. } = &op.kind
                && dead_bases.contains(&base.id())
            {
                return false;
            }
            true
        });
    }
    // 3. Replace each `alloc::fmt::format` op with the discriminant switch of
    //    per-variant constant messages.
    let mut lowered = 0;
    for site in &sites {
        if rewrite_debug_enum_fmt_site(graph, site) {
            lowered += 1;
        }
    }
    lowered
}

/// Split the format block into the discriminant switch for one recognized
/// site.  Returns `false` (leaving the graph untouched) on any threading /
/// arity mismatch.
pub(in crate::front::mir) fn rewrite_debug_enum_fmt_site(
    graph: &mut FunctionGraph,
    site: &DebugEnumFmtCollapse,
) -> bool {
    use crate::flowspace::model::ConstValue;
    use crate::front::option_variant::close_goto_mixed;
    use crate::model::{ExitCase, ExitSwitch, Link};
    // Re-locate the format op (block/op indices are stable within this
    // pass, but resolve by result var to stay robust to prior rewrites).
    let Some((bf_idx, fi)) = graph.blocks.iter().enumerate().find_map(|(bi, b)| {
        b.operations
            .iter()
            .position(|op| op.result.as_ref() == Some(&site.format_result))
            .map(|oi| (bi, oi))
    }) else {
        return false;
    };
    let bf = graph.blocks[bf_idx].id;
    // The saved single goto exit → continuation; each arm reproduces it
    // with the format-result slot sourced from the arm's constant message.
    let Some([saved_exit]) = graph.blocks[bf_idx].exits.first().map(std::slice::from_ref) else {
        return false;
    };
    let saved_exit = saved_exit.clone();
    let b_cont = saved_exit.target;

    // The enum value must be live at `bf` to read its discriminant.  It is
    // already an inputarg on the real graphs (the switch dispatch in the
    // entry block forwards `func` down every arm), so this is a no-op there;
    // decline rather than panic if a shape cannot thread it.
    if !graph.variable_defined_in_block(bf, &site.enum_value)
        && !can_thread_var_to_block(graph, bf, &site.enum_value)
    {
        return false;
    }
    // `ssa.py` `SSA_to_SSI` `record_used_var(block.exitswitch)`: the
    // switch is a use in `bf`, so it must name this block's local
    // (`Variable.copy` / `Block.renamevariables`), not the producer.
    let Some(disc) = graph.ensure_variable_at_block_local(bf, &site.enum_value) else {
        return false;
    };

    // `carried` = the distinct live Values the saved exit forwards other
    // than the format result; each threads through every arm to reach the
    // continuation.  The default (raise) arm carries nothing.
    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != site.format_result
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    // All structural checks passed; mutate the graph.

    // Drop the residual `alloc::fmt::format` op (its block tail).  A
    // fieldless enum is `Int`-valued (`tyref_to_value_type`), so its value
    // IS the discriminant — switch on it directly, no `__discriminant`
    // field read (a getfield off an integer base has no blackhole handler).
    graph.blocks[bf_idx].operations.remove(fi);

    // One arm per variant: emit the constant message and forward it to the
    // continuation through the format-result slot, threading `carried`.
    let mut links: Vec<Link> = Vec::with_capacity(site.by_discr.len() + 1);
    for (discr, ident) in &site.by_discr {
        let (arm, arm_inputs) = graph.create_block_with_arg_vars(carried.len());
        let msg = format!("{}{}{}", site.pieces[0], ident, site.pieces[1]);
        let msg_var = emit_str_const(graph, arm, &msg);
        let arm_args = match debug_enum_reproduce_exit_args(
            &saved_exit,
            &site.format_result,
            &msg_var,
            &carried,
            &arm_inputs,
        ) {
            Some(args) => args,
            None => return false,
        };
        close_goto_mixed(graph, arm, b_cont, arm_args);
        links.push(
            Link::from_variables(
                graph,
                carried.clone(),
                arm,
                Some(ExitCase::Const(ConstValue::Int(*discr))),
            )
            .with_prevblock(bf)
            .with_llexitcase_from_exitcase(),
        );
    }
    // Default arm: a closed enum's tag is always one of the above, so this
    // path "shouldn't occur" — raise the implicit `AssertionError`
    // `remove_assertion_errors` prunes.
    let (default_arm, _) = graph.create_block_with_arg_vars(0);
    graph.set_raise_implicit(
        default_arm,
        "fieldless-enum Debug over unknown discriminant",
    );
    links.push(
        Link::from_variables(
            graph,
            Vec::new(),
            default_arm,
            Some(ExitCase::Const(ConstValue::UniStr("default".into()))),
        )
        .with_prevblock(bf),
    );

    graph.block_mut(bf).exitswitch = Some(ExitSwitch::Value(disc));
    graph.recloseblock(bf, links);
    true
}

/// Reproduce the format block's saved exit args for a switch arm: the
/// format-result slot is sourced from the arm's constant message, every
/// other live Value from the arm's threaded inputarg, constants pass
/// through.  `None` if a forwarded Value was not threaded into the arm.
pub(in crate::front::mir) fn debug_enum_reproduce_exit_args(
    saved: &crate::model::Link,
    format_result: &Variable,
    msg_var: &Variable,
    carried: &[Variable],
    inputs: &[Variable],
) -> Option<Vec<LinkArg>> {
    let mut out = Vec::with_capacity(saved.args.len());
    for arg in &saved.args {
        match arg {
            LinkArg::Const(c) => out.push(LinkArg::Const(c.clone())),
            LinkArg::Value(v) if v == format_result => out.push(LinkArg::Value(msg_var.clone())),
            LinkArg::Value(v) => {
                let pos = carried.iter().position(|c| c == v)?;
                out.push(LinkArg::Value(inputs[pos].clone()));
            }
        }
    }
    Some(out)
}

/// `var` can thread to `block` iff it is already defined there or reachable
/// through a single-predecessor chain — the precondition under which
/// [`FunctionGraph::ensure_variable_at_block`] succeeds without panicking.
pub(in crate::front::mir) fn can_thread_var_to_block(
    graph: &FunctionGraph,
    block: BlockId,
    var: &Variable,
) -> bool {
    if graph.variable_defined_in_block(block, var) {
        return true;
    }
    let mut cur = block;
    let mut hops = 0usize;
    loop {
        hops += 1;
        if hops > graph.blocks.len() {
            return false;
        }
        let preds = graph.predecessors(cur);
        if preds.len() != 1 {
            return false;
        }
        let p = preds[0];
        if graph.variable_defined_in_block(p, var) {
            return true;
        }
        cur = p;
    }
}

#[cfg(test)]
mod tests;
