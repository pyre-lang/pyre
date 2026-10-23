//! Format the SSA representation as text for tests.
//!
//! Translated from `rpython/jit/codewriter/format.py`. Pyre's
//! [`SSARepr`](crate::flatten::SSARepr) holds typed [`FlatOp`] variants
//! rather than raw tuples, so the formatter produces the same textual
//! shape as upstream's `format_assembler` while reading the Rust types
//! directly.
//!
//! - Registers print as `%i<n>`, `%r<n>`, `%f<n>`.
//! - Constants print as `$<value>` (matching `format.py:23`).
//! - Labels print as `L<index>`, numbered by first printed mention —
//!   label line or jump operand, whichever comes first (matching
//!   `format.py getlabelname`).
//! - `ListOfKind` argument groups print as `I[…]`, `R[…]`, `F[…]`
//!   (matching `format.py:27`).
//! - Call descriptors print via their `Debug` repr (matching
//!   `format.py` `repr(AbstractDescr)`).
//! - When `ssarepr.insns_pos` is set the formatter prefixes each line
//!   with `'%4d  '` (matching `format.py`).
//! - The trailing `('---',)` sentinel (`format.py`) is trimmed.
//!
//! The reverse direction (`unformat_assembler`) is intentionally not
//! ported — the parity tests we run end-to-end build SSARepr through
//! the codewriter pipeline rather than parsing assembler text.  When a
//! caller actually needs to round-trip text → SSARepr the missing
//! function should be ported here mirroring `format.py:104-167` line
//! by line.

use std::collections::HashMap;
use std::fmt::Write;

use crate::codewriter::assembler::rewrite_call_kinds;
use crate::flatten::{FlatOp, Label, RegKind, RegOrConst, SSARepr};

/// `flatten.py Register.kind[0]` — single-char prefix used in
/// `int_copy`/`ref_copy`/`float_copy` opnames.
fn kind_short_name(kind: RegKind) -> &'static str {
    match kind {
        RegKind::Int => "int",
        RegKind::Ref => "ref",
        RegKind::Float => "float",
    }
}

/// `flatten.py getcolor` returns either a [`Register`] (printed
/// as `%i<n>`/`%r<n>`/`%f<n>`) or a [`crate::flowspace::model::Constant`]
/// (printed as `$<value>`).  This helper renders the union form for
/// `int_copy` source operands and `int_return` arguments.
fn regorconst_repr(arg: &RegOrConst) -> String {
    match arg {
        RegOrConst::Reg(r) => r.repr(),
        RegOrConst::Const(c) => format!("${}", c.value),
    }
}

/// `format.py getlabelname(lbl)`.
///
/// Numbering is lazy: a label takes the next ordinal the first time it is
/// *printed*, whether that mention is a jump operand (`format.py:24-25`
/// `repr(TLabel)`) or the label line itself (`format.py`).  In a
/// graph with a back edge the two orders differ — the loop header's label
/// line prints before the back edge that targets it.  `None` is the
/// `-1` sentinel the marking pass stores.
fn getlabelname(
    label: Label,
    seenlabels: &mut HashMap<Label, Option<usize>>,
    labelcount: &mut usize,
) -> usize {
    let slot = seenlabels.entry(label).or_insert(None);
    if let Some(num) = *slot {
        return num;
    }
    *labelcount += 1;
    *slot = Some(*labelcount);
    *labelcount
}

impl crate::jitcode::SsaReprDump for SSARepr {
    fn format_assembler(&self) -> String {
        format_assembler(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// `format.py format_assembler(ssarepr)`.  Per-arg kinds for
/// `OpKind::Call` argument lists resolve via `getkind(v.concretetype)`
/// read directly from each operand `Variable`'s `concretetype` cell
/// (`flowspace/model.py` `__slots__ = [..., "concretetype"]`); no
/// side-table is consulted.
///
/// **PRE-EXISTING-ADAPTATION** — upstream's `SSARepr` already holds
/// post-flatten [`crate::flatten::Register`]s (the regalloc color) by
/// the time `format_assembler` runs, so `format.py` renders
/// `'%%%s%d' % (x.kind[0], x.index)` straight off the color.  Pyre's
/// `serialize_op` (`flatten.py` analogue) defers per-op arg
/// rewriting — `FlatOp::Op` keeps the unflattened
/// [`crate::model::SpaceOperation`] with `Variable` operands — so the
/// register suffix here renders `Variable.id()` (process-wide
/// identity) rather than the regalloc color.  Convergence path:
/// colorize op operands at flatten (`flatten_list(op.args)`) so the
/// `SSARepr` becomes self-contained and this formatter reads
/// `Register.index` with neither a graph nor the regalloc result.
pub fn format_assembler(ssarepr: &SSARepr) -> String {
    // format.py:37-44: the first pass only *marks* every label that
    // appears as a jump target (`seenlabels[x.name] = -1`).  It assigns
    // no numbers — those come from `getlabelname` during the output loop
    // below.
    let mut seenlabels: HashMap<Label, Option<usize>> = HashMap::new();
    for op in &ssarepr.insns {
        match op {
            FlatOp::Jump(label)
            | FlatOp::CatchException { target: label }
            | FlatOp::GotoIfExceptionMismatch { target: label, .. }
            | FlatOp::IntBinOpJumpIfOvf { target: label, .. }
            | FlatOp::GotoIfNot { target: label, .. }
            | FlatOp::GotoIfNotOp { target: label, .. } => {
                seenlabels.insert(*label, None);
            }
            FlatOp::Switch { targets, .. } => {
                for (_, label) in targets {
                    seenlabels.insert(*label, None);
                }
            }
            _ => {}
        }
    }
    let mut labelcount = 0usize;

    // format.py:53-55:
    //   insns = ssarepr.insns
    //   if insns and insns[-1] == ('---',):
    //       insns = insns[:-1]
    let insns: &[FlatOp] = match ssarepr.insns.last() {
        Some(FlatOp::EndOfBlock) => &ssarepr.insns[..ssarepr.insns.len() - 1],
        _ => &ssarepr.insns[..],
    };

    let mut out = String::new();
    for (i, op) in insns.iter().enumerate() {
        // format.py:57-60: prefix = '%4d  ' % ssarepr._insns_pos[i] when set.
        let prefix = match &ssarepr.insns_pos {
            Some(positions) => positions
                .get(i)
                .map(|p| format!("{p:>4}  "))
                .unwrap_or_default(),
            None => String::new(),
        };
        match op {
            // format.py:61-63 — only a label the marking pass saw as a
            // jump target gets a line, and printing that line is itself a
            // first mention that can claim the next ordinal.
            FlatOp::Label(label) => {
                if seenlabels.contains_key(label) {
                    let num = getlabelname(*label, &mut seenlabels, &mut labelcount);
                    let _ = writeln!(out, "{prefix}L{num}:");
                }
            }
            FlatOp::Op(space_op) => {
                let args = op_args_repr(space_op);
                if args.is_empty() {
                    let _ = writeln!(out, "{prefix}{}", op_name(space_op));
                } else {
                    let _ = writeln!(out, "{prefix}{} {args}", op_name(space_op));
                }
            }
            FlatOp::Jump(label) => {
                let num = getlabelname(*label, &mut seenlabels, &mut labelcount);
                let _ = writeln!(out, "{prefix}goto L{num}");
            }
            FlatOp::CatchException { target } => {
                let num = getlabelname(*target, &mut seenlabels, &mut labelcount);
                let _ = writeln!(out, "{prefix}catch_exception L{num}");
            }
            FlatOp::GotoIfExceptionMismatch { llexitcase, target } => {
                let num = getlabelname(*target, &mut seenlabels, &mut labelcount);
                let _ = writeln!(
                    out,
                    "{prefix}goto_if_exception_mismatch ${llexitcase}, L{num}"
                );
            }
            FlatOp::GotoIfNot { cond, target } => {
                let num = getlabelname(*target, &mut seenlabels, &mut labelcount);
                let _ = writeln!(out, "{prefix}goto_if_not {}, L{num}", cond.repr());
            }
            FlatOp::GotoIfNotOp {
                opname,
                args,
                target,
            } => {
                let num = getlabelname(*target, &mut seenlabels, &mut labelcount);
                let arglist = args
                    .iter()
                    .map(|reg| reg.repr())
                    .collect::<Vec<_>>()
                    .join(", ");
                let _ = writeln!(out, "{prefix}goto_if_not_{opname} {arglist}, L{num}");
            }
            FlatOp::Switch { value, targets } => {
                let cases: Vec<String> = targets
                    .iter()
                    .map(|(key, label)| {
                        let num = getlabelname(*label, &mut seenlabels, &mut labelcount);
                        format!("{key}:L{num}")
                    })
                    .collect();
                let _ = writeln!(
                    out,
                    "{prefix}switch {}, <SwitchDictDescr {}>",
                    value.repr(),
                    cases.join(", ")
                );
            }
            FlatOp::IntBinOpJumpIfOvf {
                op,
                target,
                lhs,
                rhs,
                dst,
            } => {
                let opname = match op {
                    crate::flatten::IntOvfOp::Add => "int_add_jump_if_ovf",
                    crate::flatten::IntOvfOp::Sub => "int_sub_jump_if_ovf",
                    crate::flatten::IntOvfOp::Mul => "int_mul_jump_if_ovf",
                };
                let num = getlabelname(*target, &mut seenlabels, &mut labelcount);
                let _ = writeln!(
                    out,
                    "{prefix}{opname} L{num}, {}, {} -> {}",
                    lhs.repr(),
                    rhs.repr(),
                    dst.repr()
                );
            }
            // `flatten.py:333-335` — opnames are kind-prefixed
            // (`int_copy`/`ref_copy`/`float_copy`,
            // `int_push`/`ref_push`/`float_push`,
            // `int_pop`/`ref_pop`/`float_pop`).  After Phase 3 the
            // [`Register`] operand carries its kind directly, so the
            // formatter no longer reaches into a side-table — it just
            // reads `dst.kind` / `src.kind` for the prefix.
            FlatOp::Move { dst, src } => {
                let kind = kind_short_name(dst.kind);
                let _ = writeln!(
                    out,
                    "{prefix}{kind}_copy {} -> {}",
                    regorconst_repr(src),
                    dst.repr()
                );
            }
            FlatOp::Push(src) => {
                let kind = kind_short_name(src.kind);
                let _ = writeln!(out, "{prefix}{kind}_push {}", src.repr());
            }
            FlatOp::Pop(dst) => {
                let kind = kind_short_name(dst.kind);
                let _ = writeln!(out, "{prefix}{kind}_pop -> {}", dst.repr());
            }
            FlatOp::LastException { dst } => {
                let _ = writeln!(out, "{prefix}last_exception -> {}", dst.repr());
            }
            FlatOp::LastExcValue { dst } => {
                let _ = writeln!(out, "{prefix}last_exc_value -> {}", dst.repr());
            }
            FlatOp::Live { live_values } => {
                let mut names: Vec<String> = live_values.iter().map(|reg| reg.repr()).collect();
                // format.py: `if asm[0] == '-live-': lst.sort()`.
                names.sort();
                // format.py:64-79: the args (and the leading space) print only
                // `if len(asm) > 1`; an empty `-live-` renders with no trailing
                // space.
                if names.is_empty() {
                    let _ = writeln!(out, "{prefix}-live-");
                } else {
                    let _ = writeln!(out, "{prefix}-live- {}", names.join(", "));
                }
            }
            FlatOp::Reraise => {
                let _ = writeln!(out, "{prefix}reraise");
            }
            FlatOp::IntReturn(v) => {
                let _ = writeln!(out, "{prefix}int_return {}", regorconst_repr(v));
            }
            FlatOp::RefReturn(v) => {
                let _ = writeln!(out, "{prefix}ref_return {}", regorconst_repr(v));
            }
            FlatOp::FloatReturn(v) => {
                let _ = writeln!(out, "{prefix}float_return {}", regorconst_repr(v));
            }
            FlatOp::VoidReturn => {
                let _ = writeln!(out, "{prefix}void_return");
            }
            FlatOp::Raise(v) => {
                let _ = writeln!(out, "{prefix}raise {}", regorconst_repr(v));
            }
            FlatOp::EndOfBlock => {
                let _ = writeln!(out, "{prefix}---");
            }
            FlatOp::Unreachable => {
                let _ = writeln!(out, "{prefix}unreachable");
            }
        }
    }
    out
}

/// format.py `assert_format(ssarepr, expected)`.
///
/// Compares the formatted SSARepr with `expected` line by line.  When a
/// line differs we emit the same `Got:` / `Expected:` diff format as
/// upstream so failing tests are easy to read.
pub fn assert_format(ssarepr: &SSARepr, expected: &str) {
    let asm = format_assembler(ssarepr);
    let expected = if expected.is_empty() {
        String::new()
    } else {
        // Normalize multiline raw-string indentation the way
        // py.code.Source(expected).strip() does in upstream.
        normalize_expected(expected)
    };
    let asm_lines: Vec<&str> = asm.split('\n').collect();
    let exp_lines: Vec<&str> = expected.split('\n').collect();
    for (asm_line, exp_line) in asm_lines.iter().zip(exp_lines.iter()) {
        if asm_line != exp_line {
            let mut msg = String::new();
            msg.push('\n');
            let _ = writeln!(msg, "Got:      {asm_line}");
            let _ = writeln!(msg, "Expected: {exp_line}");
            let mut common = 0usize;
            for (a, e) in asm_line.chars().zip(exp_line.chars()) {
                if a == e {
                    common += 1;
                } else {
                    break;
                }
            }
            let _ = writeln!(msg, "          {}^^^^", " ".repeat(common));
            panic!("{msg}");
        }
    }
    assert_eq!(asm_lines.len(), exp_lines.len(), "line-count mismatch");
}

fn normalize_expected(expected: &str) -> String {
    // Strip the leading and trailing blank lines, then trim the common
    // indentation (similar to py.code.Source(...).strip()).
    let raw_lines: Vec<&str> = expected.split('\n').collect();
    let trimmed: Vec<&str> = raw_lines
        .iter()
        .copied()
        .skip_while(|l| l.trim().is_empty())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .skip_while(|l| l.trim().is_empty())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let indent = trimmed
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);
    let mut out = trimmed
        .iter()
        .map(|l| if l.len() >= indent { &l[indent..] } else { *l })
        .collect::<Vec<_>>()
        .join("\n");
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// `format.py split_words(line)`.
pub fn split_words(line: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut nested = 0i32;
    let padded = format!("  {line}");

    for (i, c) in line.char_indices() {
        if c == ' ' && nested == 0 {
            if !word.is_empty() {
                words.push(std::mem::take(&mut word));
            }
        } else {
            word.push(c);
            if matches!(c, '<' | '(' | '[') {
                nested += 1;
            }
            if matches!(c, ']' | ')' | '>') && padded.as_bytes().get(i..i + 4) != Some(b" -> ") {
                nested -= 1;
                assert!(nested >= 0);
            }
        }
    }
    if !word.is_empty() {
        words.push(word);
    }
    assert_eq!(nested, 0);
    words
}

fn register_repr_for_kind(suffix: u64, kind: RegKind) -> String {
    let prefix = match kind {
        RegKind::Int => 'i',
        RegKind::Ref => 'r',
        RegKind::Float => 'f',
    };
    format!("%{prefix}{suffix}")
}

/// format.py:26-27 `ListOfKind` formatter.
///
/// Upstream emits `'%s[%s]' % (x.kind[0].upper(), ', '.join(map(repr, x)))`.
/// Pyre's call-family `OpKind` variants split args into typed
/// `args_i`/`args_r`/`args_f` Vecs, so the kind char is fixed per slot —
/// it is passed in directly rather than fetched from a side-table.
/// Argument storage is `Vec<Variable>` (orthodox per `flowspace/model.py:Variable`);
/// the Variable's register suffix renders `Variable.id()` (the
/// PRE-EXISTING-ADAPTATION documented on [`format_assembler`]).
fn list_of_kind_repr_vars(kind_char: char, args: &[crate::flowspace::model::Variable]) -> String {
    let kind = match kind_char.to_ascii_lowercase() {
        'i' => RegKind::Int,
        'f' => RegKind::Float,
        _ => RegKind::Ref,
    };
    let parts: Vec<String> = args
        .iter()
        .map(|v| register_repr_for_kind(v.id(), kind))
        .collect();
    format!("{}[{}]", kind_char.to_ascii_uppercase(), parts.join(", "))
}

/// format.py:20-23 — render a `funcptr` slot.
///
/// Upstream emits `$<* struct <name>>` for `Constant(lltype.Ptr(Struct))`
/// and `$<value>` otherwise.  Pyre's codewrite-time funcptr surrogate is
/// either a symbolic [`crate::model::CallTarget`] or a runtime
/// [`crate::flowspace::model::Variable`].
fn call_target_repr(target: &crate::model::CallTarget) -> String {
    use crate::model::CallTarget;
    match target {
        CallTarget::Method {
            name,
            receiver_root,
            ..
        } => match receiver_root {
            Some(root) => format!("$<* function '{root}.{name}'>"),
            None => format!("$<* function '{name}'>"),
        },
        CallTarget::FunctionPath { segments, .. } => {
            format!("$<* function '{}'>", segments.join("."))
        }
        CallTarget::SyntheticTransparentCtor {
            name, owner_path, ..
        } => {
            if owner_path.is_empty() {
                format!("$<* synthetic-transparent-ctor '{name}'>")
            } else {
                format!(
                    "$<* synthetic-transparent-ctor '{}.{name}'>",
                    owner_path.join(".")
                )
            }
        }
        CallTarget::Indirect {
            trait_root,
            method_name,
            ..
        } => format!("$<* indirect 'dyn {trait_root}::{method_name}'>"),
        CallTarget::UnsupportedExpr => "$<unsupported call target>".to_string(),
    }
}

fn call_funcptr_repr(funcptr: &crate::model::CallFuncPtr) -> String {
    match funcptr {
        crate::model::CallFuncPtr::Target(target) => call_target_repr(target),
        // RPython's funcptr slot is `lltype.Ptr(FUNC)` (kind 'r')
        // by construction.  Pyre's lowering, however, can
        // materialize a funcptr Variable as Int when the rtyper
        // chose the integer-indexed dispatch path (e.g. opcode
        // dispatch tables); `variable_kind` reads each Variable's
        // `concretetype` cell directly via `getkind` so the funcptr
        // kind matches the upstream `getkind(v.concretetype)` slot
        // shape.  Falls back to `Ref` when the cell is unset.
        crate::model::CallFuncPtr::Value(var) => {
            let kind = variable_kind(var).unwrap_or(RegKind::Ref);
            register_repr_for_kind(var.id(), kind)
        }
    }
}

fn op_name(op: &crate::model::SpaceOperation) -> String {
    use crate::model::OpKind;
    match &op.kind {
        OpKind::Call { .. } => "call".to_string(),
        OpKind::ConstInt(_) | OpKind::ConstFnAddr { .. } => "const_int".to_string(),
        OpKind::ConstUInt(_) => "const_uint".to_string(),
        OpKind::ConstFloat(_) => "const_float".to_string(),
        OpKind::ConstStr(_) => "const_str".to_string(),
        OpKind::ConstInternedStr(_) => "const_interned_str".to_string(),
        OpKind::CallElidable {
            result_kind,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            format!(
                "call_elidable_{}_{result_kind}",
                rewrite_call_kinds(args_i, args_r, args_f, *result_kind)
            )
        }
        OpKind::CallResidual {
            result_kind,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            format!(
                "residual_call_{}_{result_kind}",
                rewrite_call_kinds(args_i, args_r, args_f, *result_kind)
            )
        }
        OpKind::CallMayForce {
            result_kind,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            format!(
                "call_may_force_{}_{result_kind}",
                rewrite_call_kinds(args_i, args_r, args_f, *result_kind)
            )
        }
        OpKind::InlineCall {
            result_kind,
            args_i,
            args_r,
            args_f,
            ..
        } => {
            format!(
                "inline_call_{}_{result_kind}",
                rewrite_call_kinds(args_i, args_r, args_f, *result_kind)
            )
        }
        OpKind::RecursiveCall { result_kind, .. } => {
            format!("recursive_call_{result_kind}")
        }
        // The Debug fallback below would spell it "guardclass"; the
        // canonical opname carries the underscore.
        OpKind::GuardClass { .. } => "guard_class".to_string(),
        // For the rest, fall back on a stable Debug-derived discriminant.
        other => format!("{:?}", other)
            .split('{')
            .next()
            .unwrap_or("?")
            .split('(')
            .next()
            .unwrap_or("?")
            .trim()
            .to_lowercase(),
    }
}

fn op_args_repr(op: &crate::model::SpaceOperation) -> String {
    use crate::model::OpKind;
    let mut out = String::new();
    match &op.kind {
        // `OpKind::Call` carries a heterogeneous argument list (no
        // per-slot kind on the variant).  Each arg's kind reads
        // straight from its `Variable.concretetype` cell via
        // [`variable_kind`] — same `getkind(v.concretetype)` source
        // PyPy uses.  Falls back to the Ref shape when the cell is
        // unset (anchor-test fixtures that build SSA shapes without
        // running the rtyper).
        OpKind::Call { args, .. } => {
            // Each Variable's register suffix renders `Variable.id()`
            // (process-wide identity) — the PRE-EXISTING-ADAPTATION
            // documented on [`format_assembler`]; upstream renders the
            // post-flatten `Register.index` color.
            let parts: Vec<String> = args
                .iter()
                .map(|v| {
                    let kind = variable_kind(v).unwrap_or(RegKind::Ref);
                    register_repr_for_kind(v.id(), kind)
                })
                .collect();
            out.push_str(&parts.join(", "));
        }
        // format.py `'$%r' % (x.value,)` — constants print as $<value>.
        OpKind::ConstInt(value) => {
            let _ = write!(out, "${value}");
        }
        OpKind::ConstFnAddr { value, .. } => {
            let _ = write!(out, "${value}");
        }
        OpKind::ConstUInt(value) => {
            let _ = write!(out, "${value}");
        }
        OpKind::ConstFloat(bits) => {
            let _ = write!(out, "${}", f64::from_bits(*bits));
        }
        OpKind::ConstStr(bytes) | OpKind::ConstInternedStr(bytes) => {
            let _ = write!(out, "${bytes:?}");
        }
        // jtransform.py `rewrite_call`:
        //   sublists = [lst_i?, lst_r?, lst_f?, calldescr?]   # only kinds present
        //   args = initialargs + sublists
        // → for residual_call/call_may_force/call_elidable upstream emits
        //   `$<funcptr>, I[…]?, R[…]?, F[…]?, <descr>` where the I/R/F
        //   slots are gated on the opname kind signature.  Pyre carries
        //   the funcptr identity on the dedicated `funcptr` field per
        //   jtransform.py `[op.args[0]] + extraargs`.
        OpKind::CallElidable {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
            ..
        }
        | OpKind::CallResidual {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
            ..
        }
        | OpKind::CallMayForce {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
            ..
        } => {
            let mut parts = vec![call_funcptr_repr(funcptr)];
            // jtransform.py `rewrite_call` — emit each ListOfKind exactly when
            // the matching kind char is in the signature, which is not the
            // same test as "the bin is non-empty": `ir` over no int arguments
            // still announces an empty `I[]`.
            let kinds = rewrite_call_kinds(args_i, args_r, args_f, *result_kind);
            if kinds.contains('i') {
                parts.push(list_of_kind_repr_vars('i', args_i));
            }
            if kinds.contains('r') {
                parts.push(list_of_kind_repr_vars('r', args_r));
            }
            if kinds.contains('f') {
                parts.push(list_of_kind_repr_vars('f', args_f));
            }
            // jtransform.py:434 — descr is the last sublist when set.
            parts.push(format!("{:?}", descriptor.extra_info));
            out.push_str(&parts.join(", "));
        }
        // jtransform.py `handle_regular_call`:
        //   args = [jitcode] + [I?, R?, F? sublists]   # only kinds present
        // → format.py:34-35 renders the JitCode object via JitCode.__repr__.
        //   Before the codewriter assigns the final dense index, fall back
        //   to the symbolic jitcode name for debugging.
        OpKind::InlineCall {
            jitcode,
            args_i,
            args_r,
            args_f,
            result_kind,
            ..
        } => {
            let head = match jitcode.try_index() {
                Some(index) => format!("<JitCode #{index}>"),
                None => format!("<JitCode {:?}>", jitcode.name),
            };
            let mut parts = vec![head];
            let kinds = rewrite_call_kinds(args_i, args_r, args_f, *result_kind);
            if kinds.contains('i') {
                parts.push(list_of_kind_repr_vars('i', args_i));
            }
            if kinds.contains('r') {
                parts.push(list_of_kind_repr_vars('r', args_r));
            }
            if kinds.contains('f') {
                parts.push(list_of_kind_repr_vars('f', args_f));
            }
            out.push_str(&parts.join(", "));
        }
        // jtransform.py `handle_recursive_call`:
        //   args = [Constant(jdindex, lltype.Signed)] + green sublists + red sublists
        // → format.py:23 renders `Constant(jdindex)` as `$<jdindex>`.
        OpKind::RecursiveCall {
            jd_index,
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            ..
        } => {
            let mut parts = vec![format!("${jd_index}")];
            parts.push(list_of_kind_repr_vars('i', greens_i));
            parts.push(list_of_kind_repr_vars('r', greens_r));
            parts.push(list_of_kind_repr_vars('f', greens_f));
            parts.push(list_of_kind_repr_vars('i', reds_i));
            parts.push(list_of_kind_repr_vars('r', reds_r));
            parts.push(list_of_kind_repr_vars('f', reds_f));
            out.push_str(&parts.join(", "));
        }
        _ => {
            // **Stub branch.**  Pyre's `OpKind` carries typed payloads
            // rather than positional argument tuples, so an upstream-
            // shaped formatter would need a per-variant projection.
            // Variants not covered here (FieldRead/FieldWrite, etc.)
            // print just the op name; extend this match when a parity
            // test demands it.
        }
    }
    // Result register suffix renders the Variable's `id()`
    // (process-wide identity) — the PRE-EXISTING-ADAPTATION documented
    // on [`format_assembler`]; upstream renders the post-flatten
    // `Register.index` color.
    let result_suffix: Option<u64> = op.result.as_ref().map(|v| v.id());
    if let Some(suffix) = result_suffix {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str("-> ");
        // RPython parity: result kind comes from the OpKind variant's
        // typed result slot.  Each producer variant pins it via
        // either `result_kind: char` (call family) or `result_ty:
        // ValueType` (BinOp/CompareOp/Cast/etc.); the [`value_type_kind`]
        // helper folds those into the canonical [`RegKind`] via
        // `getkind(concretetype)` parity.  `_ => RegKind::Ref` is a
        // last-resort fallback for the small handful of non-result-
        // bearing variants (no debug consumers exercise them today).
        let result_kind = op_result_kind(&op.kind);
        out.push_str(&register_repr_for_kind(suffix, result_kind));
    }
    out
}

/// `getkind(v.concretetype)` for a [`crate::flowspace::model::Variable`]
/// — direct reader for debug-format helpers that resolve an arg list
/// whose per-slot kind is not pinned by the variant (notably
/// [`crate::model::OpKind::Call`]).  Reads
/// `Variable.concretetype` (`flowspace/model.py`) via
/// [`crate::model::getkind`].  Returns `None` when the cell is unset
/// or the value classifies as Void / Unknown.
fn variable_kind(v: &crate::flowspace::model::Variable) -> Option<RegKind> {
    use crate::model::ConcreteType;
    let lltype = v.concretetype()?;
    match crate::model::getkind(&lltype) {
        ConcreteType::Signed => Some(RegKind::Int),
        ConcreteType::GcRef => Some(RegKind::Ref),
        ConcreteType::Float => Some(RegKind::Float),
        ConcreteType::Void | ConcreteType::Unknown => None,
    }
}

/// `getkind(v.concretetype)` parity for pyre's [`crate::model::ValueType`].
///
/// `Int | Unsigned | Bool` map to [`RegKind::Int`]; `Float` maps to
/// [`RegKind::Float`]; everything else (heap-tracking,
/// pointer-shaped) maps to [`RegKind::Ref`].
fn value_type_kind(ty: &crate::model::ValueType) -> RegKind {
    use crate::model::ValueType;
    match ty {
        ValueType::Int | ValueType::Unsigned | ValueType::Bool => RegKind::Int,
        ValueType::Float => RegKind::Float,
        _ => RegKind::Ref,
    }
}

/// `getkind(op.result.concretetype)` derived from the OpKind variant.
///
/// RPython parity: every result-bearing op has a declared result
/// type — either `result_kind: char` (call family) or
/// `result_ty`/`ty`/`item_ty: ValueType` for typed read/write
/// variants.  Pyre's `OpKind` carries the same information on each
/// variant, so the formatter can answer `getkind(result.concretetype)`
/// without consulting any side-table.  The `_ => RegKind::Ref` arm
/// only catches result-less variants whose `op.result == None`
/// branch in `op_args_repr` already short-circuits this lookup.
fn op_result_kind(kind: &crate::model::OpKind) -> RegKind {
    use crate::model::OpKind;
    match kind {
        OpKind::CallElidable { result_kind, .. }
        | OpKind::CallResidual { result_kind, .. }
        | OpKind::CallMayForce { result_kind, .. }
        | OpKind::InlineCall { result_kind, .. }
        | OpKind::RecursiveCall { result_kind, .. } => match result_kind {
            'i' => RegKind::Int,
            'f' => RegKind::Float,
            _ => RegKind::Ref,
        },
        OpKind::ConstInt(_) | OpKind::ConstFnAddr { .. } | OpKind::ConstUInt(_) => RegKind::Int,
        OpKind::ConstFloat(_) => RegKind::Float,
        OpKind::ConstStr(_) | OpKind::ConstInternedStr(_) => RegKind::Ref,
        OpKind::ConstBool(_) => RegKind::Int,
        OpKind::BinOp { result_ty, .. }
        | OpKind::UnaryOp { result_ty, .. }
        | OpKind::Call { result_ty, .. }
        | OpKind::IndirectCall { result_ty, .. } => value_type_kind(result_ty),
        OpKind::Input { ty, .. }
        | OpKind::FieldRead { ty, .. }
        | OpKind::VableFieldRead { ty, .. } => value_type_kind(ty),
        OpKind::ArrayRead { item_ty, .. }
        | OpKind::InteriorFieldRead { item_ty, .. }
        | OpKind::VableArrayRead { item_ty, .. } => value_type_kind(item_ty),
        OpKind::IsConstant { .. } | OpKind::IsVirtual { .. } | OpKind::VableArrayLen { .. } => {
            RegKind::Int
        }
        // The class word of the header, in whichever bank the read it
        // replaced was allocated to (`OpKind::GuardClass`); the formatter
        // has no regalloc to ask, and the int bank is where the class
        // pointer hints (`record_exact_class/ri`) already carry it.
        OpKind::GuardClass { .. } => RegKind::Int,
        // Result-less or pyre-only debug variants — `op_args_repr`
        // only reaches this fall-through when `op.result.is_some()`,
        // so any miss surfaces as a real coverage gap to extend.
        _ => RegKind::Ref,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flatten::{FlatOp, Label, SSARepr};

    fn empty_ssa() -> SSARepr {
        SSARepr {
            name: "test".into(),
            insns: Vec::new(),
            num_blocks: 0,
            insns_pos: None,
        }
    }

    #[test]
    fn format_jump_emits_label() {
        let mut ssa = empty_ssa();
        let target = Label(0);
        ssa.insns.push(FlatOp::Jump(target));
        let text = format_assembler(&ssa);
        assert!(text.contains("goto L1"));
    }

    #[test]
    fn format_label_uses_first_seen_numbering() {
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::Jump(Label(7)));
        ssa.insns.push(FlatOp::Label(Label(7)));
        let text = format_assembler(&ssa);
        assert!(text.contains("goto L1"));
        assert!(text.contains("L1:"));
    }

    #[test]
    fn format_label_numbering_follows_first_printed_mention() {
        // `format.py:37-44` only marks the labels that appear as jump
        // targets; `format.py getlabelname` assigns the numbers
        // lazily, so a label takes its ordinal from its first *printed*
        // mention — the label line counts just as much as a jump operand.
        //
        // In a loop the header's label line prints before the back edge
        // that targets it, so the header is L1 even though the forward
        // exit is the first target the instruction stream mentions.
        let header = Label(10);
        let exit = Label(20);
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::Label(header));
        ssa.insns.push(FlatOp::GotoIfNot {
            cond: crate::flatten::Register::new(RegKind::Int, 0),
            target: exit,
        });
        ssa.insns.push(FlatOp::Jump(header));
        ssa.insns.push(FlatOp::EndOfBlock);
        ssa.insns.push(FlatOp::Label(exit));
        assert_format(
            &ssa,
            "
            L1:
            goto_if_not %i0, L2
            goto L1
            ---
            L2:
            ",
        );
    }

    #[test]
    fn format_switch_uses_switchdictdescr_repr() {
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::Switch {
            value: crate::flatten::Register::new(RegKind::Int, 0),
            targets: vec![(4, Label(2)), (5, Label(1))],
        });
        let text = format_assembler(&ssa);
        assert!(
            text.contains("switch %i0, <SwitchDictDescr 4:L1, 5:L2>"),
            "unexpected switch format: {text}"
        );
    }

    #[test]
    fn format_end_of_block_marker() {
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::EndOfBlock);
        let text = format_assembler(&ssa);
        // format.py:54-55 trims a trailing ('---',) sentinel.
        assert_eq!(text, "");
    }

    #[test]
    fn format_end_of_block_in_middle_is_kept() {
        // Trim only when `---` is the last instruction (format.py:54-55).
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::EndOfBlock);
        ssa.insns.push(FlatOp::Jump(Label(0)));
        let text = format_assembler(&ssa);
        assert!(text.contains("---"));
        assert!(text.contains("goto L1"));
    }

    #[test]
    fn assert_format_matches_simple_program() {
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::Jump(Label(0)));
        ssa.insns.push(FlatOp::Label(Label(0)));
        ssa.insns.push(FlatOp::EndOfBlock);
        assert_format(
            &ssa,
            "
            goto L1
            L1:
            ",
        );
    }

    #[test]
    fn split_words_preserves_nested_groups() {
        assert_eq!(
            split_words("%i0, I[%i1, %i2], <SwitchDictDescr 4:L1, 5:L2> -> %r3"),
            vec![
                "%i0,",
                "I[%i1, %i2],",
                "<SwitchDictDescr 4:L1, 5:L2>",
                "->",
                "%r3",
            ]
        );
    }

    #[test]
    fn format_constint_emits_dollar_value() {
        // format.py `'$%r' % (x.value,)`.
        use crate::flowspace::model::Variable;
        use crate::model::{OpKind, SpaceOperation};
        let result_var = Variable::new();
        let result_id = result_var.id();
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::op(SpaceOperation {
            kind: OpKind::ConstInt(42),
            result: Some(result_var),
        }));
        let text = format_assembler(&ssa);
        assert!(text.contains("$42"), "expected `$42` in: {text}");
        assert!(
            text.contains(&format!("-> %i{result_id}")),
            "expected `-> %i{result_id}` in: {text}"
        );
    }

    #[test]
    fn format_residual_call_emits_descr_and_listofkind() {
        // jtransform.py rewrite_call + format.py:27,32-33.  allow-line-citation
        use crate::call::CallDescriptor;
        use crate::flowspace::model::Variable;
        use crate::model::{CallFuncPtr, CallTarget, OpKind, SpaceOperation};
        use majit_ir::descr::EffectInfo;

        let int_arg = Variable::new();
        let ref_arg = Variable::new();
        let result_var = Variable::new();
        let int_arg_id = int_arg.id();
        let ref_arg_id = ref_arg.id();
        let result_id = result_var.id();

        let mut ssa = empty_ssa();
        let funcptr = CallTarget::function_path(["foo"]);
        let descriptor = CallDescriptor::from_signature(
            &[majit_ir::value::Type::Int, majit_ir::value::Type::Ref],
            majit_ir::value::Type::Int,
            EffectInfo::default(),
        );
        ssa.insns.push(FlatOp::op(SpaceOperation {
            kind: OpKind::CallResidual {
                funcptr: CallFuncPtr::Target(funcptr),
                descriptor,
                args_i: vec![int_arg],
                args_r: vec![ref_arg],
                args_f: vec![],
                result_kind: 'i',
                indirect_targets: None,
            },
            result: Some(result_var),
        }));
        let text = format_assembler(&ssa);
        assert!(
            text.contains("residual_call_ir_i "),
            "expected residual_call_ir_i in: {text}"
        );
        // jtransform.py handle_residual_call emits funcptr as args[0], calldescr via
        // SpaceOperation.descr.  Pyre carries the funcptr identity on
        // descriptor.target and renders it as `$<* function 'name'>`
        // mirroring format.py Ptr-to-Struct repr.
        assert!(
            text.contains("$<* function 'foo'>"),
            "expected funcptr slot in: {text}"
        );
        assert!(
            text.contains(&format!("I[%i{int_arg_id}]")),
            "expected I[%i{int_arg_id}] in: {text}"
        );
        assert!(
            text.contains(&format!("R[%r{ref_arg_id}]")),
            "expected R[%r{ref_arg_id}] in: {text}"
        );
        // jtransform.py:430-433 — empty kind slots are dropped, matching
        // upstream where `kinds = "ir"` excludes the F sublist entirely.
        assert!(
            !text.contains("F["),
            "F[] must not appear when 'f' kind absent: {text}"
        );
        assert!(text.contains(&format!("-> %i{result_id}")));
    }

    #[test]
    fn format_inline_call_emits_jitcode_and_listofkind() {
        use crate::flowspace::model::Variable;
        use crate::model::{OpKind, SpaceOperation};
        let mut ssa = empty_ssa();
        let callee = std::sync::Arc::new(crate::jitcode::JitCode::new("callee"));
        callee.set_index(7);
        let red = Variable::new();
        let red_id = red.id();
        ssa.insns.push(FlatOp::op(SpaceOperation {
            kind: OpKind::InlineCall {
                jitcode: crate::jitcode::JitCodeHandle::new(callee),
                args_i: vec![],
                args_r: vec![red],
                args_f: vec![],
                result_kind: 'v',
                arg_classes: String::new(),
            },
            result: None,
        }));
        let text = format_assembler(&ssa);
        assert!(
            text.contains("inline_call_r_v "),
            "expected inline_call_r_v in: {text}"
        );
        // jtransform.py:478 stores the JitCode object as args[0]; format.py
        // renders it via JitCode.__repr__ which carries the index-keyed
        // identity.  Pyre prints it as `<JitCode #N>` so the parity test
        // sees the same shape.
        assert!(text.contains("<JitCode #7>"), "got: {text}");
        assert!(
            text.contains(&format!("R[%r{red_id}]")),
            "expected R[%r{red_id}] in: {text}"
        );
    }

    #[test]
    fn format_recursive_call_emits_jd_and_six_listofkinds() {
        use crate::flowspace::model::Variable;
        use crate::model::{OpKind, SpaceOperation};
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::op(SpaceOperation {
            kind: OpKind::RecursiveCall {
                jd_index: 0,
                greens_i: vec![Variable::new()],
                greens_r: vec![],
                greens_f: vec![],
                reds_i: vec![],
                reds_r: vec![Variable::new()],
                reds_f: vec![],
                result_kind: 'v',
                green_classes: String::new(),
                red_classes: String::new(),
            },
            result: None,
        }));
        let text = format_assembler(&ssa);
        assert!(text.contains("recursive_call_v "), "got: {text}");
        // jtransform.py:530 stores `Constant(jdindex, lltype.Signed)` as
        // args[0]; format.py:23 renders it as `$<value>`.  Pyre mirrors
        // the shape exactly: `$0` for jd_index=0.
        assert!(text.contains(" $0,"), "got: {text}");
        // Six ListOfKind groups: greens (i,r,f) + reds (i,r,f).
        let groups: Vec<&str> = text.matches('[').collect();
        assert_eq!(groups.len(), 6, "expected 6 ListOfKind groups, got: {text}");
    }

    #[test]
    fn format_with_insns_pos_prepends_position_prefix() {
        // format.py `prefix = '%4d  ' % ssarepr._insns_pos[i]`.
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::Jump(Label(0)));
        ssa.insns.push(FlatOp::Label(Label(0)));
        ssa.insns_pos = Some(vec![0, 12]);
        let text = format_assembler(&ssa);
        assert!(
            text.contains("   0  goto L1"),
            "expected '   0  goto L1' in: {text}"
        );
        assert!(
            text.contains("  12  L1:"),
            "expected '  12  L1:' in: {text}"
        );
    }
}
