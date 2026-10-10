//! Assembler — converts flattened SSARepr into a `JitCode`.
//!
//! RPython equivalent: `rpython/jit/codewriter/assembler.py` class
//! `Assembler`. The `JitCode` struct itself lives in `crate::jitcode`
//! (RPython parity: `rpython/jit/codewriter/jitcode.py`).
//!
//! **Status: partial port.** `write_insn`, `fix_labels`, the
//! `all_liveness` shared table, and the `IndirectCallTargets` sidecar
//! merge for `residual_call` are implemented in the pyre-relevant
//! subset. Descriptor operands are deduplicated through the RPython
//! `_descr_dict` shape before bytecode emission.

use parking_lot::Mutex;
use std::collections::{BTreeSet, HashMap};

use crate::call::{CallControl, extract_element_type_from_str};
use crate::flatten::{FlatOp, IntOvfOp, Label, RegKind, SSARepr};

fn reg_label(reg: crate::flatten::Register) -> String {
    reg.repr()
}

fn reg_or_const_label(value: &crate::flatten::RegOrConst) -> String {
    match value {
        crate::flatten::RegOrConst::Reg(reg) => reg_label(*reg),
        crate::flatten::RegOrConst::Const(c) => {
            let kind = crate::flatten::constant_kind(c);
            format!("Const({kind})")
        }
    }
}

fn flatop_debug_label(op: &FlatOp) -> String {
    match op {
        FlatOp::Label(label) => format!("Label({})", label.0),
        FlatOp::Op(op) => {
            let has_result = op.result.is_some();
            format!(
                "Op({}, has_result={has_result})",
                op_kind_to_opname(&op.kind)
            )
        }
        FlatOp::Jump(label) => format!("Jump({})", label.0),
        FlatOp::GotoIfNot { cond, target } => {
            format!("GotoIfNot(cond={}, target={})", reg_label(*cond), target.0)
        }
        FlatOp::GotoIfNotOp {
            opname,
            args,
            target,
        } => {
            let args = args
                .iter()
                .map(|reg| reg_label(*reg))
                .collect::<Vec<_>>()
                .join(",");
            format!("GotoIfNotOp({opname}, args=[{args}], target={})", target.0)
        }
        FlatOp::Switch { value, targets } => {
            format!(
                "Switch(value={}, targets={})",
                reg_label(*value),
                targets.len()
            )
        }
        FlatOp::IntBinOpJumpIfOvf {
            op,
            target,
            lhs,
            rhs,
            dst,
        } => format!(
            "IntBinOpJumpIfOvf({op:?}, lhs={}, rhs={}, dst={}, target={})",
            reg_label(*lhs),
            reg_label(*rhs),
            reg_label(*dst),
            target.0
        ),
        FlatOp::CatchException { target } => format!("CatchException(target={})", target.0),
        FlatOp::GotoIfExceptionMismatch { target, .. } => {
            format!("GotoIfExceptionMismatch(target={})", target.0)
        }
        FlatOp::Move { dst, src } => {
            format!(
                "Move(dst={}, src={})",
                reg_label(*dst),
                reg_or_const_label(src)
            )
        }
        FlatOp::Push(reg) => format!("Push({})", reg_label(*reg)),
        FlatOp::Pop(reg) => format!("Pop({})", reg_label(*reg)),
        FlatOp::LastException { dst } => format!("LastException(dst={})", reg_label(*dst)),
        FlatOp::LastExcValue { dst } => format!("LastExcValue(dst={})", reg_label(*dst)),
        FlatOp::Live { live_values } => format!("Live(count={})", live_values.len()),
        FlatOp::Reraise => "Reraise".to_string(),
        FlatOp::IntReturn(value) => format!("IntReturn({})", reg_or_const_label(value)),
        FlatOp::RefReturn(value) => format!("RefReturn({})", reg_or_const_label(value)),
        FlatOp::FloatReturn(value) => format!("FloatReturn({})", reg_or_const_label(value)),
        FlatOp::VoidReturn => "VoidReturn".to_string(),
        FlatOp::Raise(value) => format!("Raise({})", reg_or_const_label(value)),
        FlatOp::EndOfBlock => "EndOfBlock".to_string(),
        FlatOp::Unreachable => "Unreachable".to_string(),
    }
}

/// `flatten.py` `Register.kind[0]` — single-char prefix for opname keys.
fn kind_char_of(kind: RegKind) -> char {
    match kind {
        RegKind::Int => 'i',
        RegKind::Ref => 'r',
        RegKind::Float => 'f',
    }
}

/// Companion long form (`'int'`/`'ref'`/`'float'`) used in
/// `int_copy`/`ref_copy`/`float_copy` opname keys.
fn kind_long_name(kind: RegKind) -> &'static str {
    match kind {
        RegKind::Int => "int",
        RegKind::Ref => "ref",
        RegKind::Float => "float",
    }
}

/// `assembler.py` `USE_C_FORM` — the set of jitcode opnames whose
/// small ConstInt operands may take the inline `c` short-const argcode
/// (`assembler.py` `allow_short = (insn[0] in USE_C_FORM)`).  Listed
/// verbatim, kept identical to the runtime assembler's `use_c_form`
/// (`pyre-jit/src/jit/assembler.rs`).
///
/// In pyre's build-time flow model most operands carry no inline
/// Constant — the rtyper materialises constants through dedicated
/// `int_copy`/`ref_copy` ops and binops read registers — so only the
/// opnames that still hold an inline Constant build-time actually reach
/// the short branch: `int_copy` (`OpKind::ConstInt`/`ConstBool` and a
/// const `FlatOp::Move` source), `int_return` (a const `FlatOp::IntReturn`
/// source), `setfield_gc_i` (`OpKind::FieldWrite` with an inline int/bool
/// `LinkArg::Const`), and `jit_merge_point` (handled inline in its own
/// arm).  The remaining members are honoured by the runtime assembler,
/// which lowers array/string ops with literal small operands.
fn use_c_form(opname: &str) -> bool {
    matches!(
        opname,
        "copystrcontent"
            | "getarrayitem_gc_pure_i"
            | "getarrayitem_gc_pure_r"
            | "getarrayitem_gc_i"
            | "getarrayitem_gc_r"
            | "goto_if_not_int_eq"
            | "goto_if_not_int_ge"
            | "goto_if_not_int_gt"
            | "goto_if_not_int_le"
            | "goto_if_not_int_lt"
            | "goto_if_not_int_ne"
            | "int_add"
            | "int_and"
            | "int_copy"
            | "int_eq"
            | "int_ge"
            | "int_gt"
            | "int_le"
            | "int_lt"
            | "int_ne"
            | "int_return"
            | "int_sub"
            | "jit_merge_point"
            | "new_array"
            | "new_array_clear"
            | "newstr"
            | "setarrayitem_gc_i"
            | "setarrayitem_gc_r"
            | "setfield_gc_i"
            | "strgetitem"
            | "strsetitem"
            | "foobar"
            | "baz"
    )
}

// `reg_byte` and the `CURRENT_GRAPH` thread-local are absent because
// every FlatOp variant carries a [`crate::flatten::Register`] (or
// [`crate::flatten::RegOrConst`]) operand directly.  The assembler
// reads `r.kind` / `r.index` straight off the operand — no per-call
// kind-search, no fallback — exactly mirroring RPython's
// `Register(kind, index)` invariant from `flatten.py`.
use crate::flowspace::model::ConstValue;
use crate::jitcode::{
    BhCallDescr, ConstIRelocDescriptor, ConstIRelocKind, ConstRRelocDescriptor, JitCodeBody,
    StrConstDescriptor,
};
use crate::regalloc::RegAllocator;

pub use majit_jitcode::codewriter::assembler::*;

fn compact_exc_name(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .collect()
}

/// Address of the `interp_exceptions` row whose name derives from `class_name`.
///
/// `StopIteration` matches `EXC_STOP_ITERATION_TYPE` by dropping underscores.
/// `BaseException` is the one static spelled `EXCEPTION_TYPE`.
fn lookup_exc_static_row(rows: &[(String, i64)], class_name: &str, suffix: &str) -> Option<i64> {
    lookup_exc_static_row_named(rows, class_name, suffix).map(|(_, addr)| addr)
}

fn lookup_exc_static_row_name(
    rows: &[(String, i64)],
    class_name: &str,
    suffix: &str,
) -> Option<String> {
    lookup_exc_static_row_named(rows, class_name, suffix).map(|(name, _)| name)
}

fn lookup_exc_static_row_named(
    rows: &[(String, i64)],
    class_name: &str,
    suffix: &str,
) -> Option<(String, i64)> {
    let compact = compact_exc_name(class_name);
    let mut found: Option<(String, i64)> = None;
    for (key, addr) in rows {
        if !key.contains("interp_exceptions::") {
            continue;
        }
        let leaf = key.rsplit("::").next().unwrap_or(key.as_str());
        let matched = if suffix == "_TYPE" && leaf == "EXCEPTION_TYPE" {
            compact == "BASEEXCEPTION"
        } else if let Some(rest) = leaf.strip_prefix("EXC_") {
            let Some(body) = rest.strip_suffix(suffix) else {
                continue;
            };
            let row_compact: String = body.chars().filter(|c| *c != '_').collect();
            row_compact == compact
        } else {
            false
        };
        if !matched {
            continue;
        }
        if found.is_some() {
            panic!("exception class {class_name} matches more than one {suffix} static row");
        }
        found = Some((key.clone(), *addr));
    }
    found
}

fn exception_class_of(obj: &crate::flowspace::model::HostObject) -> Option<String> {
    if !obj.is_class() {
        return None;
    }
    let base = crate::flowspace::model::HOST_ENV.lookup_builtin("BaseException")?;
    obj.is_subclass_of(&base)
        .then(|| obj.simple_name().to_string())
}

/// Class name and message of a builtin exception instance.
///
/// `Ok(None)` message is the empty prebuilt instance. Any other argument
/// shape panics with the class name.
fn exception_instance_spec(
    obj: &crate::flowspace::model::HostObject,
) -> Option<(String, Option<Vec<u8>>)> {
    let class = obj.instance_class()?;
    let base = crate::flowspace::model::HOST_ENV.lookup_builtin("BaseException")?;
    if !class.is_subclass_of(&base) {
        return None;
    }
    let name = class.simple_name().to_string();
    let args = obj.instance_args().unwrap_or(&[]);
    let message = match args {
        [] => None,
        [crate::flowspace::model::ConstValue::UniStr(text)] => Some(text.as_bytes().to_vec()),
        [crate::flowspace::model::ConstValue::ByteStr(bytes)] => Some(bytes.clone()),
        other => panic!("raise/r: unsupported prebuilt {name} instance arguments {other:?}"),
    };
    Some((name, message))
}

/// Pytype address for a builtin exception class. `None` when `obj` is not
/// one, so other host objects keep their existing pool bits. A class with
/// no row panics: the constant must not fall back to `identity_id`.
fn exception_class_addr(
    obj: &crate::flowspace::model::HostObject,
    callcontrol: Option<&CallControl>,
    site: &str,
) -> Option<i64> {
    let name = exception_class_of(obj)?;
    let pytypes = callcontrol.map_or(&[][..], CallControl::exc_pytype_rows);
    Some(
        lookup_exc_static_row(pytypes, &name, "_TYPE").unwrap_or_else(|| {
            panic!("{site}: no interp_exceptions pytype row for exception class {name}")
        }),
    )
}

fn host_object_ref_bits(
    obj: &crate::flowspace::model::HostObject,
    callcontrol: Option<&CallControl>,
) -> i64 {
    if exception_instance_spec(obj).is_some() {
        let name = obj
            .instance_class()
            .map(|class| class.simple_name().to_string())
            .unwrap_or_default();
        panic!(
            "raise/r: exception instance {name} must be a load-time descriptor, not identity_id"
        );
    }
    if let Some(addr) = exception_class_addr(obj, callcontrol, "raise/r") {
        return addr;
    }
    obj.identity_id() as i64
}

/// The graph-reading half of `assembler.py` `Assembler`.
///
/// The state it writes into lives in `majit-jitcode` so the runtime can keep
/// an `Assembler` without the translator.
pub trait AssemblerExt {
    fn assemble(
        &mut self,
        ssarepr: &mut SSARepr,
        regallocs: &HashMap<RegKind, RegAllocator>,
    ) -> JitCodeBody;

    fn assemble_with_callcontrol(
        &mut self,
        ssarepr: &mut SSARepr,
        regallocs: &HashMap<RegKind, RegAllocator>,
        callcontrol: Option<&CallControl>,
    ) -> JitCodeBody;
}

/// The operand encoders `AssemblerExt::assemble` drives.
trait AssemblerEncode {
    fn write_insn(
        &mut self,
        op: &FlatOp,
        regallocs: &HashMap<RegKind, RegAllocator>,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    );

    fn encode_regorconst_source(
        &mut self,
        arg: &crate::flatten::RegOrConst,
        expected_kind: RegKind,
        allow_short: bool,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> (u8, char);

    fn encode_op(
        &mut self,
        op: &crate::model::SpaceOperation,
        regallocs: &HashMap<RegKind, RegAllocator>,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    );

    fn emit_list_of_kind(
        &self,
        args: &[crate::flowspace::model::Variable],
        kind: RegKind,
        regallocs: &HashMap<RegKind, RegAllocator>,
        state: &mut AssemblyState,
    );

    fn emit_call_result_arg(
        &self,
        result: Option<&crate::flowspace::model::Variable>,
        declared_result_kind: char,
        regallocs: &HashMap<RegKind, RegAllocator>,
        state: &mut AssemblyState,
        argcodes: &mut String,
    ) -> char;

    fn lookup_coloring_var(
        &self,
        var: &crate::flowspace::model::Variable,
        regallocs: &HashMap<RegKind, RegAllocator>,
    ) -> (u8, RegKind);

    fn lookup_reg_with_kind_var(
        &self,
        var: &crate::flowspace::model::Variable,
        regallocs: &HashMap<RegKind, RegAllocator>,
    ) -> (u8, char);

    fn run_coverage_audit(&self, ssarepr: &SSARepr, regallocs: &HashMap<RegKind, RegAllocator>);

    fn emit_const(
        &mut self,
        value: &ConstValue,
        kind: char,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> u8;

    fn emit_const_i_from_const(
        &mut self,
        value: &ConstValue,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> u8;

    fn resolve_const_i_value(value: &ConstValue, callcontrol: Option<&CallControl>) -> i64;

    fn emit_const_i_from_const_allow_short(
        &mut self,
        value: &ConstValue,
        allow_short: bool,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> (u8, char);

    fn emit_const_i(&mut self, value: i64, state: &mut AssemblyState) -> u8;

    fn emit_const_i_reloc(
        &mut self,
        value: i64,
        kind: ConstIRelocKind,
        state: &mut AssemblyState,
    ) -> u8;

    fn emit_fnaddr_const(
        &mut self,
        funcptr: &crate::model::CallTarget,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> u8;

    fn emit_const_r_reloc(
        &mut self,
        value: i64,
        kind: ConstIRelocKind,
        state: &mut AssemblyState,
    ) -> u8;

    fn emit_const_i_allow_short(
        &mut self,
        value: i64,
        allow_short: bool,
        state: &mut AssemblyState,
    ) -> (u8, char);

    fn emit_llexitcase(
        &mut self,
        value: &ConstValue,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> u8;

    fn emit_const_r(
        &mut self,
        value: &ConstValue,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> u8;

    fn emit_str_const_r(
        &mut self,
        bytes: Vec<u8>,
        precomputed_hash: i64,
        as_unicode_object: bool,
        state: &mut AssemblyState,
    ) -> u8;

    fn emit_unit_variant_const_r(
        &mut self,
        qualname: String,
        tag: i64,
        state: &mut AssemblyState,
    ) -> u8;

    fn emit_exc_instance_const_r(
        &mut self,
        class_name: String,
        message: Option<Vec<u8>>,
        state: &mut AssemblyState,
    ) -> u8;

    fn emit_const_r_bits(&mut self, bits: i64, state: &mut AssemblyState) -> u8;

    fn emit_type_static_or_bits(&mut self, bits: i64, state: &mut AssemblyState) -> u8;

    fn emit_type_static_const_r(&mut self, name: String, state: &mut AssemblyState) -> u8;

    fn emit_const_f(&mut self, value: &ConstValue, state: &mut AssemblyState) -> u8;
}

impl AssemblerExt for Assembler {
    /// RPython: `Assembler.assemble(ssarepr, jitcode, num_regs)`.
    ///
    /// Takes the SSARepr (flattened instruction sequence) and register
    /// allocation results, and produces a JitCode with encoded bytecode,
    /// constant pools, and register counts.
    ///
    /// RPython assembler.py:34-54.
    ///
    /// RPython codewriter.py:53-56:
    ///   ssarepr = flatten_graph(graph, regallocs)
    ///   compute_liveness(ssarepr)          ← step 3b
    ///   self.assembler.assemble(ssarepr)   ← step 4
    ///
    /// Like upstream's `Assembler.assemble`, this consumes only the
    /// `ssarepr` (plus the `regallocs` coloring) — never the graph.
    /// The kind source per operand is `Variable.concretetype`, read via
    /// the static `FunctionGraph::concretetype_of(&v)` exactly like
    /// `flatten.py getcolor` reads `getkind(v.concretetype)`.
    fn assemble(
        &mut self,
        ssarepr: &mut SSARepr,
        regallocs: &HashMap<RegKind, RegAllocator>,
    ) -> JitCodeBody {
        self.assemble_with_callcontrol(ssarepr, regallocs, None)
    }

    /// `assemble` overload with an attached [`CallControl`] — the
    /// production codewriter path threads the callcontrol so descriptor
    /// emission can reach the rtyper-built `CallDescriptor` cache.
    fn assemble_with_callcontrol(
        &mut self,
        ssarepr: &mut SSARepr,
        regallocs: &HashMap<RegKind, RegAllocator>,
        callcontrol: Option<&CallControl>,
    ) -> JitCodeBody {
        // RPython codewriter.py:56: compute_liveness(ssarepr)
        // Must run BEFORE assembly so -live- markers carry the full
        // set of alive registers.  The alive set uses
        // [`crate::flatten::Register`]-based identity, so liveness
        // also takes the regalloc result for the `Variable → Register`
        // bridge on `FlatOp::Op` operands.
        //
        // PyPy parity: liveness ALWAYS runs after flatten — there is
        // no graph-less path because every `ssarepr` arriving here is
        // produced by `flatten_graph(graph, …)` upstream, so the kind
        // source projection is always available.
        crate::liveness::compute_liveness(ssarepr, regallocs);
        self.current_graph_name = Some(ssarepr.name.clone());

        // Pyre-only diagnostic: under `MAJIT_COVERAGE_AUDIT=1` enumerate
        // every Variable referenced in `ssarepr.insns` that has no
        // regalloc coloring in any class.  Complements the
        // `MAJIT_COVERAGE_PANIC=1` path (which panics at the first gap
        // hit during `write_insn`) by surfacing the full per-graph gap
        // catalogue in one build.  Upstream RPython has no analogue —
        // the invariant is guaranteed by rtyper's `concretetype`
        // annotation so the lookup cannot miss.
        if std::env::var("MAJIT_COVERAGE_AUDIT").is_ok() {
            self.run_coverage_audit(ssarepr, regallocs);
        }

        let num_regs_i = regallocs.get(&RegKind::Int).map_or(0, |r| r.num_regs);
        let num_regs_r = regallocs.get(&RegKind::Ref).map_or(0, |r| r.num_regs);
        let num_regs_f = regallocs.get(&RegKind::Float).map_or(0, |r| r.num_regs);

        // RPython assembler.py:56-70: self.setup(ssarepr.name)
        let mut state = AssemblyState {
            code: Vec::new(),
            constants_i: Vec::new(),
            constants_r: Vec::new(),
            constants_f: Vec::new(),
            str_consts: Vec::new(),
            unit_variant_consts: Vec::new(),
            exc_instance_consts: Vec::new(),
            type_static_consts: Vec::new(),
            reloc_consts_i: Vec::new(),
            reloc_consts_r: Vec::new(),
            num_regs_i,
            num_regs_r,
            num_regs_f,
            label_positions: HashMap::new(),
            tlabel_fixups: Vec::new(),
            startpoints: indexmap::IndexSet::new(),
            jit_merge_point_offset: None,
            alllabels: indexmap::IndexSet::new(),
            resulttypes: indexmap::IndexMap::new(),
        };

        // RPython assembler.py:41-44:
        //     ssarepr._insns_pos = []
        //     for insn in ssarepr.insns:
        //         ssarepr._insns_pos.append(len(self.code))
        //         self.write_insn(insn)
        let mut insns_pos = Vec::with_capacity(ssarepr.insns.len());
        // Borrow split: clone the insn vec so we can mutate ssarepr
        // (insns_pos write) without aliasing the borrow used by the
        // write_insn loop.
        let ops = ssarepr.insns.clone();
        let debug_enabled = std::env::var("MAJIT_COVERAGE_PANIC").is_ok();
        for op in &ops {
            insns_pos.push(state.code.len());
            if debug_enabled {
                self.current_flatop_debug = Some(flatop_debug_label(op));
            }
            self.write_insn(op, regallocs, &mut state, callcontrol);
        }
        self.current_flatop_debug = None;
        ssarepr.insns_pos = Some(insns_pos);

        // RPython assembler.py: self.fix_labels()
        // Upstream `target = self.label_positions[name]` raises KeyError
        // when the label is missing — never writes a silent 0 target.
        for (label, fixup_pos) in &state.tlabel_fixups {
            // RPython `assembler.py:254 target = self.label_positions[insn[1].name]`
            // — direct dict access, raises KeyError when the TLabel
            // references a label that was never defined. Mirror with
            // a fail-loud panic instead of silently writing 0.
            let target = *state
                .label_positions
                .get(label)
                .unwrap_or_else(|| panic!("undefined TLabel {label:?} at fixup {fixup_pos}"));
            let target_u16 = target as u16;
            // RPython `assembler.py assert 0 <= target <= 0xFFFF`.
            assert!(
                target <= 0xFFFF,
                "label target {target} exceeds u16 range in {}",
                self.current_graph_name.as_deref().unwrap_or("?")
            );
            // RPython `assembler.py:252-253 assert self.code[pos] == "temp 1"`
            // — the fixup must point to two reserved placeholder
            // bytes still in range.
            assert!(
                fixup_pos + 1 < state.code.len(),
                "tlabel fixup position {fixup_pos} past end of code (len={})",
                state.code.len(),
            );
            state.code[*fixup_pos] = (target_u16 & 0xFF) as u8;
            state.code[*fixup_pos + 1] = (target_u16 >> 8) as u8;
        }
        for descr in &mut self.descrs {
            let AssemblerDescr::PendingSwitch { cases } = descr else {
                continue;
            };
            let mut const_keys_in_order: Vec<i64> = cases.iter().map(|(key, _)| *key).collect();
            const_keys_in_order.sort();
            let dict = cases
                .iter()
                .map(|(key, label)| {
                    // RPython `assembler.py:261 target =
                    // self.label_positions[switchlabel.name]` — KeyError
                    // for a missing switch case label. Same fail-loud
                    // policy as the TLabel fixup loop above.
                    let target = *state.label_positions.get(label).unwrap_or_else(|| {
                        panic!("undefined SwitchDictDescr label {label:?} for key {key}")
                    });
                    (*key, target)
                })
                .collect();
            *descr = AssemblerDescr::Ready(Box::new(crate::jitcode::BhDescr::Switch {
                dict,
                const_keys_in_order,
            }));
        }

        // RPython assembler.py `self.check_result()`, :265-269 —
        // the single-byte operand encoding addresses registers and
        // constants of one kind in one space, so their sum is what the
        // byte has to hold.
        check_result(
            (num_regs_i, state.constants_i.len()),
            (num_regs_r, state.constants_r.len()),
            (num_regs_f, state.constants_f.len()),
        );
        // RPython assembler.py:271-281: jitcode.setup(code, ...)
        // Build the body that the codewriter will commit into the
        // pre-allocated `Arc<JitCode>` shell via `set_body`.
        // RPython jitcode.py `assert num_regs_i < 256 and ...`. The
        // assembler limits register pressure via the same invariant.
        assert!(
            num_regs_i < 256 && num_regs_r < 256 && num_regs_f < 256,
            "too many registers (i={num_regs_i} r={num_regs_r} f={num_regs_f})"
        );
        // RPython assembler.py `jitcode._ssarepr = ssarepr`
        let body = JitCodeBody {
            calldescr: BhCallDescr::default(),
            code: state.code,
            constants_i: state.constants_i,
            // The builder keeps plain `i64` — nothing walks a pool that is
            // still being assembled. Interior mutability is owed only from
            // publication onward, so the conversion happens here.
            constants_r: state
                .constants_r
                .into_iter()
                .map(super::jitcode::ConstSlotR::new)
                .collect(),
            constants_f: state.constants_f,
            str_consts: state.str_consts,
            unit_variant_consts: state.unit_variant_consts,
            exc_instance_consts: state.exc_instance_consts,
            type_static_consts: state.type_static_consts,
            reloc_consts_i: state.reloc_consts_i,
            reloc_consts_r: state.reloc_consts_r,
            c_num_regs_i: num_regs_i as u8,
            c_num_regs_r: num_regs_r as u8,
            c_num_regs_f: num_regs_f as u8,
            // self.startpoints, alllabels=self.alllabels,
            // resulttypes=self.resulttypes, ...)` — assembled jitcodes
            // always carry the recorded set, never `None`. Wrap in
            // `Some(...)` so the upstream None sentinel is reserved
            // for hand-built helper jitcodes that bypass the builder
            // (jitcode.py:24 defaults).
            startpoints: Some(state.startpoints),
            jit_merge_point_offset: state.jit_merge_point_offset,
            alllabels: Some(state.alllabels),
            resulttypes: Some(state.resulttypes),
            // `jitcode.py` `JitCode._ssarepr` feeds `dump()` only. The
            // serialized body skips the field, so the prepass does not keep
            // a flattened copy of every assembled graph. In-crate tests
            // still read the portal's `jit_merge_point` off it.
            _ssarepr: if cfg!(test) {
                Some(std::sync::Arc::new(ssarepr.clone()) as _)
            } else {
                None
            },
        };

        self.count_jitcodes += 1;
        body
    }
}

impl AssemblerEncode for Assembler {
    /// RPython: `Assembler.write_insn(insn)` — assembler.py.
    ///
    /// Encodes a single FlatOp into the bytecode stream. Each instruction
    /// is encoded as: opcode_byte + argument_bytes. The opcode byte is
    /// looked up from `self.insns` using a key of the form
    /// `opname/argcodes` (RPython assembler.py:220).
    fn write_insn(
        &mut self,
        op: &FlatOp,
        regallocs: &HashMap<RegKind, RegAllocator>,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) {
        match op {
            // RPython assembler.py:143-144: Label → record bytecode position
            FlatOp::Label(label) => {
                state.label_positions.insert(*label, state.code.len());
            }

            // RPython assembler.py:146-158 `Register('-live-', ...)`
            // case in `write_insn`: emit the `live/` opcode followed
            // by the 2-byte offset returned from `_encode_liveness`.
            FlatOp::Live { live_values } => {
                self.num_liveness_ops += 1;
                let key = state.code.len();
                state.startpoints.insert(key);
                // assembler.py:151-156 `live_i, live_r, live_f` —
                // partition live registers by kind.  Each [`Register`]
                // already carries `(kind, color)` from the flatten
                // pass, so no regalloc lookup is needed here.
                let mut live_i = Vec::new();
                let mut live_r = Vec::new();
                let mut live_f = Vec::new();
                for r in live_values {
                    match r.kind {
                        RegKind::Int => live_i.push(r.index as u8),
                        RegKind::Ref => live_r.push(r.index as u8),
                        RegKind::Float => live_f.push(r.index as u8),
                    }
                }
                // assembler.py:236 `key = (frozenset(live_i), …)` —
                // `_encode_liveness` collects the inputs into `VecSet`
                // for canonical ordering and dedup, so the caller emits
                // raw push-order without an extra sort.
                // assembler.py:148 `self.code.append(chr(self.insns['live/']))`
                let opnum = self.get_opnum("live/");
                state.code.push(opnum);
                // assembler.py `self._encode_liveness(live_i, live_r, live_f)`
                // — appends the 2-byte offset into `state.code` after
                // registering or reusing the canonical entry.
                self._encode_liveness(&live_i, &live_r, &live_f, &mut state.code);
            }

            // RPython assembler.py:141-142: '---' → skip
            FlatOp::EndOfBlock => {}

            // RPython `flatten.py` `emitline("unreachable")` →
            // single-byte opcode for `bhimpl_unreachable`
            // (`blackhole.py`). Mirrors the
            // `assembler.py:140-159` general opcode path: a fresh
            // `startposition = len(self.code)` is recorded before the
            // opcode byte goes in so the `_check_no_branch_to_inside_an_op`
            // pass at `assembler.py:283` sees `unreachable/` as a valid
            // start address even though execution never reaches it.
            FlatOp::Unreachable => {
                let opnum = self.get_opnum("unreachable/");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
            }

            // RPython assembler.py:159-223: regular operation
            FlatOp::Op(inner_op) => {
                self.encode_op(inner_op, regallocs, state, callcontrol);
            }

            // RPython flatten.py: 'goto' + TLabel
            FlatOp::Jump(label) => {
                let opnum = self.get_opnum("goto/L");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                // RPython assembler.py:175-179: TLabel → record position + 2 placeholder bytes
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*label, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            FlatOp::CatchException { target } => {
                let opnum = self.get_opnum("catch_exception/L");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            FlatOp::GotoIfExceptionMismatch { llexitcase, target } => {
                // RPython `flatten.py:228-231`:
                //   emitline('goto_if_exception_mismatch',
                //            Constant(link.llexitcase,
                //                     lltype.typeOf(link.llexitcase)),
                //            TLabel(link))
                let opnum = self.get_opnum("goto_if_exception_mismatch/iL");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                let encoded_llexitcase = self.emit_llexitcase(llexitcase, state, callcontrol);
                state.code.push(encoded_llexitcase);
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            // RPython flatten.py: goto_if_not(cond, TLabel(false_path))
            // Only goto_if_not exists — no goto_if_true in RPython.
            FlatOp::GotoIfNot { cond, target } => {
                // RPython parity: `cond.kind == RegKind::Int` because
                // `block.exitswitch.concretetype == lltype.Bool` is the
                // build-time gate at `flatten.py:248`.  Pyre routes
                // every Variable-cond exitswitch installation through
                // `FunctionGraph::set_branch` (model.rs), which appends
                // a `bool` UnaryOp (flowcontext.py POP_JUMP_IF_FALSE
                // `Variable.bool().eval(self)`) so the rtyper lowers
                // the cond to a Bool register before flatten emits.
                // Fail loud if a non-Int slips through — mirrors
                // `FlatOp::Switch`'s assert below.
                assert_eq!(
                    cond.kind,
                    RegKind::Int,
                    "FlatOp::GotoIfNot.cond must be RegKind::Int \
                     (flatten.py:248 `block.exitswitch.concretetype == lltype.Bool`); \
                     got {:?}",
                    cond.kind,
                );
                let opnum = self.get_opnum("goto_if_not/iL");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(cond.index as u8);
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            // RPython `flatten.py:247` fused guard: the jitcode key is
            // `goto_if_not_<op>/<argcodes>`, where the argcodes are the
            // per-operand register kinds (`i`/`r`/`f`) followed by the
            // `L` label code (e.g. `goto_if_not_int_lt/iiL`,
            // `goto_if_not_int_is_zero/iL`).  A Bool-producing compare's
            // operands carry the matching concretetype, so the kinds
            // resolve to the keys registered in `insns.rs`'s `wellknown_bh_insns`.
            FlatOp::GotoIfNotOp {
                opname,
                args,
                target,
            } => {
                let mut argcodes = String::with_capacity(args.len() + 1);
                for arg in args {
                    argcodes.push(match arg.kind {
                        RegKind::Int => 'i',
                        RegKind::Ref => 'r',
                        RegKind::Float => 'f',
                    });
                }
                argcodes.push('L');
                if opname.starts_with("int_") && args.iter().any(|arg| arg.kind != RegKind::Int) {
                    panic!(
                        "fused goto_if_not_{opname}/{argcodes} in {} has a non-int register",
                        self.current_graph_name.as_deref().unwrap_or("?")
                    );
                }
                let opnum = self.get_opnum(&format!("goto_if_not_{opname}/{argcodes}"));
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                for arg in args {
                    state.code.push(arg.index as u8);
                }
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            FlatOp::Switch { value, targets } => {
                // `flatten.py:275-276` — `kind = getkind(block.exitswitch.
                // concretetype); assert kind == 'int'`.  The production
                // path goes through `flatten.rs::insert_exits` (which
                // already asserts `kind == 'i'` before constructing
                // `FlatOp::Switch`), so a non-Int `value.kind` here can
                // only mean a test fixture built `FlatOp::Switch`
                // directly with the wrong kind.  Fail loud — the
                // `switch/id` opcode reads the int register file, so a
                // Ref / Float index byte would silently address the
                // wrong slot at runtime.
                assert_eq!(
                    value.kind,
                    RegKind::Int,
                    "FlatOp::Switch.value must be RegKind::Int \
                     (flatten.py:275-276 `assert kind == 'int'`); got {:?}",
                    value.kind,
                );
                let descr_idx = self.emit_pending_switch_descr(targets.clone());
                let opnum = self.get_opnum("switch/id");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(value.index as u8);
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
            }

            FlatOp::IntBinOpJumpIfOvf {
                op,
                target,
                lhs,
                rhs,
                dst,
            } => {
                // `flatten.py:195-204` only synthesises
                // `int_*_jump_if_ovf` for `add_ovf` / `sub_ovf` /
                // `mul_ovf` opnames whose all three operands are
                // already `Int` by lltype construction.  Origin/main
                // matched this with a `debug_assert_eq!(kind, 'i')`
                // for lhs/rhs/dst — keep the fail-fast guard on the
                // Register payload so test fixtures that hand-build a
                // miskinded `FlatOp::IntBinOpJumpIfOvf` surface here
                // instead of garbling the bytecode at decode time.
                debug_assert_eq!(
                    lhs.kind,
                    RegKind::Int,
                    "IntBinOpJumpIfOvf.lhs must be RegKind::Int; got {:?}",
                    lhs.kind,
                );
                debug_assert_eq!(
                    rhs.kind,
                    RegKind::Int,
                    "IntBinOpJumpIfOvf.rhs must be RegKind::Int; got {:?}",
                    rhs.kind,
                );
                debug_assert_eq!(
                    dst.kind,
                    RegKind::Int,
                    "IntBinOpJumpIfOvf.dst must be RegKind::Int; got {:?}",
                    dst.kind,
                );
                let opname = match op {
                    IntOvfOp::Add => "int_add_jump_if_ovf/Lii>i",
                    IntOvfOp::Sub => "int_sub_jump_if_ovf/Lii>i",
                    IntOvfOp::Mul => "int_mul_jump_if_ovf/Lii>i",
                };
                let opnum = self.get_opnum(opname);
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
                state.code.push(lhs.index as u8);
                state.code.push(rhs.index as u8);
                state.code.push(dst.index as u8);
                state.resulttypes.insert(state.code.len(), 'i');
            }

            // RPython flatten.py `self.emitline('%s_copy' % kind,
            // v, "->", w)` — argcodes `i>i` (typed src, result marker,
            // typed dst). The `>` bears no byte in the stream; it only
            // flags the result position in the key so the blackhole
            // wire `int_copy/i>i` (in `blackhole.rs`) finds the handler
            // and `Assembler.resulttypes[pc]` is populated correctly.
            //
            // Upstream's source operand (`v`) can be either a `Register`
            // or a `Constant` (`getcolor` returns the Constant as-is at
            // flatten.py); in both cases the `assembler.py:164-174`  allow-line-citation
            // single-byte encoder shares the argcode kind letter and
            // disambiguates register vs constant at decode time via
            // `byte >= count_regs[kind]`.
            FlatOp::Move { dst, src } => {
                // `assembler.py:188-196` — every emitted instruction
                // (including `*_copy`) records its byte offset in
                // `startpoints`, so the label-fixup pass can land a
                // `Label(block)` on a block-opening copy without
                // misdescribing the bytecode boundary.  Previously
                // missed here because `FlatOp::Move` bypasses
                // `encode_op` and the per-arm emitters had each gotten
                // their own `startpoints.insert` call except this one.
                state.startpoints.insert(state.code.len());
                let kind_char = kind_char_of(dst.kind);
                let kind_name = kind_long_name(dst.kind);
                // `assembler.py:163` `allow_short = ('{kind}_copy' in
                // USE_C_FORM)` — only `int_copy` qualifies, so an int Move
                // with a small const source emits `int_copy/c>i`.
                let allow_short = use_c_form(&format!("{kind_name}_copy"));
                let (src_reg, src_argcode) =
                    self.encode_regorconst_source(src, dst.kind, allow_short, state, callcontrol);
                let key = format!("{kind_name}_copy/{src_argcode}>{kind_char}");
                let opnum = self.get_opnum(&key);
                state.code.push(opnum);
                state.code.push(src_reg);
                state.code.push(dst.index as u8);
                state.resulttypes.insert(state.code.len(), kind_char);
            }

            // RPython `flatten.py` `self.emitline('%s_push' % kind, v)`.
            FlatOp::Push(src) => {
                let kind_char = kind_char_of(src.kind);
                let kind_name = kind_long_name(src.kind);
                let key = format!("{kind_name}_push/{kind_char}");
                let opnum = self.get_opnum(&key);
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(src.index as u8);
            }

            // RPython `flatten.py` `self.emitline('%s_pop' % kind, "->", w)`.
            FlatOp::Pop(dst) => {
                let kind_char = kind_char_of(dst.kind);
                let kind_name = kind_long_name(dst.kind);
                let key = format!("{kind_name}_pop/>{kind_char}");
                let opnum = self.get_opnum(&key);
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(dst.index as u8);
                state.resulttypes.insert(state.code.len(), kind_char);
            }

            FlatOp::LastException { dst } => {
                // Parity expectation: `dst.kind == RegKind::Int`
                // (the exception class identity).  See GotoIfNot
                // notes above for the upstream-gap caveat.
                let opnum = self.get_opnum("last_exception/>i");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(dst.index as u8);
                state.resulttypes.insert(state.code.len(), 'i');
            }

            FlatOp::LastExcValue { dst } => {
                // Parity expectation: `dst.kind == RegKind::Ref`
                // (the exception instance pointer).
                let opnum = self.get_opnum("last_exc_value/>r");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(dst.index as u8);
                state.resulttypes.insert(state.code.len(), 'r');
            }

            FlatOp::Reraise => {
                let opnum = self.get_opnum("reraise/");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
            }

            // RPython `flatten.py` `make_return`.  Blackhole
            // handlers: `blackhole.py:841-863 bhimpl_{int,ref,float,void}_return`.
            // `emit_const_*` returns a byte ≥ `num_regs_{kind}` so the
            // single-byte argcode `i`/`r`/`f` suffices for both register
            // and constant sources (upstream `assembler.py:164-174`).
            FlatOp::IntReturn(v) => {
                // `int_return` is in USE_C_FORM (`assembler.py`), so a
                // small const-int source emits `int_return/c`.
                let (reg, src_argcode) = self.encode_regorconst_source(
                    v,
                    RegKind::Int,
                    use_c_form("int_return"),
                    state,
                    callcontrol,
                );
                let opnum = self.get_opnum(&format!("int_return/{src_argcode}"));
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
            }
            FlatOp::RefReturn(v) => {
                // `ref_return` is not in USE_C_FORM; ref always pools to `r`.
                let (reg, _) =
                    self.encode_regorconst_source(v, RegKind::Ref, false, state, callcontrol);
                let opnum = self.get_opnum("ref_return/r");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
            }
            FlatOp::FloatReturn(v) => {
                // `float_return` is not in USE_C_FORM; float always pools to `f`.
                let (reg, _) =
                    self.encode_regorconst_source(v, RegKind::Float, false, state, callcontrol);
                let opnum = self.get_opnum("float_return/f");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
            }
            FlatOp::VoidReturn => {
                let opnum = self.get_opnum("void_return/");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
            }
            // RPython `flatten.py` `make_return` 2-inputarg case
            // plus the `flatten.py` overflow reraise.  Both paths
            // funnel through `raise/r` — `RegOrConst::Reg` is the raised
            // exception value's Register, `RegOrConst::Const` is the
            // standard OverflowError instance.  Blackhole:
            // `blackhole.py bhimpl_raise(excvalue)`.
            FlatOp::Raise(v) => {
                // `raise` is not in USE_C_FORM; the excvalue is a Ref → `r`.
                let (reg, _) =
                    self.encode_regorconst_source(v, RegKind::Ref, false, state, callcontrol);
                let opnum = self.get_opnum("raise/r");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
            }
        }
    }

    /// Encode a [`RegOrConst`] operand into the byte stream and return
    /// its `(byte, argcode)`.
    ///
    /// `RegOrConst::Reg` carries `(kind, color)` directly — no regalloc
    /// lookup needed.  Constants emit through `emit_const` with
    /// `expected_kind` (variant-fixed) selecting the constant pool,
    /// mirroring `assembler.py:164-174` where the single-byte argcode kind
    /// letter chooses between register and constant via
    /// `byte >= count_regs[kind]`.  `allow_short` is `(opname in
    /// USE_C_FORM)` (`assembler.py`); when set, a small integer-kind
    /// Constant takes the inline `'c'` short form (`assembler.py:99-107`)
    /// instead of a pooled `'i'` byte.  Register sources and ref/float
    /// Constants always take their kind argcode.
    fn encode_regorconst_source(
        &mut self,
        arg: &crate::flatten::RegOrConst,
        expected_kind: RegKind,
        allow_short: bool,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> (u8, char) {
        match arg {
            // RPython `assembler.py:164-174`: the single-byte argcode
            // is keyed on `Register.kind`, so the source operand's
            // kind MUST match the expected kind for the surrounding
            // op (e.g. `int_copy/i>i`'s source must be Int).  PyPy
            // satisfies this by construction via
            // `flatten.py` (the source `Register` was created by
            // `getcolor(v)` against the matching `regallocs[w.kind]`
            // entry); pyre mirrors that with a strict assert so an
            // upstream kind-provenance gap surfaces here rather than
            // emitting a register byte that misrepresents its bank.
            crate::flatten::RegOrConst::Reg(r) => {
                assert_eq!(
                    r.kind, expected_kind,
                    "encode_regorconst_source: Register kind {:?} does not match \
                     variant-expected kind {expected_kind:?} (PyPy \
                     `assembler.py:164-174` requires the single-byte argcode kind \
                     and the operand kind to coincide) — graph {:?}, flatop {:?} \
                     (set MAJIT_COVERAGE_PANIC=1 to populate the flatop context)",
                    r.kind, self.current_graph_name, self.current_flatop_debug,
                );
                (r.index as u8, kind_char_of(expected_kind))
            }
            crate::flatten::RegOrConst::Const(c) => {
                // RPython `assembler.py` reads `getkind(x.concretetype)`
                // for the Constant operand.  When the Constant carries
                // a `concretetype` it MUST agree with the surrounding op's
                // `expected_kind` (the byte-stream argcode is keyed on
                // that kind, same constraint as the Register branch
                // above).  When concretetype is absent fall back to the
                // op's expected kind — that mirrors upstream's behavior
                // for synthesized constants whose kind only the caller
                // knows.
                let const_kind = crate::flatten::constant_kind(c);
                let kind_char = kind_char_of(expected_kind);
                if c.concretetype.is_some() {
                    assert_eq!(
                        const_kind, kind_char,
                        "encode_regorconst_source: Constant.concretetype kind {const_kind:?} \
                         does not match variant-expected kind {kind_char:?} (PyPy \
                         `assembler.py:168` requires `getkind(x.concretetype)` to coincide \
                         with the surrounding op's kind)",
                    );
                }
                // Only the int bank has a short-const form (argcode `c`);
                // ref/float Constants always take their pooled `r`/`f` byte.
                if kind_char == 'i' {
                    self.emit_const_i_from_const_allow_short(
                        &c.value,
                        allow_short,
                        state,
                        callcontrol,
                    )
                } else {
                    (
                        self.emit_const(&c.value, kind_char, state, callcontrol),
                        kind_char,
                    )
                }
            }
        }
    }

    /// RPython assembler.py:159-223: encode one SpaceOperation.
    ///
    /// The encoding for each instruction is:
    /// [opcode_byte][arg1_byte][arg2_byte]...[->][result_byte]
    ///
    /// Where args are:
    /// - Register: 1 byte (index), argcode = kind char ('i','r','f')
    /// - Constant: 1 byte (pool index), argcode = kind char
    /// - TLabel: 2 bytes (u16 LE offset), argcode = 'L'
    /// - ListOfKind: 1 byte (len) + items, argcode = uppercase kind
    /// - Descr: 2 bytes (u16 LE index), argcode = 'd'
    fn encode_op(
        &mut self,
        op: &crate::model::SpaceOperation,
        regallocs: &HashMap<RegKind, RegAllocator>,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) {
        use crate::model::OpKind;

        let startposition = state.code.len();
        state.code.push(0); // placeholder for opcode byte
        state.startpoints.insert(startposition);

        let mut argcodes = String::new();

        match &op.kind {
            // RPython flatten.py keeps inputargs on Block.inputargs and does
            // not serialize them as bytecode operations.
            OpKind::Input { .. } => {
                panic!("OpKind::Input must be eliminated before assembly");
            }
            // RPython: inline_call → [jitcode, I[...], R[...], F[...]]
            OpKind::InlineCall {
                jitcode,
                args_i,
                args_r,
                args_f,
                result_kind,
                ..
            } => {
                // RPython assembler.py:197-207: jitcode → descrs[index]
                // The JitCode object IS the descriptor for inline_call.
                let descr_idx = self.emit_pending_jitcode_descr(jitcode.clone());
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython jtransform.py: rewrite_call
                // Only emit the kind sublists that are in 'kinds'.
                let kinds = rewrite_call_kinds(args_i, args_r, args_f, *result_kind);
                if kinds.contains('i') {
                    self.emit_list_of_kind(args_i, RegKind::Int, regallocs, state);
                    argcodes.push('I');
                }
                if kinds.contains('r') {
                    self.emit_list_of_kind(args_r, RegKind::Ref, regallocs, state);
                    argcodes.push('R');
                }
                if kinds.contains('f') {
                    self.emit_list_of_kind(args_f, RegKind::Float, regallocs, state);
                    argcodes.push('F');
                }
                // Result — see residual_call note below: derive the
                // key-level `reskind` from regalloc so `_r_i` / `_r_r`
                // match the actual `>X` argcode suffix.
                let result_key_kind = self.emit_call_result_arg(
                    op.result.as_ref(),
                    *result_kind,
                    regallocs,
                    state,
                    &mut argcodes,
                );
                // RPython jtransform.py:434: inline_call_{kinds}_{reskind}
                let key = format!("inline_call_{kinds}_{result_key_kind}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // RPython: recursive_call → [jd_index, G_I, G_R, G_F, R_I, R_R, R_F]
            //
            // `bhimpl_recursive_call_{i,r,f,v}` declares jd_index as
            // `@arguments("self", "i", ...)` (blackhole.py) so
            // the canonical argcode is `i` (register read). `emit_const_i`
            // returns a register-index into the int constant pool; the
            // dispatch side `bh.registers_i[code[p]]` reads the jd_index
            // back out. RPython does not include `recursive_call` in
            // `USE_C_FORM` (assembler.py), so the `c` short-const
            // form is not permitted here.
            OpKind::RecursiveCall {
                jd_index,
                greens_i,
                greens_r,
                greens_f,
                reds_i,
                reds_r,
                reds_f,
                result_kind,
                green_classes: _,
                red_classes: _,
            } => {
                let idx = self.emit_const_i(*jd_index as i64, state);
                state.code.push(idx);
                argcodes.push('i');
                // green lists
                self.emit_list_of_kind(greens_i, RegKind::Int, regallocs, state);
                argcodes.push('I');
                self.emit_list_of_kind(greens_r, RegKind::Ref, regallocs, state);
                argcodes.push('R');
                self.emit_list_of_kind(greens_f, RegKind::Float, regallocs, state);
                argcodes.push('F');
                // red lists
                self.emit_list_of_kind(reds_i, RegKind::Int, regallocs, state);
                argcodes.push('I');
                self.emit_list_of_kind(reds_r, RegKind::Ref, regallocs, state);
                argcodes.push('R');
                self.emit_list_of_kind(reds_f, RegKind::Float, regallocs, state);
                argcodes.push('F');
                let result_key_kind = self.emit_call_result_arg(
                    op.result.as_ref(),
                    *result_kind,
                    regallocs,
                    state,
                    &mut argcodes,
                );
                let key = format!("recursive_call_{result_key_kind}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // RPython: residual_call/call_may_force/call_elidable
            // → [funcptr, calldescr, I[...], R[...], F[...]]
            // RPython jtransform.py: rewrite_call splits args
            // by kind via make_three_lists.
            OpKind::CallResidual {
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
            }
            | OpKind::CallElidable {
                funcptr,
                descriptor,
                args_i,
                args_r,
                args_f,
                result_kind,
                ..
            } => {
                // RPython `assembler.py:208-209`: the sidecar
                // `IndirectCallTargets(lst)` on a `residual_call`
                // merges into the global `Assembler.indirectcalltargets`
                // set so the metainterp can later look up jitcodes by
                // funcptr address during runtime dispatch.  Only
                // `OpKind::CallResidual` carries the sidecar today.
                if let OpKind::CallResidual {
                    indirect_targets: Some(t),
                    ..
                } = &op.kind
                {
                    self.indirectcalltargets.extend(t.lst.iter().cloned());
                }
                let base = match &op.kind {
                    OpKind::CallMayForce { .. } => "call_may_force",
                    OpKind::CallElidable { .. } => "call_elidable",
                    _ => "residual_call",
                };
                // RPython `jtransform.py` `rewrite_call` emits args
                // by kind (I, R, F) first, then the calldescr, producing
                // keys like `residual_call_ir_r/iIRd>r`. jtransform now
                // materializes direct-call funcptrs as `ConstFnAddr` values,
                // so every post-jtransform call op reaches the assembler
                // as `CallFuncPtr::Value(...)` and encodes the orthodox
                // leading `i` operand.
                match funcptr {
                    crate::model::CallFuncPtr::Value(var) => {
                        let (reg, kc) = self.lookup_reg_with_kind_var(var, regallocs);
                        state.code.push(reg);
                        argcodes.push(kc);
                    }
                    crate::model::CallFuncPtr::Target(target) => {
                        panic!("call op reached assembler without materialized funcptr: {target}");
                    }
                }
                // RPython `assembler.py:197-203`: resolve the descriptor
                // through `_descr_dict` before writing the two bytes. The
                // bytes are still written AFTER the I/R/F lists to match
                // `jtransform.py:422-431` ordering: `iIRFd` / `iIRd` / `iRd`.
                let calldescr = descriptor.to_bh_calldescr();
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::Call { calldescr });
                // RPython jtransform.py:422-431: kind-separated sublists
                let kinds = rewrite_call_kinds(args_i, args_r, args_f, *result_kind);
                if kinds.contains('i') {
                    self.emit_list_of_kind(args_i, RegKind::Int, regallocs, state);
                    argcodes.push('I');
                }
                if kinds.contains('r') {
                    self.emit_list_of_kind(args_r, RegKind::Ref, regallocs, state);
                    argcodes.push('R');
                }
                if kinds.contains('f') {
                    self.emit_list_of_kind(args_f, RegKind::Float, regallocs, state);
                    argcodes.push('F');
                }
                // RPython assembler.py:197-207: descriptor as 2-byte index,
                // emitted last per jtransform.py:422-431 ordering so the
                // blackhole key suffix is `...d>k`.
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // Result
                // RPython `residual_call_r_r` / `residual_call_r_i` /
                // `residual_call_r_v` are *different* bhimpls
                // (`blackhole.py bhimpl_residual_call_r_i`): the `_r` / `_i` / `_v`
                // suffix encodes the actual result kind. When pyre's
                // rtyper (`translator::rtyper::legacy_resolve::resolve_types`)
                // upgrades a call result's concrete type to `Signed`
                // (e.g. via `is_int_arith` backward constraint), the
                // regalloc-assigned register class diverges from the
                // op struct's original `result_kind`. Derive the key
                // name suffix from the regalloc-determined class so
                // `base_{kinds}_{reskind}` stays consistent with the
                // argcode `>X` suffix. If no result, fall back to `v`.
                let result_key_kind = self.emit_call_result_arg(
                    op.result.as_ref(),
                    *result_kind,
                    regallocs,
                    state,
                    &mut argcodes,
                );
                // RPython jtransform.py:434: {base}_{kinds}_{reskind}
                let key = format!("{base}_{kinds}_{result_key_kind}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // RPython `rpython/jit/codewriter/assembler.py:164-174`: ConstInt
            // is NOT a separate op — Constants appear as arguments to other
            // instructions via `emit_const`. Pyre's model forces constants
            // through a standalone materialization op since operands are
            // always `Variable`; lowering that limitation is deferred
            // (requires op-level constant operands). Until then, materialise
            // through `int_copy` and reuse the canonical `bhimpl_int_copy`
            // handler: `int_copy` is in USE_C_FORM (`assembler.py`), so a
            // small value (`-128..=127`) takes the inline `c` byte
            // (`int_copy/c>i`) and a larger one a pool-region `i` slot
            // (`int_copy/i>i`).
            OpKind::ConstInt(val) => {
                // assembler.py `allow_short = ('int_copy' in USE_C_FORM)` →
                // a small constant takes the inline `c` byte (`int_copy/c>i`).
                let (idx, src_argcode) =
                    self.emit_const_i_allow_short(*val, use_c_form("int_copy"), state);
                state.code.push(idx);
                argcodes.push(src_argcode);
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("int_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::ConstFnAddr {
                value,
                path,
                symbolic,
            } => {
                // `assembler.py emit_const`: a function pointer is a
                // symbolic (`AddressAsInt` / `lltype` funcptr), never the
                // short `c` form (`"Symbolics cannot be compared!"`).
                let idx = self.emit_const_i_reloc(
                    *value,
                    ConstIRelocKind::FnAddr {
                        path: path.clone(),
                        symbolic: *symbolic,
                    },
                    state,
                );
                state.code.push(idx);
                argcodes.push('i');
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("int_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::ConstUInt(val) => {
                let (idx, src_argcode) =
                    self.emit_const_i_allow_short(*val as i64, use_c_form("int_copy"), state);
                state.code.push(idx);
                argcodes.push(src_argcode);
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("int_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // RPython folds `lltype.Bool` to kind `'int'` at codewriter
            // (`flatten.py:getkind`), so `ConstBool` materialises through
            // the same `int_copy/i>i` path as `ConstInt`. The bool value
            // collapses to 0/1 in the int constant pool and the
            // canonical `bhimpl_int_copy` handler runs in the blackhole.
            OpKind::ConstBool(val) => {
                // Bool folds to kind `int`; `int_copy` is in USE_C_FORM, so the
                // 0/1 value always fits the inline `c` byte (`int_copy/c>i`).
                let (idx, src_argcode) =
                    self.emit_const_i_allow_short(*val as i64, use_c_form("int_copy"), state);
                state.code.push(idx);
                argcodes.push(src_argcode);
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("int_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // Ref-constant materialization mirrors `ConstInt`: the
            // `HostObject`'s `identity_id()` goes through
            // `emit_const_r` (the shared `'r'` constant pool) and the
            // pool-region register index is moved into the SSA
            // destination via `ref_copy`.  Emitted only for
            // unit-variant `SyntheticTransparentCtor` results that the
            // pre-jtransform fold (`translator/rtyper/unit_variant_fold.rs`)
            // rewrote from `OpKind::Call { args: [] }` (PyPy
            // `rtyper/rpbc.py::SingleFrozenPBCRepr` parity).
            OpKind::ConstRef(obj) => {
                let const_value = crate::flowspace::model::ConstValue::HostObject(obj.clone());
                let idx = self.emit_const_r(&const_value, state, callcontrol);
                state.code.push(idx);
                argcodes.push('r');
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("ref_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::ConstStr(bytes) => {
                let p = crate::translator::rtyper::lltypesystem::rstr::const_str_cache_llstr(bytes)
                    .expect("prebuilt STR constant must materialize");
                let const_value = crate::flowspace::model::ConstValue::LLPtr(Box::new(p));
                let idx = self.emit_const_r(&const_value, state, callcontrol);
                state.code.push(idx);
                argcodes.push('r');
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("ref_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::ConstInternedStr(bytes) => {
                let hash = crate::translator::rtyper::lltypesystem::rstr::ll_strhash_value(bytes);
                let idx = self.emit_str_const_r(bytes.clone(), hash, true, state);
                state.code.push(idx);
                argcodes.push('r');
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("ref_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::ConstRefNull => {
                let const_value = crate::flowspace::model::ConstValue::LLAddress(
                    crate::translator::rtyper::lltypesystem::lltype::_address::Null,
                );
                let idx = self.emit_const_r(&const_value, state, callcontrol);
                state.code.push(idx);
                argcodes.push('r');
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("ref_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::ConstRefAddr(addr) => {
                let idx = self.emit_type_static_or_bits(*addr, state);
                state.code.push(idx);
                argcodes.push('r');
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("ref_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // Float-constant materialization mirrors `ConstInt`: the
            // bit pattern goes through `emit_const_f` (the same pool
            // every `'f'` constant uses) and the resulting pool-region
            // register index is moved into the SSA destination via
            // `float_copy`.
            OpKind::ConstFloat(bits) => {
                let const_value = crate::flowspace::model::ConstValue::Float(*bits);
                let idx = self.emit_const_f(&const_value, state);
                state.code.push(idx);
                argcodes.push('f');
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("float_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // Field/array operations: encode registers + descriptor.
            // RPython assembler.py:197-207: AbstractDescr → 2-byte index.
            // Field operations: register + descriptor.
            // RPython assembler.py:197-207: AbstractDescr → 2-byte index.
            OpKind::FieldRead {
                base,
                field,
                ty,
                pure,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                assert!(
                    kc == 'r' || kc == 'i',
                    "getfield base must be Ref (gc) or Int (raw), got {kc:?} — graph {:?}",
                    self.current_graph_name,
                );
                let is_gc = kc == 'r';
                state.code.push(reg);
                argcodes.push(kc);
                let descr_idx = self.emit_ready_descr(fielddescrof(field, ty, callcontrol, is_gc));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython `bhimpl_getfield_gc_{i,r,f}` canonical keys key
                // off the RESULT register's kind (`@arguments("cpu", "r",
                // "d", returns="X")`), not the declared field type —
                // declared field `ty` can be pyre-only Void/State/Unknown
                // while the SSA result register is always i/r/f after
                // regalloc. Using the result kind keeps the opname
                // aligned with the `>X` argcode the runtime dispatches on.
                // `rewrite_op_getfield` drops `RESULT is lltype.Void`; a
                // surviving Void result has no register.
                let result_kind = if let Some(result) = op.result.as_ref() {
                    if crate::model::FunctionGraph::concretetype_of(result)
                        == crate::model::ConcreteType::Void
                    {
                        'v'
                    } else {
                        argcodes.push('>');
                        let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                        argcodes.push(kc);
                        state.code.push(reg);
                        kc
                    }
                } else {
                    'v'
                };
                // `jtransform.py rewrite_op_getfield`: a non-pure raw
                // ref load is refused. A pure one is re-emitted as
                // `getfield_raw_r` with the `_pure` suffix stripped.
                if !is_gc && result_kind == 'r' && !*pure {
                    panic!(
                        "getfield_raw_r (without _pure) not supported — graph {:?}",
                        self.current_graph_name
                    );
                }
                let mut opname = if is_gc {
                    format!("getfield_gc_{result_kind}")
                } else {
                    format!("getfield_raw_{result_kind}")
                };
                if is_gc && *pure {
                    opname.push_str("_pure");
                }
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            // RPython `jtransform.py rewrite_op_malloc`: a fixed-size
            // GC struct without a vtable lowers to `new(descr)`.
            OpKind::New { owner } => {
                let opname = "new";
                let spec = bh_size_spec_from_callcontrol(
                    callcontrol.expect("new assembly requires a CallControl"),
                    owner,
                )
                .unwrap_or_else(|| {
                    panic!("{opname}: no struct layout registered for owner {owner:?}")
                });
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::Size {
                    size: spec.size,
                    type_id: spec.type_id,
                    vtable: 0,
                    owner: String::new(),
                    all_fielddescrs: spec.all_fielddescrs,
                    is_gc_managed: spec.is_gc_managed,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                let result = op.result.as_ref().expect("new must produce a result");
                let (reg, kind) = self.lookup_reg_with_kind_var(result, regallocs);
                assert_eq!(kind, 'r', "new result must use the ref register bank");
                state.code.push(reg);
                argcodes.push('>');
                argcodes.push('r');
                let key = format!("{opname}/{argcodes}");
                state.code[startposition] = self.get_opnum(&key);
            }
            // RPython `new_array_clear(v_length, arraydescr)` — the cleared
            // fixed-size array allocation `do_fixed_newlist_clear` emits
            // (`jtransform.py`).  `bhimpl_new_array_clear`
            // (`blackhole.py`, `@arguments("cpu", "i", "d",
            // returns="r")`) gives the canonical key `new_array_clear/id>r`:
            // length (Int) + arraydescr + ref result.  The arraydescr is the
            // same `arraydescrof(item_ty, array_type_id, len_offset=Some(0))`
            // shape `ArrayRead`/`ArrayWrite` mint for the length-prefixed
            // items block.
            OpKind::NewArray {
                length,
                item_ty,
                array_type_id,
            }
            | OpKind::NewArrayClear {
                length,
                item_ty,
                array_type_id,
            } => {
                let opname = if matches!(op.kind, OpKind::NewArray { .. }) {
                    "new_array"
                } else {
                    "new_array_clear"
                };
                let (reg, kc) = self.lookup_reg_with_kind_var(length, regallocs);
                assert_eq!(
                    kc, 'i',
                    "new_array_clear length must be int-kind, got {kc:?}",
                );
                state.code.push(reg);
                argcodes.push(kc);
                let descr_idx = self.emit_ready_descr(arraydescrof(
                    item_ty,
                    array_type_id,
                    Some(0),
                    callcontrol,
                ));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                let result = op
                    .result
                    .as_ref()
                    .expect("new_array_clear must produce a result");
                let (reg, kind) = self.lookup_reg_with_kind_var(result, regallocs);
                assert_eq!(
                    kind, 'r',
                    "new_array_clear result must use the ref register bank"
                );
                state.code.push(reg);
                argcodes.push('>');
                argcodes.push('r');
                let key = format!("{opname}/{argcodes}");
                state.code[startposition] = self.get_opnum(&key);
            }
            // `newlist_clear/idddd>r` (`handler_newlist_clear`,
            // blackhole.rs) lowers the resizable-list allocation
            // `do_resizable_newlist` emits (blackhole.py:1173-1180): a
            // `list` GcStruct header wrapping a cleared items array.  The
            // handler reads length (Int) then FOUR descriptors in this
            // order — structdescr, lengthdescr, itemsdescr, arraydescr —
            // then the ref result.  The four descrs must be pushed in that
            // exact read order or the handler misreads them.  The list
            // header (`GcStruct("list", ("length", Signed), ("items",
            // Ptr(GcArray(ITEM))))`) is resizable and carries no vtable, so
            // its Size descr takes vtable 0.
            OpKind::NewListClear {
                length,
                item_ty,
                array_type_id,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(length, regallocs);
                assert_eq!(kc, 'i', "newlist_clear length must be int-kind, got {kc:?}",);
                state.code.push(reg);
                argcodes.push(kc);
                // structdescr: the `list` header Size descr.  Mirror the
                // `OpKind::New` arm — a resizable list header has no vtable.
                let spec = bh_size_spec_from_callcontrol(
                    callcontrol.expect("newlist_clear assembly requires a CallControl"),
                    "list",
                )
                .unwrap_or_else(|| {
                    panic!("newlist_clear: no struct layout registered for owner \"list\"")
                });
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::Size {
                    size: spec.size,
                    type_id: spec.type_id,
                    vtable: 0,
                    owner: String::new(),
                    all_fielddescrs: spec.all_fielddescrs,
                    is_gc_managed: spec.is_gc_managed,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // lengthdescr: fielddescr for `list.length` (Signed word).
                // Mirror the `FieldWrite` arm's fielddescr interning — the
                // local free `fielddescrof` builds the `BhDescr::Field`,
                // resolving the offset off the registered `list` layout.
                let length_field =
                    crate::model::FieldDescriptor::new("length", Some("list".to_string()));
                let descr_idx = self.emit_ready_descr(fielddescrof(
                    &length_field,
                    &crate::model::ValueType::Int,
                    callcontrol,
                    true,
                ));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // itemsdescr: fielddescr for `list.items` (Ptr to array).
                let items_field =
                    crate::model::FieldDescriptor::new("items", Some("list".to_string()));
                let descr_idx = self.emit_ready_descr(fielddescrof(
                    &items_field,
                    &crate::model::ValueType::Ref(None),
                    callcontrol,
                    true,
                ));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // arraydescr: the length-prefixed items block, identical to
                // the `NewArrayClear` arm.
                let descr_idx = self.emit_ready_descr(arraydescrof(
                    item_ty,
                    array_type_id,
                    Some(0),
                    callcontrol,
                ));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                let result = op
                    .result
                    .as_ref()
                    .expect("newlist_clear must produce a result");
                let (reg, kind) = self.lookup_reg_with_kind_var(result, regallocs);
                assert_eq!(
                    kind, 'r',
                    "newlist_clear result must use the ref register bank"
                );
                state.code.push(reg);
                argcodes.push('>');
                argcodes.push('r');
                let key = format!("newlist_clear/{argcodes}");
                state.code[startposition] = self.get_opnum(&key);
            }
            // Boxing GC allocation (`fuse_boxing_alloc`).  Mirrors the runtime
            // tracer oracle (`box_trace.rs trace_box_float`): a `new_with_vtable`
            // carrying ONLY a size descriptor and a fresh ref-kind result, no
            // register operands.  `bh_size_spec_from_callcontrol` resolves the
            // struct size / gc type-id from `owner` via the runtime `gc_cache`
            // Arc (`path_hash(owner)` keys `_cache_size`).  The `vtable` (type
            // pointer) is NOT in that Arc — `_cache_size` carries struct size
            // only — so it travels on the op (captured by `fuse_boxing_alloc`
            // from the dropped `ob_header.ob_type` store) and is what the
            // runtime stamps into the new object's header
            // (`bh_new_with_vtable` writes the type word when it is nonzero,
            // and the class word when the backend was given an offset).
            // A zero vtable would silently leave `ob_type` null, so it
            // fails loud here.
            OpKind::NewWithVtable { owner, vtable } => {
                let spec = bh_size_spec_from_callcontrol(
                    callcontrol.expect("new_with_vtable assembly requires a CallControl"),
                    owner,
                )
                .unwrap_or_else(|| {
                    panic!("new_with_vtable: no struct layout registered for owner {owner:?}")
                });
                // A registered type static cannot travel as this process's
                // address: the descr pool is loaded in another process.
                // The sentinel is the one `constants_r` already uses, and
                // `owner` carries the name the runtime map resolves. An
                // address that was never interned keeps the raw word.
                // vtable 0 is `rewrite_op_malloc`'s size-descr-only form:
                // `descr.get_vtable()` is filled from the published group.
                let (vtable_word, type_static_name) = if *vtable == 0 {
                    (0, String::new())
                } else {
                    match self.type_static_size_words(*vtable as i64) {
                        Some((word, name)) => (word, name),
                        None => (*vtable as u64, String::new()),
                    }
                };
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::Size {
                    size: spec.size,
                    type_id: spec.type_id,
                    vtable: vtable_word,
                    // Type-static name while `vtable` is a sentinel; empty
                    // when the address was not interned. The gc_cache hit
                    // keys on `type_id`, not this field.
                    owner: type_static_name,
                    all_fielddescrs: spec.all_fielddescrs,
                    // Round-trip the GC-header flag off the resolved struct
                    // layout: a `new_with_vtable` boxes a header-carrying
                    // object, so its size descr must keep `GUARD_GC_TYPE`
                    // gating (a header-less raw struct would clear it).
                    is_gc_managed: spec.is_gc_managed,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // The allocation result is ALWAYS a fresh GC ref ('r').  fresh-reference:
                // assert the regalloc result kind structurally (mirror the
                // getarrayitem invariant), so an Int-banked result fails loud at
                // emit time instead of silently miscompiling.
                let result = op
                    .result
                    .as_ref()
                    .expect("new_with_vtable must produce a result register");
                argcodes.push('>');
                let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                assert_eq!(
                    kc, 'r',
                    "new_with_vtable result must be ref-kind ('r'), got '{kc}' for owner {owner:?}"
                );
                argcodes.push(kc);
                state.code.push(reg);
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            // RPython `rpython/jit/codewriter/jtransform.py:546` emits
            // `int_guard_value(op.args[0])` where `op.args[0]` is already a
            // `Ptr(FuncType)` integer after rtype.  Rust `&dyn Trait` is a
            // fat pointer so the rtyper-equivalent layer
            // (`translator/rtyper/rclass.rs::class_get_method_ptr`) emits
            // `OpKind::VtableMethodPtr(receiver)` with the
            // `(trait_root, method_name)` pair; the assembler encodes it
            // as `vtable_method_ptr/rd>i` carrying a `BhDescr::VtableMethod`.
            // The result register is the integer function address that
            // `int_guard_value` and the subsequent `residual_call_*`
            // consume — backend lowering of the actual vtable slot read
            // is not yet implemented.
            // blackhole.py `@arguments("cpu", "r", returns="i")
            // bhimpl_guard_class` — one ref register in, the class out.
            // The result bank follows the register the original read was
            // allocated to (`guard_class/r>i` or `guard_class/r>r`), see
            // `OpKind::GuardClass`.
            OpKind::GuardClass { base } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                // The base is the object whose header word is read, so it is a
                // ref register; `bhimpl_guard_class` takes `"r"` and both
                // registered keys (`guard_class/r>i`, `guard_class/r>r`) spell
                // it that way.  An Int-banked base would form a key nothing
                // registers, so fail loud at emit time.
                assert_eq!(
                    kc, 'r',
                    "guard_class base must be ref-kind ('r'), got '{kc}'"
                );
                state.code.push(reg);
                argcodes.push(kc);
                let result = op
                    .result
                    .as_ref()
                    .expect("guard_class must produce a result register");
                argcodes.push('>');
                let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                assert!(
                    kc == 'i' || kc == 'r',
                    "guard_class result must be int- or ref-kind, got '{kc}'"
                );
                argcodes.push(kc);
                state.code.push(reg);
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VtableMethodPtr {
                receiver,
                trait_root,
                method_name,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(receiver, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::VtableMethod {
                    trait_root: trait_root.clone(),
                    method_name: method_name.clone(),
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            // RPython `rpython/jit/codewriter/jtransform.py:901-903` emits
            // `record_quasiimmut_field(v_inst, fielddescr, mutatefielddescr)`
            // — a register followed by two descrs.  The blackhole counterpart
            // `bhimpl_record_quasiimmut_field(struct, fielddescr,
            // mutatefielddescr)` (`rpython/jit/metainterp/blackhole.py`)
            // expects argcodes `rdd`.
            OpKind::RecordQuasiImmutField {
                base,
                field,
                mutate_field,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                for fd in [field, mutate_field] {
                    let descr_idx = self.emit_ready_descr(fielddescrof(
                        fd,
                        &crate::model::ValueType::Unknown,
                        callcontrol,
                        true,
                    ));
                    state.code.push((descr_idx & 0xFF) as u8);
                    state.code.push((descr_idx >> 8) as u8);
                    argcodes.push('d');
                }
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::FieldWrite {
                base,
                value,
                field,
                ty,
            } => {
                // `rewrite_op_setfield` drops a void value. A unit payload
                // that still reaches here has no coloring; emit nothing.
                if let crate::model::LinkArg::Value(var) = value
                    && crate::model::FunctionGraph::concretetype_of(var)
                        == crate::model::ConcreteType::Void
                {
                    state.code.pop();
                    state.startpoints.shift_remove(&startposition);
                    return;
                }
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                assert!(
                    kc == 'r' || kc == 'i',
                    "setfield base must be Ref (gc) or Int (raw), got {kc:?} — graph {:?}",
                    self.current_graph_name,
                );
                let is_gc = kc == 'r';
                state.code.push(reg);
                argcodes.push(kc);
                // RPython `bhimpl_setfield_gc_{i,r,f}` canonical keys key
                // off the VALUE's kind (`@arguments("cpu", "r", "X",
                // "d")`), not the declared field type — declared field
                // `ty` can be pyre-only Void/State/Unknown.  jtransform's
                // rewrite_op_setfield derives the value kind from
                // `getkind(v_newvalue.concretetype)`; mirror that via
                // `constant_kind` (concretetype, falling back to the Value
                // variant).  A Variable value is a register of its SSA
                // kind.  An int Constant takes the short `c` byte
                // (`setfield_gc_i/rcd`) when small or a pool `i` slot
                // (`/rid`) when wide (`assembler.py:99-107`, `setfield_gc`
                // ∈ USE_C_FORM at `assembler.py`); a ref/float Constant
                // takes its pooled `r`/`f` byte (`assembler.py:168`).
                // `setfield_raw_*` is not in USE_C_FORM.
                let value_kind = match value {
                    crate::model::LinkArg::Value(var) => {
                        let (reg, kc) = self.lookup_reg_with_kind_var(var, regallocs);
                        state.code.push(reg);
                        argcodes.push(kc);
                        kc
                    }
                    crate::model::LinkArg::Const(c) => {
                        let kind = crate::flatten::constant_kind(c);
                        if kind == 'i' {
                            let (byte, argcode) = self.emit_const_i_from_const_allow_short(
                                &c.value,
                                use_c_form(if is_gc {
                                    "setfield_gc_i"
                                } else {
                                    "setfield_raw_i"
                                }),
                                state,
                                callcontrol,
                            );
                            state.code.push(byte);
                            argcodes.push(argcode);
                        } else {
                            let byte = self.emit_const(&c.value, kind, state, callcontrol);
                            state.code.push(byte);
                            argcodes.push(kind);
                        }
                        kind
                    }
                };
                if !is_gc && value_kind == 'r' {
                    panic!(
                        "setfield_raw_r not supported (jtransform.py rewrite_op_setfield) \
                         — graph {:?}",
                        self.current_graph_name,
                    );
                }
                let descr_idx = self.emit_ready_descr(fielddescrof(field, ty, callcontrol, is_gc));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                let opname = if is_gc {
                    format!("setfield_gc_{value_kind}")
                } else {
                    format!("setfield_raw_{value_kind}")
                };
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::RawLoad {
                base,
                offset,
                item_ty: _,
                itemsize,
                is_item_signed,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                assert_eq!(
                    kc, 'i',
                    "raw_load base must be int-kind (a raw address), got {kc:?} — \
                     graph {:?}",
                    self.current_graph_name,
                );
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind_var(offset, regallocs);
                assert_eq!(
                    kc, 'i',
                    "raw_load byte offset must be int-kind, got {kc:?} — graph {:?}",
                    self.current_graph_name,
                );
                state.code.push(reg);
                argcodes.push(kc);
                let result = op
                    .result
                    .as_ref()
                    .expect("raw_load produces the loaded element");
                let (result_reg, result_kind) = self.lookup_reg_with_kind_var(result, regallocs);
                let descr_ty = if result_kind == 'f' {
                    crate::model::ValueType::Float
                } else {
                    crate::model::ValueType::Int
                };
                let descr_idx = self.emit_ready_descr(raw_carray_descrof(
                    &descr_ty,
                    *itemsize,
                    *is_item_signed,
                ));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                argcodes.push('>');
                argcodes.push(result_kind);
                state.code.push(result_reg);
                let key = format!("raw_load_{result_kind}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::RawStore {
                base,
                offset,
                value,
                item_ty: _,
                itemsize,
                is_item_signed,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                assert_eq!(
                    kc, 'i',
                    "raw_store base must be int-kind (a raw address), got {kc:?} — \
                     graph {:?}",
                    self.current_graph_name,
                );
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind_var(offset, regallocs);
                assert_eq!(
                    kc, 'i',
                    "raw_store byte offset must be int-kind, got {kc:?} — graph {:?}",
                    self.current_graph_name,
                );
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, value_kind) = self.lookup_reg_with_kind_var(value, regallocs);
                assert!(
                    matches!(value_kind, 'i' | 'f'),
                    "raw_store value must be int- or float-kind, got {value_kind:?} — \
                     graph {:?}",
                    self.current_graph_name,
                );
                state.code.push(reg);
                argcodes.push(value_kind);
                let descr_ty = if value_kind == 'f' {
                    crate::model::ValueType::Float
                } else {
                    crate::model::ValueType::Int
                };
                let descr_idx = self.emit_ready_descr(raw_carray_descrof(
                    &descr_ty,
                    *itemsize,
                    *is_item_signed,
                ));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                let key = format!("raw_store_{value_kind}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::ArrayRead {
                base,
                index,
                item_ty,
                array_type_id,
                nolength,
                pure,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                assert!(
                    kc == 'r' || kc == 'i',
                    "getarrayitem base must be Ref (gc) or Int (raw), got {kc:?} — graph {:?}",
                    self.current_graph_name,
                );
                let is_gc = kc == 'r';
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind_var(index, regallocs);
                // Array indices are typed Signed, so the index register is
                // always the int bank.  `bhimpl_getarrayitem_gc_*` only
                // dispatches the `i`-index argcode; a ref/float index is a
                // kind-flow bug upstream of the codewriter — fail here rather
                // than emit an unconsumable ref-index (`rrd`/`ird`) shape.
                //
                // The array's identity is part of the report: one graph can
                // hold several `getarrayitem_gc`s and only some of them carry
                // the bad index, so without it the message names the graph and
                // leaves the op to be found by bisection.
                assert_eq!(
                    kc, 'i',
                    "getarrayitem_gc array index must be int-kind, got {kc:?} — \
                     graph {:?}, array_type_id {array_type_id:?}, item_ty \
                     {item_ty:?}, flatop {:?} (set MAJIT_COVERAGE_PANIC=1 to \
                     populate the flatop context)",
                    self.current_graph_name, self.current_flatop_debug,
                );
                state.code.push(reg);
                argcodes.push(kc);
                // descr.py:359-362 + ARRAY_INSIDE._hints.get('nolength',
                // False): the producer (model::OpKind::ArrayRead) carries
                // the layout bit. `nolength=true` → no length header
                // (`lendescr=None`); `nolength=false` → length word at
                // offset 0 (`lendescr=Some(0)`).
                let len_offset = if *nolength { None } else { Some(0) };
                let descr_idx = self.emit_ready_descr(arraydescrof(
                    item_ty,
                    array_type_id,
                    len_offset,
                    callcontrol,
                ));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython `bhimpl_getarrayitem_gc_{i,r,f}` keys off the
                // result register's kind — same rationale as getfield_gc_*.
                let result_kind = if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                    kc
                } else {
                    'v'
                };
                // `jtransform.py rewrite_op_getarrayitem`: `_gckind != 'gc'`
                // emits `getarrayitem_raw_*` and strips `_pure`.  A
                // `getarrayitem_raw_r` is refused.
                if !is_gc && result_kind == 'r' {
                    panic!(
                        "getarrayitem_raw_r not supported \
                         (jtransform.py rewrite_op_getarrayitem) — graph {:?}",
                        self.current_graph_name,
                    );
                }
                // `getarrayitem_gc_{i,r,f}_pure` for a foldable/immutable
                // element load (`ll_getitem_foldable_nonneg`, rlist.py);
                // `blackhole.py:1339-1341` aliases `bhimpl_getarrayitem_gc_*_pure
                // = bhimpl_getarrayitem_gc_*`, so the `rid>X` argcodes are
                // identical to the non-pure form — only the opname differs.
                let opname = if is_gc && *pure {
                    format!("getarrayitem_gc_{result_kind}_pure")
                } else if is_gc {
                    format!("getarrayitem_gc_{result_kind}")
                } else {
                    format!("getarrayitem_raw_{result_kind}")
                };
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::ArrayWrite {
                base,
                index,
                value,
                item_ty,
                array_type_id,
                nolength,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                assert!(
                    kc == 'r' || kc == 'i',
                    "setarrayitem base must be Ref (gc) or Int (raw), got {kc:?} — graph {:?}",
                    self.current_graph_name,
                );
                let is_gc = kc == 'r';
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind_var(index, regallocs);
                // Array indices are typed Signed, so the index register is
                // always the int bank.  `bhimpl_setarrayitem_gc_*` only
                // dispatches the `i`-index argcode; a ref/float index is a
                // kind-flow bug upstream of the codewriter — fail here rather
                // than emit an unconsumable ref-index (`rrrd`) shape.
                assert_eq!(
                    kc, 'i',
                    "setarrayitem_gc array index must be int-kind, got {kc:?}",
                );
                state.code.push(reg);
                argcodes.push(kc);
                // RPython `bhimpl_setarrayitem_gc_{i,r,f}` keys off the
                // VALUE's kind; jtransform.py:803 derives it from
                // `getkind(op.args[2].concretetype)` and passes the operand
                // (Variable or Constant) verbatim.  Mirror the FieldWrite
                // c-form: a Variable is a register; an int Constant takes
                // the short `c` byte (`setarrayitem_gc_i/ricd`) when small
                // (`setarrayitem_gc_i` ∈ USE_C_FORM, assembler.py) or a
                // pool `i` slot otherwise; a ref/float Constant takes its
                // pooled byte.
                let value_kind = match value {
                    crate::model::LinkArg::Value(var) => {
                        let (reg, kc) = self.lookup_reg_with_kind_var(var, regallocs);
                        state.code.push(reg);
                        argcodes.push(kc);
                        kc
                    }
                    crate::model::LinkArg::Const(c) => {
                        let kind = crate::flatten::constant_kind(c);
                        if kind == 'i' {
                            let (byte, argcode) = self.emit_const_i_from_const_allow_short(
                                &c.value,
                                use_c_form(if is_gc {
                                    "setarrayitem_gc_i"
                                } else {
                                    "setarrayitem_raw_i"
                                }),
                                state,
                                callcontrol,
                            );
                            state.code.push(byte);
                            argcodes.push(argcode);
                        } else {
                            let byte = self.emit_const(&c.value, kind, state, callcontrol);
                            state.code.push(byte);
                            argcodes.push(kind);
                        }
                        kind
                    }
                };
                // pyre source-level array operations are emitted from
                // `Vec<T>` / GcArray-backed layouts that always carry a
                // length header at offset 0 (rust-source / codewriter
                // descr.py:359-362 + ARRAY_INSIDE._hints.get('nolength',
                // False): the producer carries the layout bit via
                // `OpKind::ArrayWrite::nolength`; same encoding rule as
                // ArrayRead above.
                let len_offset = if *nolength { None } else { Some(0) };
                let descr_idx = self.emit_ready_descr(arraydescrof(
                    item_ty,
                    array_type_id,
                    len_offset,
                    callcontrol,
                ));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                if !is_gc && value_kind == 'r' {
                    panic!(
                        "setarrayitem_raw_r not supported \
                         (jtransform.py rewrite_op_setarrayitem) — graph {:?}",
                        self.current_graph_name,
                    );
                }
                // RPython `bhimpl_setarrayitem_gc_{i,r,f}` keys off the
                // value register's kind — same rationale as setfield_gc_*.
                let opname = if is_gc {
                    format!("setarrayitem_gc_{value_kind}")
                } else {
                    format!("setarrayitem_raw_{value_kind}")
                };
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            // `jtransform.py rewrite_op_getinteriorfield`: GcArray of Struct
            // lowers to `getinteriorfield_gc_{i,r,f}(v_inst, v_index, descr)`.
            OpKind::InteriorFieldRead {
                base,
                index,
                field,
                item_ty: _,
                array_type_id,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                assert_eq!(
                    kc, 'r',
                    "getinteriorfield base must be Ref (gc), got {kc:?} — graph {:?}",
                    self.current_graph_name,
                );
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind_var(index, regallocs);
                assert_eq!(
                    kc, 'i',
                    "getinteriorfield index must be int-kind, got {kc:?} — \
                     graph {:?}, array_type_id {array_type_id:?}",
                    self.current_graph_name,
                );
                state.code.push(reg);
                argcodes.push(kc);
                let descr = interiorfield_bh_descr(
                    callcontrol,
                    array_type_id,
                    &field.name,
                    &self.current_graph_name,
                );
                let descr_idx = self.emit_ready_descr(descr);
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // `bhimpl_getinteriorfield_gc_{i,r,f}` keys off the result
                // register kind (`@arguments("cpu", "r", "i", "d", returns="X")`),
                // the same way `rewrite_op_getfield` keys `getfield_gc_*`.
                let result_kind = if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                    kc
                } else {
                    'v'
                };
                let opname = format!("getinteriorfield_gc_{result_kind}");
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            // `jtransform.py rewrite_op_setinteriorfield`:
            // `setinteriorfield_gc_{i,r,f}(v_inst, v_index, v_value, descr)`.
            OpKind::InteriorFieldWrite {
                base,
                index,
                field,
                value,
                item_ty: _,
                array_type_id,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                assert_eq!(
                    kc, 'r',
                    "setinteriorfield base must be Ref (gc), got {kc:?} — graph {:?}",
                    self.current_graph_name,
                );
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind_var(index, regallocs);
                assert_eq!(
                    kc, 'i',
                    "setinteriorfield index must be int-kind, got {kc:?} — \
                     graph {:?}, array_type_id {array_type_id:?}",
                    self.current_graph_name,
                );
                state.code.push(reg);
                argcodes.push(kc);
                // `rewrite_op_setinteriorfield` passes `v_value` through;
                // the op stores a register (`ArrayWrite`'s variable arm).
                // `setinteriorfield_gc_*` is not in `USE_C_FORM` (`assembler.py`).
                let (reg, value_kind) = self.lookup_reg_with_kind_var(value, regallocs);
                state.code.push(reg);
                argcodes.push(value_kind);
                let descr = interiorfield_bh_descr(
                    callcontrol,
                    array_type_id,
                    &field.name,
                    &self.current_graph_name,
                );
                let descr_idx = self.emit_ready_descr(descr);
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                let opname = format!("setinteriorfield_gc_{value_kind}");
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            // `arraylen_gc(array, arraydescr)` — `len(l.items)` reads the
            // GcArray length header. Operand shape `rd>i`: base reg + descr
            // + int result. The descr is `arraydescrof(ARRAY)` with a
            // length word (`jtransform.py rewrite_op_getarraysize`); a
            // `nolength` ARRAY has `lendescr is None` and blackhole
            // `bh_arraylen_gc` panics (llmodel.py).
            OpKind::ArrayLen {
                base,
                array_type_id,
                nolength,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                // `jtransform.py rewrite_op_getarraysize` asserts
                // `ARRAY._gckind == 'gc'`.  An Int-bank base is not
                // `arraylen_gc`.
                assert_eq!(
                    kc, 'r',
                    "arraylen_gc base must use the Ref register bank \
                     (blackhole.py bhimpl_arraylen_gc @arguments(\"cpu\", \"r\", \"d\", returns=\"i\")) \
                     — graph {:?}",
                    self.current_graph_name,
                );
                state.code.push(reg);
                argcodes.push(kc);
                let len_offset = if *nolength { None } else { Some(0) };
                let descr = arraydescrof(
                    &crate::model::ValueType::Ref(None),
                    array_type_id,
                    len_offset,
                    callcontrol,
                );
                // `get_array_descr` answers a cache hit with the first
                // mint of this ARRAY identity and drops a later
                // `nolength=false` request, so the live descr — not the
                // flag on this op — is what blackhole will read.
                match &descr {
                    crate::jitcode::BhDescr::Array {
                        len_offset: Some(_),
                        ..
                    } => {}
                    crate::jitcode::BhDescr::Array {
                        len_offset: None,
                        array_type_id: atid,
                        ..
                    } => panic!(
                        "arraylen_gc requires ArrayDescr.lendescr \
                         (llmodel.py bh_arraylen_gc); graph {:?} \
                         array_type_id {:?} requested nolength={}",
                        self.current_graph_name, atid, *nolength
                    ),
                    other => panic!("arraylen_gc descr must be Array, got {other:?}"),
                }
                let descr_idx = self.emit_ready_descr(descr);
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // `bhimpl_arraylen_gc` always returns int.
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("arraylen_gc/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            // Vable field/array: encode the base register followed by the
            // field_index descriptor, matching blackhole.py @arguments("r", "d").
            OpKind::VableFieldRead {
                base, field_index, ..
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                assert_eq!(
                    kc, 'r',
                    "virtualizable base must use the Ref register bank: RPython BlackholeInterpreter.bhimpl_getfield_vable_* declares @arguments(\"r\", \"d\")"
                );
                state.code.push(reg);
                argcodes.push(kc);
                // RPython: vable field → VableField descriptor (index, not byte offset).
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::VableField {
                    index: *field_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython `bhimpl_getfield_vable_{i,r,f}` canonical keys
                // (blackhole.py bhimpl_getfield_vable_i) match on the RESULT register
                // kind. See FieldRead above for the Void/State/Unknown
                // rationale — the pyre-only declared ty can be Void
                // while the SSA result register is always i/r/f.
                let result_kind = if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                    kc
                } else {
                    'v'
                };
                let opname = format!("getfield_vable_{result_kind}");
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VableFieldWrite {
                base,
                field_index,
                value,
                ..
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                assert_eq!(
                    kc, 'r',
                    "virtualizable base must use the Ref register bank: RPython BlackholeInterpreter.bhimpl_setfield_vable_* declares @arguments(\"r\", ..., \"d\")"
                );
                state.code.push(reg);
                argcodes.push(kc);
                // `setfield_vable_i` is not in USE_C_FORM
                // (`assembler.py:312-345`): a constant value never takes
                // the short `c` byte, always a pool slot.  The value kind
                // follows `getkind(v_newvalue.concretetype)` via
                // `constant_kind` — int → pool `i` (`setfield_vable_i/rid`),
                // ref/float → pool `r`/`f` (`assembler.py:168`).  A register
                // value keeps its own SSA kind.
                let value_kind = match value {
                    crate::model::LinkArg::Value(var) => {
                        let (reg, kc) = self.lookup_reg_with_kind_var(var, regallocs);
                        state.code.push(reg);
                        argcodes.push(kc);
                        kc
                    }
                    crate::model::LinkArg::Const(c) => {
                        let kind = crate::flatten::constant_kind(c);
                        let byte = self.emit_const(&c.value, kind, state, callcontrol);
                        state.code.push(byte);
                        argcodes.push(kind);
                        kind
                    }
                };
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::VableField {
                    index: *field_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython `bhimpl_setfield_vable_{i,r,f}` canonical keys
                // (blackhole.py bhimpl_setfield_vable_i) match on the VALUE register's
                // kind. Same rationale as setfield_gc_*.
                let opname = format!("setfield_vable_{value_kind}");
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VableArrayRead {
                base,
                array_index,
                elem_index,
                item_ty,
                array_itemsize,
                array_is_signed,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                assert_eq!(
                    kc, 'r',
                    "virtualizable base must use the Ref register bank: RPython BlackholeInterpreter.bhimpl_getarrayitem_vable_* declares @arguments(\"r\", \"i\", \"d\", \"d\")"
                );
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind_var(elem_index, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                // RPython: two descriptors — fielddescr (vable array field) + arraydescr.
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::VableArray {
                    index: *array_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // Second descriptor: arraydescr from VirtualizableInfo.array_descrs.
                let descr_idx2 = self.emit_ready_descr(vable_arraydescrof(
                    item_ty,
                    *array_itemsize,
                    *array_is_signed,
                ));
                state.code.push((descr_idx2 & 0xFF) as u8);
                state.code.push((descr_idx2 >> 8) as u8);
                argcodes.push('d');
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VableArrayWrite {
                base,
                array_index,
                elem_index,
                value,
                item_ty,
                array_itemsize,
                array_is_signed,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                assert_eq!(
                    kc, 'r',
                    "virtualizable base must use the Ref register bank: RPython BlackholeInterpreter.bhimpl_setarrayitem_vable_* declares @arguments(\"r\", \"i\", ..., \"d\", \"d\")"
                );
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind_var(elem_index, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                // `jtransform.py` `rewrite_op_setarrayitem` passes the value
                // through. A constant is a pool slot, same as
                // `setfield_vable_*` (`assembler.py` constant registers).
                let value_kind = match value {
                    crate::model::LinkArg::Value(var) => {
                        let (reg, kc) = self.lookup_reg_with_kind_var(var, regallocs);
                        state.code.push(reg);
                        argcodes.push(kc);
                        kc
                    }
                    crate::model::LinkArg::Const(c) => {
                        let kind = crate::flatten::constant_kind(c);
                        let byte = self.emit_const(&c.value, kind, state, callcontrol);
                        state.code.push(byte);
                        argcodes.push(kind);
                        kind
                    }
                };
                // RPython: two descriptors — fielddescr (vable array field) + arraydescr.
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::VableArray {
                    index: *array_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // Second descriptor: arraydescr from VirtualizableInfo.array_descrs.
                let descr_idx2 = self.emit_ready_descr(vable_arraydescrof(
                    item_ty,
                    *array_itemsize,
                    *array_is_signed,
                ));
                state.code.push((descr_idx2 & 0xFF) as u8);
                state.code.push((descr_idx2 >> 8) as u8);
                argcodes.push('d');
                let opname = format!("setarrayitem_vable_{value_kind}");
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VableArrayLen {
                base,
                array_index,
                item_ty,
                array_itemsize,
                array_is_signed,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                assert_eq!(
                    kc, 'r',
                    "virtualizable base must use the Ref register bank: RPython BlackholeInterpreter.bhimpl_arraylen_vable declares @arguments(\"r\", \"d\", \"d\")"
                );
                state.code.push(reg);
                argcodes.push(kc);
                // The same descr pair its read and write siblings carry, in
                // the same order: fielddescr (vable array field) + arraydescr.
                // `arraylen_vable/rdd>i` takes the length off the first at
                // run time, but `vable_array_index_pair_at` reads both and
                // rejects anything that is not `(VableArray, Array)`.
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::VableArray {
                    index: *array_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                let descr_idx2 = self.emit_ready_descr(vable_arraydescrof(
                    item_ty,
                    *array_itemsize,
                    *array_is_signed,
                ));
                state.code.push((descr_idx2 & 0xFF) as u8);
                state.code.push((descr_idx2 >> 8) as u8);
                argcodes.push('d');
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VableForce { base } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                assert_eq!(kc, 'r', "hint_force_virtualizable expects a Ref base");
                state.code.push(reg);
                argcodes.push(kc);
                let opnum = self.get_opnum("hint_force_virtualizable/r");
                state.code[startposition] = opnum;
            }

            // RPython jtransform.py handle_jit_marker__loop_header
            // emits `SpaceOperation('loop_header', [c_index], None)`; upstream
            // assembler.py encodes that Constant via `emit_const(allow_short=
            // False)` which registers it in `constants_i` and emits a single
            // byte register index (argcodes `i`). The bhimpl signature
            // (`blackhole.py @arguments("i")`) looks the byte up in
            // `registers_i`. The canonical runtime key is `loop_header/i`
            // (`majit-metainterp/src/jitcode`); `emit_const_i`
            // returns `num_regs_i + pool_idx` which the runtime resolves
            // back to the constant via `registers_i[byte]`. Emitting via
            // the generic fallback would push zero operand bytes (because
            // `op_variable_refs(LoopHeader)` is empty), misaligning the
            // dispatch cursor.
            // jtransform.py `_rewrite_op_cond_call` — `conditional_call_ir_v/iiIRd`.
            // Condition and funcptr are int registers; force_ir always
            // emits both I and R lists, then the calldescr.
            OpKind::ConditionalCall {
                condition,
                funcptr,
                descriptor,
                args_i,
                args_r,
                args_f,
            } => {
                assert!(
                    args_f.is_empty(),
                    "conditional_call does not support float args"
                );
                let (reg, kc) = self.lookup_reg_with_kind_var(condition, regallocs);
                assert_eq!(kc, 'i', "conditional_call condition is an int");
                state.code.push(reg);
                argcodes.push(kc);
                // `assembler.py emit_const` on the funcptr Constant from
                // `_rewrite_op_cond_call` / `handle_residual_call`.
                let func_byte = self.emit_fnaddr_const(funcptr, state, callcontrol);
                state.code.push(func_byte);
                argcodes.push('i');
                self.emit_list_of_kind(args_i, RegKind::Int, regallocs, state);
                argcodes.push('I');
                self.emit_list_of_kind(args_r, RegKind::Ref, regallocs, state);
                argcodes.push('R');
                let calldescr = descriptor.to_bh_calldescr();
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::Call { calldescr });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                let key = format!("conditional_call_ir_v/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // jtransform.py `_rewrite_op_cond_call` (is_value=True) —
            // `conditional_call_value_ir_{k}/{k}iIRd>{k}`. force_ir always
            // emits both I and R lists. Decoder order matches
            // `bhimpl_conditional_call_value_ir_{i,r}` and
            // `pyjitpl.py _opimpl_conditional_call_value`: value, funcptr
            // int const, I list, R list, calldescr, result.
            OpKind::ConditionalCallValue {
                value,
                funcptr,
                descriptor,
                args_i,
                args_r,
                args_f,
                result_kind,
            } => {
                assert!(
                    args_f.is_empty(),
                    "conditional_call_value does not support float args"
                );
                let (reg, kc) = self.lookup_reg_with_kind_var(value, regallocs);
                assert!(
                    kc == 'i' || kc == 'r',
                    "conditional_call_value lead is an int or a ref"
                );
                state.code.push(reg);
                argcodes.push(kc);
                // Same funcptr Constant channel as `conditional_call_ir_v`
                // and residual_call (`assembler.py emit_const`).
                let func_byte = self.emit_fnaddr_const(funcptr, state, callcontrol);
                state.code.push(func_byte);
                argcodes.push('i');
                self.emit_list_of_kind(args_i, RegKind::Int, regallocs, state);
                argcodes.push('I');
                self.emit_list_of_kind(args_r, RegKind::Ref, regallocs, state);
                argcodes.push('R');
                let calldescr = descriptor.to_bh_calldescr();
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::Call { calldescr });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                let result_key_kind = self.emit_call_result_arg(
                    op.result.as_ref(),
                    *result_kind,
                    regallocs,
                    state,
                    &mut argcodes,
                );
                let key = format!("conditional_call_value_ir_{result_key_kind}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            OpKind::LoopHeader { jitdriver_index } => {
                let reg_byte = self.emit_const_i(*jitdriver_index as i64, state);
                state.code.push(reg_byte);
                argcodes.push('i');
                let opnum = self.get_opnum("loop_header/i");
                state.code[startposition] = opnum;
            }

            // RPython jtransform.py handle_jit_marker__jit_merge_point
            // emits `SpaceOperation('jit_merge_point',
            //   [Constant(jdindex), greens_i, greens_r, greens_f,
            //    reds_i, reds_r, reds_f], None)`. Upstream bhimpl signature
            // (`blackhole.py @arguments("self","i","I","R","F",
            // "I","R","F")`) reads jdindex + six typed register lists, each
            // encoded as `[len:u8][reg:u8 * N]` (assembler.py:181-196 ListOfKind).
            // pyre's runtime (`blackhole.rs`'s `handler_jit_merge_point_c`)
            // consumes exactly this
            // six-list shape. The canonical runtime key is
            // `jit_merge_point/cIRFIRF` for signed-byte jitdriver indices or
            // `jit_merge_point/iIRFIRF` for constant-pool jitdriver indices.
            // The generic fallback would flatten SSA register bytes without the
            // length prefix and without the jdindex byte, corrupting the
            // stream.
            OpKind::JitMergePoint {
                jitdriver_index,
                greens_i,
                greens_r,
                greens_f,
                reds_i,
                reds_r,
                reds_f,
                green_classes: _,
                red_classes: _,
            } => {
                // This LLBC route enforces `jtransform.py`'s at-most-one-marker
                // contract per portal graph. The sibling
                // `JitCodeBuilder::jit_merge_point` route tolerates later
                // markers while retaining the first offset; a body reaching
                // this route twice means two markers were lowered into one
                // graph and the offset below would name whichever won.
                assert!(
                    state.jit_merge_point_offset.is_none(),
                    "a second jit_merge_point in one assembled body",
                );
                state.jit_merge_point_offset = Some(startposition);
                let jdindex_value = *jitdriver_index as i64;
                let jdindex_argcode = if (-128..=127).contains(&jdindex_value) {
                    state.code.push(jdindex_value as i8 as u8);
                    'c'
                } else {
                    let jdindex_byte = self.emit_const_i(jdindex_value, state);
                    state.code.push(jdindex_byte);
                    'i'
                };
                self.emit_list_of_kind(greens_i, RegKind::Int, regallocs, state);
                self.emit_list_of_kind(greens_r, RegKind::Ref, regallocs, state);
                self.emit_list_of_kind(greens_f, RegKind::Float, regallocs, state);
                self.emit_list_of_kind(reds_i, RegKind::Int, regallocs, state);
                self.emit_list_of_kind(reds_r, RegKind::Ref, regallocs, state);
                self.emit_list_of_kind(reds_f, RegKind::Float, regallocs, state);
                argcodes.push(jdindex_argcode);
                argcodes.push('I');
                argcodes.push('R');
                argcodes.push('F');
                argcodes.push('I');
                argcodes.push('R');
                argcodes.push('F');
                let key = format!("jit_merge_point/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // Default: encode operand registers + result register (no descriptor)
            other => {
                let mut operand_kinds = String::new();
                for v in crate::inline::op_variable_refs(other) {
                    if crate::model::FunctionGraph::concretetype_of(&v)
                        == crate::model::ConcreteType::Void
                    {
                        continue;
                    }
                    let (reg, kind_char) = self.lookup_reg_with_kind_var(&v, regallocs);
                    state.code.push(reg);
                    argcodes.push(kind_char);
                    operand_kinds.push(kind_char);
                }
                // A result Variable's declared `concretetype` can be
                // `Void` here for the same reason `emit_call_result_arg`
                // guards it: `getkind(op.result.concretetype)` is the
                // obligation `flatten.py`'s `serialize_op` enforces, and
                // this default arm has no declared-kind side to consult
                // at all, so it must ask the Variable directly. A Void
                // result has no register to color; skip the `>` argcode
                // rather than let `lookup_reg_with_kind_var` reach it.
                if let Some(result) = op.result.as_ref()
                    && crate::model::FunctionGraph::concretetype_of(result)
                        != crate::model::ConcreteType::Void
                {
                    argcodes.push('>');
                    let (reg, kind_char) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kind_char);
                    state.code.push(reg);
                }
                let opname = op_kind_to_opname_with_kinds(other, &operand_kinds);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
        }

        // RPython assembler.py:217-219: record result type position.
        // If argcodes contains '>', the last char is the result kind,
        // and we record the current code length as the result position.
        if argcodes.contains('>')
            && let Some(reskind) = argcodes.chars().last()
        {
            state.resulttypes.insert(state.code.len(), reskind);
        }
    }

    /// Emit a ListOfKind: [u8 count][reg0][reg1]...
    ///
    /// RPython `assembler.py:181-196`: every item in the
    /// `ListOfKind(kind, [...])` shares the list's `kind` per
    /// construction (`flatten.py ListOfKind` carries `kind` as
    /// an attribute and the constructors only accept matching
    /// Registers).  Pyre asserts the same invariant strictly: each
    /// item resolves through `lookup_coloring_var` and its kind must
    /// equal the list's `kind`.  A mismatch surfaces as a hard panic
    /// — the upstream contract has no escape hatch.
    fn emit_list_of_kind(
        &self,
        args: &[crate::flowspace::model::Variable],
        kind: RegKind,
        regallocs: &HashMap<RegKind, RegAllocator>,
        state: &mut AssemblyState,
    ) {
        // RPython `assembler.py` writes the count as a single byte and
        // does not silently truncate — clipping to 255 while still
        // emitting every item would desync the decoder by N - 255 bytes.
        // Fail loud here so the producer's contract violation surfaces
        // at the codewriter rather than as garbled bytecode downstream.
        assert!(
            args.len() < 256,
            "emit_list_of_kind: {} entries exceed the u8 count byte \
             (kind {kind:?}); RPython parity requires the count to fit \
             in a single byte",
            args.len(),
        );
        let args: Vec<_> = args
            .iter()
            .filter(|v| {
                crate::model::FunctionGraph::concretetype_of(v) != crate::model::ConcreteType::Void
            })
            .cloned()
            .collect();
        state.code.push(args.len() as u8);
        for v in &args {
            let (reg, item_kind) = self.lookup_coloring_var(v, regallocs);
            assert_eq!(
                item_kind, kind,
                "emit_list_of_kind: item {v:?} has kind {item_kind:?} but the \
                 surrounding `ListOfKind` declares {kind:?} (graph {:?}; \
                 PyPy `flatten.py:35-51` keeps every item's kind aligned with \
                 the list's `kind` attribute)",
                self.current_graph_name,
            );
            state.code.push(reg);
        }
    }

    /// RPython `flatten.py` emits the trailing `-> result` only for
    /// non-Void result variables. `jtransform.py:433` still computes
    /// `reskind = getkind(op.result.concretetype)[0]`, so a call op may
    /// carry a Void result placeholder while the bytecode opcode must be
    /// `*_v` with no `>` argcode.
    fn emit_call_result_arg(
        &self,
        result: Option<&crate::flowspace::model::Variable>,
        declared_result_kind: char,
        regallocs: &HashMap<RegKind, RegAllocator>,
        state: &mut AssemblyState,
        argcodes: &mut String,
    ) -> char {
        if declared_result_kind == 'v' {
            return 'v';
        }
        let Some(result) = result else {
            return declared_result_kind;
        };
        // `declared_result_kind` is the call target's declared signature
        // and can disagree with the result Variable's own `concretetype`
        // (e.g. a synthetic ctor's `result_ty` staying `Ref` for a
        // destination type that resolves to `Void`). `getkind(op.result.
        // concretetype)` is the obligation `flatten.py`'s `serialize_op`
        // enforces on the Variable, not the declared kind, so check it
        // directly: a Void Variable has no register to color and must
        // not reach `lookup_coloring_var`.
        if crate::model::FunctionGraph::concretetype_of(result) == crate::model::ConcreteType::Void
        {
            return 'v';
        }
        argcodes.push('>');
        let (reg, kind) = self.lookup_coloring_var(result, regallocs);
        let kind_char = match kind {
            RegKind::Int => 'i',
            RegKind::Ref => 'r',
            RegKind::Float => 'f',
        };
        argcodes.push(kind_char);
        state.code.push(reg);
        kind_char
    }

    /// Resolve `(register_index, kind)` for a `&Variable`.
    ///
    /// **RPython invariant** (`flatten.py getcolor`): every
    /// Variable has exactly one `(kind, color)` via
    /// `getkind(v.concretetype)` + `regallocs[kind]`.  This helper
    /// reads the declared kind directly from `Variable.concretetype`
    /// (`rtyper.py setconcretetype v.concretetype = ...`) and looks up the color
    /// strictly in `regallocs[kind].coloring[var]` (orthodox per
    /// `tool/algo/regalloc.py coloring: dict[Variable, int]`) —
    /// a hard panic on miss.
    ///
    /// When the kind cannot be derived (test fixtures whose Variables
    /// lack a populated `concretetype`), the helper falls back to a
    /// [`KINDS`]-ordered scan with a multi-class panic that still
    /// preserves "exactly one class per value" semantics.  RPython
    /// has no equivalent fallback because every assembler call comes
    /// from the typed flatten output; the fallback is documented
    /// divergence pending the migration of `encode_op` slot lookups
    /// to the strict `(v, expected_kind)` form.
    fn lookup_coloring_var(
        &self,
        var: &crate::flowspace::model::Variable,
        regallocs: &HashMap<RegKind, RegAllocator>,
    ) -> (u8, RegKind) {
        // Strict path: `regallocs[kind]` supplies the color when
        // `Variable.concretetype` declares a non-Void/Unknown kind
        // (`flatten.py:386-387`).  A miss panics so an upstream
        // type ↔ regalloc mismatch surfaces immediately.
        use crate::model::ConcreteType;
        let declared = crate::model::FunctionGraph::concretetype_of(var);
        let kind = match declared {
            ConcreteType::Signed => Some(RegKind::Int),
            ConcreteType::GcRef => Some(RegKind::Ref),
            ConcreteType::Float => Some(RegKind::Float),
            ConcreteType::Void | ConcreteType::Unknown => None,
        };
        if let Some(kind) = kind {
            let ra = regallocs.get(&kind).unwrap_or_else(|| {
                panic!(
                    "lookup_coloring: type-state declared kind {kind:?} for {var:?} \
                     but regallocs map is missing the entry (graph={:?}, op={:?})",
                    self.current_graph_name, self.current_flatop_debug,
                )
            });
            let color = ra.coloring.get(var).copied().unwrap_or_else(|| {
                let other_classes: Vec<_> = [RegKind::Int, RegKind::Ref, RegKind::Float]
                    .iter()
                    .filter(|k| **k != kind)
                    .filter(|k| {
                        regallocs
                            .get(*k)
                            .is_some_and(|ra| ra.coloring.contains_key(var))
                    })
                    .copied()
                    .collect();
                panic!(
                    "lookup_coloring: type-state declared kind {kind:?} for {var:?} \
                     but regallocs[{kind:?}] has no coloring (other classes with a \
                     coloring: {other_classes:?}; graph={:?}, op={:?})",
                    self.current_graph_name, self.current_flatop_debug,
                )
            });
            return (color as u8, kind);
        }
        // No declared kind (Void / Unknown, or test fixture whose
        // Variable lacks a populated `concretetype`).  Fall back to
        // a KINDS-ordered scan with a single-class assertion.
        let mut found: Option<(u8, RegKind)> = None;
        for kind in [RegKind::Int, RegKind::Ref, RegKind::Float] {
            if let Some(ra) = regallocs.get(&kind)
                && let Some(&color) = ra.coloring.get(var)
            {
                if let Some((_, prev)) = found {
                    panic!(
                        "lookup_coloring: value {var:?} colored in multiple regalloc \
                             classes ({prev:?} and {kind:?}) — RPython `getkind` must \
                             give exactly one (graph={:?}, op={:?})",
                        self.current_graph_name, self.current_flatop_debug,
                    );
                }
                found = Some((color as u8, kind));
            }
        }
        if let Some(result) = found {
            return result;
        }
        let class_coverage: Vec<_> = [RegKind::Int, RegKind::Ref, RegKind::Float]
            .iter()
            .filter_map(|k| regallocs.get(k).map(|ra| (*k, ra)))
            .map(|(k, ra)| (k, ra.coloring.len()))
            .collect();
        panic!(
            "lookup_coloring: value {var:?} has no coloring in any regalloc class \
             (graph={:?}, op={:?}, regalloc_coverage={:?})",
            self.current_graph_name, self.current_flatop_debug, class_coverage,
        );
    }

    /// Look up the register index (u8) AND kind character for a
    /// Variable — orthodox per `tool/algo/regalloc.py:31`
    /// `coloring: dict[Variable, int]`.  Returns
    /// (register_index, kind_char) where kind_char ∈ {'i','r','f'}.
    fn lookup_reg_with_kind_var(
        &self,
        var: &crate::flowspace::model::Variable,
        regallocs: &HashMap<RegKind, RegAllocator>,
    ) -> (u8, char) {
        let (color, kind) = self.lookup_coloring_var(var, regallocs);
        let kind_char = match kind {
            RegKind::Int => 'i',
            RegKind::Ref => 'r',
            RegKind::Float => 'f',
        };
        (color, kind_char)
    }

    /// Eagerly walk every `FlatOp` in `ssarepr.insns` and report every
    /// `Variable` that lacks a regalloc coloring in any class.
    ///
    /// Pyre-only diagnostic — RPython's `assembler.py` never needs
    /// this because `rtyper` guarantees that every `Variable`'s
    /// `concretetype` produces exactly one `(kind, color)` via
    /// `getkind()` + `regalloc.py`.  Pyre's annotator / rtyper still
    /// has known coverage gaps, and the
    /// `lookup_reg_with_kind` fallback silently emits a `(0, 'r')`
    /// default at write time — which masks how many distinct gaps
    /// exist per graph.  `MAJIT_COVERAGE_PANIC=1` aborts at the first
    /// gap, losing the rest; this walker enumerates them all up
    /// front so the full gap catalogue surfaces in a single build.
    ///
    /// Output goes through `cargo:warning=` so the build script
    /// runner (`build.rs`) surfaces each line to the user.
    fn run_coverage_audit(&self, ssarepr: &SSARepr, regallocs: &HashMap<RegKind, RegAllocator>) {
        // For each Variable, track: has a def site (result of some op),
        // count of direct operand uses, count of Live markers mentioning
        // it.  Live-only gaps (no def, no operand use) point at backward
        // liveness leakage; uses-without-def at missing rtyper coverage;
        // def-without-coverage at regalloc class mismatch.
        #[derive(Default)]
        struct ValueSites {
            has_def: bool,
            use_count: usize,
            live_count: usize,
            first_use_tag: Option<&'static str>,
        }

        fn opkind_tag(kind: &crate::model::OpKind) -> &'static str {
            use crate::model::OpKind;
            match kind {
                OpKind::Input { .. } => "Input",
                OpKind::RawLoad { .. } => "RawLoad",
                OpKind::RawStore { .. } => "RawStore",
                OpKind::ConstInt(_) => "ConstInt",
                OpKind::ConstFnAddr { .. } => "ConstFnAddr",
                OpKind::ConstUInt(_) => "ConstUInt",
                OpKind::ConstInt128(_) => "ConstInt128",
                OpKind::ConstUInt128(_) => "ConstUInt128",
                OpKind::ConstSingleFloat(_) => "ConstSingleFloat",
                OpKind::ConstBool(_) => "ConstBool",
                OpKind::ConstSymbolic { .. } => "ConstSymbolic",
                OpKind::ConstFloat(_) => "ConstFloat",
                OpKind::ConstStr(_) => "ConstStr",
                OpKind::ConstInternedStr(_) => "ConstInternedStr",
                OpKind::ConstRef(_) => "ConstRef",
                OpKind::ConstRefNull => "ConstRefNull",
                OpKind::ConstNone => "ConstNone",
                OpKind::ConstRefAddr(_) => "ConstRefAddr",
                OpKind::FieldRead { .. } => "FieldRead",
                OpKind::FieldWrite { .. } => "FieldWrite",
                OpKind::ArrayRead { .. } => "ArrayRead",
                OpKind::ArrayLen { .. } => "ArrayLen",
                OpKind::ArrayWrite { .. } => "ArrayWrite",
                OpKind::InteriorFieldRead { .. } => "InteriorFieldRead",
                OpKind::InteriorFieldWrite { .. } => "InteriorFieldWrite",
                OpKind::Call { .. } => "Call",
                OpKind::GuardTrue { .. } => "GuardTrue",
                OpKind::GuardFalse { .. } => "GuardFalse",
                OpKind::GuardValue { .. } => "GuardValue",
                OpKind::GuardClass { .. } => "GuardClass",
                OpKind::VtableMethodPtr { .. } => "VtableMethodPtr",
                OpKind::IndirectCall { .. } => "IndirectCall",
                OpKind::VableFieldRead { .. } => "VableFieldRead",
                OpKind::VableFieldWrite { .. } => "VableFieldWrite",
                OpKind::VableArrayRead { .. } => "VableArrayRead",
                OpKind::VableArrayWrite { .. } => "VableArrayWrite",
                OpKind::VableArrayLen { .. } => "VableArrayLen",
                OpKind::BinOp { .. } => "BinOp",
                OpKind::UnaryOp { .. } => "UnaryOp",
                OpKind::VableForce { .. } => "VableForce",
                OpKind::Hint { .. } => "Hint",
                OpKind::CallElidable { .. } => "CallElidable",
                OpKind::CallResidual { .. } => "CallResidual",
                OpKind::CallMayForce { .. } => "CallMayForce",
                OpKind::InlineCall { .. } => "InlineCall",
                OpKind::RecursiveCall { .. } => "RecursiveCall",
                OpKind::JitDebug { .. } => "JitDebug",
                OpKind::AssertGreen { .. } => "AssertGreen",
                OpKind::CurrentTraceLength => "CurrentTraceLength",
                OpKind::IsConstant { .. } => "IsConstant",
                OpKind::IsVirtual { .. } => "IsVirtual",
                OpKind::IsInstance { .. } => "IsInstance",
                OpKind::ConditionalCall { .. } => "ConditionalCall",
                OpKind::ConditionalCallValue { .. } => "ConditionalCallValue",
                OpKind::RecordKnownResult { .. } => "RecordKnownResult",
                OpKind::RecordQuasiImmutField { .. } => "RecordQuasiImmutField",
                OpKind::Live => "Live",
                OpKind::JitMergePoint { .. } => "JitMergePoint",
                OpKind::LoopHeader { .. } => "LoopHeader",
                OpKind::Abort { .. } => "Abort",
                OpKind::NewTuple { .. } => "NewTuple",
                OpKind::NewList { .. } => "NewList",
                OpKind::GetSlice { .. } => "GetSlice",
                OpKind::New { .. } => "New",
                OpKind::RawMalloc { .. } => "RawMalloc",
                OpKind::RawFree { .. } => "RawFree",
                OpKind::NewWithVtable { .. } => "NewWithVtable",
                OpKind::NewArray { .. } => "NewArray",
                OpKind::NewArrayClear { .. } => "NewArrayClear",
                OpKind::NewListClear { .. } => "NewListClear",
                OpKind::LoweredBlackholeOp { .. } => "LoweredBlackholeOp",
                OpKind::LoadStatic { .. } => "LoadStatic",
            }
        }
        #[expect(
            clippy::mutable_key_type,
            reason = "Variable equality and hashing use immutable identity IDs while annotations are excluded; this exactly models RPython object-identity dictionary keys without a divergent side table"
        )]
        let mut sites: std::collections::HashMap<
            crate::flowspace::model::Variable,
            ValueSites,
        > = std::collections::HashMap::new();

        for op in &ssarepr.insns {
            match op {
                FlatOp::Op(inner) => {
                    let tag = opkind_tag(&inner.kind);
                    if let Some(r) = inner.result.as_ref() {
                        sites.entry(r.clone()).or_default().has_def = true;
                    }
                    // `op_variable_refs` walks the orthodox Variable
                    // operands (RPython `SpaceOperation.args:
                    // Vec<Hlvalue>`) directly out of OpKind.
                    for v in crate::inline::op_variable_refs(&inner.kind) {
                        let s = sites.entry(v).or_default();
                        s.use_count += 1;
                        s.first_use_tag.get_or_insert(tag);
                    }
                }
                FlatOp::GotoIfNot { .. }
                | FlatOp::GotoIfNotOp { .. }
                | FlatOp::Switch { .. }
                | FlatOp::IntBinOpJumpIfOvf { .. } => {
                    // Guard ops carry [`Register`] operands
                    // (post-regalloc identity); not tracked by the
                    // pre-regalloc Variable audit.
                }
                FlatOp::Move { .. } | FlatOp::Push(_) | FlatOp::Pop(_) => {
                    // Move/Push/Pop carry [`Register`]
                    // (post-regalloc identity) rather than pre-regalloc
                    // Variables, so the audit (which is keyed on
                    // pre-regalloc Variables) can no longer attribute
                    // uses/defs to a specific source Variable here.  The coverage
                    // gap-finder still reads SpaceOperation arguments
                    // via the surrounding match arms; cycle-break and
                    // copy register operands are by construction
                    // covered, so dropping them from the audit is
                    // safe.
                }
                FlatOp::LastException { .. } | FlatOp::LastExcValue { .. } => {
                    // Register operand carries (kind, color);
                    // not tracked by the pre-regalloc Variable audit.
                }
                FlatOp::IntReturn(_)
                | FlatOp::RefReturn(_)
                | FlatOp::FloatReturn(_)
                | FlatOp::Raise(_) => {
                    // Operand is RegOrConst (Register or Constant);
                    // not tracked by the pre-regalloc Variable audit.
                }
                FlatOp::Live { .. } => {
                    // Live carries [`Register`]s; not tracked by the
                    // pre-regalloc Variable audit.
                }
                FlatOp::Label(_)
                | FlatOp::Jump(_)
                | FlatOp::VoidReturn
                | FlatOp::EndOfBlock
                | FlatOp::Unreachable
                | FlatOp::CatchException { .. }
                | FlatOp::GotoIfExceptionMismatch { .. }
                | FlatOp::Reraise => {}
            }
        }

        let mut gaps: Vec<(&crate::flowspace::model::Variable, &ValueSites)> = Vec::new();
        for (var, s) in &sites {
            // The Variable-keyed `coloring` map is the orthodox
            // source per `tool/algo/regalloc.py:31`; if none of the
            // three classes registered this Variable, it surfaces as
            // a coverage gap.
            let covered = [RegKind::Int, RegKind::Ref, RegKind::Float]
                .iter()
                .any(|k| {
                    regallocs
                        .get(k)
                        .is_some_and(|ra| ra.coloring.contains_key(var))
                });
            if !covered {
                gaps.push((var, s));
            }
        }
        gaps.sort_by_key(|(var, _)| var.id());

        if gaps.is_empty() {
            return;
        }

        let class_coverage: Vec<(RegKind, usize)> = [RegKind::Int, RegKind::Ref, RegKind::Float]
            .iter()
            .filter_map(|k| regallocs.get(k).map(|ra| (*k, ra.coloring.len())))
            .collect();
        println!(
            "cargo:warning=[MAJIT_COVERAGE_AUDIT] graph={:?} gaps={} regalloc_coverage={:?}",
            ssarepr.name,
            gaps.len(),
            class_coverage,
        );
        for (v, s) in &gaps {
            let first_use = s.first_use_tag.unwrap_or("<no use>");
            println!(
                "cargo:warning=  - {v:?} def={} uses={} live={} first_use={}",
                s.has_def, s.use_count, s.live_count, first_use,
            );
        }
    }

    /// RPython assembler.py: emit_const for integer constants.
    /// Adds to constant pool and returns the index byte.
    fn emit_const(
        &mut self,
        value: &ConstValue,
        kind: char,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> u8 {
        match kind {
            'i' => self.emit_const_i_from_const(value, state, callcontrol),
            'r' => self.emit_const_r(value, state, callcontrol),
            'f' => self.emit_const_f(value, state),
            other => std::panic::panic_any(AssemblerError::message(format!(
                "unknown constant kind {other:?} for {value:?}"
            ))),
        }
    }

    fn emit_const_i_from_const(
        &mut self,
        value: &ConstValue,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> u8 {
        let value = Self::resolve_const_i_value(value, callcontrol);
        self.emit_const_i(value, state)
    }

    /// Resolve an integer-kind [`ConstValue`] to its concrete `i64`
    /// (`assembler.py` value extraction).  `llmemory` address offsets
    /// are symbolic; resolve to the concrete byte size at code emission.
    /// Struct field offsets / sizes come from the `CallControl`'s struct
    /// layouts (it implements `OffsetLayout`); the layout-free offsets
    /// resolve even without a `callcontrol`.
    fn resolve_const_i_value(value: &ConstValue, callcontrol: Option<&CallControl>) -> i64 {
        match value {
            ConstValue::Int(n) => *n,
            ConstValue::Bool(b) => *b as i64,
            ConstValue::SpecTag(tag) => *tag as i64,
            // assembler.py::Assembler.emit_const converts raw pointers via
            // cast_ptr_to_adr/adr2int. Null needs no runtime relocation.
            ConstValue::LLPtr(ptr) if !ptr.nonzero() => {
                assert_eq!(
                    ptr._TYPE._gckind(),
                    crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw,
                    "integer pointer constants must be raw"
                );
                0
            }
            ConstValue::LLAddress(
                crate::translator::rtyper::lltypesystem::lltype::_address::Null,
            ) => 0,
            // Symbolic inheritance-id marker: emit the resolved id. When
            // only the eager `value` is present it is the concrete id;
            // `cdef_id` reserves the key for a numbering resolution.
            ConstValue::InheritanceId { value, .. } => *value,
            ConstValue::AddressOffset(offset) => {
                let resolved = match callcontrol {
                    Some(cc) => offset.byte_size(cc),
                    None => offset.byte_size(&NoStructLayout),
                };
                resolved.unwrap_or_else(|err| panic!("emit_const_i: {err}"))
            }
            other => std::panic::panic_any(AssemblerError::message(format!(
                "integer-kind constant not supported by emit_const_i: {other:?}"
            ))),
        }
    }

    /// `assembler.py:99-107` short-const for an integer-kind Constant
    /// operand: resolve to `i64`, then route through
    /// [`Self::emit_const_i_allow_short`].  Returns `(byte, argcode)` —
    /// `'c'` when the value took the inline short form, else `'i'`.
    fn emit_const_i_from_const_allow_short(
        &mut self,
        value: &ConstValue,
        allow_short: bool,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> (u8, char) {
        let value = Self::resolve_const_i_value(value, callcontrol);
        self.emit_const_i_allow_short(value, allow_short, state)
    }

    fn emit_const_i(&mut self, value: i64, state: &mut AssemblyState) -> u8 {
        // Check if already in pool. Relocatable slots keep their own
        // identity (`assembler.py emit_const` keys on the Constant
        // object, not the integer bits), so an ordinary int that
        // happens to equal a funcptr/static address does not share a
        // slot with it.
        for (i, &existing) in state.constants_i.iter().enumerate() {
            if existing == value && !state.slot_i_is_reloc(i) {
                return const_pool_slot(state.num_regs_i, i);
            }
        }
        // Add to pool: index = num_regs + pool_position
        state.constants_i.push(value);
        const_pool_slot(state.num_regs_i, state.constants_i.len() - 1)
    }

    fn emit_const_i_reloc(
        &mut self,
        value: i64,
        kind: ConstIRelocKind,
        state: &mut AssemblyState,
    ) -> u8 {
        // `assembler.py Assembler.emit_const` keys `constants_dict` on
        // `(kind, Constant(value_key))` where `value_key` is the
        // symbolic address object, so one symbol owns one slot. The
        // incoming `value` is the build-time word for that symbol; a
        // hit must reuse the stored word.
        if let Some(existing) = state.reloc_consts_i.iter().find(|d| d.kind == kind) {
            debug_assert_eq!(
                state.constants_i[existing.constants_i_index], value,
                "one symbol must bind one build address"
            );
            return const_pool_slot(state.num_regs_i, existing.constants_i_index);
        }
        let constants_i_index = state.constants_i.len();
        state.constants_i.push(value);
        state.reloc_consts_i.push(ConstIRelocDescriptor {
            constants_i_index,
            kind,
        });
        const_pool_slot(state.num_regs_i, constants_i_index)
    }

    /// Funcptr Constant of `residual_call` / `conditional_call*` /
    /// `conditional_call_value*`. `_rewrite_op_cond_call` and
    /// `handle_residual_call` pass the same `Constant`; `assembler.py
    /// emit_const` stores the symbolic object. `ConstFnAddr` is the
    /// materialized spelling of that object; this is the in-operand
    /// spelling. Both write `{ path, symbolic }` via `emit_const_i_reloc`.
    fn emit_fnaddr_const(
        &mut self,
        funcptr: &crate::model::CallTarget,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> u8 {
        let (fnaddr, path, symbolic) = match callcontrol {
            Some(cc) => {
                let binding = cc.fnaddr_binding_for_target(funcptr);
                (binding.addr, binding.path, binding.symbolic)
            }
            None => (
                crate::call::symbolic_fnaddr_for_target(funcptr),
                crate::call::symbolic_fnaddr_path_for_target(funcptr),
                true,
            ),
        };
        self.emit_const_i_reloc(fnaddr, ConstIRelocKind::FnAddr { path, symbolic }, state)
    }

    fn emit_const_r_reloc(
        &mut self,
        value: i64,
        kind: ConstIRelocKind,
        state: &mut AssemblyState,
    ) -> u8 {
        // Same reuse as `emit_const_i_reloc`: `Assembler.emit_const`
        // keys `constants_dict` on the symbolic address, so one
        // symbol owns one slot.
        if let Some(existing) = state.reloc_consts_r.iter().find(|d| d.kind == kind) {
            debug_assert_eq!(
                state.constants_r[existing.constants_r_index], value,
                "one symbol must bind one build address"
            );
            return const_pool_slot(state.num_regs_r, existing.constants_r_index);
        }
        let constants_r_index = state.constants_r.len();
        state.constants_r.push(value);
        state.reloc_consts_r.push(ConstRRelocDescriptor {
            constants_r_index,
            kind,
        });
        const_pool_slot(state.num_regs_r, constants_r_index)
    }

    /// `assembler.py` — the `allow_short` branch of `emit_const`.
    /// When `allow_short` (the surrounding opname is in [`use_c_form`])
    /// and `value` fits in signed-i8 range, emit it inline as one byte
    /// (`assembler.py` `self.code.append(chr(value & 0xFF))`) and
    /// return argcode `'c'`; otherwise fall back to the constant pool and
    /// return argcode `'i'`.  Returns `(byte, argcode)` — the caller
    /// pushes `byte` and appends `argcode`.
    fn emit_const_i_allow_short(
        &mut self,
        value: i64,
        allow_short: bool,
        state: &mut AssemblyState,
    ) -> (u8, char) {
        if allow_short && (-128..=127).contains(&value) {
            (value as i8 as u8, 'c')
        } else {
            (self.emit_const_i(value, state), 'i')
        }
    }

    fn emit_llexitcase(
        &mut self,
        value: &ConstValue,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> u8 {
        match value {
            ConstValue::Int(value) => self.emit_const_i(*value, state),
            ConstValue::HostObject(obj) => {
                let bits = exception_class_addr(obj, callcontrol, "goto_if_exception_mismatch")
                    .unwrap_or_else(|| {
                        panic!("goto_if_exception_mismatch: llexitcase {obj:?} is not an exception class")
                    });
                let name = exception_class_of(obj).expect("exception class llexitcase");
                let pytypes = callcontrol.map_or(&[][..], CallControl::exc_pytype_rows);
                let row_name =
                    lookup_exc_static_row_name(pytypes, &name, "_TYPE").unwrap_or_else(|| {
                        panic!(
                            "goto_if_exception_mismatch: no interp_exceptions pytype row for {name}"
                        )
                    });
                self.emit_const_i_reloc(bits, ConstIRelocKind::StaticAddr { name: row_name }, state)
            }
            other => {
                panic!("goto_if_exception_mismatch: unsupported llexitcase constant {other:?}")
            }
        }
    }

    fn emit_const_r(
        &mut self,
        value: &ConstValue,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> u8 {
        // A prebuilt `Ptr(STR)` constant cannot be baked as a runtime
        // address: the translator runs in a separate build-script process,
        // so the `_ptr` target is process-local garbage at runtime.  Record
        // the string's bytes + precomputed hash for runtime materialization
        // and pool a non-canonical sentinel that the jitcode-load pass
        // overwrites with the live immortal STR GcStruct address.
        if let ConstValue::LLPtr(p) = value
            && let Some((bytes, hash)) =
                crate::translator::rtyper::lltypesystem::rstr::prebuilt_str_bytes_and_hash(p)
        {
            return self.emit_str_const_r(bytes, hash, false, state);
        }
        // A unit-variant prebuilt singleton is likewise process-local
        // (`rpbc.py SingleFrozenPBCRepr`'s prebuilt instance): the
        // `HostObject` lives only in the translator, so pool a
        // non-canonical sentinel plus a descriptor for the runtime load
        // pass to overwrite with the immortal discriminant cell.
        //
        // The discriminant is a class-level constant on the interned
        // variant host (`ClassDesc.classdict` / `rclass.py`
        // `initialize_prebuilt_data` `classdesc.read_attribute`).  Recover
        // `(qualname, tag)` from the constant itself so `emit_const_r`
        // does not need a side table of instance identities.
        if let ConstValue::HostObject(obj) = value {
            if let Some((qualname, tag)) =
                crate::translator::rtyper::unit_variant_fold::unit_variant_const_from_host(obj)
            {
                return self.emit_unit_variant_const_r(qualname, tag, state);
            }
        }
        if let ConstValue::HostObject(obj) = value
            && let Some((class_name, message)) = exception_instance_spec(obj)
        {
            return self.emit_exc_instance_const_r(class_name, message, state);
        }
        if let ConstValue::HostObject(obj) = value {
            let type_name = self
                .type_static_const_by_addr(obj.identity_id() as i64)
                .map(str::to_string);
            if let Some(name) = type_name {
                return self.emit_type_static_const_r(name, state);
            }
        }
        let bits = match value {
            // assembler.py::Assembler.emit_const casts ref constants to
            // GCREF and gives every typed nullptr the same None pool key.
            ConstValue::LLPtr(ptr) if !ptr.nonzero() => {
                assert_eq!(
                    ptr._TYPE._gckind(),
                    crate::translator::rtyper::lltypesystem::lltype::GcKind::Gc,
                    "reference pointer constants must be GC-managed"
                );
                0
            }
            ConstValue::HostObject(obj) => host_object_ref_bits(obj, callcontrol),
            ConstValue::LLAddress(
                crate::translator::rtyper::lltypesystem::lltype::_address::Null,
            ) => 0,
            other => panic!("raise/r constant pool does not support {other:?}"),
        };
        self.emit_type_static_or_bits(bits, state)
    }

    /// Record a prebuilt-string constant for runtime materialization and
    /// pool its sentinel, returning the `r`-bank register index.  Identical
    /// literals (by bytes) share one descriptor and one sentinel, so
    /// [`Self::emit_const_r_bits`]' dedup-by-bits collapses them to a single
    /// pool entry — the analog of upstream `emit_const`'s `value_key` dedup
    /// (`assembler.py:116`).
    fn emit_str_const_r(
        &mut self,
        bytes: Vec<u8>,
        precomputed_hash: i64,
        as_unicode_object: bool,
        state: &mut AssemblyState,
    ) -> u8 {
        if let Some(ordinal) = state
            .str_consts
            .iter()
            .position(|d| d.bytes == bytes && d.as_unicode_object == as_unicode_object)
        {
            return self.emit_const_r_bits(str_const_sentinel(ordinal), state);
        }
        let ordinal = state.str_consts.len();
        let constants_r_index = state.constants_r.len();
        let reg = self.emit_const_r_bits(str_const_sentinel(ordinal), state);
        debug_assert_eq!(
            state.constants_r.len(),
            constants_r_index + 1,
            "a fresh str-const sentinel must push a new constants_r slot"
        );
        state.str_consts.push(StrConstDescriptor {
            constants_r_index,
            bytes,
            precomputed_hash,
            as_unicode_object,
        });
        reg
    }

    /// Record a unit-variant singleton constant for runtime
    /// materialization and pool its sentinel —
    /// [`Self::emit_str_const_r`]'s shape for prebuilt variant
    /// instances.  Identical variants (by qualname) share one
    /// descriptor and one sentinel.
    fn emit_unit_variant_const_r(
        &mut self,
        qualname: String,
        tag: i64,
        state: &mut AssemblyState,
    ) -> u8 {
        if let Some(ordinal) = state
            .unit_variant_consts
            .iter()
            .position(|d| d.qualname == qualname)
        {
            return self.emit_const_r_bits(unit_variant_const_sentinel(ordinal), state);
        }
        let ordinal = state.unit_variant_consts.len();
        let constants_r_index = state.constants_r.len();
        let reg = self.emit_const_r_bits(unit_variant_const_sentinel(ordinal), state);
        debug_assert_eq!(
            state.constants_r.len(),
            constants_r_index + 1,
            "a fresh unit-variant sentinel must push a new constants_r slot"
        );
        state
            .unit_variant_consts
            .push(super::jitcode::UnitVariantConstDescriptor {
                constants_r_index,
                qualname,
                tag,
            });
        reg
    }

    /// Record a prebuilt exception instance and pool its sentinel.
    /// Identical `(class, message)` pairs share one descriptor.
    fn emit_exc_instance_const_r(
        &mut self,
        class_name: String,
        message: Option<Vec<u8>>,
        state: &mut AssemblyState,
    ) -> u8 {
        if let Some(ordinal) = state
            .exc_instance_consts
            .iter()
            .position(|d| d.class_name == class_name && d.message == message)
        {
            return self.emit_const_r_bits(exc_instance_const_sentinel(ordinal), state);
        }
        let ordinal = state.exc_instance_consts.len();
        let constants_r_index = state.constants_r.len();
        let reg = self.emit_const_r_bits(exc_instance_const_sentinel(ordinal), state);
        debug_assert_eq!(
            state.constants_r.len(),
            constants_r_index + 1,
            "a fresh exception-instance sentinel must push a new constants_r slot"
        );
        state
            .exc_instance_consts
            .push(super::jitcode::ExcInstanceConstDescriptor {
                constants_r_index,
                class_name,
                message,
            });
        reg
    }

    /// Pool a type-static sentinel when `bits` names an interned
    /// `PyType` singleton; a named reloc when it names a host-static
    /// ref; otherwise pool the raw bits.
    fn emit_type_static_or_bits(&mut self, bits: i64, state: &mut AssemblyState) -> u8 {
        if let Some(name) = self.type_static_const_by_addr(bits) {
            return self.emit_type_static_const_r(name.to_string(), state);
        }
        if let Some(name) = self.static_ref_by_addr(bits) {
            return self.emit_const_r_reloc(
                bits,
                ConstIRelocKind::StaticAddr {
                    name: name.to_string(),
                },
                state,
            );
        }
        self.emit_const_r_bits(bits, state)
    }

    /// Record a host `PyType` singleton for runtime materialization and
    /// pool its sentinel — [`Self::emit_str_const_r`]'s shape for type
    /// statics.  Identical names share one descriptor and one sentinel.
    fn emit_type_static_const_r(&mut self, name: String, state: &mut AssemblyState) -> u8 {
        if let Some(ordinal) = state.type_static_consts.iter().position(|d| d.name == name) {
            return self.emit_const_r_bits(type_static_const_sentinel(ordinal), state);
        }
        let ordinal = state.type_static_consts.len();
        let constants_r_index = state.constants_r.len();
        let reg = self.emit_const_r_bits(type_static_const_sentinel(ordinal), state);
        debug_assert_eq!(
            state.constants_r.len(),
            constants_r_index + 1,
            "a fresh type-static sentinel must push a new constants_r slot"
        );
        state
            .type_static_consts
            .push(super::jitcode::TypeStaticConstDescriptor {
                constants_r_index,
                name,
            });
        reg
    }

    fn emit_const_r_bits(&mut self, bits: i64, state: &mut AssemblyState) -> u8 {
        for (i, &existing) in state.constants_r.iter().enumerate() {
            if existing == bits && !state.slot_r_is_reloc(i) {
                return const_pool_slot(state.num_regs_r, i);
            }
        }
        state.constants_r.push(bits);
        const_pool_slot(state.num_regs_r, state.constants_r.len() - 1)
    }

    fn emit_const_f(&mut self, value: &ConstValue, state: &mut AssemblyState) -> u8 {
        let bits = match value {
            ConstValue::Float(bits) => *bits as i64,
            other => panic!("float constant pool does not support {other:?}"),
        };
        if let Some(index) = state
            .constants_f
            .iter()
            .position(|&existing| existing == bits)
        {
            return const_pool_slot(state.num_regs_f, index);
        }
        state.constants_f.push(bits);
        const_pool_slot(state.num_regs_f, state.constants_f.len() - 1)
    }
}

/// RPython `jtransform.py rewrite_call` — the call-family opcode kind suffix:
/// ```text
/// if lst_f or reskind == 'f': kinds = 'irf'
/// elif lst_i or force_ir: kinds = 'ir'
/// else: kinds = 'r'
/// ```
/// One of exactly three signatures, chosen by the widest bin in play rather
/// than by which bins are occupied: `"irf"` when a float appears among the
/// arguments or as the result, else `"ir"` when an int does, else `"r"`.  So
/// `(args_i=[a], args_r=[], args_f=[])` is `"ir"` and an all-empty call is
/// `"r"` — the `r` bin is named even when it is empty, because the sublists
/// the signature announces are positional and the reader counts them.
///
/// The result kind is a signature input, not just the suffix after it: a
/// float result forces `irf` even with no float argument
/// (`test_jtransform.py` `if RESTYPE == lltype.Float: with_f = True`).
/// Without that a `&self -> f64` shape (empty `args_i`/`args_f`, single
/// `args_r`, `result_kind='f'`) would map to a pyre-only `_r_f` handler with
/// no RPython `bhimpl_*_r_f` counterpart (`blackhole.py` only has
/// `_r_{i,r,v}` / `_ir_*` / `_irf_*`).
///
/// The emitter and the text formatter (`codewriter::format`) both spell the
/// opname from this one rule, so a printed trace names the opcode the
/// bytecode carries.
pub(crate) fn rewrite_call_kinds<T, U, V>(
    args_i: &[T],
    _args_r: &[U],
    args_f: &[V],
    result_kind: char,
) -> &'static str {
    if !args_f.is_empty() || result_kind == 'f' {
        "irf"
    } else if !args_i.is_empty() {
        "ir"
    } else {
        "r"
    }
}

/// Operand byte of a constant-pool entry: `count_regs[kind] +
/// pool_index` (`assembler.py:132`).  Registers and constants of one
/// kind share a single byte-wide address space, so the slot is bounded
/// by `assembler.py assert 0 <= val < 256, "too many constants"`.
/// Without the bound a wider pool wraps modulo 256 and silently aliases
/// a live register or an earlier constant.
fn const_pool_slot(num_regs: usize, pool_index: usize) -> u8 {
    let val = num_regs + pool_index;
    assert!(val < 256, "too many constants");
    val as u8
}

/// `assembler.py check_result()` — "Limitation of the number of
/// registers, from the single-byte encoding".  Each pair is
/// `(count_regs[kind], len(constants_kind))`.
fn check_result(int: (usize, usize), reference: (usize, usize), float: (usize, usize)) {
    for (kind, (num_regs, num_consts)) in [("int", int), ("ref", reference), ("float", float)] {
        assert!(
            num_regs + num_consts <= 256,
            "too many {kind} registers and constants \
             ({num_regs} + {num_consts} > 256)"
        );
    }
}

/// `STR_CONST_SENTINEL_BASE | ordinal` — see [`STR_CONST_SENTINEL_BASE`].
fn str_const_sentinel(ordinal: usize) -> i64 {
    debug_assert!(
        (ordinal as u64) < (1u64 << 48),
        "too many prebuilt-string constants in one jitcode"
    );
    STR_CONST_SENTINEL_BASE | ordinal as i64
}

/// `UNIT_VARIANT_CONST_SENTINEL_BASE | ordinal` — see
/// [`UNIT_VARIANT_CONST_SENTINEL_BASE`].
fn unit_variant_const_sentinel(ordinal: usize) -> i64 {
    debug_assert!(
        (ordinal as u64) < (1u64 << 48),
        "too many unit-variant constants in one jitcode"
    );
    UNIT_VARIANT_CONST_SENTINEL_BASE | ordinal as i64
}

fn exc_instance_const_sentinel(ordinal: usize) -> i64 {
    debug_assert!(
        (ordinal as u64) < (1u64 << 48),
        "too many prebuilt exception instances in one jitcode"
    );
    EXC_INSTANCE_CONST_SENTINEL_BASE | ordinal as i64
}

/// Per-assembly state (RPython: Assembler.setup() fields).
struct AssemblyState {
    code: Vec<u8>,
    constants_i: Vec<i64>,
    constants_r: Vec<i64>,
    constants_f: Vec<i64>,
    /// Prebuilt-string constants recorded while assembling, committed to
    /// [`JitCodeBody::str_consts`].  Parallel to `constants_r`: each entry
    /// owns the `constants_r` slot named by its `constants_r_index`.
    str_consts: Vec<StrConstDescriptor>,
    /// Unit-variant singleton constants recorded while assembling,
    /// committed to [`JitCodeBody::unit_variant_consts`]; same ownership
    /// contract as `str_consts`.
    unit_variant_consts: Vec<super::jitcode::UnitVariantConstDescriptor>,
    exc_instance_consts: Vec<super::jitcode::ExcInstanceConstDescriptor>,
    /// Host `PyType` singleton constants recorded while assembling,
    /// committed to [`JitCodeBody::type_static_consts`].
    type_static_consts: Vec<super::jitcode::TypeStaticConstDescriptor>,
    /// Relocatable `constants_i` slots, committed to
    /// [`JitCodeBody::reloc_consts_i`].
    reloc_consts_i: Vec<ConstIRelocDescriptor>,
    /// Relocatable `constants_r` slots, committed to
    /// [`JitCodeBody::reloc_consts_r`].
    reloc_consts_r: Vec<ConstRRelocDescriptor>,
    num_regs_i: usize,
    num_regs_r: usize,
    num_regs_f: usize,
    label_positions: HashMap<Label, usize>,
    tlabel_fixups: Vec<(Label, usize)>,
    startpoints: indexmap::IndexSet<usize>,
    /// Where the `jit_merge_point` opcode byte went, for the one body that
    /// carries the marker. Recorded at the reserved opcode position rather
    /// than recovered later: an operand byte can equal the opcode byte, so
    /// only the encoder knows which position is an instruction start.
    jit_merge_point_offset: Option<usize>,
    /// RPython assembler.py:176: positions in bytecode where TLabel operands
    /// are written. Used by JitCode.follow_jump() for verification.
    alllabels: indexmap::IndexSet<usize>,
    /// RPython assembler.py:217-219: map from bytecode offset (after `->`)
    /// to result kind character. Recorded when encoding result registers.
    resulttypes: indexmap::IndexMap<usize, char>,
}

impl AssemblyState {
    fn slot_i_is_reloc(&self, index: usize) -> bool {
        self.reloc_consts_i
            .iter()
            .any(|d| d.constants_i_index == index)
    }

    fn slot_r_is_reloc(&self, index: usize) -> bool {
        self.reloc_consts_r
            .iter()
            .any(|d| d.constants_r_index == index)
    }
}

/// RPython: getkind(v.concretetype)[0] → 'i', 'r', 'f', 'v'.
fn value_type_to_kind(ty: &crate::model::ValueType) -> char {
    use crate::model::ValueType;
    match ty {
        // RPython `getkind(Bool/Unsigned)` returns `'int'` (`lloperation.
        // py:108`); BoolRepr's lowleveltype is `Bool` and IntegerRepr
        // shares register class with Signed/Unsigned — all `'i'` for
        // the codewriter.
        ValueType::Int | ValueType::Unsigned | ValueType::Bool => 'i',
        ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder => 'r',
        ValueType::Float => 'f',
        ValueType::Void | ValueType::State | ValueType::Unknown => 'v',
        ValueType::Int128 | ValueType::UInt128 => {
            panic!("getkind: 128-bit integer type is too large (history.py:62)")
        }
        ValueType::SingleFloat => panic!(
            "getkind: SingleFloat is not supported (history.py:61) — majit is \
             the `supports_singlefloats = False` CPU (backend/model.py:20), so \
             the codewriter policy refuses a graph carrying one; reaching here \
             means a carrier escaped `collect_declared_value_types`"
        ),
    }
}

/// `i`/`r`/`f`/`v` → `int`/`ref`/`float`/`void` for opname formation.
/// Mirrors RPython `bhimpl_<kind>_*` naming where the prefix is the full
/// kind word — `bhimpl_int_guard_value`, `bhimpl_ref_isvirtual`, etc. —
/// not the single-character argcode used inside the `/argcodes` suffix.
fn kind_char_to_name(c: char) -> &'static str {
    match c {
        'i' => "int",
        'r' => "ref",
        'f' => "float",
        _ => panic!(
            "kind_char_to_name: invalid kind char {c:?} — only 'i'/'r'/'f' \
             are valid for typed opname prefixes (void is not an operand kind)"
        ),
    }
}

/// Alignment of a scalar of `field_size` bytes, for the textual struct-layout
/// walkers below.
///
/// The cap is the target's *maximum scalar alignment*, not its word:
/// `u64`/`f64` align to 8 on wasm32 even though a pointer there is 4 bytes,
/// so this must NOT be rewritten to `layout::target_word_size()` — that would
/// under-align every 8-byte field on a 32-bit target and shift the offsets
/// this walker derives.
fn scalar_align(field_size: usize) -> usize {
    field_size.min(8)
}

fn value_type_to_itemsize(ty: &crate::model::ValueType) -> usize {
    use crate::model::ValueType;
    match ty {
        // A pointer strides by the target word, not the build host's
        // (`symbolic.py WORD = sizeof(lltype.Signed)`).  The sibling
        // fallback in `arraydescrof` already reads it this way.
        ValueType::Ref(_) => crate::layout::target_word_size(),
        // `lltype.Bool` is one byte (`descr.py` takes the field's real
        // size from `symbolic.get_field_token`), and so is a Rust `bool` —
        // the spelling-keyed twin already answers 1 for it
        // (`type_flag_from_str` `"u8" | "bool"`).
        ValueType::Bool => 1,
        // `Int`/`Unsigned` deliberately keep 8.  They are `lltype.Signed` /
        // `lltype.Unsigned` on paper, but the MIR front-end folds narrower
        // *and* 64-bit Rust integers into them (see `model.rs ValueType`),
        // so the width is genuinely unknown here.  Over-reading inside the
        // struct is recoverable; narrowing an `i64`/`u64` field to the
        // wasm32 word would truncate it.  Resolving this needs the
        // front-end to carry the source width, not a guess at this seam.
        ValueType::Int | ValueType::Unsigned => 8,
        ValueType::Float => 8,
        _ => 8,
    }
}

/// `descr.py get_type_flag` over a `ValueType`, for the no-layout
/// fallback.
///
/// Cannot go through `ArrayFlag::from_field_type`: that reads the descriptor
/// IR type, which collapses `Bool` / `Unsigned` onto `Type::Int` to track the
/// register class (`value_type_to_ir_type_for_descr`), and would then report
/// every unsigned field as `Signed`.  The signedness has to come from the
/// `ValueType` itself, the way the spelling-keyed `type_flag_from_str` reads
/// it off `u*` vs `i*`.
fn value_type_to_field_flag(ty: &crate::model::ValueType) -> majit_ir::descr::ArrayFlag {
    use crate::model::ValueType;
    use majit_ir::descr::ArrayFlag;
    match ty {
        ValueType::Ref(_) => ArrayFlag::Pointer,
        ValueType::Float => ArrayFlag::Float,
        ValueType::Bool | ValueType::Unsigned | ValueType::UInt128 => ArrayFlag::Unsigned,
        ValueType::Void => ArrayFlag::Void,
        _ => ArrayFlag::Signed,
    }
}

fn value_type_to_ir_type_for_descr(ty: &crate::model::ValueType) -> majit_ir::value::Type {
    match ty {
        // `getkind(BOOL_TYPE)` returns `'int'` (`lloperation.py _freeze_`);
        // `getkind(Unsigned) == 'int'` per `lltype.py` — descriptor IR
        // type tracks the register class so Bool/Unsigned alias to Int
        // rather than falling into the wildcard Ref branch.
        crate::model::ValueType::Int
        | crate::model::ValueType::Bool
        | crate::model::ValueType::Unsigned => majit_ir::value::Type::Int,
        crate::model::ValueType::Float => majit_ir::value::Type::Float,
        crate::model::ValueType::Void => majit_ir::value::Type::Void,
        _ => majit_ir::value::Type::Ref,
    }
}

/// `descr.py get_type_flag`, over a Rust type spelling.
///
/// The authoritative copy is [`super::call::get_type_flag`]; this one serves
/// the layout walkers below.  Word-sized spellings must resolve through
/// `layout::target_word_size()` in both: this crate is the build-time host
/// prepass, so `size_of::<usize>()` here would bake the *host's* 8 into a
/// wasm32 descr.
///
/// The two deliberately still differ on `f32`.  `get_type_flag` follows
/// `descr.py` and lands SingleFloat int-banked
/// (`ArrayFlag::Unsigned` / `Type::Int`), then restores the `'f'` width
/// marker via `concrete_type` on the ArrayDescr it builds
/// (`call.rs` `elem_ref == Some("f32")`).  `BhDescr::Array` carries no
/// `concrete_type` field to restore it with, so adopting the int-banked
/// answer here would lose the f32-ness outright.  Merging the two is
/// blocked on `BhDescr::Array` gaining that marker.
fn type_flag_from_str(
    type_str: &str,
) -> (majit_ir::descr::ArrayFlag, majit_ir::value::Type, usize) {
    use majit_ir::descr::ArrayFlag;
    let word = crate::layout::target_word_size();
    match type_str {
        // descr.py raw Ptr parity; see call.rs::get_type_flag.
        "*const u8" | "*const CellFamily" | "*mut CellFamily" => {
            (ArrayFlag::Unsigned, majit_ir::value::Type::Int, word)
        }
        // descr.py get_type_flag: raw Ptr → FLAG_UNSIGNED. Same recognizer
        // as `Bookkeeper::project_rust_vec` / `rrustvec.rs rust_vec_lltype`.
        s if majit_ir::rvec::rust_vec_item_kind_for_spelling(s, word).is_some() => {
            (ArrayFlag::Unsigned, majit_ir::value::Type::Int, word)
        }
        s if crate::fat_ptr_layout::spelling_is_dyn_fat_ptr(s) => {
            (ArrayFlag::Pointer, majit_ir::value::Type::Ref, 2 * word)
        }
        s if s.starts_with('&')
            || s.starts_with("Box<")
            || s.starts_with("Arc<")
            || s.starts_with("Rc<")
            || s.starts_with("Option<")
            || s == "String" =>
        {
            (ArrayFlag::Pointer, majit_ir::value::Type::Ref, word)
        }
        // `{cap, ptr, len}` inline. Same arm as `get_type_flag`.
        s if s.starts_with("Vec<") => (ArrayFlag::Struct, majit_ir::value::Type::Ref, 3 * word),
        "f64" => (ArrayFlag::Float, majit_ir::value::Type::Float, 8),
        "f32" => (ArrayFlag::Float, majit_ir::value::Type::Float, 4),
        "i64" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 8),
        "isize" => (ArrayFlag::Signed, majit_ir::value::Type::Int, word),
        "i32" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 4),
        "i16" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 2),
        "i8" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 1),
        "u64" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 8),
        "usize" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, word),
        // `llmemory.GCREF` (`VirtualizableInstanceRepr._setup_repr_llfields`).
        "GCREF" => (ArrayFlag::Pointer, majit_ir::value::Type::Ref, word),
        "u32" | "char" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 4),
        "u16" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 2),
        "u8" | "bool" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 1),
        "()" => (ArrayFlag::Void, majit_ir::value::Type::Void, 0),
        // Unknown type — treat as GC pointer (conservative), matching
        // `get_type_flag`'s wildcard.
        _ => (ArrayFlag::Pointer, majit_ir::value::Type::Ref, word),
    }
}

fn fallback_field_layout(
    ty: &crate::model::ValueType,
) -> (
    usize,
    majit_ir::value::Type,
    majit_ir::descr::ArrayFlag,
    bool,
) {
    let field_type = value_type_to_ir_type_for_descr(ty);
    let field_size = value_type_to_itemsize(ty);
    let field_flag = value_type_to_field_flag(ty);
    let is_signed = field_flag == majit_ir::descr::ArrayFlag::Signed;
    (field_size, field_type, field_flag, is_signed)
}

fn bh_field_name(owner: &str, field_name: &str) -> String {
    if owner.is_empty() || field_name.contains('.') {
        field_name.to_string()
    } else {
        format!("{owner}.{field_name}")
    }
}

enum BhFieldLookup {
    Parent(crate::jitcode::BhFieldSpec),
    Layout(crate::codewriter::call::StructFieldLayout),
    Missing,
}

fn bh_field_lookup(cc: &CallControl, field: &crate::model::FieldDescriptor) -> BhFieldLookup {
    let Some(owner) = field.owner_root.as_deref() else {
        return BhFieldLookup::Missing;
    };
    if let Some(parent_spec) = bh_size_spec_from_callcontrol(cc, owner) {
        let full_name = bh_field_name(owner, &field.name);
        // `all_fielddescrs` is flattened depth-first, so a nested
        // `Outer.inner.x` precedes the `Outer.x` the caller named. A
        // single `find` over the disjunction would let that nested row
        // win on the dotted-suffix arm alone; run the exact spellings to
        // exhaustion first and keep the suffix match as the fallback.
        let dotted_suffix = format!(".{}", field.name);
        let matched = parent_spec
            .all_fielddescrs
            .iter()
            .find(|spec| spec.name == full_name || spec.field_key() == field.name)
            .or_else(|| {
                parent_spec
                    .all_fielddescrs
                    .iter()
                    .find(|spec| spec.name.ends_with(&dotted_suffix))
            });
        if let Some(spec) = matched {
            return BhFieldLookup::Parent(spec.clone());
        }
    }
    cc.struct_layout_for(owner)
        .and_then(|layout| {
            layout
                .fields
                .iter()
                .find(|row| row.name == field.name)
                .cloned()
        })
        .map(BhFieldLookup::Layout)
        .unwrap_or(BhFieldLookup::Missing)
}

/// Offset and size of a by-value substructure that has no flattened field
/// descriptor of its own.
///
/// This deliberately shares `fielddescrof`'s lookup: an exact parent
/// `all_fielddescrs` match wins, and only a layout-row fallback carrying the
/// struct flag denotes the `getsubstruct` shape. Size is the layout row's
/// byte size, so `rewrite_op_getfield` can tell a transparent one-word
/// newtype (an ordinary load) from the first nested parent struct (identity).
pub(crate) fn inline_substruct_field_layout(
    cc: &crate::codewriter::call::CallControl,
    field: &crate::model::FieldDescriptor,
) -> Option<(usize, usize)> {
    match bh_field_lookup(cc, field) {
        BhFieldLookup::Layout(row) if row.flag == majit_ir::descr::ArrayFlag::Struct => {
            Some((row.offset, row.size))
        }
        BhFieldLookup::Parent(_) | BhFieldLookup::Layout(_) | BhFieldLookup::Missing => None,
    }
}

/// Return the byte offset of a by-value substructure that has no flattened
/// field descriptor of its own.
pub(crate) fn inline_substruct_field_offset(
    cc: &crate::codewriter::call::CallControl,
    field: &crate::model::FieldDescriptor,
) -> Option<usize> {
    inline_substruct_field_layout(cc, field).map(|(offset, _size)| offset)
}

#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
)]
fn bh_field_spec_from_parts(
    index: u32,
    owner: &str,
    field_name: &str,
    offset: usize,
    field_size: usize,
    field_type: majit_ir::value::Type,
    field_flag: majit_ir::descr::ArrayFlag,
    is_immutable: bool,
    is_quasi_immutable: bool,
    index_in_parent: usize,
) -> crate::jitcode::BhFieldSpec {
    crate::jitcode::BhFieldSpec {
        index,
        field_key: field_name.to_string(),
        name: bh_field_name(owner, field_name),
        offset,
        field_size,
        field_type,
        field_flag,
        is_field_signed: field_flag == majit_ir::descr::ArrayFlag::Signed,
        is_immutable,
        is_quasi_immutable,
        index_in_parent,
        // Built from an owner name and a field name, with no layout in reach:
        // there is nothing here to declare with, so the rebuilding side falls
        // back to the name.
        is_class_word: None,
    }
}

/// Empty layout source used when no `CallControl` is threaded into
/// constant emission. The layout-free symbolic offsets (primitive
/// `ItemOffset`, `CompositeOffset`, and the standard array tokens) still
/// resolve; struct field offsets / sizes report as missing.
struct NoStructLayout;
impl crate::translator::rtyper::lltypesystem::llmemory::OffsetLayout for NoStructLayout {
    fn field_offset(&self, _struct_name: &str, _fldname: &str) -> Option<i64> {
        None
    }
    fn struct_size(&self, _struct_name: &str) -> Option<i64> {
        None
    }
}

/// Whether `owner` names one of the three enum variants `front/mir` gives an
/// explicit `[tag@0 | payload@8]` shell to.
///
/// `front/mir` admits aliases from the bare `Result::Ok` up to the fully
/// qualified standard-library spelling. The spelling alone is insufficient:
/// a crate root named `option` can define `option::Option::Some` too. Resolve
/// the candidate and the fully-qualified `core`/`std` anchors through the
/// StructId registry and require their defining identities to agree.
///
/// A lone leaf (`Ok`) is not enough: the shortest form `front/mir` produces
/// names the enum as well as the variant.
fn is_explicit_shell_variant_owner(owner: &str) -> bool {
    const SHELLED: [&str; 6] = [
        "core::result::Result::Ok",
        "std::result::Result::Ok",
        "core::result::Result::Err",
        "std::result::Result::Err",
        "core::option::Option::Some",
        "std::option::Option::Some",
    ];
    let Some(owner_id) = majit_ir::descr::struct_template_id_for_name(owner) else {
        return false;
    };
    SHELLED
        .into_iter()
        .any(|anchor| majit_ir::descr::struct_template_id_for_name(anchor) == Some(owner_id))
}

/// `BhSizeSpec.type_id` for one owner spelling.
///
/// `descr.py` `get_size_descr` keys on the lltype STRUCT. A registered
/// `StructId` is that object for a GC struct. Otherwise hash the
/// crate-stripped definition path with [`definition_path_type_id`], the
/// same function `majit_metainterp::__majit_struct_type_id` uses after
/// `type_name::<T>()`.
fn bh_size_type_id(owner: &str, layout_owner: &str, is_gc_managed: bool) -> u64 {
    if is_gc_managed {
        majit_ir::descr::struct_id_for_name(owner)
            .map(majit_ir::descr::StructId::as_u64)
            .unwrap_or_else(|| definition_path_type_id(layout_owner, is_gc_managed))
    } else {
        definition_path_type_id(layout_owner, is_gc_managed)
    }
}

/// Hash a crate-stripped definition path for a low-level struct identity.
///
/// Charon's module origin drops the crate boundary (`front/mir.rs`
/// `strip_crate_prefix`). `path_hash_for_gc_kind` is the hasher; raw
/// layouts append the `"raw"` discriminator so `GcStruct(T)` and
/// `Struct(T)` stay distinct (`descr.py` `get_size_descr`).
pub fn definition_path_type_id(definition_path: &str, is_gc_managed: bool) -> u64 {
    majit_ir::descr::path_hash_for_gc_kind(definition_path, is_gc_managed)
}

pub(crate) fn bh_size_spec_from_callcontrol(
    cc: &CallControl,
    owner: &str,
) -> Option<crate::jitcode::BhSizeSpec> {
    if owner.is_empty() {
        return None;
    }
    let canonical_owner = majit_ir::descr::canonical_struct_name(owner);
    let owner = canonical_owner.as_str();
    // RPython keys SizeDescrs on the concrete low-level STRUCT object. A Rust
    // generic TypeDecl is only a template: `Option<usize>::Some` and
    // `Option<BinOpKind>::Some` can have different payload banks and therefore
    // cannot share a field-descr cache entry. Prefer Charon's concrete
    // instantiation layout and fall back to the nominal template only for the
    // layout lookup; descriptor identity remains the full concrete owner.
    let template_owner = majit_ir::descr::strip_generic_args(owner);
    let template_owner = template_owner.as_ref();
    let layout_owner = if cc.struct_layout_for(owner).is_some() {
        owner
    } else {
        template_owner
    };
    let mut size = cc
        .struct_layout_for(layout_owner)
        .map(|layout| layout.size)
        .or_else(|| heuristic_struct_size_for_bh(cc, layout_owner))?;
    // `rclass.py` lays a subclass's inherited fields out before its own
    // fields.  Annotation metadata deliberately keeps enum-variant attrs
    // variant-local, so reconstruct that inherited physical slot here for
    // the explicit Result shell.  PtrInfo indexes virtual fields by
    // `index_in_parent`; without the base tag in this list both the tag and
    // payload are slot 0 and the payload replaces the tag despite living at
    // byte offset 8.
    // `Option::Some` is on this list for the same reason its two `Result`
    // siblings are: `front/mir.rs` gives it the same explicit shell, so the
    // blackhole must reconstruct the same inherited tag slot or the payload
    // takes slot 0 here while living at byte 8 there.  `None` carries no
    // payload row, so it needs no reconstruction.
    let result_variant = is_explicit_shell_variant_owner(template_owner);
    // The bare Charon variant can contain only the inherited enum tag while
    // the annotator keeps the payload row on the concrete instantiation
    // (`Result<PyObjectRef, PyError>::Ok`).  RPython has one concrete low-level
    // STRUCT at this point, whose embedded base and variant fields are flattened
    // together by `heaptracker.all_fielddescrs`.  Prefer the instantiated rows
    // for Result variants, falling back to the template only when no such rows
    // exist, so the payload participates in that same flattened slot list.
    let mut all_fielddescrs = if result_variant {
        let instantiated = bh_result_variant_field_specs(cc, owner);
        if instantiated.is_empty() {
            bh_result_variant_field_specs(cc, template_owner)
        } else {
            instantiated
        }
    } else {
        let instantiated = bh_all_field_specs_for_struct(cc, owner);
        if instantiated.is_empty() {
            bh_all_field_specs_for_struct(cc, template_owner)
        } else {
            instantiated
        }
    };
    if result_variant {
        size = size.max(16);
    }
    if result_variant
        && !all_fielddescrs
            .iter()
            .any(|field| field.field_key() == "__discriminant")
    {
        for (index, field) in all_fielddescrs.iter_mut().enumerate() {
            field.index = (index + 1) as u32;
            field.index_in_parent = index + 1;
        }
        all_fielddescrs.insert(
            0,
            bh_field_spec_from_parts(
                0,
                template_owner,
                "__discriminant",
                0,
                8,
                majit_ir::value::Type::Int,
                majit_ir::descr::ArrayFlag::Signed,
                false,
                false,
                0,
            ),
        );
    }
    if result_variant {
        // `heaptracker.get_fielddescr_index_in` counts through the embedded
        // base before the variant's own fields.  Source registries assign
        // indices within each owner, so normalize the combined physical list
        // after adding the inherited tag.  Offset order is the low-level
        // STRUCT order for this explicit shell (tag at 0, payload at 8).
        all_fielddescrs.sort_by_key(|field| {
            (
                field.offset,
                usize::from(field.field_key() != "__discriminant"),
            )
        });
        for (index, field) in all_fielddescrs.iter_mut().enumerate() {
            field.index = index as u32;
            field.index_in_parent = index;
        }
    }
    // `rclass.py InstanceRepr._setup_repr` seats a sum-type variant subclass's
    // own fields after the base's, and pyre registers that base carrying the
    // enum's synthetic `__discriminant` row alone (`front/mir.rs`).
    // `Rvalue::Discriminant`
    // reads the tag through the base identity while the allocation is the
    // variant, and `PtrInfo` indexes a virtual's fields by `index_in_parent`,
    // so a variant list without the inherited row gives the tag and the first
    // payload slot 0 alike: the payload store replaces the tag, and every
    // later read of either resolves to a slot holding the other.  The explicit
    // shell reconstructs that row above; a native enum's variant inherits the
    // same row, at the tag's own recorded offset and width rather than the
    // shell's word, and keeps it first the way the walk that counts through
    // the embedded base does.
    if !result_variant
        && !all_fielddescrs
            .iter()
            .any(|field| field.field_key() == "__discriminant")
        && let Some(tag) = inherited_enum_tag_spec(cc, template_owner)
    {
        for (index, field) in all_fielddescrs.iter_mut().enumerate() {
            field.index = (index + 1) as u32;
            field.index_in_parent = index + 1;
        }
        all_fielddescrs.insert(0, tag);
    }
    let violating_fields = all_fielddescrs
        .iter()
        .filter(|field| field.offset + field.field_size > size)
        .collect::<Vec<_>>();
    if !violating_fields.is_empty() {
        static WARNED_LAYOUT_OWNERS: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
        let mut warned = WARNED_LAYOUT_OWNERS.lock();
        if warned.insert(layout_owner.to_string()) {
            let count = warned.len();
            let rows = violating_fields
                .iter()
                .map(|field| {
                    format!(
                        "{:?} (offset {}, field_size {})",
                        field.name, field.offset, field.field_size
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!(
                "[majit-translate] struct layout warning #{count}: owner {layout_owner:?} has \
                 parent size {size}; out-of-bounds field rows: {rows}"
            );
        }
    }
    let (is_gc_managed, headerless) = cc
        .struct_storage_for(owner)
        .or_else(|| cc.struct_storage_for(template_owner))
        .unwrap_or((true, false));
    Some(crate::jitcode::BhSizeSpec {
        size,
        is_gc_managed,
        headerless,
        // `descr.py get_size_descr` keys `_cache_size[STRUCT]` on
        // the lltype STRUCT object identity.  Pyre's analogue is
        // `path_hash_for_gc_kind(owner, is_gc_managed)` per
        // `majit_ir::descr::path_hash_for_gc_kind` doc
        // (in `majit-ir/src/descr.rs`): the analyzer side hashes
        // `field.owner_root`, the runtime macro hashes
        // `concat!(module_path!(), "::", stringify!(Struct))`.  The
        // analyzer hashes `owner` with the same raw-struct discriminator so
        // analyzer-side `BhSizeSpec` and runtime-side `__majit_type_id`
        // produce the same `LLType::Struct(u64)` cache key in
        // `gc_cache._cache_size`.
        // MUST NOT truncate to u32 — `path_hash` has 64-bit range and
        // `as u32` collisions approach certainty around 2^16 distinct
        // structs (birthday paradox), whereas PyPy's `id(STRUCT)` never
        // aliases.  The rare hash-to-zero case (1 in 2^64) is handled
        // by `simple_descr_group_from_bh_size`'s no-identity branch.
        // `layout_owner` is the template spelling used to look up the one
        // physical field layout.  The low-level STRUCT identity is still the
        // concrete monomorphization (`Option<i64>::Some`, not
        // `Option::Some`), exactly as RPython's `GcStruct` object remains
        // distinct even when two reprs share a shape.  Field descriptors
        // already preserve this identity in `fielddescrof`; allocations must
        // use the same key or a freshly virtualized object and its first
        // setfield acquire different parents.
        type_id: bh_size_type_id(owner, layout_owner, is_gc_managed),
        vtable: 0,
        owner: String::new(),
        all_fielddescrs,
    })
}

/// The enum base's `__discriminant` row that `variant_owner` inherits, or
/// `None` when `variant_owner` does not name a variant of a tag-carrying enum.
///
/// The evidence is the `__discriminant`-only base registration `front/mir.rs`
/// writes for every payload enum whose tag width Charon spelled -- a plain
/// struct whose path merely contains `::` registers its own rows under that
/// key and fails the shape test, and an enum whose tag width was recorded but
/// is unspellable registers no base at all, which is the state that wants a
/// declining read rather than a reconstructed slot.
///
/// The row is named under the *variant*, the way the explicit shell's own tag
/// row is: it is the variant's slot 0, and what identifies it as the base's
/// field is the inheritance edge the read side resolves. Charon can spell the
/// variant with a crate prefix or only the enum leaf while the discriminant
/// read uses the crate-stripped canonical base. Rebuild the variant owner from
/// that leaf's registered origin so both spell the same inheritance edge.
fn inherited_enum_tag_spec(
    cc: &CallControl,
    variant_owner: &str,
) -> Option<crate::jitcode::BhFieldSpec> {
    let (base_owner, variant) = variant_owner.rsplit_once("::")?;
    let entries = cc.struct_field_entries(base_owner)?;
    if entries.len() != 1 || entries[0].0 != "__discriminant" {
        return None;
    }
    let tag = bh_all_field_specs_for_struct(cc, base_owner)
        .into_iter()
        .find(|spec| spec.field_key() == "__discriminant")?;
    let base_leaf = base_owner.rsplit("::").next().unwrap_or(base_owner);
    let canonical_variant_owner = format!(
        "{}::{variant}",
        majit_ir::descr::canonical_struct_name(base_leaf)
    );
    Some(bh_field_spec_from_parts(
        0,
        &canonical_variant_owner,
        "__discriminant",
        tag.offset,
        tag.field_size,
        tag.field_type,
        tag.field_flag,
        tag.is_immutable,
        tag.is_quasi_immutable,
        0,
    ))
}

fn bh_all_field_specs_for_struct(
    cc: &CallControl,
    owner: &str,
) -> Vec<crate::jitcode::BhFieldSpec> {
    let mut specs = Vec::new();
    bh_all_field_specs_for_struct_into(cc, owner, owner, "", 0, &mut specs);
    specs
}

/// Build the variant-owned portion of the explicit `Result<T, PyError>` /
/// `Option<T>` shell.  The front end deliberately represents these as one
/// inherited tag word followed by one-word payload fields (`front/mir.rs`'s
/// `explicit_sum_shell`), rather than as Rust's native enum layout.  The
/// shared nominal layout registry cannot describe each generic payload at
/// once, so read the concrete instantiation's rows directly and place them
/// after the inherited tag, matching RPython's concrete low-level STRUCT.
fn bh_result_variant_field_specs(
    cc: &CallControl,
    owner: &str,
) -> Vec<crate::jitcode::BhFieldSpec> {
    let Some(fields) = cc.struct_field_entries(owner) else {
        return Vec::new();
    };
    let mut specs = Vec::new();
    let mut offset = 8usize;
    for (field_name, field_type_str) in fields {
        // This builder lays out the variant's own payload rows, which start
        // past the inherited tag at byte 8.  The registry can also carry the
        // enum base's synthetic `__discriminant` row (`layout.rs`,
        // `front/semantic.rs`); emitting it here would place the tag at
        // offset 8 alongside the payload AND make the caller's
        // `any(field_key() == "__discriminant")` check skip the slot-0 /
        // offset-0 tag it synthesizes.  Leave the tag to the caller.
        if field_name == "__discriminant" {
            continue;
        }
        let (field_flag, field_type, field_size) = type_flag_from_str(field_type_str);
        if field_type == majit_ir::value::Type::Void || field_size == 0 {
            continue;
        }
        let align = scalar_align(field_size);
        offset = (offset + align - 1) & !(align - 1);
        let index = specs.len() + 1;
        let rank = cc.field_immutability(Some(owner), field_name);
        specs.push(bh_field_spec_from_parts(
            index as u32,
            owner,
            field_name,
            offset,
            field_size,
            field_type,
            field_flag,
            rank.is_some(),
            rank.map(|r| r.is_quasi_immutable()).unwrap_or(false),
            index,
        ));
        offset += field_size;
    }
    specs
}

/// RPython `heaptracker.all_fielddescrs(STRUCT, res=...)` parity port:
/// recursively walks `STRUCT._names`, skipping `Void` / `typeptr` /
/// `c__pad`, and recursing into nested-struct fields so their leaf
/// fielddescrs land in the same flat `res` list with `index_in_parent`
/// matching `heaptracker.get_fielddescr_index_in()` (`heaptracker.py`).
///
/// Pyre keeps both a structured `struct_layouts` cache and a textual
/// `struct_field_entries` registry; the layout path doesn't carry the
/// nested-struct type name, so we cross-reference the entries registry
/// to recover the inner owner string before recursing.
fn bh_all_field_specs_for_struct_into(
    cc: &CallControl,
    root_owner: &str,
    owner: &str,
    field_prefix: &str,
    base_offset: usize,
    specs: &mut Vec<crate::jitcode::BhFieldSpec>,
) {
    if let Some(layout) = cc.struct_layout_for(owner) {
        // The textual entries registry carries the inner-struct type name
        // for nested fields; match by field name to recover the owner
        // string when recursing.  Cloned out of `cc` so the immutable
        // borrow does not collide with the recursive call below.
        let entries: Vec<(String, String)> = cc
            .struct_field_entries(owner)
            .map(|fs| fs.to_vec())
            .unwrap_or_default();
        for fl in &layout.fields {
            // Header words are skipped against the owner that declares them,
            // which is the nested owner once the walk has descended. They are
            // not part of the positional census (`heaptracker.py all_fielddescrs`), so a
            // list that keeps them numbers every following field two slots
            // past where `get_fielddescr_index_in` puts it — and the index a
            // descr is minted with is then a position in a different list
            // than the one attached to it as the parent.
            if fl.field_type == majit_ir::value::Type::Void
                || crate::codewriter::heaptracker::is_header_word(owner, &fl.name)
                || fl.name.starts_with("c__pad")
            {
                continue;
            }
            if fl.flag == majit_ir::descr::ArrayFlag::Struct {
                // `heaptracker.py:68-69 isinstance(FIELD, lltype.Struct):
                //  all_fielddescrs(gccache, FIELD, only_gc, res, get_field_descr)`.
                if let Some(inner_owner) = entries
                    .iter()
                    .find(|(name, _)| name == &fl.name)
                    .map(|(_, ty)| ty.as_str())
                {
                    let nested_prefix = format!("{field_prefix}{}.", fl.name);
                    bh_all_field_specs_for_struct_into(
                        cc,
                        root_owner,
                        inner_owner,
                        &nested_prefix,
                        base_offset + fl.offset,
                        specs,
                    );
                }
                continue;
            }
            let index_in_parent = specs.len();
            let field_key = format!("{field_prefix}{}", fl.name);
            specs.push(bh_field_spec_from_parts(
                index_in_parent as u32,
                root_owner,
                &field_key,
                base_offset + fl.offset,
                fl.size,
                fl.field_type,
                fl.flag,
                fl.is_immutable(),
                fl.is_quasi_immutable(),
                index_in_parent,
            ));
        }
        return;
    }

    let Some(fields) = cc.struct_field_entries(owner).map(|fs| fs.to_vec()) else {
        return;
    };
    let mut offset = 0usize;
    for (field_name, field_type_str) in &fields {
        let (field_flag, field_type, field_size) = if cc.is_known_struct(field_type_str) {
            (
                majit_ir::descr::ArrayFlag::Struct,
                majit_ir::value::Type::Ref,
                cc.struct_layout_for(field_type_str)
                    .map(|layout| layout.size)
                    .unwrap_or(std::mem::size_of::<usize>()),
            )
        } else {
            type_flag_from_str(field_type_str)
        };
        if field_type == majit_ir::value::Type::Void || field_size == 0 {
            continue;
        }
        let align = scalar_align(field_size);
        offset = (offset + align - 1) & !(align - 1);
        let is_skipped_field = crate::codewriter::heaptracker::is_header_word(owner, field_name)
            || field_name.starts_with("c__pad");
        if !is_skipped_field {
            if field_flag == majit_ir::descr::ArrayFlag::Struct {
                // `heaptracker.py:68-69` recursive flatten for nested
                // structs.  `field_type_str` is the inner owner name in
                // this textual path.
                let nested_prefix = format!("{field_prefix}{field_name}.");
                bh_all_field_specs_for_struct_into(
                    cc,
                    root_owner,
                    field_type_str,
                    &nested_prefix,
                    base_offset + offset,
                    specs,
                );
            } else {
                let index_in_parent = specs.len();
                let rank = cc.field_immutability(Some(owner), field_name);
                let field_key = format!("{field_prefix}{field_name}");
                specs.push(bh_field_spec_from_parts(
                    index_in_parent as u32,
                    root_owner,
                    &field_key,
                    base_offset + offset,
                    field_size,
                    field_type,
                    field_flag,
                    rank.is_some(),
                    rank.map(|r| r.is_quasi_immutable()).unwrap_or(false),
                    index_in_parent,
                ));
            }
        }
        offset += field_size;
    }
}

fn heuristic_struct_size_for_bh(cc: &CallControl, owner: &str) -> Option<usize> {
    let fields = cc.struct_field_entries(owner)?;
    let mut offset = 0usize;
    let mut max_align = 0usize;
    for (_, field_type_str) in fields {
        let field_size = if cc.is_known_struct(field_type_str) {
            cc.struct_layout_for(field_type_str)
                .map(|layout| layout.size)
                .unwrap_or(std::mem::size_of::<usize>())
        } else {
            let (_, field_type, size) = type_flag_from_str(field_type_str);
            if field_type == majit_ir::value::Type::Void || size == 0 {
                continue;
            }
            size
        };
        // A zero-sized field (e.g. a known struct whose layout size is 0)
        // occupies no space and imposes no alignment; skip it so the
        // `align - 1` masking below never underflows on `align == 0`. The
        // final `max_align.max(1)` already floors overall alignment at 1.
        if field_size == 0 {
            continue;
        }
        let align = scalar_align(field_size);
        max_align = max_align.max(align);
        offset = (offset + align - 1) & !(align - 1);
        offset += field_size;
    }
    if offset == 0 {
        return Some(0);
    }
    let align = max_align.max(1);
    Some((offset + align - 1) & !(align - 1))
}

/// The one slot of `fields` at `offset`, or `None` when no field or more than
/// one sits there.
///
/// A flattened layout puts an inline aggregate and its first leaf at the same
/// address (`heaptracker.py:68-69` recurses into a nested `lltype.Struct`
/// without minting a descr for the container), so an offset can name two
/// fields. Where it does, there is no answer to give and inventing one names a
/// sibling — `descr.rs find_index_in_parent` refuses the same way.
fn unique_slot_at_offset(fields: &[crate::jitcode::BhFieldSpec], offset: usize) -> Option<usize> {
    let mut at_offset = fields
        .iter()
        .enumerate()
        .filter(|(_, f)| f.offset == offset);
    let (idx, _) = at_offset.next()?;
    at_offset.next().is_none().then_some(idx)
}

fn transparent_scalar_field<'a>(
    parent: &'a crate::jitcode::BhSizeSpec,
    field_name: &str,
    offset: usize,
    expected_type: majit_ir::value::Type,
) -> Option<&'a crate::jitcode::BhFieldSpec> {
    let prefix = format!("{field_name}.");
    let mut matches = parent.all_fielddescrs.iter().filter(|spec| {
        spec.offset == offset
            && spec.field_type == expected_type
            && spec.field_key().starts_with(&prefix)
    });
    let field = matches.next()?;
    matches.next().is_none().then_some(field)
}

fn canonical_field_owner(field: &crate::model::FieldDescriptor) -> Option<String> {
    let owner = field.owner_root.as_deref()?;
    let canonical = majit_ir::descr::canonical_struct_name(owner);
    if !owner.contains("::") {
        return Some(canonical);
    }
    let leaf = owner.rsplit("::").next().unwrap_or(owner);
    let leaf_canonical = majit_ir::descr::canonical_struct_name(leaf);
    let owner_id = field
        .owner_id
        .or_else(|| majit_ir::descr::struct_id_for_name(owner));
    if leaf_canonical != leaf
        && owner_id.is_some()
        && majit_ir::descr::struct_id_for_name(&leaf_canonical) == owner_id
    {
        Some(leaf_canonical)
    } else {
        Some(canonical)
    }
}

fn apply_scalar_field_word(
    word: crate::model::ScalarFieldWord,
    offset: &mut usize,
    field_size: &mut usize,
    field_type: &mut majit_ir::value::Type,
    field_flag: &mut majit_ir::descr::ArrayFlag,
    is_field_signed: &mut bool,
) {
    use crate::model::ScalarFieldWidth;
    *offset = word.offset;
    *field_size = word.size;
    match word.width {
        ScalarFieldWidth::Signed => {
            *field_type = majit_ir::value::Type::Int;
            *field_flag = majit_ir::descr::ArrayFlag::Signed;
            *is_field_signed = true;
        }
        ScalarFieldWidth::Unsigned => {
            *field_type = majit_ir::value::Type::Int;
            *field_flag = majit_ir::descr::ArrayFlag::Unsigned;
            *is_field_signed = false;
        }
        ScalarFieldWidth::Float => {
            *field_type = majit_ir::value::Type::Float;
            *field_flag = majit_ir::descr::ArrayFlag::Float;
            *is_field_signed = false;
        }
    }
}

/// Host byte offset of `field` when the base is a raw address.
///
/// The variant owner is `{enum_root}::{Variant}`; the host table is
/// registered on the enum root and cloned onto the variant. A struct is
/// variant 0 and has no `::` to strip.
fn raw_host_field_offset(
    cc: Option<&CallControl>,
    field: &crate::model::FieldDescriptor,
) -> Option<usize> {
    let (variant, index) = field.host_index?;
    let cc = cc?;
    let owner = field.owner_root.as_deref()?;
    let host = cc.host_layout_for(owner).or_else(|| {
        let root = owner.rsplit_once("::")?.0;
        cc.host_layout_for(root)
    })?;
    Some(
        *host
            .variant_field_offsets
            .get(variant as usize)?
            .get(index as usize)? as usize,
    )
}

fn fielddescrof(
    field: &crate::model::FieldDescriptor,
    ty: &crate::model::ValueType,
    callcontrol: Option<&CallControl>,
    is_gc: bool,
) -> crate::jitcode::BhDescr {
    let (mut offset, mut field_size, mut field_type, mut field_flag, mut is_field_signed) = {
        let (field_size, field_type, field_flag, signed) = fallback_field_layout(ty);
        (0, field_size, field_type, field_flag, signed)
    };
    let mut is_immutable = false;
    let mut is_quasi_immutable = false;
    // `None` until a lookup arm actually resolves a slot. The minted descr still
    // carries `0` for the unresolved case (`unwrap_or(0)` at the bottom), so the
    // emitted `BhDescr::Field` is unchanged; what the `Option` adds is that the
    // two states stop being the same bytes while they are still separable.
    // Written as a literal `0`, an unresolved mint and a mint that resolved to
    // slot 0 are indistinguishable the moment this function returns, and every
    // census downstream inherits that.
    let mut index_in_parent: Option<usize> = None;
    let mut parent = None;
    let mut field_key = if let Some(owner) = field.owner_root.as_deref() {
        let prefix = format!("{owner}.");
        field
            .name
            .strip_prefix(&prefix)
            .unwrap_or(&field.name)
            .to_string()
    } else {
        field.name.clone()
    };

    let canonical_owner = canonical_field_owner(field);
    let mut layout_resolved = false;
    if let (Some(cc), Some(owner)) = (callcontrol, canonical_owner.as_deref()) {
        parent = bh_size_spec_from_callcontrol(cc, owner);
        // RPython `descr.py` keys both the SizeDescr and
        // FieldDescr caches on the low-level STRUCT object, never on the
        // spelling used to reach that object.  The Rust front end can carry
        // the same external type as `core::result::Result<T, E>` at a
        // synthetic constructor and `result::Result<T, E>` at a MIR
        // discriminant read (the crate prefix is stripped from the latter).
        // Preserve the source-attached StructId across that spelling split so
        // both operations resolve through one parent SizeDescr and therefore
        // one `__discriminant` FieldDescr.
        if let Some(parent_spec) = parent.as_mut()
            && let Some(owner_id) = field
                .owner_id
                .or_else(|| majit_ir::descr::struct_id_for_name(owner))
        {
            parent_spec.type_id = if parent_spec.is_gc_managed {
                owner_id.as_u64()
            } else {
                // StructId identifies the nominal declaration only. Raw and
                // GC variants of that declaration are distinct lltypes, so
                // retain the storage discriminator installed above.
                majit_ir::descr::path_hash_for_gc_kind(owner, false)
            };
        }
        let field_lookup = bh_field_lookup(cc, field);
        let mut found_parent_field = matches!(field_lookup, BhFieldLookup::Parent(_));
        match field_lookup {
            BhFieldLookup::Parent(spec) => {
                offset = spec.offset;
                field_size = spec.field_size;
                field_type = spec.field_type;
                field_flag = spec.field_flag;
                is_field_signed = spec.is_field_signed;
                is_immutable = spec.is_immutable;
                is_quasi_immutable = spec.is_quasi_immutable;
                index_in_parent = Some(spec.index_in_parent);
                layout_resolved = true;
            }
            BhFieldLookup::Layout(layout_field) => {
                if layout_field.flag == majit_ir::descr::ArrayFlag::Struct
                    && field_type != majit_ir::value::Type::Ref
                    && let Some(spec) = parent.as_ref().and_then(|parent| {
                        transparent_scalar_field(
                            parent,
                            &field.name,
                            layout_field.offset,
                            field_type,
                        )
                    })
                {
                    offset = spec.offset;
                    field_size = spec.field_size;
                    field_type = spec.field_type;
                    field_flag = spec.field_flag;
                    is_field_signed = spec.is_field_signed;
                    is_immutable = spec.is_immutable;
                    is_quasi_immutable = spec.is_quasi_immutable;
                    index_in_parent = Some(spec.index_in_parent);
                    found_parent_field = true;
                } else {
                    offset = layout_field.offset;
                    field_size = layout_field.size;
                    field_type = layout_field.field_type;
                    field_flag = layout_field.flag;
                    is_field_signed = field_flag == majit_ir::descr::ArrayFlag::Signed;
                    is_immutable = layout_field.is_immutable();
                    is_quasi_immutable = layout_field.is_quasi_immutable();
                }
                layout_resolved = true;
            }
            BhFieldLookup::Missing => {
                if let Some((
                    computed_offset,
                    computed_size,
                    computed_type,
                    computed_flag,
                    computed_signed,
                )) = heuristic_field_layout(cc, owner, &field.name)
                {
                    offset = computed_offset;
                    field_size = computed_size;
                    field_type = computed_type;
                    field_flag = computed_flag;
                    is_field_signed = computed_signed;
                    layout_resolved = true;
                }
            }
        }
        // An opaque dependency struct has a layout and no field list, so
        // the name lookup misses. The front records that scalar's byte
        // offset on the descriptor. A parent, layout, or heuristic hit
        // already named the slot and keeps its own offset. Apply the word
        // before `unique_slot_at_offset` so the slot follows the offset
        // that is actually emitted.
        if !layout_resolved && let Some(word) = field.scalar_word {
            apply_scalar_field_word(
                word,
                &mut offset,
                &mut field_size,
                &mut field_type,
                &mut field_flag,
                &mut is_field_signed,
            );
        }
        if let Some(rank) = cc.field_immutability(Some(owner), &field_key) {
            is_immutable = rank.is_immutable();
            is_quasi_immutable = rank.is_quasi_immutable();
        }
        // Both fallbacks above take `offset` from a second source — the
        // `struct_layout_for` registry or `heuristic_field_layout` — and neither
        // touches `index_in_parent`, which is still its `0` initialiser while
        // `parent` carries the whole flattened list.  Slot 0 of that list is
        // some other field, and `all_fielddescrs()[index_in_parent]` is what
        // `optimizeopt/info.rs force_box` indexes by, so the descr would name
        // one field and address another.  Resolve the slot against the parent
        // that is actually attached, the way `descr.py:228
        // heaptracker.get_fielddescr_index_in` derives it from the STRUCT
        // itself.
        //
        // Only the byte offset is available here — reaching this arm means the
        // name lookup above already missed — and offset is not an identity: a
        // flattened inline aggregate shares an address with its first leaf
        // (`heaptracker.py`).  So resolve only when exactly one field sits
        // there, and otherwise leave the caller's number rather than name a
        // sibling.
        //
        // The name arm runs first and covers `found_parent_field` too.  That
        // arm carries a number STORED on the spec it matched, and the list that
        // number counts is not always the list `parent` holds: `bh_field_lookup`
        // probes the owner spelling the field descriptor arrived with, while
        // `parent` is built for the canonical one, and the two flatten
        // differently whenever only one of them reaches the field registry a
        // by-value member's leaves are enumerated through.  Measured on this
        // tree, 23 field descrs claimed a slot the list attached to them gives
        // to a sibling; `builtins::WritableBuffer.owner_slot` (an `Int`) claimed
        // slot 2, which its own parent gives to `address` (a `Ref`).  A
        // consumer that resolves a field through `all_fielddescrs[index]` — the
        // optimizer's `init_fields` slot space — then reads one bank's value
        // out of the other's slot.  `descr.py:228` has no such split because
        // one walker answers both questions upstream; taking the position from
        // the attached list restores that.
        if let Some(parent_spec) = parent.as_ref()
            && let Some(pos) = parent_spec
                .all_fielddescrs
                .iter()
                .position(|spec| spec.field_key() == field_key)
        {
            index_in_parent = Some(pos);
        } else if !found_parent_field
            && let Some(parent_spec) = parent.as_ref()
            && let Some(pos) = unique_slot_at_offset(&parent_spec.all_fielddescrs, offset)
        {
            index_in_parent = Some(pos);
        }
    } else if let Some(word) = field.scalar_word {
        apply_scalar_field_word(
            word,
            &mut offset,
            &mut field_size,
            &mut field_type,
            &mut field_flag,
            &mut is_field_signed,
        );
    }

    // Outside every `parent` check above, deliberately. This is the only census
    // that fires when the mint never had a usable parent at all — the case where
    // the placeholder survives unrepaired and both of the existing counters are
    // silent by construction. See `census_mint_index_provenance`.
    majit_ir::descr::census_mint_index_provenance(index_in_parent.is_some());

    // The descr's two halves, counted where they are produced rather than at a
    // reader that would repair them (`GcCache::derive_index_in_parent`
    // re-derives by name and overwrites, so it can only report repairs it
    // already applied — see `field_position_jit_stats`).
    //
    // Still meaningful after the re-resolution above, which only covers the
    // fallback arms: the `found_parent_field` arm takes `index_in_parent` from
    // the matched spec's STORED number, and nothing here guarantees that number
    // equals the spec's position in the list it was stored in.
    //
    // Name first, and an ambiguous offset is not counted at all — a descr that
    // names its field correctly while sharing an address with a sibling is not
    // misplaced, and counting it would put a false reading behind a gate.
    if let Some(parent_spec) = parent.as_ref()
        && let Some(pos) = parent_spec
            .all_fielddescrs
            .iter()
            .position(|spec| spec.field_key() == field_key)
            .or_else(|| unique_slot_at_offset(&parent_spec.all_fielddescrs, offset))
    {
        majit_ir::descr::census_attached_index(pos, index_in_parent);
    }
    // A raw base addresses the host layout (`rewrite_op_getfield` on a
    // raw `Ptr`). The explicit sum shell keeps the tag at 0 and payloads
    // at `8 + i * 8`; those rows are the gc representation. The census
    // above still sees the shell slot. `vec_part` is added afterwards.
    if !is_gc && let Some(host_offset) = raw_host_field_offset(callcontrol, field) {
        offset = host_offset;
    }
    // `rlist.py` `ll_getitem_fast` is `l.ll_items()[index]` and `ll_length`
    // is `l.length`. A Rust `Vec<T>` is those two words inside the value.
    // `Box<[T]>` is a fat pointer: data word, then length word
    // (`fat_ptr_layout::probe`). Charon records neither split. Applied
    // after the slot census so the parent slot stays the field's.
    if let Some(part) = field.vec_part {
        let vec_layout = crate::vec_layout::probe();
        let fat_layout = crate::fat_ptr_layout::probe();
        let word = crate::layout::target_word_size();
        let (add, ty, flag, suffix) = match part {
            crate::model::VecFieldPart::Buf => (
                vec_layout.ptr_offset,
                majit_ir::value::Type::Ref,
                majit_ir::descr::ArrayFlag::Pointer,
                ".buf",
            ),
            crate::model::VecFieldPart::Len => (
                vec_layout.len_offset,
                majit_ir::value::Type::Int,
                majit_ir::descr::ArrayFlag::Unsigned,
                ".len",
            ),
            crate::model::VecFieldPart::FatData => {
                // A raw owner's data word is an address. `ArrayFlag::Pointer`
                // asks for `getfield_raw_r`, which a raw base refuses.
                if is_gc {
                    (
                        fat_layout.data_offset,
                        majit_ir::value::Type::Ref,
                        majit_ir::descr::ArrayFlag::Pointer,
                        ".data",
                    )
                } else {
                    (
                        fat_layout.data_offset,
                        majit_ir::value::Type::Int,
                        majit_ir::descr::ArrayFlag::Unsigned,
                        ".data",
                    )
                }
            }
            crate::model::VecFieldPart::FatLen => (
                fat_layout.len_offset,
                majit_ir::value::Type::Int,
                majit_ir::descr::ArrayFlag::Unsigned,
                ".len",
            ),
            // `&dyn Trait`: the metadata word is the vtable pointer, not a
            // length. Same slot as `FatLen`, loaded as a pointer.
            crate::model::VecFieldPart::FatVtable => (
                fat_layout.len_offset,
                majit_ir::value::Type::Ref,
                majit_ir::descr::ArrayFlag::Pointer,
                ".vtable",
            ),
        };
        offset = offset.saturating_add(add);
        field_size = word;
        field_type = ty;
        field_flag = flag;
        is_field_signed = false;
        field_key = format!("{field_key}{suffix}");
    }
    crate::jitcode::BhDescr::Field {
        offset,
        field_size,
        field_type,
        field_flag,
        is_field_signed,
        is_immutable,
        is_quasi_immutable,
        // Carried as-is: the `None` is the whole point. Collapsing it to `0`
        // here is what made the split underivable at the reader, and this is
        // the last place that still knows the difference.
        index_in_parent,
        parent: parent.map(std::sync::Arc::new),
        name: field_key,
        owner: canonical_owner.unwrap_or_default(),
    }
}

fn heuristic_field_layout(
    cc: &CallControl,
    owner: &str,
    field_name: &str,
) -> Option<(
    usize,
    usize,
    majit_ir::value::Type,
    majit_ir::descr::ArrayFlag,
    bool,
)> {
    let fields = cc.struct_field_entries(owner)?;
    let mut offset = 0usize;
    for (name, type_str) in fields {
        let (flag, field_type, mut field_size) = if cc.is_known_struct(type_str) {
            (
                majit_ir::descr::ArrayFlag::Struct,
                majit_ir::value::Type::Ref,
                cc.struct_layout_for(type_str)
                    .map(|layout| layout.size)
                    .unwrap_or(std::mem::size_of::<usize>()),
            )
        } else {
            type_flag_from_str(type_str)
        };
        if field_type == majit_ir::value::Type::Void || field_size == 0 {
            continue;
        }
        let align = scalar_align(field_size);
        offset = (offset + align - 1) & !(align - 1);
        if name == field_name {
            return Some((
                offset,
                field_size,
                field_type,
                flag,
                flag == majit_ir::descr::ArrayFlag::Signed,
            ));
        }
        field_size = field_size.max(1);
        offset += field_size;
    }
    None
}

/// `cpu.arraydescrof(rffi.CArray(T))` for a raw element access
/// (`jtransform.py rewrite_op_raw_store`): no length header, non-GC, item width and
/// signedness explicit because [`crate::model::ValueType`] collapses widths.
fn raw_carray_descrof(
    ty: &crate::model::ValueType,
    itemsize: usize,
    is_item_signed: bool,
) -> crate::jitcode::BhDescr {
    let prefix = match ty {
        crate::model::ValueType::Float => "f",
        _ if is_item_signed => "i",
        _ => "u",
    };
    crate::jitcode::BhDescr::Array {
        base_size: 0,
        itemsize,
        len_offset: None,
        type_id: 0,
        gc_type_id: 0,
        item_type: value_type_to_ir_type_for_descr(ty),
        is_array_of_pointers: false,
        is_array_of_structs: false,
        is_item_signed,
        // Raw memory carries no heapcache bookkeeping, so no EffectInfo
        // heap-invalidation index applies.
        ei_index: u32::MAX,
        // Distinct raw widths must stay on distinct runtime descr slots.
        array_type_id: Some(format!("majit::raw_carray_{prefix}{}", itemsize * 8)),
        interior_fields: Vec::new(),
        is_gc_managed: false,
    }
}

/// `cpu.interiorfielddescrof(ARRAY, fieldname)` (`descr.py`) wrapped as
/// the `BhDescr::InteriorField` `emit_ready_descr` stores.
///
/// `idx` is `DescrIndexRegistry::interiorfield_index`, the same
/// `(array_type_id, fieldname)` namespace `push_interior_effect` and
/// `heaptracker::all_interiorfielddescrs` pass to `interiorfielddescrof`.
/// An access with no descr would address the wrong bytes, so a missing
/// `CallControl` or a `None` mint panics instead of encoding bare registers.
fn interiorfield_bh_descr(
    callcontrol: Option<&CallControl>,
    array_type_id: &Option<String>,
    field_name: &str,
    graph_name: &Option<String>,
) -> crate::jitcode::BhDescr {
    let Some(cc) = callcontrol else {
        panic!(
            "interior field access requires a CallControl \
             (jtransform.py rewrite_op_getinteriorfield) — \
             graph {graph_name:?}, array {array_type_id:?}, field {field_name}"
        );
    };
    let idx = cc
        .descr_indices
        .interiorfield_index(array_type_id, field_name);
    let Some(descr) = cc.interiorfielddescrof(idx, array_type_id, field_name) else {
        panic!(
            "interiorfielddescrof returned no descr \
             (jtransform.py rewrite_op_getinteriorfield) — \
             graph {graph_name:?}, array {array_type_id:?}, field {field_name}"
        );
    };
    let Some(ifd) = descr.as_interior_field_descr() else {
        panic!(
            "interiorfielddescrof did not return an InteriorFieldDescr — \
             graph {graph_name:?}, array {array_type_id:?}, field {field_name}"
        );
    };
    crate::jitcode::BhDescr::from_interior_field_descr(ifd)
}

/// jtransform.py:773,802 cpu.arraydescrof(ARRAY) equivalent.
///
/// Determines the full ArrayDescr shape from the array element type.
/// When `array_type_id` is available (e.g. `Vec<i32>` → element `i32`),
/// the result is exact. Fallback uses descr.py get_type_flag()
/// semantics: Int → FLAG_SIGNED, Float/Ref → FLAG_UNSIGNED/FLAG_FLOAT.
///
/// When `callcontrol` is present, this routes through
/// `CallControl::arraydescrof_for_type` and carries the EffectInfo
/// `ei_index` across the BhDescr boundary.  The `callcontrol == None`
/// fallback is descriptor-shape-only: it must not be used for EffectInfo raw
/// sets because there is no codewriter-side array namespace to publish.
fn arraydescrof(
    ty: &crate::model::ValueType,
    array_type_id: &Option<String>,
    len_offset: Option<usize>,
    callcontrol: Option<&CallControl>,
) -> crate::jitcode::BhDescr {
    let ir_type = value_type_to_ir_type_for_descr(ty);
    if let Some(cc) = callcontrol {
        // Route through `arraydescrof_for_type` so the bytecode emit
        // path shares the same `(item_ty, array_type_id, len_offset) → ei_index`
        // table as `writeanalyze` in `call.rs`; otherwise
        // every emit-time descr lands at `ei_index = 0` and aliases
        // distinct ARRAY identities at `force_from_effectinfo`
        // (`heap.py:540-560`, `heap.rs`'s `array_effect_index`).
        let descr = cc.arraydescrof_for_type(ty, array_type_id, ir_type, len_offset);
        let array_descr = descr
            .as_array_descr()
            .expect("CallControl::arraydescrof must return an ArrayDescr");
        return crate::jitcode::BhDescr::Array {
            base_size: array_descr.base_size(),
            itemsize: array_descr.item_size(),
            len_offset: array_descr.len_descr().map(|fd| fd.offset()),
            // `descr.py` cache identity — `ArrayDescr.cache_key()`
            // returns the u64 `path_hash(array_type_id)` slot the analyzer
            // stamped at `gc_cache.get_array_descr` cache-miss-mint.
            // Round-trips through `_cache_array[LLType::Array(cache_key)]`
            // on the runtime side.
            type_id: array_descr.cache_key(),
            gc_type_id: array_descr.type_id(),
            item_type: array_descr.item_type(),
            is_array_of_pointers: array_descr.is_array_of_pointers(),
            is_array_of_structs: array_descr.is_array_of_structs(),
            is_item_signed: array_descr.is_item_signed(),
            // descr.py:465 compute_bitstrings — carry the SimpleArrayDescr's
            // ei_index across the BhDescr boundary so make_descr_from_bh
            // republishes it on the runtime SimpleArrayDescr.
            ei_index: descr.get_ei_index(),
            // `cpu.arraydescrof` keys one ARRAY. `[i64]` / `GcArray<i64>`
            // (and the f64 pair) are that one lltype, so the pool string
            // is the canonical spelling `get_array_descr` hashed.
            array_type_id: array_type_id
                .as_ref()
                .map(|id| crate::front::typestr::canonical_array_type_id(id).into_owned()),
            interior_fields: super::jitcode::bh_interior_field_specs_from_array_descr(array_descr),
            is_gc_managed: array_descr.is_gc_managed(),
        };
    }

    // Primary path: extract element type from the array type identity
    // (our equivalent of `ARRAY.OF` in RPython).
    let (flag, item_type, itemsize) = if let Some(elem) = array_type_id
        .as_deref()
        .and_then(extract_element_type_from_str)
    {
        type_flag_from_str(elem.as_str())
    } else {
        match ty {
            crate::model::ValueType::Int => (
                majit_ir::descr::ArrayFlag::Signed,
                majit_ir::value::Type::Int,
                8,
            ),
            crate::model::ValueType::Float => (
                majit_ir::descr::ArrayFlag::Float,
                majit_ir::value::Type::Float,
                8,
            ),
            crate::model::ValueType::Ref(_) => (
                majit_ir::descr::ArrayFlag::Pointer,
                majit_ir::value::Type::Ref,
                // Pointer elements stride by the target word, not the
                // build host's.
                crate::layout::target_word_size(),
            ),
            _ => (
                majit_ir::descr::ArrayFlag::Unsigned,
                majit_ir::value::Type::Int,
                8,
            ),
        }
    };
    // descr.py:354/359-362 + symbolic.get_array_token — basesize follows
    // the lltype's nolength flag: nolength → items at offset 0;
    // length-prefixed → items past the header at len_offset + WORD.
    //
    // This codewriter-less fallback is shape-only and serves GC arrays, whose
    // items sit flat at the length word (`GcTypedArray`) — it does NOT
    // element-align. The word must be the target's, not the host's, so a
    // cross-target extraction (build host ≠ run target) bakes the right stride;
    // the `CallControl` path above already routes through `array_items_base`.
    let base_size = match len_offset {
        None => 0,
        Some(off) => off + crate::layout::target_word_size(),
    };
    crate::jitcode::BhDescr::Array {
        base_size,
        itemsize,
        len_offset,
        type_id: 0,
        gc_type_id: 0,
        item_type,
        is_array_of_pointers: flag == majit_ir::descr::ArrayFlag::Pointer,
        is_array_of_structs: flag == majit_ir::descr::ArrayFlag::Struct,
        is_item_signed: flag == majit_ir::descr::ArrayFlag::Signed,
        // No CallControl-side `array_index` for the fallback path.  This
        // descriptor is shape-only for codewriter-less emission helpers; any
        // path that needs EffectInfo heap invalidation must pass CallControl
        // so `arraydescrof_for_type` can publish the real `ei_index`.
        ei_index: u32::MAX,
        // Same canonical spelling as the CallControl path: `[i64]` and
        // `GcArray<i64>` (and the f64 pair) are one ARRAY.
        array_type_id: array_type_id
            .as_ref()
            .map(|id| crate::front::typestr::canonical_array_type_id(id).into_owned()),
        interior_fields: Vec::new(),
        // Shape-only fallback is for GC arrays; the raw `pool_arrays`
        // base is minted only via `add_ptr_array_descr`.
        is_gc_managed: true,
    }
}

fn vable_arraydescrof(
    ty: &crate::model::ValueType,
    itemsize: usize,
    is_item_signed: bool,
) -> crate::jitcode::BhDescr {
    let item_type = value_type_to_ir_type_for_descr(ty);
    // The runtime block behind a vable array is `FixedObjectArray`: a length
    // word with items flat at the target word (`FIXED_ARRAY_ITEMS_OFFSET`),
    // holding word-wide `PyObjectRef` items. An element wider than the word
    // would need an items-base decision this mint has never had to make
    // (flat word vs element-aligned — the `TypedItemsBlock` /
    // `GcTypedArray` split in `call.rs`); fail the build instead of
    // silently picking a base.
    assert!(
        itemsize <= crate::layout::target_word_size(),
        "vable array itemsize {itemsize} exceeds the target word \
         {} — the flat-word items base below would under-align it; \
         decide the items base against the runtime block before widening",
        crate::layout::target_word_size(),
    );
    crate::jitcode::BhDescr::Array {
        // A virtualizable frame array is length-prefixed by a single word
        // (`len_offset = Some(0)`), so its items start at the target word — the
        // build host's word would mis-stride the block on a narrower target.
        base_size: crate::layout::target_word_size(),
        itemsize,
        len_offset: Some(0),
        // No layout identity, and neither zero is a placeholder to fill in.
        // `PyFrame.locals_cells_stack_w` is reached through two allocators —
        // `alloc_frame_locals_array`'s GC arm stamps the object-array tid into
        // the header, its `alloc_fixed_array_with_header` arm (taken for a
        // frame the collector does not own, and as the GC arm's own
        // out-of-memory fallback) leaves the prepended header zeroed — so one
        // tid cannot describe every block a trace will meet. Stamping either
        // slot would put a `GUARD_GC_TYPE` on the short-preamble entry that is
        // false for the other arm's blocks. `ArrayPtrInfo::make_guards`
        // (`optimizeopt/info.rs`) instead refuses to build the entry, which
        // costs the one unrolled attempt `unroll_free_retry_rescued` counts and
        // keeps the guard honest.
        type_id: 0,
        gc_type_id: 0,
        item_type,
        is_array_of_pointers: matches!(item_type, majit_ir::value::Type::Ref),
        is_array_of_structs: false,
        is_item_signed,
        // vable array slots are codewriter-known per-vinfo descrs, not
        // EffectInfo-keyed; ei_index stays unset.
        ei_index: u32::MAX,
        // vable array slots have no source-level array_type_id;
        // distinct vable indices are already disambiguated by the
        // parent `VableArray { index }` variant carried alongside.
        array_type_id: None,
        interior_fields: Vec::new(),
        // Virtualizable frame array slots are GC-managed frame fields;
        // preserve the existing GUARD_GC_TYPE behavior.
        is_gc_managed: true,
    }
}

/// Convert OpKind to an opname string for the assembler's instruction table.
/// RPython: the opname comes from SpaceOperation.opname.
/// Convert OpKind to a typed opname matching RPython's jtransform output.
///
/// RPython jtransform produces fully-qualified names like `getfield_vable_i`,
/// `setfield_gc_r`, `int_add`. The kind suffix comes from the result type
/// or value type of the operation.
/// Variant of [`op_kind_to_opname`] that routes operand-kind-sensitive
/// names through the proper RPython opname.  Specifically:
///
/// `OpKind::UnaryOp { op: "bool", .. }` is the truthify operator
/// pyre's frontend emits from the `&&`/`||`/`!` desugar (the bool
/// switch discriminator).  Upstream RPython lowers `bool` per the
/// operand's repr at rtyper time:
///
/// - `IntegerRepr.rtype_bool` → `genop("int_is_true", ...)`
///   (`rint.py`)
/// - `PtrRepr.rtype_bool` → `genop("ptr_nonzero", ...)`
///   (`rmodel.py::PtrRepr.rtype_bool`)
/// - `FloatRepr.rtype_bool` → `genop("float_ne", ..., 0.0)`
///   (`rfloat.py:191`); the `/f>i` shape comes from the float
///   compare, but pyre's truthify uses `float_ne` against zero
///   directly so the assembler key is `float_ne/ff>i` after the
///   constant pool emits the 0.0.  No `bool` op survives at the
///   `f` operand kind here today, so the float arm is a defensive
///   placeholder.
///
/// The unconditional `int_<op>` prefix in [`op_kind_to_opname`]
/// would name these `int_bool/i>i` and `int_bool/r>i`, which has no
/// blackhole handler (RPython has no `int_bool` opname) and trips
/// the strict-coverage `default_bh_builder_unwired_set_matches_task_85_snapshot`
/// guard.  Routing on the operand kind here keeps the legacy/codewriter
/// path producing handler-backed opnames without requiring a full
/// rtyper port for every prebuilt graph.
fn op_kind_to_opname_with_kinds(kind: &crate::model::OpKind, operand_kinds: &str) -> String {
    use crate::model::OpKind;
    if let OpKind::UnaryOp { op, .. } = kind
        && op == "bool"
    {
        return match operand_kinds {
            "i" => "int_is_true".into(),
            "r" => "ptr_nonzero".into(),
            // RPython `jtransform.py rewrite_op_float_is_true`
            // collapses both `bool/f` and `float_is_true/f` to
            // `float_ne(x, 0.0)` upstream of the assembler — pyre's
            // jtransform mirror in `rewrite_operation` covers
            // both surfaces, so an `f` operand reaching here means
            // the rewrite was skipped.  Fail loud rather than emit
            // a `float_is_true` opname the backend does not register
            // (`rpython/jit/codewriter/jtransform.py` is
            // unconditional, so pyre matches that invariant here).
            "f" => unreachable!(
                "OpKind::UnaryOp {{ op: \"bool\", .. }} over an `f` operand must be \
                 rewritten to float_ne in jtransform — see jtransform.rs:917"
            ),
            _ => format!("int_{op}"),
        };
    }
    // RPython parity (`rtyper/rmodel.py`, pairtype(Repr, Repr).rtype_is_,
    // and `jtransform.py`'s pointer comparisons): equality, inequality,
    // and identity on Ref operands lower to `ptr_eq` / `ptr_ne`, not the
    // integer form. Float arithmetic carries the full `float_*` opname
    // through `op_kind_to_opname`'s BinOp arm already; here we route the
    // high-level labels emitted by the MIR front-end when they meet two Ref
    // operands. Mixed `ri` / `ir` Ref-Int shapes
    // remain a kind-flow gap (see `default_bh_builder_unwired_set_*`
    // snapshot) — surfaced via the canonical fallthrough below.
    if let OpKind::BinOp { op, .. } = kind {
        match (op.as_str(), operand_kinds) {
            ("eq", "rr") => return "ptr_eq".into(),
            ("ne", "rr") => return "ptr_ne".into(),
            ("is_", "rr") => return "ptr_eq".into(),
            // `jtransform.py` `rewrite_op_ptr_eq` / `_rewrite_nongc_ptrs`:
            // equality of non-GC pointers (`Ptr(FuncType)`, both in the int
            // bank) is `int_eq`. A mixed `ri`/`ir` shape is a kind-flow gap.
            ("is_", "ii") => return "int_eq".into(),
            // The other operand was `Void` and was not encoded. The
            // remaining word compared by identity is `int_is_zero`
            // (`bhimpl_int_is_zero`), not `int_` + `is_`.
            ("is_", "i") => return "int_is_zero".into(),
            _ => {}
        }
    }
    op_kind_to_opname(kind)
}

fn op_kind_to_opname(kind: &crate::model::OpKind) -> String {
    use crate::model::OpKind;
    match kind {
        OpKind::Input { ty, .. } => format!("input_{}", value_type_to_kind(ty)),
        OpKind::RawLoad { item_ty, .. } => match item_ty {
            crate::model::ValueType::Float => "raw_load_f".into(),
            _ => "raw_load_i".into(),
        },
        OpKind::RawStore { item_ty, .. } => match item_ty {
            crate::model::ValueType::Float => "raw_store_f".into(),
            _ => "raw_store_i".into(),
        },
        // RPython: ConstInt is NOT a standalone op; see encode_op comment.
        // Pyre materialises constants as an int_copy from pool-region reg.
        OpKind::ConstInt(_) | OpKind::ConstFnAddr { .. } | OpKind::ConstUInt(_) => {
            "int_copy".into()
        }
        OpKind::ConstInt128(_) | OpKind::ConstUInt128(_) => {
            panic!("getkind: 128-bit integer constant is too large (history.py:62)")
        }
        OpKind::ConstSingleFloat(_) => {
            panic!(
                "getkind: SingleFloat constant is not supported (history.py:61) — \
                 the codewriter policy refuses a graph carrying one"
            )
        }
        // RPython folds `lltype.Bool` into kind `'int'`
        // (`flatten.py:getkind`), so the bool constant materialises
        // through the same `int_copy` path as `ConstInt`.
        OpKind::ConstBool(_) => "int_copy".into(),
        // The `_we_are_jitted` symbolic is folded to `ConstBool(true)` by
        // `jtransform` before assembly, so it never reaches opname
        // emission.
        OpKind::ConstSymbolic { .. } => {
            unreachable!("OpKind::ConstSymbolic must be folded by jtransform before assembly")
        }
        // Mirrors `ConstInt` — the constant is materialised through the
        // shared `constants_f` pool, then a `float_copy` op moves it into
        // the SSA destination register.
        OpKind::ConstFloat(_) => "float_copy".into(),
        // Ref-bank analog of `ConstInt`. The singleton HostObject's
        // `identity_id()` enters the shared `constants_r` pool via
        // `emit_const_r`, then a `ref_copy/r>r` op moves it into the
        // SSA destination register.
        OpKind::ConstStr(_)
        | OpKind::ConstInternedStr(_)
        | OpKind::ConstRef(_)
        | OpKind::ConstRefNull
        | OpKind::ConstRefAddr(_) => "ref_copy".into(),
        // `ConstNone` (Void `None`) is const-inlined by the flowspace
        // adapter. jtransform also erases its legacy definition after
        // rtyping, including graphs that used the legacy walker. RPython
        // represents it only as Constant(None, Void), not an instruction.
        // A surviving definition is a missed transformation; the distinctive
        // handler-less `const_none` opname trips
        // `default_bh_builder_unwired_set_matches_task_85_snapshot` loudly
        // rather than mis-materialising a Void into a register.
        OpKind::ConstNone => "const_none".into(),
        // RPython: getfield_gc_i, getfield_gc_r, getfield_gc_f and `_pure`
        // variants from jtransform.py rewrite_op_getfield().
        OpKind::FieldRead { ty, pure, .. } => {
            let mut opname = format!("getfield_gc_{}", value_type_to_kind(ty));
            if *pure {
                opname.push_str("_pure");
            }
            opname
        }
        OpKind::FieldWrite { ty, .. } => format!("setfield_gc_{}", value_type_to_kind(ty)),
        OpKind::New { .. } => "new".into(),
        OpKind::NewWithVtable { .. } => "new_with_vtable".into(),
        // `NewArrayClear` (`new_array_clear/id>r`) and `NewListClear`
        // (`newlist_clear/idddd>r`) both carry descriptor operands and are
        // encoded by their dedicated `encode_op` arms, never the
        // descriptor-less default path that calls this helper — these arms
        // exist only to keep the opname map exhaustive.
        OpKind::NewArray { .. } => "new_array".into(),
        OpKind::NewArrayClear { .. } => "new_array_clear".into(),
        OpKind::NewListClear { .. } => "newlist_clear".into(),
        // RPython: getarrayitem_gc_i etc.
        OpKind::ArrayRead { item_ty, .. } => {
            format!("getarrayitem_gc_{}", value_type_to_kind(item_ty))
        }
        OpKind::ArrayWrite { item_ty, .. } => {
            format!("setarrayitem_gc_{}", value_type_to_kind(item_ty))
        }
        OpKind::ArrayLen { .. } => "arraylen_gc".into(),
        // RPython: getinteriorfield_gc_i etc.
        OpKind::InteriorFieldRead { item_ty, .. } => {
            format!("getinteriorfield_gc_{}", value_type_to_kind(item_ty))
        }
        OpKind::InteriorFieldWrite { item_ty, .. } => {
            format!("setinteriorfield_gc_{}", value_type_to_kind(item_ty))
        }
        OpKind::Call { result_ty, .. } => {
            format!("direct_call_{}", value_type_to_kind(result_ty))
        }
        OpKind::GuardTrue { .. } => "guard_true".into(),
        OpKind::GuardFalse { .. } => "guard_false".into(),
        // jtransform.py handle_getfield_typeptr: `SpaceOperation('guard_class', [op.args[0]], op.result)`.
        OpKind::GuardClass { .. } => "guard_class".into(),
        OpKind::GuardValue { kind_char, .. } => {
            // `rpython/jit/codewriter/jtransform.py:611` emits one of
            // `int_guard_value` / `ref_guard_value` / `float_guard_value`
            // depending on the `getkind()` of the guarded arg.
            //
            // RPython also emits `str_guard_value` for `promote_string`
            // (jit.py) / `promote_unicode` (jit.py), but pyre
            // panics in those rewrite arms (pyre-object lacks an
            // `rstr.STR` / `rstr.UNICODE` GC layout) so `kind_char` is
            // always one of `'i'` / `'r'` / `'f'` here.
            format!("{}_guard_value", kind_char_to_name(*kind_char))
        }
        // RPython: getfield_vable_i, getfield_vable_r, getfield_vable_f
        OpKind::VableFieldRead { ty, .. } => {
            format!("getfield_vable_{}", value_type_to_kind(ty))
        }
        OpKind::VableFieldWrite { ty, .. } => {
            format!("setfield_vable_{}", value_type_to_kind(ty))
        }
        // RPython: getarrayitem_vable_i etc.
        OpKind::VableArrayRead { item_ty, .. } => {
            format!("getarrayitem_vable_{}", value_type_to_kind(item_ty))
        }
        OpKind::VableArrayWrite { item_ty, .. } => {
            format!("setarrayitem_vable_{}", value_type_to_kind(item_ty))
        }
        // `jtransform.py:814-817` — one opname with no kind suffix: a
        // length is a `Signed` whatever the element kind is.
        OpKind::VableArrayLen { .. } => "arraylen_vable".into(),
        // RPython `blackhole.py bhimpl_int_and` canonical opnames for bitwise ints are
        // `int_and` / `int_or` / `int_xor`. When an `OpKind::BinOp.op`
        // arrives spelled with Rust's `syn::BinOp` trait names
        // (`bitand`/`bitor`/`bitxor`) for source faithfulness, rename them
        // here at the emission boundary instead of duplicating wire entries
        // in the blackhole dispatch table.
        OpKind::BinOp { op, .. } => match op.as_str() {
            "bitand" => "int_and".into(),
            "bitor" => "int_or".into(),
            "bitxor" => "int_xor".into(),
            // RPython `jtransform.py rewrite_op_ptr_eq` produces these opnames as-is —
            // do not prefix with `int_`.
            "ptr_eq" | "ptr_ne" | "instance_ptr_eq" | "instance_ptr_ne" => op.clone(),
            // jtransform-rewritten float operands carry the full RPython
            // opname (`float_add` / `float_lt` / etc.) — preserve as-is.
            s if s.starts_with("float_") => op.clone(),
            // Already-canonical unsigned low-level ops the front-end emits
            // directly (`uint_mul_high` / `uint_lt` from
            // `front::checked_arith_uint`); pass through so the `int_` prefix
            // does not double up (`int_uint_lt` has no `insns` entry).
            s if s.starts_with("uint_") => op.clone(),
            // Canonical llop names (`int_add`, `int_mul`, `int_lt`,
            // `int_floordiv`) are already final. Prefixing again emits
            // `int_int_*`, which blackhole.py does not dispatch.
            s if s.starts_with("int_") => op.clone(),
            _ => format!("int_{op}"),
        },
        // RPython `blackhole.py`: bitwise NOT on i64 is `int_invert`.
        // pyre's front-end uses Rust's `syn::UnOp::Not` spelling `not` for
        // both logical-not and bitwise-not (they share the `!` token at the
        // AST level); canonicalize to `int_invert` at the emission boundary.
        //
        // `bool` (truthify) lacks per-operand-kind dispatch here — see
        // [`op_kind_to_opname_with_kinds`] which routes `bool` to
        // `int_is_true` / `ptr_nonzero` / `float_ne` based on the
        // operand's actual register-class.
        OpKind::UnaryOp { op, .. } => match op.as_str() {
            "not" => "int_invert".into(),
            // Already-canonical opnames produced by the rtyper / cast
            // family / jtransform rewrites — preserve as-is so the
            // unconditional `int_` prefix below does not double up
            // (`int_is_true` would otherwise become `int_int_is_true`,
            // which has no blackhole handler).  RPython's
            // `rint.py:rtype_int__Bool` emits `int_is_true` directly,
            // and `ptr_nonzero` / `same_as` / cast_* are similarly
            // already-canonical.
            "int_is_true" | "ptr_nonzero" | "ptr_iszero" | "same_as" => op.clone(),
            s if s.starts_with("int_") || s.starts_with("uint_") => op.clone(),
            s if s.starts_with("float_") || s.starts_with("cast_") => op.clone(),
            // rfloat.py `float2longlong` / `longlong2float` emit these
            // llops by name.  The `int_` fallthrough would mint
            // `int_convert_*`, which has no blackhole handler and no
            // walker arm (pyjitpl.py's generated unary loop spells the
            // unprefixed convert).
            s if s.starts_with("convert_") => op.clone(),
            _ => format!("int_{op}"),
        },
        OpKind::VableForce { .. } => "hint_force_virtualizable".into(),
        // `Hint` is consumed by `jtransform::rewrite_op_hint` (lowered to
        // `same_as` / `VableForce` / `*_guard_value`) before assembly, so it
        // never reaches opname emission.
        OpKind::Hint { .. } => {
            unreachable!("OpKind::Hint must be rewritten by jtransform before assembly")
        }
        // jtransform.py:1731-1743 — jit.* builtin ops
        OpKind::JitDebug { .. } => "jit_debug".into(),
        OpKind::AssertGreen { kind_char, .. } => {
            format!("{}_assert_green", kind_char_to_name(*kind_char))
        }
        OpKind::CurrentTraceLength => "current_trace_length".into(),
        OpKind::IsConstant { kind_char, .. } => {
            format!("{}_isconstant", kind_char_to_name(*kind_char))
        }
        OpKind::IsVirtual { kind_char, .. } => {
            format!("{}_isvirtual", kind_char_to_name(*kind_char))
        }
        // The rtyper (in `rtyper.rs`) dispatches the flowspace
        // `"isinstance"` opname through `InstanceRepr::rtype_isinstance`,
        // which lowers it into a direct_call to the per-class
        // `ll_isinstance_const_*` or unspecialised `ll_isinstance`
        // helper.  By the time the codewriter assembler runs every
        // `OpKind::IsInstance` should already have been replaced with
        // that direct_call, so reaching this arm signals an rtyper
        // gap.  Fail loudly rather than emit an `"isinstance"` opname
        // with no corresponding entry in `insns.rs::wellknown_bh_insns`.
        OpKind::IsInstance { .. } => panic!(
            "OpKind::IsInstance reached the JitCode assembler; the rtyper \
             must lower isinstance to ll_isinstance* direct_call before \
             codewriter emission (rtyper.rs:2035 / \
             InstanceRepr::rtype_isinstance)"
        ),
        OpKind::RecordKnownResult { result_kind, .. } => {
            format!("record_known_result_{result_kind}")
        }
        // jtransform.py `_rewrite_op_cond_call` — dump prefix. Dedicated
        // encode_op arms emit the full `conditional_call_ir_v` /
        // `conditional_call_value_ir_{k}` keys; this is the debug name.
        OpKind::ConditionalCall { .. } => "conditional_call".into(),
        OpKind::ConditionalCallValue { result_kind, .. } => {
            format!("conditional_call_value_ir_{result_kind}")
        }
        OpKind::Live => "live".into(),
        // jtransform.py:1707,1718 — jit_merge_point / loop_header markers.
        OpKind::JitMergePoint { .. } => "jit_merge_point".into(),
        OpKind::LoopHeader { .. } => "loop_header".into(),
        // Call variants are handled by encode_op directly, not here.
        OpKind::CallElidable { .. } => "call_elidable".into(),
        OpKind::CallResidual { .. } => "residual_call".into(),
        OpKind::CallMayForce { .. } => "call_may_force".into(),
        OpKind::InlineCall { .. } => "inline_call".into(),
        OpKind::RecursiveCall { .. } => "recursive_call".into(),
        // RPython: no dedicated opname — the vtable entry becomes the `funcptr`
        // Variable that `int_guard_value` + `residual_call_*` consume.
        OpKind::VtableMethodPtr { .. } => "vtable_method_ptr".into(),
        OpKind::IndirectCall { .. } => "indirect_call".into(),
        // jtransform.py:901-903 — `record_quasiimmut_field(v_inst, descr, descr1)`.
        OpKind::RecordQuasiImmutField { .. } => "record_quasiimmut_field".into(),
        OpKind::Abort { .. } => "abort".into(),
        OpKind::NewTuple { .. } => "newtuple".into(),
        // High-level `newlist(item0, …, itemN)` does not exist as a jitcode
        // op. `rtype_newlist` (`rlist.py`) expands it to `ll_newlist` plus
        // `ll_setitem_fast`, and `do_fixed_newlist` /
        // `do_fixed_list_setitem` (`jtransform.py`) turn those into
        // `new_array_clear` + `setarrayitem_gc`. `rewrite_fixed_newlist`
        // performs that expansion on the model graph. Reaching this arm
        // would emit `newlist/<r…>r`, which has no blackhole handler.
        OpKind::NewList { .. } => panic!(
            "OpKind::NewList reached the JitCode assembler; jtransform must \
             expand newlist to new_array_clear + setarrayitem_gc \
             (rewrite_fixed_newlist / do_fixed_newlist)"
        ),
        // `getslice` never reaches the assembler in a lifted graph — the
        // rtyper's `rtype_getslice` replaces it with a `direct_call` to the
        // `ll_listslice_*` helper, and the gated front recognizer keeps it
        // out of graphs that drop to the legacy walker.  The opname exists
        // only for exhaustiveness / diagnostics.
        OpKind::GetSlice { .. } => "getslice".into(),
        // The opname-dispatch spine lowers the rtyper helper graphs to
        // register-shaped blackhole insns, carrying the resolved opname
        // (`strlen`/`strgetitem`/`strsetitem`/`newstr`/…) verbatim.  The
        // blackhole interpreter resolves the handler by this name, and
        // `Assembler::get_opnum` assigns its byte dynamically when the
        // opname has no fixed `insns.rs` slot.
        OpKind::LoweredBlackholeOp { opname, .. } => opname.clone(),
        // RPython's codewriter never sees a crate-static carrier:
        // LOAD_GLOBAL is resolved to a Constant before JitCode assembly.
        // The flowspace adapter must either fold `LoadStatic` to a
        // constant `same_as` that rtyper/removenoops eliminates, or reject
        // the unresolved static before this layer.  Emitting `same_as/*`
        // here would create an opcode with no blackhole handler.
        OpKind::LoadStatic { segments, .. } => {
            panic!("unresolved LoadStatic reached JitCode assembly: {segments:?}")
        }
        // `jtransform.py _rewrite_raw_malloc` rewrites a fixed-size raw
        // malloc to a residual direct call before assembly.
        OpKind::RawMalloc { .. } => {
            panic!(
                "raw malloc reached JitCode assembly; jtransform.py _rewrite_raw_malloc rewrites it first"
            )
        }
        // `jtransform.py rewrite_op_free` rewrites raw free before assembly.
        OpKind::RawFree { .. } => {
            panic!(
                "raw free reached JitCode assembly; jtransform.py rewrite_op_free rewrites it first"
            )
        }
    }
}

pub use majit_ir::CallInfoCollection;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{ConstValue, HostObject};
    use crate::regalloc;

    /// `descr.py get_type_flag` of the canonical RustVec header-pointer
    /// spellings is FLAG_UNSIGNED / Int / word, matching `get_type_flag`.
    #[test]
    fn rust_vec_header_pointer_type_flag_is_int_banked() {
        use majit_ir::descr::ArrayFlag;
        use majit_ir::value::Type;

        let word = crate::layout::target_word_size();
        let expected = (ArrayFlag::Unsigned, Type::Int, word);
        for spelling in ["*mut Vec<usize>", "*mut Vec<*mut u8>", "*mut Vec<f64>"] {
            assert_eq!(type_flag_from_str(spelling), expected, "{spelling}");
        }
    }

    /// Index and len of an inline `Vec<u8>` / `Vec<i64>` field are
    /// `getfield` of the measured buffer pointer or length word, then a
    /// headerless array op. `[u8]` stays length-prefixed; the Vec identity
    /// does not.
    #[test]
    fn vec_field_index_and_len_use_buf_and_len_words() {
        use crate::call::{CallControl, StructFieldLayout, StructLayout};
        use crate::model::{FieldDescriptor, ValueType, VecFieldPart};
        use majit_ir::descr::ArrayFlag;
        use majit_ir::value::Type;

        let layout = crate::vec_layout::probe();
        let word = crate::layout::target_word_size();
        assert_ne!(layout.ptr_offset, layout.len_offset);
        assert_ne!(layout.ptr_offset, layout.cap_offset);
        assert!(layout.ptr_offset < 3 * word);
        assert!(layout.len_offset < 3 * word);

        let owner = "vec_component_layout::Holder";
        let owner_id = majit_ir::descr::StructId::from_canonical(owner);
        let _guard =
            crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
                (owner.to_string(), Some(owner_id)),
            ]));
        let mut cc = CallControl::new();
        cc.set_struct_layout(
            owner_id,
            StructLayout {
                size: 6 * word,
                align: word,
                gckind: crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw,
                fields: vec![
                    StructFieldLayout {
                        name: "bytes".into(),
                        offset: 0,
                        size: 3 * word,
                        flag: ArrayFlag::Struct,
                        field_type: Type::Ref,
                        rank: None,
                    },
                    StructFieldLayout {
                        name: "words".into(),
                        offset: 3 * word,
                        size: 3 * word,
                        flag: ArrayFlag::Struct,
                        field_type: Type::Ref,
                        rank: None,
                    },
                ],
                host: None,
                ll_struct: std::cell::RefCell::new(None),
                ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
            },
        );

        let check = |field: &str, field_off: usize, part: VecFieldPart, id: &str, item: usize| {
            let descr = fielddescrof(
                &FieldDescriptor::new(field, Some(owner.into()))
                    .with_owner_id(Some(owner_id))
                    .with_vec_part(part),
                &match part {
                    VecFieldPart::Buf | VecFieldPart::FatData | VecFieldPart::FatVtable => {
                        ValueType::Ref(None)
                    }
                    VecFieldPart::Len | VecFieldPart::FatLen => ValueType::Int,
                },
                Some(&cc),
                true,
            );
            let (offset, size, flag, name) = match descr {
                crate::jitcode::BhDescr::Field {
                    offset,
                    field_size,
                    field_flag,
                    name,
                    ..
                } => (offset, field_size, field_flag, name),
                other => panic!("expected field descr, got {other:?}"),
            };
            let fat = crate::fat_ptr_layout::probe();
            let add = match part {
                VecFieldPart::Buf => layout.ptr_offset,
                VecFieldPart::Len => layout.len_offset,
                VecFieldPart::FatData => fat.data_offset,
                VecFieldPart::FatLen | VecFieldPart::FatVtable => fat.len_offset,
            };
            assert_eq!(offset, field_off + add, "{field} {part:?}");
            assert_eq!(size, word);
            let expect_flag = match part {
                VecFieldPart::Buf | VecFieldPart::FatData | VecFieldPart::FatVtable => {
                    ArrayFlag::Pointer
                }
                VecFieldPart::Len | VecFieldPart::FatLen => ArrayFlag::Unsigned,
            };
            assert_eq!(flag, expect_flag);
            assert!(name.ends_with(match part {
                VecFieldPart::Buf => ".buf",
                VecFieldPart::Len | VecFieldPart::FatLen => ".len",
                VecFieldPart::FatData => ".data",
                VecFieldPart::FatVtable => ".vtable",
            }));
            if matches!(part, VecFieldPart::Buf | VecFieldPart::FatData) {
                assert!(crate::front::typestr::nolength_from_array_type_id(Some(id)));
                let array = arraydescrof(&ValueType::Int, &Some(id.to_string()), None, Some(&cc));
                match array {
                    crate::jitcode::BhDescr::Array {
                        base_size,
                        itemsize,
                        len_offset,
                        ..
                    } => {
                        assert_eq!(base_size, 0, "{id} buffer has no length header");
                        assert_eq!(itemsize, item, "{id}");
                        assert_eq!(len_offset, None);
                    }
                    other => panic!("expected array descr, got {other:?}"),
                }
            }
        };
        check("bytes", 0, VecFieldPart::Buf, "Vec<u8>", 1);
        check("bytes", 0, VecFieldPart::Len, "Vec<u8>", 1);
        check("words", 3 * word, VecFieldPart::Buf, "Vec<i64>", word);
        check("words", 3 * word, VecFieldPart::Len, "Vec<i64>", word);
        check("bytes", 0, VecFieldPart::FatData, "Vec<u8>", 1);
        check("bytes", 0, VecFieldPart::FatLen, "Vec<u8>", 1);
        check("imp", 0, VecFieldPart::FatVtable, "Vec<u8>", 1);
        // A pointer to the Vec (Box<Vec<_>> after the box load) adds nothing
        // but the component offset.
        let boxed = fielddescrof(
            &FieldDescriptor::new("buf", Some("alloc::vec::Vec".into()))
                .with_vec_part(VecFieldPart::Buf),
            &ValueType::Ref(None),
            Some(&cc),
            true,
        );
        match boxed {
            crate::jitcode::BhDescr::Field {
                offset, field_size, ..
            } => {
                assert_eq!(offset, layout.ptr_offset);
                assert_eq!(field_size, word);
            }
            other => panic!("expected field descr, got {other:?}"),
        }
    }

    #[test]
    fn surviving_ref_identity_uses_rpython_ptr_eq_opname() {
        let lhs = crate::flowspace::model::Variable::new();
        let rhs = crate::flowspace::model::Variable::new();
        let identity = crate::model::OpKind::BinOp {
            op: "is_".into(),
            lhs,
            rhs,
            result_ty: crate::model::ValueType::Int,
        };

        assert_eq!(op_kind_to_opname_with_kinds(&identity, "rr"), "ptr_eq");
    }

    #[test]
    fn convert_float_bytes_keeps_the_rpython_llop_name() {
        let operand = crate::flowspace::model::Variable::new();
        for (op, result_ty) in [
            (
                "convert_float_bytes_to_longlong",
                crate::model::ValueType::Int,
            ),
            (
                "convert_longlong_bytes_to_float",
                crate::model::ValueType::Float,
            ),
        ] {
            let kind = crate::model::OpKind::UnaryOp {
                op: op.into(),
                operand: operand.clone(),
                result_ty,
            };
            assert_eq!(op_kind_to_opname(&kind), op);
        }
    }

    #[test]
    fn front_signext_binop_encodes_as_int_signext_ii() {
        use crate::flatten::flatten_graph;
        use crate::model::{FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("signext_i32");
        let value = push_input_var(&mut graph, "value", ValueType::Int);
        let nbytes = push_input_var(&mut graph, "nbytes", ValueType::Int);
        let extended = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "signext".into(),
                    lhs: value.clone(),
                    rhs: nbytes.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(extended.clone()));
        FunctionGraph::set_concretetype_of_inline(&value, crate::model::ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&nbytes, crate::model::ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&extended, crate::model::ConcreteType::Signed);

        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);
        assert!(
            asm.insns.contains_key("int_signext/ii>i"),
            "front signext BinOp must encode as int_signext/ii>i, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        let byte = *asm.insns.get("int_signext/ii>i").expect("int_signext/ii>i");
        assert_eq!(byte, crate::insns::BC_INT_SIGNEXT);
        assert!(
            body.code.contains(&byte),
            "assembled code must contain the int_signext opcode byte"
        );
    }

    #[test]
    fn integer_add_with_a_signed_result_is_int_add_ii() {
        use crate::flatten::flatten_graph;
        use crate::model::{ConcreteType, FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("int_add_signed");
        let lhs = push_input_var(&mut graph, "lhs", ValueType::Int);
        let rhs = push_input_var(&mut graph, "rhs", ValueType::Int);
        let sum = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(sum.clone()));
        FunctionGraph::set_concretetype_of_inline(&lhs, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&sum, ConcreteType::Signed);

        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);
        assert!(
            !asm.insns.contains_key("int_add/ii>r"),
            "an int add must not take a ref result bank: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        let byte = *asm.insns.get("int_add/ii>i").expect("int_add/ii>i");
        assert_eq!(byte, crate::insns::BC_INT_ADD);
        assert!(body.code.contains(&byte));
    }

    #[test]
    fn canonical_int_binop_is_not_double_prefixed() {
        let lhs = crate::flowspace::model::Variable::new();
        let rhs = crate::flowspace::model::Variable::new();
        for name in ["int_add", "int_mul", "int_lt", "int_floordiv"] {
            let kind = crate::model::OpKind::BinOp {
                op: name.into(),
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                result_ty: crate::model::ValueType::Int,
            };
            assert_eq!(op_kind_to_opname(&kind), name);
        }
    }

    #[test]
    fn instance_ptr_eq_is_not_prefixed_as_an_int_binop() {
        let lhs = crate::flowspace::model::Variable::new();
        let rhs = crate::flowspace::model::Variable::new();
        for name in ["instance_ptr_eq", "instance_ptr_ne", "ptr_eq", "ptr_ne"] {
            let kind = crate::model::OpKind::BinOp {
                op: name.into(),
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                result_ty: crate::model::ValueType::Int,
            };
            assert_eq!(op_kind_to_opname(&kind), name);
        }
    }

    #[test]
    fn vable_arraydescrof_pins_the_flat_word_base_for_pointer_items() {
        // `FixedObjectArray`: length word at 0, word-wide pointer items flat
        // at the word. On the host both words are `size_of::<usize>()`.
        let word = crate::layout::target_word_size();
        let descr = vable_arraydescrof(&crate::model::ValueType::Ref(None), word, false);
        let crate::jitcode::BhDescr::Array {
            base_size,
            itemsize,
            len_offset,
            is_array_of_pointers,
            ..
        } = descr
        else {
            panic!("vable_arraydescrof must mint a BhDescr::Array");
        };
        assert_eq!(base_size, word);
        assert_eq!(itemsize, word);
        assert_eq!(len_offset, Some(0));
        assert!(is_array_of_pointers);
    }

    #[test]
    #[should_panic(expected = "exceeds the target word")]
    fn vable_arraydescrof_rejects_items_wider_than_the_word() {
        // A wider-than-word element would be under-aligned by the flat-word
        // base; the mint must fail the build rather than pick a base.
        let word = crate::layout::target_word_size();
        let _ = vable_arraydescrof(&crate::model::ValueType::Int, word * 2, true);
    }

    /// `cpu.arraydescrof(GcArray(OBJECTPTR))`: once the ARRAY is in
    /// `get_array_descr`'s cache, both the codewriter identity and the
    /// published `wrappeditems` layout resolve to that descr, whose GC
    /// tid is nonzero (`init_array_descr`). An identity-less mint stays
    /// at `type_id = 0`.
    #[test]
    fn arraydescrof_object_gcarray_identity_has_nonzero_type_id_when_registered() {
        use crate::call::CallControl;

        let cc = CallControl::new();
        let id = crate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID;
        let first = arraydescrof(
            &crate::model::ValueType::Ref(None),
            &Some(id.to_string()),
            Some(0),
            Some(&cc),
        );
        let crate::jitcode::BhDescr::Array {
            gc_type_id,
            type_id,
            array_type_id,
            ..
        } = first
        else {
            panic!("arraydescrof must mint a BhDescr::Array");
        };
        assert_ne!(
            gc_type_id, 0,
            "registered GcArray(OBJECTPTR) descr must carry a collector tid"
        );
        assert_ne!(
            type_id, 0,
            "ARRAY identity must hash to a nonzero cache key"
        );
        assert_eq!(array_type_id.as_deref(), Some(id));

        let wrapped = arraydescrof(
            &crate::model::ValueType::Ref(None),
            &Some("[*mut PyObject]".to_string()),
            Some(0),
            Some(&cc),
        );
        let crate::jitcode::BhDescr::Array {
            gc_type_id: wrapped_tid,
            type_id: wrapped_key,
            array_type_id: wrapped_id,
            ..
        } = wrapped
        else {
            panic!("arraydescrof of the wrappeditems layout must mint a BhDescr::Array");
        };
        assert_eq!(wrapped_tid, gc_type_id);
        assert_eq!(wrapped_key, type_id);
        assert_eq!(wrapped_id.as_deref(), Some(id));
    }

    #[test]
    fn fielddescrof_uses_struct_id_across_owner_spellings() {
        use crate::call::CallControl;
        use crate::model::FieldDescriptor;

        let mut cc = CallControl::new();
        let mut struct_fields = crate::front::StructFieldRegistry::default();
        for owner in ["core::result::Result", "result::Result"] {
            struct_fields.fields.insert(
                owner.to_string(),
                vec![("__discriminant".to_string(), "i64".to_string())],
            );
        }
        cc.set_struct_fields(struct_fields);
        let template_id = majit_ir::descr::StructId::from_canonical("result::Result");
        let owner_id = template_id.instantiate("<i64,PyError>");
        let parent_type_id = |owner: &str| {
            let field = FieldDescriptor::new("__discriminant", Some(owner.to_string()))
                .with_owner_id(Some(owner_id));
            let crate::jitcode::BhDescr::Field {
                parent: Some(parent),
                ..
            } = fielddescrof(&field, &crate::model::ValueType::Int, Some(&cc), true)
            else {
                panic!("Result discriminant must carry a parent descriptor");
            };
            parent.type_id
        };

        assert_eq!(
            parent_type_id("core::result::Result<i64,PyError>"),
            parent_type_id("result::Result<i64,PyError>"),
        );
        assert_eq!(
            parent_type_id("core::result::Result<i64,PyError>"),
            owner_id.as_u64(),
        );
    }

    use crate::test_support::register_struct_ids_serialized;

    fn standard_shell_struct_ids() -> HashMap<String, Option<majit_ir::descr::StructId>> {
        let option_some = majit_ir::descr::StructId::from_canonical("core::option::Option::Some");
        let result_ok = majit_ir::descr::StructId::from_canonical("core::result::Result::Ok");
        let result_err = majit_ir::descr::StructId::from_canonical("core::result::Result::Err");
        let mut ids = HashMap::new();
        for name in [
            "Option::Some",
            "option::Option::Some",
            "std::option::Option::Some",
            "core::option::Option::Some",
        ] {
            ids.insert(name.to_string(), Some(option_some));
        }
        for (identity, names) in [
            (
                result_ok,
                [
                    "Result::Ok",
                    "result::Result::Ok",
                    "std::result::Result::Ok",
                    "core::result::Result::Ok",
                ],
            ),
            (
                result_err,
                [
                    "Result::Err",
                    "result::Result::Err",
                    "std::result::Result::Err",
                    "core::result::Result::Err",
                ],
            ),
        ] {
            for name in names {
                ids.insert(name.to_string(), Some(identity));
            }
        }
        ids
    }

    #[test]
    fn fielddescrof_resolves_the_slot_when_the_offset_comes_from_the_layout_registry() {
        use crate::call::{CallControl, StructFieldLayout, StructLayout};
        use crate::model::FieldDescriptor;

        let owner = "assembler_fielddescrof_layout_registry_test::Owner";
        let owner_id = majit_ir::descr::StructId::from_canonical(owner);
        let _registry =
            register_struct_ids_serialized(HashMap::from([(owner.to_string(), Some(owner_id))]));

        let mut cc = CallControl::new();
        let mut struct_fields = crate::front::StructFieldRegistry::default();
        struct_fields.fields.insert(
            owner.to_string(),
            vec![
                ("visible_zero".to_string(), "i64".to_string()),
                ("visible_eight".to_string(), "i64".to_string()),
            ],
        );
        cc.set_struct_fields(struct_fields);
        cc.set_struct_layout(
            owner_id,
            StructLayout {
                size: 16,
                align: 8,
                gckind: crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw,
                fields: vec![
                    StructFieldLayout {
                        name: "visible_zero".to_string(),
                        offset: 0,
                        size: 8,
                        flag: majit_ir::descr::ArrayFlag::Signed,
                        field_type: majit_ir::value::Type::Int,
                        rank: None,
                    },
                    StructFieldLayout {
                        name: "visible_eight".to_string(),
                        offset: 8,
                        size: 8,
                        flag: majit_ir::descr::ArrayFlag::Signed,
                        field_type: majit_ir::value::Type::Int,
                        rank: None,
                    },
                    // `bh_size_spec_from_callcontrol` omits void fields from
                    // its flattened list, leaving this layout-only name to
                    // exercise fielddescrof's registry fallback.
                    StructFieldLayout {
                        name: "hidden".to_string(),
                        offset: 8,
                        size: 0,
                        flag: majit_ir::descr::ArrayFlag::Void,
                        field_type: majit_ir::value::Type::Void,
                        rank: None,
                    },
                ],
                host: None,
                ll_struct: std::cell::RefCell::new(None),
                ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
            },
        );

        let crate::jitcode::BhDescr::Field {
            offset,
            index_in_parent,
            parent,
            ..
        } = fielddescrof(
            &FieldDescriptor::new("hidden", Some(owner.to_string())),
            &crate::model::ValueType::Int,
            Some(&cc),
            true,
        )
        else {
            panic!("fielddescrof must produce a field descriptor");
        };

        assert_eq!(offset, 8);
        assert_eq!(index_in_parent, Some(1));
        assert_eq!(
            parent.as_ref().unwrap().all_fielddescrs[index_in_parent.unwrap()].offset,
            offset
        );
    }

    #[test]
    fn fielddescrof_uses_the_scalar_word_when_the_owner_is_unknown() {
        use crate::call::CallControl;
        use crate::model::{FieldDescriptor, ScalarFieldWidth, ScalarFieldWord};

        let owner = "majit_metainterp::pyjitpl::frame::MIFrame";
        let field = FieldDescriptor::new("__pos_3", Some(owner.to_string())).with_scalar_word(
            ScalarFieldWord {
                offset: 392,
                size: 8,
                width: ScalarFieldWidth::Unsigned,
            },
        );
        let crate::jitcode::BhDescr::Field {
            offset,
            field_size,
            field_type,
            field_flag,
            is_field_signed,
            index_in_parent,
            ..
        } = fielddescrof(
            &field,
            &crate::model::ValueType::Unsigned,
            Some(&CallControl::new()),
            true,
        )
        else {
            panic!("fielddescrof must produce a field descriptor");
        };
        assert_eq!(offset, 392);
        assert_eq!(field_size, 8);
        assert!(!is_field_signed);
        assert_eq!(field_type, majit_ir::value::Type::Int);
        assert_eq!(field_flag, majit_ir::descr::ArrayFlag::Unsigned);
        assert_eq!(index_in_parent, None);
    }

    #[test]
    fn fielddescrof_keeps_a_named_field_ahead_of_the_scalar_word() {
        use crate::call::CallControl;
        use crate::model::{FieldDescriptor, ScalarFieldWidth, ScalarFieldWord};

        let owner = "majit_metainterp::pyjitpl::frame::MIFrame";
        let mut cc = CallControl::new();
        let mut struct_fields = crate::front::StructFieldRegistry::default();
        struct_fields.fields.insert(
            owner.to_string(),
            vec![("code_cursor".to_string(), "usize".to_string())],
        );
        cc.set_struct_fields(struct_fields);
        let field = FieldDescriptor::new("code_cursor", Some(owner.to_string())).with_scalar_word(
            ScalarFieldWord {
                offset: 392,
                size: 8,
                width: ScalarFieldWidth::Unsigned,
            },
        );
        let crate::jitcode::BhDescr::Field {
            offset, field_size, ..
        } = fielddescrof(&field, &crate::model::ValueType::Unsigned, Some(&cc), true)
        else {
            panic!("fielddescrof must produce a field descriptor");
        };
        assert_eq!(offset, 0);
        assert_eq!(field_size, crate::layout::target_word_size());
    }

    /// The pair the mint census exists for: two `fielddescrof` calls whose
    /// `index_in_parent` is the SAME NUMBER and means opposite things.
    ///
    /// The first half is the load-bearing one. Both descrs carry the INTEGER
    /// `0` — one because no lookup arm ever ran, one because the field really
    /// is at slot 0 — so any counter that reads only the value
    /// (`census_attached_index`, `derive_index_in_parent`,
    /// `census_spec_positions`) is blind to the difference by construction, not
    /// by oversight. Asserting that the integers agree is what makes the rest
    /// mean anything: a census that splits
    /// two already-distinguishable cases would prove nothing.
    #[test]
    fn mint_census_splits_a_placeholder_index_from_a_real_slot_zero() {
        use crate::call::{CallControl, StructFieldLayout, StructLayout};
        use crate::model::FieldDescriptor;

        let owner = "assembler_mint_census_test::Owner";
        let owner_id = majit_ir::descr::StructId::from_canonical(owner);
        let _registry =
            register_struct_ids_serialized(HashMap::from([(owner.to_string(), Some(owner_id))]));

        let mut cc = CallControl::new();
        let mut struct_fields = crate::front::StructFieldRegistry::default();
        struct_fields.fields.insert(
            owner.to_string(),
            vec![("visible_zero".to_string(), "i64".to_string())],
        );
        cc.set_struct_fields(struct_fields);
        cc.set_struct_layout(
            owner_id,
            StructLayout {
                size: 8,
                align: 8,
                gckind: crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw,
                fields: vec![StructFieldLayout {
                    name: "visible_zero".to_string(),
                    offset: 0,
                    size: 8,
                    flag: majit_ir::descr::ArrayFlag::Signed,
                    field_type: majit_ir::value::Type::Int,
                    rank: None,
                }],
                host: None,
                ll_struct: std::cell::RefCell::new(None),
                ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
            },
        );

        let index_of = |descr: crate::jitcode::BhDescr| {
            let crate::jitcode::BhDescr::Field {
                index_in_parent, ..
            } = descr
            else {
                panic!("fielddescrof must produce a field descriptor");
            };
            index_in_parent
        };

        // A real slot-0 claim: the name is in the flattened list at position 0,
        // so the `Parent` arm writes the index.
        let before = majit_ir::descr::GcCache::mint_index_census();
        let claimed_index = index_of(fielddescrof(
            &FieldDescriptor::new("visible_zero", Some(owner.to_string())),
            &crate::model::ValueType::Int,
            Some(&cc),
            true,
        ));
        let after_claimed = majit_ir::descr::GcCache::mint_index_census();

        // A placeholder: with no `CallControl` the whole lookup block is
        // skipped, so nothing ever writes the index and the initialiser rides
        // out on the descr.
        let placeholder_index = index_of(fielddescrof(
            &FieldDescriptor::new("visible_zero", Some(owner.to_string())),
            &crate::model::ValueType::Int,
            None,
            true,
        ));
        let after_placeholder = majit_ir::descr::GcCache::mint_index_census();

        // The two mints must land on the SAME integer and still be separable,
        // or neither half of this test is testing anything: identical integers
        // are what made the population underivable, and the `Option` is what
        // makes them tellable apart. Assert both, in that order.
        assert_eq!(
            claimed_index.unwrap_or(0),
            placeholder_index.unwrap_or(0),
            "the two mints must agree on the integer, or this pair is splitting \
             something that was already visible without the provenance bit"
        );
        assert_eq!(claimed_index, Some(0), "the lookup arm resolved slot 0");
        assert_eq!(
            placeholder_index, None,
            "a mint whose lookup block never ran must carry no claim; if this \
             is `Some(0)` the provenance was flattened somewhere on the way out"
        );

        // And separable on the census: each call moved its own arm.
        //
        // `>=`, not `==`, and the reason is measured rather than defensive —
        // the counters are process-global and other tests in this binary mint
        // field descrs concurrently, so an exact delta read `[3, 0]` where the
        // call itself contributed `[1, 0]`. Concurrency can only ADD, so a
        // lower bound on the arm the call must move stays sound: an instrument
        // wired to one constant arm, or wired to the wrong one, leaves the
        // expected arm at zero and fails here.
        //
        // What this cannot assert, for the same reason: that the call did
        // NOT also move the other arm. No assertion over a shared counter can,
        // while the suite runs in parallel. The deterministic half above is the
        // one that does not depend on it.
        assert!(
            after_claimed[0] - before[0] >= 1,
            "a resolved slot must count as claimed"
        );
        assert!(
            after_placeholder[1] - after_claimed[1] >= 1,
            "a mint that resolved nothing must count as a placeholder"
        );
    }

    #[test]
    fn explicit_result_variant_size_inherits_discriminant_slot() {
        use crate::call::CallControl;

        let _registry = register_struct_ids_serialized(standard_shell_struct_ids());
        let mut cc = CallControl::new();
        let mut struct_fields = crate::front::StructFieldRegistry::default();
        struct_fields.fields.insert(
            "core::result::Result::Ok".to_string(),
            vec![("__pos_0".to_string(), "??TypeVar".to_string())],
        );
        cc.set_struct_fields(struct_fields);

        let spec = bh_size_spec_from_callcontrol(&cc, "core::result::Result<i64,PyError>::Ok")
            .expect("explicit Result variant size");
        assert_eq!(spec.size, 16);
        assert_eq!(spec.all_fielddescrs.len(), 2);
        assert_eq!(spec.all_fielddescrs[0].field_key(), "__discriminant");
        assert_eq!(spec.all_fielddescrs[0].offset, 0);
        assert_eq!(spec.all_fielddescrs[0].index_in_parent, 0);
        assert_eq!(spec.all_fielddescrs[1].field_key(), "__pos_0");
        assert_eq!(spec.all_fielddescrs[1].index_in_parent, 1);
    }

    #[test]
    fn instantiated_result_payload_follows_template_discriminant_slot() {
        use crate::call::CallControl;

        let _registry = register_struct_ids_serialized(standard_shell_struct_ids());
        let owner = "core::result::Result<*mut PyObject,PyError>::Ok";
        let mut cc = CallControl::new();
        let mut struct_fields = crate::front::StructFieldRegistry::default();
        // The production registry keeps the enum-root tag on the template,
        // while the concrete payload row belongs to the instantiation.
        struct_fields.fields.insert(
            "core::result::Result::Ok".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        struct_fields.fields.insert(
            owner.to_string(),
            vec![("__pos_0".to_string(), "*mut PyObject".to_string())],
        );
        cc.set_struct_fields(struct_fields);

        let spec =
            bh_size_spec_from_callcontrol(&cc, owner).expect("instantiated Result variant size");
        assert_eq!(spec.size, 16);
        assert_eq!(spec.all_fielddescrs.len(), 2);
        assert_eq!(spec.all_fielddescrs[0].field_key(), "__discriminant");
        assert_eq!(spec.all_fielddescrs[0].offset, 0);
        assert_eq!(spec.all_fielddescrs[0].index_in_parent, 0);
        assert_eq!(spec.all_fielddescrs[1].field_key(), "__pos_0");
        assert_eq!(spec.all_fielddescrs[1].offset, 8);
        assert_eq!(spec.all_fielddescrs[1].field_size, 8);
        assert_eq!(
            spec.all_fielddescrs[1].field_flag,
            majit_ir::descr::ArrayFlag::Pointer
        );
        assert_eq!(spec.all_fielddescrs[1].index_in_parent, 1);
    }

    #[test]
    fn instantiated_sum_allocation_keeps_its_concrete_struct_identity() {
        use crate::call::CallControl;

        let owner = "core::option::Option<i64>::Some";
        let template_name = "core::option::Option::Some";
        let template_id = majit_ir::descr::StructId::from_canonical(template_name);
        let concrete_id = template_id.instantiate("<i64>");
        let _registry = register_struct_ids_serialized(standard_shell_struct_ids());
        let mut cc = CallControl::new();
        let mut struct_fields = crate::front::StructFieldRegistry::default();
        struct_fields.fields.insert(
            template_name.to_string(),
            vec![("__pos_0".to_string(), "i64".to_string())],
        );
        cc.set_struct_fields(struct_fields);

        let spec =
            bh_size_spec_from_callcontrol(&cc, owner).expect("instantiated Option variant size");
        assert_eq!(spec.type_id, concrete_id.as_u64());
        assert_eq!(spec.all_fielddescrs.len(), 2);
    }

    #[test]
    fn native_enum_variant_inherited_tag_uses_the_canonical_base_path() {
        use crate::call::CallControl;

        let _registry = crate::test_support::register_struct_registries_serialized(
            HashMap::new(),
            HashMap::from([
                ("BinOperand".to_string(), "grain::bytecode::op".to_string()),
                ("Union".to_string(), "types::dynamic".to_string()),
            ]),
        );
        let mut cc = CallControl::new();
        let mut struct_fields = crate::front::StructFieldRegistry::default();
        for (base, variant) in [
            (
                "rhai::grain::bytecode::op::BinOperand",
                "rhai::grain::bytecode::op::BinOperand::Local",
            ),
            ("Union", "Union::Int"),
        ] {
            struct_fields.fields.insert(
                base.to_string(),
                vec![("__discriminant".to_string(), "u8".to_string())],
            );
            struct_fields.fields.insert(
                variant.to_string(),
                vec![("__pos_0".to_string(), "i64".to_string())],
            );
        }
        cc.set_struct_fields(struct_fields);

        let bin_operand =
            bh_size_spec_from_callcontrol(&cc, "rhai::grain::bytecode::op::BinOperand::Local")
                .expect("BinOperand variant size");
        assert_eq!(
            bin_operand.all_fielddescrs[0].name,
            "grain::bytecode::op::BinOperand::Local.__discriminant"
        );

        let union = bh_size_spec_from_callcontrol(&cc, "Union::Int").expect("Union variant size");
        assert_eq!(
            union.all_fielddescrs[0].name,
            "types::dynamic::Union::Int.__discriminant"
        );
    }

    /// A user enum whose path merely ends in `option::Option::Some` gets no
    /// shell: `front::option_ctor` never writes a tag for it, so its payload
    /// stays where its own registered layout puts it.
    #[test]
    fn a_lookalike_variant_owner_keeps_its_native_layout() {
        use crate::call::CallControl;

        let owner = "option::Option<*mut PyObject>::Some";
        let mut ids = standard_shell_struct_ids();
        ids.insert(
            "option::Option::Some".to_string(),
            Some(majit_ir::descr::StructId::from_canonical(
                "user_crate::option::Option::Some",
            )),
        );
        let _registry = register_struct_ids_serialized(ids);
        let mut cc = CallControl::new();
        let mut struct_fields = crate::front::StructFieldRegistry::default();
        struct_fields.fields.insert(
            "option::Option::Some".to_string(),
            vec![("__pos_0".to_string(), "*mut PyObject".to_string())],
        );
        cc.set_struct_fields(struct_fields);

        let spec = bh_size_spec_from_callcontrol(&cc, owner).expect("lookalike variant size");
        assert_eq!(spec.size, 8);
        assert_eq!(spec.all_fielddescrs.len(), 1);
        assert_eq!(spec.all_fielddescrs[0].field_key(), "__pos_0");
        assert_eq!(spec.all_fielddescrs[0].offset, 0);
        assert_eq!(spec.all_fielddescrs[0].index_in_parent, 0);
    }

    /// A shelled variant that is not `Result`/`Option` carries the inherited
    /// tag as one of its own registry rows, so the generic walker produces the
    /// flattened `heaptracker.all_fielddescrs` list — base field first, payload
    /// after it — without the reconstruction `is_explicit_shell_variant_owner`
    /// gates.  The payload must not claim slot 0: that is the slot the tag's
    /// own descr claims, and a virtual keyed on `index_in_parent` would let the
    /// payload store replace the tag store.
    #[test]
    fn a_shelled_variant_row_places_the_payload_past_the_inherited_tag() {
        use crate::call::CallControl;

        let owner = "pyopcode::StepResult::CloseLoop";
        let mut cc = CallControl::new();
        let mut struct_fields = crate::front::StructFieldRegistry::default();
        struct_fields.fields.insert(
            "pyopcode::StepResult".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        struct_fields.fields.insert(
            owner.to_string(),
            vec![
                ("__discriminant".to_string(), "i64".to_string()),
                ("loop_header_pc".to_string(), "i64".to_string()),
            ],
        );
        cc.set_struct_fields(struct_fields);

        assert!(
            !is_explicit_shell_variant_owner(owner),
            "this owner is served by the registry row, not by the reconstruction"
        );
        let spec = bh_size_spec_from_callcontrol(&cc, owner).expect("shelled variant size");
        assert_eq!(spec.all_fielddescrs.len(), 2);
        assert_eq!(spec.all_fielddescrs[0].field_key(), "__discriminant");
        assert_eq!(spec.all_fielddescrs[0].offset, 0);
        assert_eq!(spec.all_fielddescrs[0].index_in_parent, 0);
        assert_eq!(spec.all_fielddescrs[1].field_key(), "loop_header_pc");
        assert_eq!(spec.all_fielddescrs[1].offset, 8);
        assert_eq!(spec.all_fielddescrs[1].index_in_parent, 1);
    }

    /// The reconstruction gate covers the generic `Result`/`Option` shell
    /// alone, in every spelling `front::mir` mints for it and no other.  It is
    /// needed only there: the enum-root tag stays on the template while the
    /// concrete payload row belongs to the instantiation, so no single registry
    /// entry carries both and the tag slot has to be synthesized.  Every other
    /// shelled variant registers its own inherited `__discriminant` row
    /// (`front/mir.rs`) and takes the generic walker instead —
    /// `a_shelled_variant_row_places_the_payload_past_the_inherited_tag`.
    /// `synthetic_result_ctor_identity_accepts_qualified_spellings` pins the
    /// same list on the producer side.
    #[test]
    fn the_shelled_variant_owners_are_every_rooted_spelling() {
        let _registry = register_struct_ids_serialized(standard_shell_struct_ids());
        for owner in [
            "Result::Ok",
            "result::Result::Ok",
            "std::result::Result::Ok",
            "core::result::Result::Ok",
            "Result::Err",
            "result::Result::Err",
            "std::result::Result::Err",
            "core::result::Result::Err",
            "Option::Some",
            "option::Option::Some",
            "std::option::Option::Some",
            "core::option::Option::Some",
        ] {
            assert!(is_explicit_shell_variant_owner(owner), "{owner}");
        }
        for owner in [
            "Ok",
            "Some",
            "Option::None",
            "StepResult::Continue",
            "mycrate::option::Option::Some",
            "mycrate::core::option::Option::Some",
        ] {
            assert!(!is_explicit_shell_variant_owner(owner), "{owner}");
        }
    }

    #[test]
    fn a_short_standard_library_lookalike_is_not_a_shell_by_spelling() {
        let mut ids = standard_shell_struct_ids();
        ids.insert(
            "option::Option::Some".to_string(),
            Some(majit_ir::descr::StructId::from_canonical(
                "user_crate::option::Option::Some",
            )),
        );
        let _registry = register_struct_ids_serialized(ids);
        assert!(!is_explicit_shell_variant_owner("option::Option::Some"));
        assert!(is_explicit_shell_variant_owner(
            "core::option::Option::Some"
        ));
    }

    #[test]
    fn emit_const_unknown_kind_raises_assembler_error() {
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut asm = Assembler::new();
            let mut state = empty_state();
            let _ = asm.emit_const(&ConstValue::Int(1), 'x', &mut state, None);
        }))
        .expect_err("unsupported constant kind must raise");

        let err = panic
            .downcast_ref::<AssemblerError>()
            .expect("panic payload must be AssemblerError");
        assert!(err.to_string().contains("unknown constant kind"));
    }

    #[test]
    fn emit_const_i_unsupported_value_raises_assembler_error() {
        let panic = std::panic::catch_unwind(|| {
            let _ = Assembler::resolve_const_i_value(&ConstValue::byte_str("nope"), None);
        })
        .expect_err("unsupported int constant must raise");

        let err = panic
            .downcast_ref::<AssemblerError>()
            .expect("panic payload must be AssemblerError");
        assert!(
            err.to_string()
                .contains("integer-kind constant not supported")
        );
    }

    fn empty_regallocs() -> HashMap<RegKind, regalloc::RegAllocator> {
        let mut regallocs = HashMap::new();
        regallocs.insert(
            RegKind::Int,
            regalloc::RegAllocator {
                coloring: regalloc::Coloring::default(),
                num_regs: 0,
            },
        );
        regallocs.insert(
            RegKind::Ref,
            regalloc::RegAllocator {
                coloring: regalloc::Coloring::default(),
                num_regs: 0,
            },
        );
        regallocs.insert(
            RegKind::Float,
            regalloc::RegAllocator {
                coloring: regalloc::Coloring::default(),
                num_regs: 0,
            },
        );
        regallocs
    }

    fn empty_state() -> AssemblyState {
        AssemblyState {
            code: Vec::new(),
            constants_i: Vec::new(),
            constants_r: Vec::new(),
            constants_f: Vec::new(),
            str_consts: Vec::new(),
            unit_variant_consts: Vec::new(),
            exc_instance_consts: Vec::new(),
            type_static_consts: Vec::new(),
            reloc_consts_i: Vec::new(),
            reloc_consts_r: Vec::new(),
            num_regs_i: 4,
            num_regs_r: 0,
            num_regs_f: 0,
            label_positions: HashMap::new(),
            tlabel_fixups: Vec::new(),
            startpoints: indexmap::IndexSet::new(),
            jit_merge_point_offset: None,
            alllabels: indexmap::IndexSet::new(),
            resulttypes: indexmap::IndexMap::new(),
        }
    }

    fn push_input_var(
        graph: &mut crate::model::FunctionGraph,
        name: &str,
        ty: crate::model::ValueType,
    ) -> crate::flowspace::model::Variable {
        let block = graph.startblock;
        let var = graph
            .push_op_var(
                block,
                crate::model::OpKind::Input {
                    name: name.into(),
                    ty,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(block, var.clone());
        var
    }

    #[test]
    fn use_c_form_matches_assembler_py_membership() {
        // `assembler.py` USE_C_FORM members reachable build-time …
        assert!(use_c_form("int_copy"));
        assert!(use_c_form("int_return"));
        assert!(use_c_form("jit_merge_point"));
        // … and non-members that must keep the pooled `i`/`r`/`f` argcode.
        assert!(!use_c_form("ref_copy"));
        assert!(!use_c_form("float_copy"));
        assert!(!use_c_form("getfield_gc_i"));
    }

    #[test]
    fn setfield_of_void_value_encodes_no_opcode() {
        use crate::model::{FunctionGraph, OpKind, SpaceOperation};

        let mut graph = FunctionGraph::new("unit_payload");
        let base = graph.alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
        let unit = graph.alloc_value_var_with_type(crate::model::ConcreteType::Void);
        let op = SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base,
                field: crate::model::FieldDescriptor::new("payload", None),
                value: crate::model::LinkArg::Value(unit),
                ty: crate::model::ValueType::Void,
            },
        };
        let mut asm = Assembler::new();
        let mut state = empty_state();
        asm.encode_op(&op, &HashMap::new(), &mut state, None);
        assert!(state.code.is_empty());
    }

    /// `jtransform.py _rewrite_op_cond_call` void form: condition, funcptr
    /// int const, I list, R list, calldescr. Canonical key
    /// `conditional_call_ir_v/iiIRd`.
    #[test]
    fn assemble_conditional_call_ir_v_emits_canonical_iiird_bytes() {
        use crate::call::CallDescriptor;
        use crate::model::{CallTarget, FunctionGraph, OpKind, SpaceOperation};
        use majit_ir::effectinfo::EffectInfo;
        use majit_ir::value::Type;

        let condition = crate::flowspace::model::Variable::new();
        FunctionGraph::set_concretetype_of_inline(&condition, crate::model::ConcreteType::Signed);

        let mut regallocs = empty_regallocs();
        let int_ra = regallocs.get_mut(&RegKind::Int).unwrap();
        int_ra.coloring.insert(condition.clone(), 0);
        int_ra.num_regs = 1;

        let op = SpaceOperation {
            result: None,
            kind: OpKind::ConditionalCall {
                condition: condition.clone(),
                funcptr: CallTarget::function_path([
                    "pyre_object",
                    "listobject",
                    "ll_list_int_resize_hint_really",
                ]),
                descriptor: CallDescriptor::from_signature(
                    &[Type::Ref],
                    Type::Void,
                    EffectInfo::default(),
                ),
                args_i: vec![],
                args_r: vec![],
                args_f: vec![],
            },
        };
        let mut asm = Assembler::new();
        let mut state = empty_state();
        asm.encode_op(&op, &regallocs, &mut state, None);

        let key = "conditional_call_ir_v/iiIRd";
        assert!(
            asm.insns.contains_key(key),
            "expected {key}, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        // opcode, cond, func const, I count, R count, descr u16
        assert_eq!(state.code.len(), 7);
        assert_eq!(state.code[0], asm.insns[key]);
        assert_eq!(state.code[1], 0);
        assert_eq!(state.code[2], state.num_regs_i as u8);
        assert_eq!(state.code[3], 0);
        assert_eq!(state.code[4], 0);
        assert_eq!(u16::from_le_bytes([state.code[5], state.code[6]]), 0);
        assert_eq!(state.reloc_consts_i.len(), 1);
        assert_eq!(state.reloc_consts_i[0].constants_i_index, 0);
        assert!(
            matches!(
                &state.reloc_consts_i[0].kind,
                ConstIRelocKind::FnAddr { symbolic: true, .. }
            ),
            "conditional_call funcptr must take the FnAddr constant channel, got {:?}",
            state.reloc_consts_i[0].kind
        );
    }

    /// `jtransform.py _rewrite_op_cond_call` value form /
    /// `bhimpl_conditional_call_value_ir_r`: value, funcptr int const,
    /// I list, R list, 2 descr bytes, result. Canonical key
    /// `conditional_call_value_ir_r/riIRd>r`.
    #[test]
    fn assemble_conditional_call_value_ir_r_emits_canonical_riird_bytes() {
        use crate::call::CallDescriptor;
        use crate::model::{CallTarget, FunctionGraph, OpKind, SpaceOperation};
        use majit_ir::effectinfo::EffectInfo;
        use majit_ir::value::Type;

        let value = crate::flowspace::model::Variable::new();
        let arg = crate::flowspace::model::Variable::new();
        let result = crate::flowspace::model::Variable::new();
        FunctionGraph::set_concretetype_of_inline(&value, crate::model::ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&arg, crate::model::ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&result, crate::model::ConcreteType::GcRef);

        let mut regallocs = empty_regallocs();
        let ref_ra = regallocs.get_mut(&RegKind::Ref).unwrap();
        ref_ra.coloring.insert(value.clone(), 0);
        ref_ra.coloring.insert(arg.clone(), 1);
        ref_ra.coloring.insert(result.clone(), 2);
        ref_ra.num_regs = 3;

        let op = SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::ConditionalCallValue {
                value: value.clone(),
                funcptr: CallTarget::function_path([
                    "pyre_object",
                    "unicodeobject",
                    "w_str_compute_index_storage",
                ]),
                descriptor: CallDescriptor::from_signature(
                    &[Type::Ref],
                    Type::Ref,
                    EffectInfo::default(),
                ),
                args_i: vec![],
                args_r: vec![arg.clone()],
                args_f: vec![],
                result_kind: 'r',
            },
        };
        let mut asm = Assembler::new();
        let mut state = empty_state();
        asm.encode_op(&op, &regallocs, &mut state, None);

        let key = "conditional_call_value_ir_r/riIRd>r";
        assert!(
            asm.insns.contains_key(key),
            "expected {key}, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        // opcode, value, func const, I count, R count, r0, descr u16, result
        assert_eq!(state.code.len(), 9);
        assert_eq!(state.code[0], asm.insns[key]);
        assert_eq!(state.code[1], 0);
        assert_eq!(state.code[2], state.num_regs_i as u8);
        assert_eq!(state.code[3], 0);
        assert_eq!(state.code[4], 1);
        assert_eq!(state.code[5], 1);
        assert_eq!(u16::from_le_bytes([state.code[6], state.code[7]]), 0);
        assert_eq!(state.code[8], 2);
        assert_eq!(state.reloc_consts_i.len(), 1);
        assert_eq!(state.reloc_consts_i[0].constants_i_index, 0);
        assert!(
            matches!(
                &state.reloc_consts_i[0].kind,
                ConstIRelocKind::FnAddr { symbolic: true, .. }
            ),
            "conditional_call_value funcptr must take the FnAddr constant channel, got {:?}",
            state.reloc_consts_i[0].kind
        );
    }

    /// Int-result twin of `assemble_conditional_call_value_ir_r_*`.
    /// Canonical key `conditional_call_value_ir_i/iiIRd>i`.
    #[test]
    fn assemble_conditional_call_value_ir_i_emits_canonical_iiird_bytes() {
        use crate::call::CallDescriptor;
        use crate::model::{CallTarget, FunctionGraph, OpKind, SpaceOperation};
        use majit_ir::effectinfo::EffectInfo;
        use majit_ir::value::Type;

        let value = crate::flowspace::model::Variable::new();
        let arg = crate::flowspace::model::Variable::new();
        let result = crate::flowspace::model::Variable::new();
        FunctionGraph::set_concretetype_of_inline(&value, crate::model::ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&arg, crate::model::ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&result, crate::model::ConcreteType::Signed);

        let mut regallocs = empty_regallocs();
        let int_ra = regallocs.get_mut(&RegKind::Int).unwrap();
        int_ra.coloring.insert(value.clone(), 0);
        int_ra.coloring.insert(arg.clone(), 1);
        int_ra.coloring.insert(result.clone(), 2);
        int_ra.num_regs = 3;

        let op = SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::ConditionalCallValue {
                value: value.clone(),
                funcptr: CallTarget::function_path(["rstr", "ll_strhash"]),
                descriptor: CallDescriptor::from_signature(
                    &[Type::Int],
                    Type::Int,
                    EffectInfo::default(),
                ),
                args_i: vec![arg.clone()],
                args_r: vec![],
                args_f: vec![],
                result_kind: 'i',
            },
        };
        let mut asm = Assembler::new();
        let mut state = empty_state();
        asm.encode_op(&op, &regallocs, &mut state, None);

        let key = "conditional_call_value_ir_i/iiIRd>i";
        assert!(
            asm.insns.contains_key(key),
            "expected {key}, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        // opcode, value, func const, I count, i1, R count, descr u16, result
        assert_eq!(state.code.len(), 9);
        assert_eq!(state.code[0], asm.insns[key]);
        assert_eq!(state.code[1], 0);
        assert_eq!(state.code[2], state.num_regs_i as u8);
        assert_eq!(state.code[3], 1);
        assert_eq!(state.code[4], 1);
        assert_eq!(state.code[5], 0);
        assert_eq!(u16::from_le_bytes([state.code[6], state.code[7]]), 0);
        assert_eq!(state.code[8], 2);
        assert_eq!(state.reloc_consts_i.len(), 1);
        assert!(matches!(
            &state.reloc_consts_i[0].kind,
            ConstIRelocKind::FnAddr { symbolic: true, .. }
        ));
    }

    #[test]
    fn emit_const_i_allow_short_uses_inline_byte_for_small_ints() {
        // `assembler.py:99-107` — with `allow_short`, a signed-i8 value is
        // emitted inline as one byte (argcode `c`) and never pooled.
        let mut asm = Assembler::new();
        let mut state = empty_state();
        assert_eq!(
            asm.emit_const_i_allow_short(5, true, &mut state),
            (5u8, 'c')
        );
        // signed-i8 boundaries: 127 and -128 both stay inline.
        assert_eq!(
            asm.emit_const_i_allow_short(127, true, &mut state),
            (127u8, 'c')
        );
        assert_eq!(
            asm.emit_const_i_allow_short(-128, true, &mut state),
            (128u8, 'c') // -128 as i8 as u8
        );
        assert!(
            state.constants_i.is_empty(),
            "the short form must not touch the int constant pool"
        );

        // Out of signed-i8 range → constant pool, argcode `i`. First pool
        // slot is `num_regs_i + 0`.
        let (byte, argcode) = asm.emit_const_i_allow_short(128, true, &mut state);
        assert_eq!(argcode, 'i');
        assert_eq!(byte, state.num_regs_i as u8);
        assert_eq!(state.constants_i, vec![128]);

        // `allow_short = false` (opname not in USE_C_FORM) pools even small
        // ints — second pool slot is `num_regs_i + 1`.
        let (byte, argcode) = asm.emit_const_i_allow_short(5, false, &mut state);
        assert_eq!(argcode, 'i');
        assert_eq!(byte, (state.num_regs_i + 1) as u8);
        assert_eq!(state.constants_i, vec![128, 5]);
    }

    #[test]
    fn emit_const_i_does_not_share_a_slot_with_a_relocatable_fnaddr() {
        let mut asm = Assembler::new();
        let mut state = empty_state();
        let fnaddr = 0x1000_i64;
        let reloc = asm.emit_const_i_reloc(
            fnaddr,
            ConstIRelocKind::FnAddr {
                path: "helpers::foo".into(),
                symbolic: false,
            },
            &mut state,
        );
        let ordinary = asm.emit_const_i(fnaddr, &mut state);
        assert_ne!(
            reloc, ordinary,
            "an ordinary integer equal to a funcptr must keep its own slot"
        );
        assert_eq!(state.constants_i, vec![fnaddr, fnaddr]);
        assert_eq!(state.reloc_consts_i.len(), 1);
        assert_eq!(state.reloc_consts_i[0].constants_i_index, 0);
        let reloc_again = asm.emit_const_i_reloc(
            fnaddr,
            ConstIRelocKind::FnAddr {
                path: "helpers::foo".into(),
                symbolic: false,
            },
            &mut state,
        );
        assert_eq!(reloc_again, reloc, "same path reuses the reloc slot");
        assert_eq!(state.constants_i.len(), 2);
    }

    #[test]
    fn emit_const_r_does_not_share_a_slot_with_a_relocatable_static() {
        let mut asm = Assembler::new();
        let mut state = empty_state();
        asm.intern_static_ref_addrs(&[("dictstrategy::EMPTY", 0x1000)]);
        let reloc = asm.emit_type_static_or_bits(0x1000, &mut state);
        let ordinary = asm.emit_const_r_bits(0x1000, &mut state);
        assert_ne!(
            reloc, ordinary,
            "an ordinary ref equal to a host-static addr must keep its own slot"
        );
        assert_eq!(state.constants_r, vec![0x1000, 0x1000]);
        assert_eq!(state.reloc_consts_r.len(), 1);
        assert_eq!(state.reloc_consts_r[0].constants_r_index, 0);
        assert_eq!(
            state.reloc_consts_r[0].kind,
            ConstIRelocKind::StaticAddr {
                name: "dictstrategy::EMPTY".into()
            }
        );
        let reloc_again = asm.emit_type_static_or_bits(0x1000, &mut state);
        assert_eq!(reloc_again, reloc, "same name reuses the reloc slot");
        assert_eq!(state.constants_r.len(), 2);
    }

    #[test]
    fn encode_regorconst_source_threads_short_argcode() {
        use crate::flatten::{RegOrConst, Register};
        use crate::flowspace::model::{ConstValue, Constant};

        let mut asm = Assembler::new();
        let mut state = empty_state();
        // A small int Constant in a USE_C_FORM op (`allow_short = true`,
        // e.g. `int_return`/`int_copy`) takes the inline `c` argcode.
        let arg = RegOrConst::Const(Constant::new(ConstValue::Int(42)));
        assert_eq!(
            asm.encode_regorconst_source(&arg, RegKind::Int, true, &mut state, None),
            (42u8, 'c')
        );
        assert!(state.constants_i.is_empty());
        // The same Constant without `allow_short` pools to `i`.
        assert_eq!(
            asm.encode_regorconst_source(&arg, RegKind::Int, false, &mut state, None),
            (state.num_regs_i as u8, 'i')
        );
        assert_eq!(state.constants_i, vec![42]);
        // A register source always takes its kind argcode, never `c`.
        let reg = RegOrConst::Reg(Register::new(RegKind::Int, 3));
        assert_eq!(
            asm.encode_regorconst_source(&reg, RegKind::Int, true, &mut state, None),
            (3u8, 'i')
        );
    }

    #[test]
    fn get_opnum_setdefault_allocates_dynamic_bytes_for_unregistered_keys() {
        // RPython parity: `assembler.py:220 self.insns.setdefault(key,
        // len(self.insns))` allocates a fresh byte when the key has
        // not been seen before.  Pyre skips reserved canonical /
        // extension bytes so those mappings keep their compile-time-
        // stable bytes for the runtime walker, but translator-only
        // keys still get assigned a unique byte instead of panicking.
        let mut asm = Assembler::new();
        let expected_first = (0..=u8::MAX)
            .find(|byte| !crate::insns::is_reserved_opcode_byte(*byte))
            .expect("expected at least one non-reserved opcode byte");
        let first = asm.get_opnum("translator_only_unknown_key/254");
        assert_eq!(
            first, expected_first,
            "first unregistered key should land on the lowest \
             non-reserved byte"
        );
        assert!(!crate::insns::is_reserved_opcode_byte(first));
        let expected_second = ((first as u16 + 1)..=u8::MAX as u16)
            .map(|byte| byte as u8)
            .find(|byte| !crate::insns::is_reserved_opcode_byte(*byte))
            .expect("expected a second non-reserved opcode byte");
        let second = asm.get_opnum("translator_only_other_key/254");
        assert_eq!(second, expected_second);
        assert!(!crate::insns::is_reserved_opcode_byte(second));
        // Re-querying the same key returns the cached byte, matching
        // `setdefault`'s dict semantics.
        let first_again = asm.get_opnum("translator_only_unknown_key/254");
        assert_eq!(first_again, first);
        // Canonical keys keep their reserved bytes regardless of the
        // dynamic counter.
        let live = asm.get_opnum("live/");
        assert_eq!(
            live,
            crate::insns::insn_byte("live/"),
            "canonical keys must keep their reserved BC_* bytes",
        );

        // Typed opname variants that are registered as canonical bytes
        // must resolve to their reserved BC_* values through get_opnum.
        for (key, canonical) in [
            ("int_guard_value/i", "int_guard_value/i"),
            ("ref_guard_value/r", "ref_guard_value/r"),
            ("float_guard_value/f", "float_guard_value/f"),
        ] {
            let byte = asm.get_opnum(key);
            let expected = crate::insns::insn_byte(canonical);
            assert_eq!(
                byte, expected,
                "{key} must map to its canonical BC_* byte {expected}",
            );
            assert!(
                crate::insns::is_reserved_opcode_byte(byte),
                "{key} byte {byte} must be reserved",
            );
        }

        // Typed opnames not in the canonical table get dynamic bytes.
        for key in [
            "int_assert_green/i",
            "ref_assert_green/r",
            "float_assert_green/f",
            "int_isconstant/i",
            "ref_isconstant/r",
            "float_isconstant/f",
            "ref_isvirtual/r",
        ] {
            assert!(
                crate::insns::insn_byte_opt(key).is_none(),
                "{key} should not be in the canonical table",
            );
            let byte = asm.get_opnum(key);
            assert!(
                !crate::insns::is_reserved_opcode_byte(byte),
                "{key} should get a non-reserved dynamic byte, got {byte}",
            );
        }
    }

    #[test]
    fn emit_descr_reuses_rpython_descr_dict_index() {
        let mut asm = Assembler::new();

        let first = asm.emit_ready_descr(crate::jitcode::BhDescr::Field {
            offset: 0,
            field_size: 8,
            field_type: majit_ir::value::Type::Ref,
            field_flag: majit_ir::descr::ArrayFlag::Pointer,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: Some(0),
            parent: None,
            name: "value".into(),
            owner: "Cell".into(),
        });
        let repeated = asm.emit_ready_descr(crate::jitcode::BhDescr::Field {
            offset: 0,
            field_size: 8,
            field_type: majit_ir::value::Type::Ref,
            field_flag: majit_ir::descr::ArrayFlag::Pointer,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: Some(0),
            parent: None,
            name: "value".into(),
            owner: "Cell".into(),
        });
        let other = asm.emit_ready_descr(crate::jitcode::BhDescr::Field {
            offset: 0,
            field_size: 8,
            field_type: majit_ir::value::Type::Ref,
            field_flag: majit_ir::descr::ArrayFlag::Pointer,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: Some(1),
            parent: None,
            name: "mutate_value".into(),
            owner: "Cell".into(),
        });

        assert_eq!(first, repeated);
        assert_ne!(first, other);
        assert_eq!(asm.snapshot_descrs().len(), 2);
    }

    #[test]
    fn emit_call_descr_key_uses_full_rpython_calldescr_shape() {
        let mut asm = Assembler::new();
        let effect = majit_ir::descr::EffectInfo::MOST_GENERAL;

        let signed_int =
            crate::jitcode::BhCallDescr::from_arg_classes("i".to_string(), 'i', effect.clone());
        let signed_int_repeat =
            crate::jitcode::BhCallDescr::from_arg_classes("i".to_string(), 'i', effect.clone());
        let single_float =
            crate::jitcode::BhCallDescr::from_arg_classes("i".to_string(), 'S', effect.clone());
        let unsigned_int = crate::jitcode::BhCallDescr {
            arg_classes: "i".to_string(),
            result_type: 'i',
            result_signed: false,
            result_size: 8,
            result_erased: crate::jitcode::CallResultErasedKey::Unsigned,
            void_word_abi: false,
            extra_info: effect.clone(),
            translated_effect_info_id: None,
            call_stub: std::sync::OnceLock::new(),
        };
        let raw_address = crate::jitcode::BhCallDescr {
            arg_classes: "i".to_string(),
            result_type: 'i',
            result_signed: false,
            result_size: 8,
            result_erased: crate::jitcode::CallResultErasedKey::Address,
            void_word_abi: false,
            extra_info: effect,
            translated_effect_info_id: None,
            call_stub: std::sync::OnceLock::new(),
        };

        assert_eq!(single_float.result_type, 'S');
        assert_eq!(single_float.result_size, 4);
        assert!(!single_float.result_signed);
        assert_eq!(
            single_float.result_erased,
            crate::jitcode::CallResultErasedKey::SingleFloat,
        );

        let signed_idx = asm.emit_ready_descr(crate::jitcode::BhDescr::Call {
            calldescr: signed_int,
        });
        let signed_repeat_idx = asm.emit_ready_descr(crate::jitcode::BhDescr::Call {
            calldescr: signed_int_repeat,
        });
        let single_idx = asm.emit_ready_descr(crate::jitcode::BhDescr::Call {
            calldescr: single_float,
        });
        let unsigned_idx = asm.emit_ready_descr(crate::jitcode::BhDescr::Call {
            calldescr: unsigned_int,
        });
        let address_idx = asm.emit_ready_descr(crate::jitcode::BhDescr::Call {
            calldescr: raw_address,
        });

        assert_eq!(signed_idx, signed_repeat_idx);
        assert_ne!(signed_idx, single_idx);
        assert_ne!(signed_idx, unsigned_idx);
        assert_ne!(unsigned_idx, address_idx);
        assert_eq!(asm.snapshot_descrs().len(), 4);
    }

    #[test]
    fn jit_merge_point_and_loop_header_opnames() {
        // jtransform.py `op1 = SpaceOperation('jit_merge_point', args, None)`
        let merge = crate::model::OpKind::JitMergePoint {
            jitdriver_index: 0,
            greens_i: vec![],
            greens_r: vec![],
            greens_f: vec![],
            reds_i: vec![],
            reds_r: vec![],
            reds_f: vec![],
            green_classes: String::new(),
            red_classes: String::new(),
        };
        assert_eq!(op_kind_to_opname(&merge), "jit_merge_point");
        // jtransform.py `SpaceOperation('loop_header', [c_index], None)`
        let header = crate::model::OpKind::LoopHeader { jitdriver_index: 0 };
        assert_eq!(op_kind_to_opname(&header), "loop_header");
    }

    /// `OpKind::NewListClear` (`opimpl_newlist_clear`, pyjitpl.py)
    /// carries the same `{length, item_ty, array_type_id}` shape as
    /// `NewArrayClear`.  Its length operand must survive the inline var
    /// remap while the descriptor fields (`item_ty`/`array_type_id`) copy
    /// through unchanged, and it must map to the opname `newlist_clear`.
    #[test]
    fn newlist_clear_survives_remap_and_names() {
        use crate::flowspace::model::Variable;
        use crate::model::{OpKind, ValueType};

        let length = Variable::new();
        let remapped_length = Variable::new();
        // The remapped length must be a *distinct* SSA var so the assert
        // proves the length field was rewritten, not merely preserved.
        assert_ne!(length, remapped_length);

        // `array_type_id` is the ITEMS ARRAY identity (`cpu.arraydescrof(
        // ARRAY)`), not the enclosing struct name — use a plausible array id.
        let kind = OpKind::NewListClear {
            length: length.clone(),
            item_ty: ValueType::Ref(None),
            array_type_id: Some("list_items".to_string()),
        };

        // Run it through the inline var remap: length is rewritten via the
        // closure, the descriptor fields clone through unchanged.
        let remapped_length_c = remapped_length.clone();
        let length_c = length.clone();
        let remap = move |v: &Variable| {
            if *v == length_c {
                remapped_length_c.clone()
            } else {
                v.clone()
            }
        };
        let remapped = crate::inline::remap_op_kind(&kind, &remap);
        match &remapped {
            OpKind::NewListClear {
                length: l,
                item_ty,
                array_type_id,
            } => {
                assert_eq!(*l, remapped_length, "length must be rewritten by remap");
                assert!(matches!(item_ty, ValueType::Ref(None)));
                assert_eq!(array_type_id.as_deref(), Some("list_items"));
            }
            other => panic!("expected NewListClear, got {other:?}"),
        }

        // Only the length is an SSA operand; the descrs are not values.
        let refs = crate::inline::op_variable_refs(&kind);
        assert_eq!(refs, vec![length.clone()]);

        // Allocation is never pure — CSE must not coalesce two of them.
        assert!(!crate::inline::is_pure_op(&kind));

        // opname map: `newlist_clear`.
        assert_eq!(op_kind_to_opname(&kind), "newlist_clear");
    }

    /// The offset has to be recorded where the opcode byte goes, and a
    /// consumer has to be able to trust it without re-scanning: an operand
    /// byte may equal the opcode byte, so a scan can land on the wrong
    /// position. The merge point here is deliberately not the first
    /// instruction — an offset of 0 is what a recorder that never ran would
    /// also produce, so a body whose marker sits at 0 cannot tell the two
    /// apart.
    #[test]
    fn a_merge_point_records_the_offset_of_its_own_opcode_byte() {
        use crate::codewriter::insns::{BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C};
        let mut flat = SSARepr {
            name: "portal".into(),
            insns: vec![
                FlatOp::Live {
                    live_values: vec![],
                },
                FlatOp::op(crate::model::SpaceOperation {
                    result: None,
                    kind: crate::model::OpKind::JitMergePoint {
                        jitdriver_index: 0,
                        greens_i: vec![],
                        greens_r: vec![],
                        greens_f: vec![],
                        reds_i: vec![],
                        reds_r: vec![],
                        reds_f: vec![],
                        green_classes: String::new(),
                        red_classes: String::new(),
                    },
                }),
            ],
            num_blocks: 1,
            insns_pos: None,
        };
        let regallocs = empty_regallocs();
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);

        let offset = body
            .jit_merge_point_offset
            .expect("an assembled merge point records where it went");
        assert_ne!(offset, 0, "the `-live-` prefix precedes the marker");
        let opcode = body.code[offset];
        assert!(
            opcode == BC_JIT_MERGE_POINT || opcode == BC_JIT_MERGE_POINT_C,
            "offset {offset} holds {opcode}, not a merge-point opcode",
        );
    }

    /// Every other jitcode answers `None`, so a consumer reading the field
    /// can tell "this body has no marker" from "this body has one at 0".
    #[test]
    fn a_body_without_a_merge_point_records_no_offset() {
        let mut flat = SSARepr {
            name: "callee".into(),
            insns: vec![FlatOp::Live {
                live_values: vec![],
            }],
            num_blocks: 1,
            insns_pos: None,
        };
        let regallocs = empty_regallocs();
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);
        assert_eq!(body.jit_merge_point_offset, None);
    }

    #[test]
    fn assemble_basic() {
        let mut flat = SSARepr {
            name: "test".into(),
            insns: vec![],
            num_blocks: 1,
            insns_pos: None,
        };

        let regallocs = empty_regallocs();
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);

        assert_eq!(flat.name, "test");
        assert_eq!(body.c_num_regs_i as usize, 0);
        assert_eq!(body.c_num_regs_r as usize, 0);
        assert_eq!(body.c_num_regs_f as usize, 0);
        assert_eq!(asm.count_jitcodes(), 1);
    }

    #[test]
    fn assemble_fused_goto_if_not_routes_to_reserved_opcode() {
        // the GotoIfNotOp lowering: a `FlatOp::GotoIfNotOp` assembles through the
        // registered `goto_if_not_<op>/<argcodes>` jitcode key — the
        // per-arg register kinds form the argcodes, and the key must
        // resolve to the reserved opcode the metainterp blackhole
        // dispatches, NOT a freshly-minted dynamic byte (which would
        // have no consumer).
        use crate::flatten::Register;
        let regallocs = empty_regallocs();

        // Binary compare: `int_lt` with two int operands → `iiL`.
        let mut flat = SSARepr {
            name: "fused".into(),
            insns: vec![
                FlatOp::GotoIfNotOp {
                    opname: "int_lt".into(),
                    args: vec![
                        Register::new(RegKind::Int, 0),
                        Register::new(RegKind::Int, 1),
                    ],
                    target: Label(0),
                },
                FlatOp::Label(Label(0)),
            ],
            num_blocks: 1,
            insns_pos: None,
        };
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
        assert_eq!(
            asm.insns.get("goto_if_not_int_lt/iiL").copied(),
            Some(crate::insns::BC_GOTO_IF_NOT_INT_LT),
            "binary int compare must route to the reserved fused opcode",
        );

        // Unary test: `int_is_zero` with one int operand → `iL`.
        let mut flat = SSARepr {
            name: "fused_unary".into(),
            insns: vec![
                FlatOp::GotoIfNotOp {
                    opname: "int_is_zero".into(),
                    args: vec![Register::new(RegKind::Int, 0)],
                    target: Label(0),
                },
                FlatOp::Label(Label(0)),
            ],
            num_blocks: 1,
            insns_pos: None,
        };
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
        assert_eq!(
            asm.insns.get("goto_if_not_int_is_zero/iL").copied(),
            Some(crate::insns::BC_GOTO_IF_NOT_INT_IS_ZERO),
            "unary int test must route to the reserved fused opcode",
        );
    }

    #[test]
    fn assemble_ref_return_with_host_object_constant() {
        let module = HostObject::new_module("hello");
        let mut flat = SSARepr {
            name: "return_host_object".into(),
            insns: vec![FlatOp::RefReturn(crate::flatten::RegOrConst::Const(
                crate::flowspace::model::Constant::new(ConstValue::HostObject(module.clone())),
            ))],
            num_blocks: 1,
            insns_pos: None,
        };

        let regallocs = empty_regallocs();
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);

        assert_eq!(body.constants_r, vec![module.identity_id() as i64]);
        assert!(asm.insns.contains_key("ref_return/r"));
    }

    fn cc_with_exc_rows(pytypes: &[(&str, i64)]) -> crate::call::CallControl {
        let mut cc = crate::call::CallControl::new();
        cc.set_exc_pytype_rows(
            pytypes
                .iter()
                .map(|(key, addr)| ((*key).to_string(), *addr))
                .collect(),
        );
        cc
    }

    #[test]
    fn exception_class_llexitcase_emits_pytype_row_not_identity() {
        use crate::flatten::{FlatOp, Label, SSARepr};
        use crate::flowspace::model::HOST_ENV;

        let stop = HOST_ENV
            .lookup_exception_class("StopIteration")
            .expect("StopIteration");
        let base = HOST_ENV
            .lookup_exception_class("BaseException")
            .expect("BaseException");
        let exc = HOST_ENV
            .lookup_exception_class("Exception")
            .expect("Exception");
        let cc = cc_with_exc_rows(&[
            ("interp_exceptions::EXCEPTION_TYPE", 0x10),
            ("interp_exceptions::EXC_EXCEPTION_TYPE", 0x20),
            ("interp_exceptions::EXC_STOP_ITERATION_TYPE", 0x30),
        ]);
        let mut flat = SSARepr {
            name: "mismatch".into(),
            insns: vec![
                FlatOp::GotoIfExceptionMismatch {
                    llexitcase: ConstValue::HostObject(stop.clone()),
                    target: Label(0),
                },
                FlatOp::GotoIfExceptionMismatch {
                    llexitcase: ConstValue::HostObject(base.clone()),
                    target: Label(0),
                },
                FlatOp::GotoIfExceptionMismatch {
                    llexitcase: ConstValue::HostObject(exc.clone()),
                    target: Label(0),
                },
                FlatOp::Label(Label(0)),
            ],
            num_blocks: 1,
            insns_pos: None,
        };
        let mut asm = Assembler::new();
        let body = asm.assemble_with_callcontrol(&mut flat, &empty_regallocs(), Some(&cc));
        assert_eq!(body.constants_i, vec![0x30, 0x10, 0x20]);
        for class in [&stop, &base, &exc] {
            assert!(!body.constants_i.contains(&(class.identity_id() as i64)));
        }
    }

    #[test]
    fn assertion_error_message_instance_emits_sentinel_not_identity() {
        use crate::flatten::{FlatOp, RegOrConst, SSARepr};
        use crate::flowspace::model::{ConstValue, Constant, HOST_ENV, HostObject};

        let class = HOST_ENV
            .lookup_exception_class("AssertionError")
            .expect("AssertionError");
        let message = "implicit AssertionError shouldn't occur";
        let inst = HostObject::new_instance(class, vec![ConstValue::UniStr(message.to_string())]);
        let mut flat = SSARepr {
            name: "raise_assert".into(),
            insns: vec![FlatOp::Raise(RegOrConst::Const(Constant::new(
                ConstValue::HostObject(inst.clone()),
            )))],
            num_blocks: 1,
            insns_pos: None,
        };
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &empty_regallocs());
        assert_eq!(
            body.constants_r,
            vec![crate::assembler::EXC_INSTANCE_CONST_SENTINEL_BASE]
        );
        assert_ne!(
            crate::assembler::EXC_INSTANCE_CONST_SENTINEL_BASE,
            inst.identity_id() as i64
        );
        assert_eq!(body.exc_instance_consts.len(), 1);
        assert_eq!(body.exc_instance_consts[0].class_name, "AssertionError");
        assert_eq!(
            body.exc_instance_consts[0].message.as_deref(),
            Some(message.as_bytes())
        );
        assert_eq!(body.exc_instance_consts[0].constants_r_index, 0);
    }

    #[test]
    #[should_panic(expected = "no interp_exceptions pytype row for exception class ValueError")]
    fn exception_class_without_pytype_row_panics() {
        use crate::flatten::{FlatOp, Label, SSARepr};
        use crate::flowspace::model::HOST_ENV;

        let value = HOST_ENV
            .lookup_exception_class("ValueError")
            .expect("ValueError");
        let cc = cc_with_exc_rows(&[("interp_exceptions::EXC_STOP_ITERATION_TYPE", 0x30)]);
        let mut flat = SSARepr {
            name: "missing".into(),
            insns: vec![
                FlatOp::GotoIfExceptionMismatch {
                    llexitcase: ConstValue::HostObject(value),
                    target: Label(0),
                },
                FlatOp::Label(Label(0)),
            ],
            num_blocks: 1,
            insns_pos: None,
        };
        let mut asm = Assembler::new();
        let _ = asm.assemble_with_callcontrol(&mut flat, &empty_regallocs(), Some(&cc));
    }

    #[test]
    #[should_panic(expected = "integer pointer constants must be raw")]
    fn integer_constants_reject_null_gc_pointer() {
        let null = crate::translator::rtyper::lltypesystem::lltype::nullptr(
            crate::translator::rtyper::rclass::OBJECT.clone(),
        )
        .unwrap();
        Assembler::resolve_const_i_value(&ConstValue::LLPtr(Box::new(null)), None);
    }

    #[test]
    #[should_panic(expected = "reference pointer constants must be GC-managed")]
    fn reference_constants_reject_null_raw_pointer() {
        let null = crate::translator::rtyper::lltypesystem::lltype::nullptr(
            crate::translator::rtyper::rclass::OBJECT_VTABLE.clone(),
        )
        .unwrap();
        Assembler::new().emit_const_r(&ConstValue::LLPtr(Box::new(null)), &mut empty_state(), None);
    }

    #[test]
    fn assemble_int_return_with_typed_null_class_pointer() {
        use crate::translator::rtyper::lltypesystem::lltype::{self, LowLevelType};

        let classtype = crate::translator::rtyper::rclass::CLASSTYPE.clone();
        let LowLevelType::Ptr(classptr) = &classtype else {
            unreachable!()
        };
        let null = lltype::nullptr(classptr.TO.clone().into()).expect("null CLASSTYPE");
        let mut flat = SSARepr {
            name: "return_null_class".into(),
            insns: vec![FlatOp::IntReturn(crate::flatten::RegOrConst::Const(
                crate::flowspace::model::Constant::with_concretetype(
                    ConstValue::LLPtr(Box::new(null)),
                    classtype,
                ),
            ))],
            num_blocks: 1,
            insns_pos: None,
        };
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &empty_regallocs());
        // Assembler.emit_const uses USE_C_FORM for int_return: raw NULL is
        // the short integer zero, not a constant-pool entry.
        assert!(body.constants_i.is_empty());
        assert!(body.constants_r.is_empty());
        assert!(asm.insns.contains_key("int_return/c"));

        let null_address = ConstValue::LLAddress(lltype::_address::Null);
        let mut state = empty_state();
        assert_eq!(
            asm.emit_const_i_from_const_allow_short(&null_address, true, &mut state, None),
            (0, 'c'),
        );
        assert!(state.constants_i.is_empty());
    }

    #[test]
    fn emit_const_r_deduplicates_typed_nulls() {
        use crate::translator::rtyper::lltypesystem::{lltype, rstr};

        let mut asm = Assembler::new();
        let mut state = empty_state();
        state.num_regs_r = 3;
        for container in [
            crate::translator::rtyper::rclass::OBJECT.clone(),
            rstr::STR.clone(),
        ] {
            let null = lltype::nullptr(container).expect("null GC pointer");
            let slot = asm.emit_const_r(&ConstValue::LLPtr(Box::new(null)), &mut state, None);
            assert_eq!(
                slot, 3,
                "typed nulls share one slot after the Ref registers"
            );
        }
        assert_eq!(state.constants_r, vec![0]);
        assert!(state.str_consts.is_empty());
        assert!(state.unit_variant_consts.is_empty());
    }

    #[test]
    fn assemble_ref_return_with_typed_null_constant() {
        use crate::translator::rtyper::lltypesystem::lltype::{self, LowLevelType};

        let LowLevelType::Ptr(objectptr) = crate::translator::rtyper::rclass::OBJECTPTR.clone()
        else {
            unreachable!()
        };
        let null = lltype::nullptr(objectptr.TO.clone().into()).expect("null OBJECTPTR");
        let mut flat = SSARepr {
            name: "return_typed_null".into(),
            insns: vec![FlatOp::RefReturn(crate::flatten::RegOrConst::Const(
                crate::flowspace::model::Constant::with_concretetype(
                    ConstValue::LLPtr(Box::new(null)),
                    crate::translator::rtyper::rclass::OBJECTPTR.clone(),
                ),
            ))],
            num_blocks: 1,
            insns_pos: None,
        };
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &empty_regallocs());
        assert_eq!(body.constants_r, vec![0]);
        assert!(body.str_consts.is_empty());
        assert!(body.unit_variant_consts.is_empty());
        assert!(asm.insns.contains_key("ref_return/r"));
    }

    #[test]
    fn assemble_ref_return_with_unit_variant_constant() {
        // A `RefReturn` of an interned unit-variant singleton drives
        // `ConstValue::HostObject` through `emit_const_r` into the
        // deferred sentinel path and commits the recorded descriptor into
        // `JitCodeBody.unit_variant_consts` — the build-side half of the
        // loop the runtime `materialize_unit_variant_consts` pass closes.
        use crate::annotator::bookkeeper::Bookkeeper;
        use std::rc::Rc;
        let bk = Rc::new(Bookkeeper::new());
        let instance =
            crate::translator::rtyper::unit_variant_fold::intern_unit_variant_prebuilt_instance(
                &bk,
                "TestUnitVariantEnum.Only",
                Some(7),
            )
            .expect("unit variant instance");
        let expected_class = instance
            .instance_class()
            .expect("prebuilt instance has a class host")
            .clone();
        let mut flat = SSARepr {
            name: "return_unit_variant".into(),
            insns: vec![FlatOp::RefReturn(crate::flatten::RegOrConst::Const(
                crate::flowspace::model::Constant::new(ConstValue::HostObject(instance)),
            ))],
            num_blocks: 1,
            insns_pos: None,
        };
        let regallocs = empty_regallocs();
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);
        assert_eq!(body.unit_variant_consts.len(), 1);
        let d = &body.unit_variant_consts[0];
        assert_eq!(d.tag, 7);
        assert_eq!(d.qualname, expected_class.qualname());
        assert_eq!(
            body.constants_r[d.constants_r_index].get(),
            UNIT_VARIANT_CONST_SENTINEL_BASE,
        );
    }

    #[test]
    fn assemble_ref_return_with_bookkeeper_unit_variant_constant() {
        // Production order: `fold_unit_variant_ctors(graph, Some(&bk))`
        // then assemble.  `emit_const_r` recovers `(qualname, tag)` from
        // the interned variant class host.
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::model::{CallTarget, FunctionGraph, OpKind, ValueType};
        use std::rc::Rc;

        let bk = Rc::new(Bookkeeper::new());
        let mut graph = FunctionGraph::new("return_bk_unit_variant");
        let entry = graph.startblock;
        graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_enum_variant_ctor(
                        vec!["JitAction".into()],
                        "Return",
                        1,
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("JitAction::Return".into())),
                },
                true,
            )
            .expect("ctor result");
        crate::translator::rtyper::unit_variant_fold::fold_unit_variant_ctors(
            &mut graph,
            Some(&bk),
        );
        let instance = graph
            .blocks
            .iter()
            .find_map(|block| {
                block.operations.iter().find_map(|op| match &op.kind {
                    OpKind::ConstRef(obj) => Some(obj.clone()),
                    _ => None,
                })
            })
            .expect("fold rewrote the ctor to ConstRef");
        let expected_class = bk.intern_enum_variant_host("JitAction", "Return");
        let mut flat = SSARepr {
            name: "return_bk_unit_variant".into(),
            insns: vec![FlatOp::RefReturn(crate::flatten::RegOrConst::Const(
                crate::flowspace::model::Constant::new(ConstValue::HostObject(instance)),
            ))],
            num_blocks: 1,
            insns_pos: None,
        };
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &empty_regallocs());
        assert_eq!(body.unit_variant_consts.len(), 1);
        let d = &body.unit_variant_consts[0];
        assert_eq!(d.tag, 1);
        assert_eq!(d.qualname, expected_class.qualname());
        assert_eq!(
            body.constants_r[d.constants_r_index].get(),
            UNIT_VARIANT_CONST_SENTINEL_BASE,
        );
    }

    #[test]
    fn emit_const_r_records_type_static_descriptor_and_dedups() {
        let mut state = empty_state();
        let mut asm = Assembler::new();
        asm.intern_type_static_addrs(&[("pyobject::INT_TYPE", 0x1020_3040)]);
        let reg = asm.emit_type_static_or_bits(0x1020_3040, &mut state);
        assert_eq!(state.type_static_consts.len(), 1);
        assert_eq!(state.type_static_consts[0].name, "pyobject::INT_TYPE");
        let idx = state.type_static_consts[0].constants_r_index;
        assert_eq!(state.constants_r[idx], TYPE_STATIC_CONST_SENTINEL_BASE);
        let reg2 = asm.emit_type_static_or_bits(0x1020_3040, &mut state);
        assert_eq!(reg, reg2);
        assert_eq!(state.type_static_consts.len(), 1);
    }

    #[test]
    fn type_static_size_words_uses_the_constant_sentinel() {
        let mut asm = Assembler::new();
        asm.intern_type_static_addrs(&[
            ("pyobject::INT_TYPE", 0x1020_3040),
            ("pyobject::FLOAT_TYPE", 0x1020_3080),
        ]);
        let (word, name) = asm
            .type_static_size_words(0x1020_3040)
            .expect("interned type static");
        assert_eq!(name, "pyobject::INT_TYPE");
        assert_eq!(word, TYPE_STATIC_CONST_SENTINEL_BASE as u64);
        let (word, name) = asm.type_static_size_words(0x1020_3080).unwrap();
        assert_eq!(name, "pyobject::FLOAT_TYPE");
        assert_eq!(word, (TYPE_STATIC_CONST_SENTINEL_BASE | 1) as u64);
        assert!(asm.type_static_size_words(0x9999).is_none());
        // A second intern of the same address keeps the first name and ordinal.
        asm.intern_type_static_addrs(&[("other::INT_TYPE", 0x1020_3040)]);
        let (word, name) = asm.type_static_size_words(0x1020_3040).unwrap();
        assert_eq!(name, "pyobject::INT_TYPE");
        assert_eq!(word, TYPE_STATIC_CONST_SENTINEL_BASE as u64);
    }

    #[test]
    fn assemble_ref_return_with_prebuilt_string_constant() {
        use crate::translator::rtyper::lltypesystem::rstr::{
            const_str_cache_llstr, ll_strhash_value,
        };
        // A `RefReturn` of a prebuilt `Ptr(STR)` constant drives a real
        // `ConstValue::LLPtr(STR)` through the whole assembler emit path
        // (encode_regorconst_source -> emit_const('r') -> emit_const_r ->
        // emit_str_const_r) and commits the recorded descriptor into the
        // produced `JitCodeBody.str_consts`.  This is the build-side half of
        // the deferred-materialization loop the runtime `materialize_str_consts`
        // pass closes (the runtime half is covered in pyre-jit-trace).
        let mut flat = SSARepr {
            name: "return_prebuilt_string".into(),
            insns: vec![FlatOp::RefReturn(crate::flatten::RegOrConst::Const(
                crate::flowspace::model::Constant::new(ConstValue::LLPtr(Box::new(
                    const_str_cache_llstr(b"hi").expect("cache llstr"),
                ))),
            ))],
            num_blocks: 1,
            insns_pos: None,
        };

        let regallocs = empty_regallocs();
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);

        // The committed body carries the deferred descriptor, and the
        // `constants_r` slot it names holds the non-canonical sentinel the
        // runtime load pass overwrites with the live STR address.
        assert_eq!(body.str_consts.len(), 1);
        assert_eq!(body.str_consts[0].bytes, b"hi".to_vec());
        assert_eq!(body.str_consts[0].precomputed_hash, ll_strhash_value(b"hi"));
        let idx = body.str_consts[0].constants_r_index;
        assert_eq!(body.constants_r[idx], str_const_sentinel(0));
        assert!(asm.insns.contains_key("ref_return/r"));
    }

    #[test]
    fn emit_const_r_records_prebuilt_string_descriptor_and_dedups() {
        use crate::translator::rtyper::lltypesystem::rstr::{
            const_str_cache_llstr, ll_strhash_value,
        };
        let str_const =
            |s: &[u8]| ConstValue::LLPtr(Box::new(const_str_cache_llstr(s).expect("cache llstr")));
        let mut asm = Assembler::new();
        let mut state = empty_state();

        // A prebuilt `Ptr(STR)` no longer panics: it records a descriptor
        // and pools a sentinel slot for runtime materialization.
        let reg_a = asm.emit_const_r(&str_const(b"hello"), &mut state, None);
        assert_eq!(state.str_consts.len(), 1);
        assert_eq!(state.str_consts[0].bytes, b"hello".to_vec());
        assert_eq!(
            state.str_consts[0].precomputed_hash,
            ll_strhash_value(b"hello")
        );
        assert_eq!(state.str_consts[0].constants_r_index, 0);
        assert_eq!(state.constants_r, vec![str_const_sentinel(0)]);

        // The same literal dedups to the same descriptor + pool slot.
        let reg_a2 = asm.emit_const_r(&str_const(b"hello"), &mut state, None);
        assert_eq!(reg_a2, reg_a);
        assert_eq!(state.str_consts.len(), 1);
        assert_eq!(state.constants_r.len(), 1);

        // A distinct literal gets a distinct descriptor + sentinel slot.
        let reg_b = asm.emit_const_r(&str_const(b"world"), &mut state, None);
        assert_ne!(reg_b, reg_a);
        assert_eq!(state.str_consts.len(), 2);
        assert_eq!(state.str_consts[1].bytes, b"world".to_vec());
        assert_eq!(state.str_consts[1].constants_r_index, 1);
        assert_eq!(
            state.constants_r,
            vec![str_const_sentinel(0), str_const_sentinel(1)]
        );
    }

    #[test]
    fn assemble_with_registers() {
        use crate::model::{FunctionGraph, OpKind, ValueType};
        // Build graph for regalloc (regalloc operates on graph, not SSARepr)
        let mut graph = FunctionGraph::new("add");
        let entry = graph.startblock;
        let v0_var = push_input_var(&mut graph, "a", ValueType::Int);
        let v1_var = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: v0_var.clone(),
                    rhs: v0_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let v2_var = push_input_var(&mut graph, "r", ValueType::Ref(None));
        graph.set_return(entry, Some(v1_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&v0_var, crate::model::ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&v1_var, crate::model::ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&v2_var, crate::model::ConcreteType::GcRef);

        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = SSARepr {
            name: "add".into(),
            insns: vec![],
            num_blocks: 1,
            insns_pos: None,
        };
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);

        // v0 dies when v1 is defined → they share a register → 1 int reg
        assert_eq!(body.c_num_regs_i as usize, 1);
        assert_eq!(body.c_num_regs_r as usize, 1);
        assert_eq!(body.c_num_regs_f as usize, 0);
    }

    /// `OpKind::RecordQuasiImmutField` must lower to a single opcode
    /// keyed `record_quasiimmut_field/rdd`, with the field+mutate
    /// FieldDescriptor pair pushed as two `BhDescr::Field` entries — see
    /// `rpython/jit/codewriter/jtransform.py:901-903` and
    /// `rpython/jit/metainterp/blackhole.py:1537-1539`.
    #[test]
    fn assembles_record_quasiimmut_field_with_two_descrs() {
        use crate::call::CallControl;
        use crate::flatten::flatten_graph;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FieldDescriptor, FunctionGraph, ImmutableRank, OpKind, ValueType};

        let mut cc = CallControl::new();
        cc.immutable_fields_by_struct.insert(
            "Cell".to_string(),
            vec![("value".to_string(), ImmutableRank::QuasiImmutable)],
        );

        let mut graph = FunctionGraph::new("read_cell");
        let base_var = push_input_var(&mut graph, "cell", ValueType::Ref(None));
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base_var.clone(),
                    field: FieldDescriptor::new("value", Some("Cell".to_string())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let mut rewritten = transformer.transform(&graph).graph;

        FunctionGraph::set_concretetype_of_inline(&base_var, crate::model::ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&result_var, crate::model::ConcreteType::Signed);
        regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
        let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
        let mut flat = flatten_graph(&rewritten, &mut regallocs);
        // `assemble` derives every operand kind from the variable
        // concretetype + the `regallocs` coloring (lookup_coloring_var,
        // flatten.py:386-387); pass the same canonical-exceptblock-
        // augmented graph and regallocs that the flatten pass consumed.
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        let key_count = asm
            .insns
            .keys()
            .filter(|k| k.starts_with("record_quasiimmut_field/"))
            .count();
        assert_eq!(
            key_count,
            1,
            "expected exactly one record_quasiimmut_field/* key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("record_quasiimmut_field/rdd"),
            "expected key record_quasiimmut_field/rdd, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("getfield_gc_i_pure/rd>i"),
            "expected pure getfield opcode, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        // Two BhDescr::Field entries — for `value` and `mutate_value`.
        let descrs = asm.snapshot_descrs();
        let field_descr_names: Vec<&str> = descrs
            .iter()
            .filter_map(|d| match d {
                crate::jitcode::BhDescr::Field { name, owner, .. } if owner == "Cell" => {
                    Some(name.as_str())
                }
                _ => None,
            })
            .collect();
        assert!(
            field_descr_names.contains(&"value") && field_descr_names.contains(&"mutate_value"),
            "expected Field descrs for `value` + `mutate_value`, got {:?}",
            field_descr_names
        );
    }

    /// `OpKind::NewListClear` assembles to the wired `newlist_clear/idddd>r`
    /// opcode: length (Int) + four descriptors (structdescr, lengthdescr,
    /// itemsdescr, arraydescr) + ref result.  The `list` GcStruct header
    /// layout (`length@0`, `items@8`) is registered on the CallControl so
    /// the struct/field descrs resolve, and the four descr slots must be
    /// interned in exactly that read order (`handler_newlist_clear`,
    /// blackhole.rs).
    #[test]
    fn assembles_newlist_clear_with_four_descrs_in_read_order() {
        use crate::call::CallControl;
        use crate::flatten::flatten_graph;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FunctionGraph, OpKind, ValueType};

        // Mirror `rtyper_synthesised_list_struct_resolves_its_field_descrs`
        // (call.rs): register the rtyper-synthesised `GcStruct("list",
        // ("length", Signed), ("items", Ptr(GcArray(ITEM))))` shape so the
        // structdescr and the `length`/`items` fielddescrs resolve.
        let mut cc = CallControl::new();
        let mut registry = crate::front::StructFieldRegistry::default();
        registry.fields.insert(
            "list".to_string(),
            vec![
                ("length".to_string(), "i64".to_string()),
                ("items".to_string(), "&[i64]".to_string()),
            ],
        );
        cc.set_struct_fields(registry);

        let mut graph = FunctionGraph::new("newlist_clear_graph");
        let length_var = push_input_var(&mut graph, "n", ValueType::Int);
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::NewListClear {
                    length: length_var.clone(),
                    item_ty: ValueType::Int,
                    array_type_id: Some("list_items".to_string()),
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(
            &length_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(
            &result_var,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let mut rewritten = transformer.transform(&graph).graph;
        regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
        let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
        let mut flat = flatten_graph(&rewritten, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble_with_callcontrol(&mut flat, &regallocs, Some(&cc));

        // The key must be exactly `newlist_clear/idddd>r` — four `d` descr
        // slots in the length/struct-length-items-array order.  Any drift
        // (wrong descr count) would mint a different dynamic opname.
        assert!(
            asm.insns.contains_key("newlist_clear/idddd>r"),
            "expected key newlist_clear/idddd>r, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );

        // Four descrs interned in read order: Size (struct) → Field
        // (length) → Field (items) → Array (items block).
        let descrs = asm.snapshot_descrs();
        let list_size = descrs.iter().position(
            |d| matches!(d, crate::jitcode::BhDescr::Size { vtable, .. } if *vtable == 0),
        );
        assert!(
            list_size.is_some(),
            "expected a vtable-0 Size descr for the list header, got {descrs:?}"
        );
        let length_pos = descrs.iter().position(
            |d| matches!(d, crate::jitcode::BhDescr::Field { name, .. } if name == "length"),
        );
        let items_pos = descrs.iter().position(
            |d| matches!(d, crate::jitcode::BhDescr::Field { name, .. } if name == "items"),
        );
        let array_pos = descrs
            .iter()
            .position(|d| matches!(d, crate::jitcode::BhDescr::Array { .. }));
        let (length_pos, items_pos, array_pos) = (
            length_pos.expect("length fielddescr interned"),
            items_pos.expect("items fielddescr interned"),
            array_pos.expect("array descr interned"),
        );
        assert!(
            list_size.unwrap() < length_pos && length_pos < items_pos && items_pos < array_pos,
            "descrs must intern in struct/length/items/array order, got {descrs:?}"
        );

        // The `list` header layout resolved off the registered struct:
        // length@0 (Signed word) and items@8 (pointer to the array block).
        let field_at = |name: &str| {
            descrs.iter().find_map(|d| match d {
                crate::jitcode::BhDescr::Field {
                    name: n, offset, ..
                } if n == name => Some(*offset),
                _ => None,
            })
        };
        assert_eq!(field_at("length"), Some(0), "length is the first word");
        assert_eq!(field_at("items"), Some(8), "items follows the length word");
    }

    #[test]
    fn assemble_typed_writes_use_canonical_non_v_opnames() {
        use crate::flatten::flatten_graph;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FieldDescriptor, FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("typed_writes");
        let base_var = push_input_var(&mut graph, "obj", ValueType::Ref(None));
        let index_var = push_input_var(&mut graph, "i", ValueType::Int);
        let value_var = push_input_var(&mut graph, "v", ValueType::Int);
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldWrite {
                base: base_var.clone(),
                field: FieldDescriptor::new("x", Some("Point".into())),
                value: crate::model::LinkArg::Value(value_var.clone()),
                ty: ValueType::Unknown,
            },
            false,
        );
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayWrite {
                base: base_var.clone(),
                index: index_var.clone(),
                value: crate::model::LinkArg::Value(value_var.clone()),
                item_ty: ValueType::Unknown,
                array_type_id: None,
                nolength: false,
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        // Publish kinds to graph cells before jtransform.  Variable
        // Rc-shares the concretetype cell across clones, so the
        // cloned rewritten graph picks up the same kinds.
        FunctionGraph::set_concretetype_of_inline(
            &base_var,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );
        FunctionGraph::set_concretetype_of_inline(
            &index_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(
            &value_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let mut rewritten = Transformer::new(&config).transform(&graph).graph;
        regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
        let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
        let mut flat = flatten_graph(&rewritten, &mut regallocs);

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        assert!(
            asm.insns.contains_key("setfield_gc_i/rid"),
            "expected canonical setfield_gc_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("setarrayitem_gc_i/riid"),
            "expected canonical setarrayitem_gc_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setfield_gc_v/rid"),
            "unexpected setfield_gc_v key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setfield_gc_v/iid"),
            "unexpected setfield_gc_v/iid key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setfield_gc_v/ird"),
            "unexpected setfield_gc_v/ird key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setarrayitem_gc_v/riid"),
            "unexpected setarrayitem_gc_v key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setarrayitem_gc_v/iiid"),
            "unexpected setarrayitem_gc_v/iiid key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn assemble_setfield_const_int_uses_c_form() {
        use crate::flatten::flatten_graph;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FieldDescriptor, FunctionGraph, LinkArg, OpKind, ValueType};

        // A `setfield_gc_i` whose value is an inline integer Constant takes
        // the short `c` byte (`setfield_gc_i/rcd`) when it fits a signed
        // byte, else a pool `i` slot (`setfield_gc_i/rid`).  `setfield_gc`
        // ∈ USE_C_FORM, `assembler.py`.
        let build = |value: i64| -> Assembler {
            let mut graph = FunctionGraph::new("const_setfield");
            let base_var = push_input_var(&mut graph, "obj", ValueType::Ref(None));
            graph.push_op_var(
                graph.startblock,
                OpKind::FieldWrite {
                    base: base_var.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    value: LinkArg::Const(Constant::new(ConstValue::Int(value))),
                    ty: ValueType::Unknown,
                },
                false,
            );
            graph.set_return(graph.startblock, None);
            FunctionGraph::set_concretetype_of_inline(
                &base_var,
                crate::codewriter::type_state::ConcreteType::GcRef,
            );
            let config = GraphTransformConfig::default();
            let mut rewritten = Transformer::new(&config).transform(&graph).graph;
            regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
            let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
            let mut flat = flatten_graph(&rewritten, &mut regallocs);
            let mut asm = Assembler::new();
            let _ = asm.assemble(&mut flat, &regallocs);
            asm
        };

        let small = build(7);
        assert!(
            small.insns.contains_key("setfield_gc_i/rcd"),
            "small const setfield should take the c-form, got {:?}",
            small.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !small.insns.contains_key("setfield_gc_i/rid"),
            "small const setfield should not materialise an i-register, got {:?}",
            small.insns.keys().collect::<Vec<_>>()
        );

        let wide = build(100_000);
        assert!(
            wide.insns.contains_key("setfield_gc_i/rid"),
            "wide const setfield should take the pool i-form, got {:?}",
            wide.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !wide.insns.contains_key("setfield_gc_i/rcd"),
            "wide const setfield should not take the c-form, got {:?}",
            wide.insns.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn assemble_setfield_vable_const_int_uses_pool_form() {
        use crate::flatten::flatten_graph;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::jtransform::{GraphTransformConfig, Transformer, VirtualizableFieldDescriptor};
        use crate::model::{FieldDescriptor, FunctionGraph, LinkArg, OpKind, ValueType};

        // A `setfield_gc_i` whose field is virtualizable rewrites to
        // `setfield_vable_i` (`jtransform.py:921-927`).  `setfield_vable_i`
        // is NOT in USE_C_FORM (`assembler.py`), so an inline
        // integer Constant value always takes the pool `i` slot
        // (`setfield_vable_i/rid`) — never the short `c` byte — even when
        // it fits a signed byte.
        let build = |value: i64| -> Assembler {
            let mut graph = FunctionGraph::new("const_setfield_vable");
            let base_var = push_input_var(&mut graph, "frame", ValueType::Ref(None));
            graph.push_op_var(
                graph.startblock,
                OpKind::FieldWrite {
                    base: base_var.clone(),
                    field: FieldDescriptor::new("depth", Some("Frame".into())),
                    value: LinkArg::Const(Constant::new(ConstValue::Int(value))),
                    ty: ValueType::Int,
                },
                false,
            );
            graph.set_return(graph.startblock, None);
            FunctionGraph::set_concretetype_of_inline(
                &base_var,
                crate::codewriter::type_state::ConcreteType::GcRef,
            );
            let config = GraphTransformConfig {
                vable_fields: vec![VirtualizableFieldDescriptor::new(
                    "depth",
                    Some("Frame".into()),
                    0,
                )],
                ..Default::default()
            };
            let mut rewritten = Transformer::new(&config).transform(&graph).graph;
            regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
            let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
            let mut flat = flatten_graph(&rewritten, &mut regallocs);
            let mut asm = Assembler::new();
            let _ = asm.assemble(&mut flat, &regallocs);
            asm
        };

        for value in [7, 100_000] {
            let asm = build(value);
            assert!(
                asm.insns.contains_key("setfield_vable_i/rid"),
                "vable const setfield (value={value}) should take the pool i-form, got {:?}",
                asm.insns.keys().collect::<Vec<_>>()
            );
            assert!(
                !asm.insns.contains_key("setfield_vable_i/rcd"),
                "vable const setfield (value={value}) must never take the c-form, got {:?}",
                asm.insns.keys().collect::<Vec<_>>()
            );
        }
    }

    /// An Int-bank FieldRead is `getfield_raw`, matching
    /// `jtransform.py rewrite_op_getfield` `_gckind == 'raw'`.
    #[test]
    fn assemble_int_bank_field_read_emits_getfield_raw() {
        use crate::flatten::flatten_graph;
        use crate::model::{FieldDescriptor, FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("int_getfield_base");
        let base_var = push_input_var(&mut graph, "obj", ValueType::Int);
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base_var.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .expect("field read has a result");
        graph.set_return(graph.startblock, Some(result.clone()));
        FunctionGraph::set_concretetype_of_inline(
            &base_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(
            &result,
            crate::codewriter::type_state::ConcreteType::Signed,
        );

        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
        assert!(
            asm.insns.contains_key("getfield_raw_i/id>i"),
            "Int-bank FieldRead must assemble as getfield_raw_i/id>i, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.keys().any(|k| k.starts_with("getfield_gc_")),
            "Int-bank FieldRead must not emit getfield_gc, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    /// `&mut S` of a Raw struct is the address integer. Field access assembles
    /// as `getfield_raw_i` / `setfield_raw_i` at the struct's own offsets, and
    /// the argument stays in an `i` register.
    #[test]
    fn raw_mut_struct_field_access_assembles_getfield_raw_at_host_offsets() {
        use crate::call::{CallControl, StructFieldLayout, StructLayout};
        use crate::model::{FunctionGraph, OpKind, ValueType};
        use majit_ir::descr::ArrayFlag;
        use majit_ir::value::Type;

        let span = || {
            serde_json::json!({
                "data": {
                    "file_id": 0,
                    "beg": {"line": 1, "col": 0},
                    "end": {"line": 1, "col": 1}
                }
            })
        };
        let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
        let s_ty = serde_json::json!({
            "Adt": {
                "id": 0,
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        });
        let ref_ty = serde_json::json!({"Ref": ["Erased", s_ty, "Mut"]});
        let place = |kind: serde_json::Value, ty: &serde_json::Value| serde_json::json!({"kind": kind, "ty": ty});
        let local =
            |index: u64, ty: &serde_json::Value| place(serde_json::json!({"Local": index}), ty);
        let arg = local(1, &ref_ty);
        let deref = place(serde_json::json!({"Projection": [arg, "Deref"]}), &s_ty);
        let field = |index: u64| {
            place(
                serde_json::json!({"Projection": [deref.clone(), {"Field": [null, index]}]}),
                &i64_ty,
            )
        };
        let use_copy = |src: serde_json::Value| serde_json::json!({"Use": [{"Copy": src}, "No"]});
        let assign = |dest: serde_json::Value, rvalue: serde_json::Value| serde_json::json!({"span": span(), "kind": {"Assign": [dest, rvalue]}});
        let item_meta = |path: &[&str]| {
            serde_json::json!({
                "name": path.iter().map(|seg| serde_json::json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
                "span": span(),
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": true
            })
        };
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "rawptr",
                "type_decls": [{
                    "def_id": 0,
                    "item_meta": item_meta(&["rawptr", "S"]),
                    "kind": {"Struct": [
                        {"name": "a", "ty": i64_ty},
                        {"name": "b", "ty": i64_ty}
                    ]},
                    "src": "Normal",
                    "layout": [{
                        "key": "host",
                        "value": {
                            "size": 16,
                            "align": 8,
                            "variant_layouts": [{"field_offsets": [0, 8]}],
                            "repr": {"transparent": false}
                        }
                    }]
                }],
                "fun_decls": [{
                    "def_id": 0,
                    "item_meta": item_meta(&["rawptr", "bump"]),
                    "signature": {
                        "is_unsafe": false,
                        "inputs": [ref_ty],
                        "output": i64_ty
                    },
                    "src": "Normal",
                    "body": {
                        "Unstructured": {
                            "span": span(),
                            "locals": {
                                "arg_count": 1,
                                "locals": [
                                    {"index": 0, "name": null, "span": span(), "ty": i64_ty},
                                    {"index": 1, "name": "p", "span": span(), "ty": ref_ty},
                                    {"index": 2, "name": "tmp", "span": span(), "ty": i64_ty}
                                ]
                            },
                            "body": [{
                                "statements": [
                                    assign(local(2, &i64_ty), use_copy(field(0))),
                                    assign(field(1), use_copy(local(2, &i64_ty))),
                                    assign(local(0, &i64_ty), use_copy(local(2, &i64_ty)))
                                ],
                                "terminator": {"span": span(), "kind": "Return"}
                            }]
                        }
                    }
                }],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        let llbc = majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
            .expect("raw pointer fixture parses");
        let fd = llbc
            .iter_local_fns()
            .find(|fd| fd.item_meta.name_path().ends_with("::bump"))
            .expect("bump");
        let context = crate::front::mir::LowerContext::new(&llbc);
        let mut graph = crate::front::mir::lower_fun_decl(&context, fd).expect("lower bump");

        let input = graph
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .find_map(|op| match (&op.result, &op.kind) {
                (
                    Some(result),
                    OpKind::Input {
                        name,
                        ty,
                        class_root,
                    },
                ) if name == "p" => {
                    assert_eq!(
                        ty,
                        &ValueType::Ref(None),
                        "a borrowed raw struct stays a reference"
                    );
                    assert_eq!(class_root.as_deref(), Some("S"));
                    Some(result.clone())
                }
                _ => None,
            })
            .expect("p input");
        let mut saw_read = false;
        let mut saw_write = false;
        let mut read_result = None;
        for op in graph
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
        {
            assert!(
                crate::model::cast_instance_root(&op.kind).is_none(),
                "a raw address is not cast onto a gc shell"
            );
            match &op.kind {
                OpKind::FieldRead {
                    base, field, ty, ..
                } => {
                    assert_eq!(base, &input);
                    assert_eq!(field.name, "a");
                    assert_eq!(field.owner_root.as_deref(), Some("S"));
                    assert_eq!(ty, &ValueType::Int);
                    saw_read = true;
                    read_result = op.result.clone();
                }
                OpKind::FieldWrite {
                    base, field, value, ..
                } => {
                    assert_eq!(base, &input);
                    assert_eq!(field.name, "b");
                    assert_eq!(field.owner_root.as_deref(), Some("S"));
                    assert_eq!(value.as_variable(), read_result.as_ref());
                    saw_write = true;
                }
                _ => {}
            }
        }
        assert!(saw_read && saw_write, "bump reads S.a and writes S.b");
        let read_result = read_result.expect("field read result");
        FunctionGraph::set_concretetype_of_inline(
            &input,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(
            &read_result,
            crate::codewriter::type_state::ConcreteType::Signed,
        );

        let owner_id = majit_ir::descr::StructId::from_canonical("S");
        let _guard =
            crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
                ("S".to_string(), Some(owner_id)),
            ]));
        let mut cc = CallControl::new();
        cc.set_struct_layout(
            owner_id,
            StructLayout {
                size: 16,
                align: 8,
                gckind: crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw,
                fields: vec![
                    StructFieldLayout {
                        name: "a".into(),
                        offset: 0,
                        size: 8,
                        flag: ArrayFlag::Signed,
                        field_type: Type::Int,
                        rank: None,
                    },
                    StructFieldLayout {
                        name: "b".into(),
                        offset: 8,
                        size: 8,
                        flag: ArrayFlag::Signed,
                        field_type: Type::Int,
                        rank: None,
                    },
                ],
                host: None,
                ll_struct: std::cell::RefCell::new(None),
                ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
            },
        );
        crate::codewriter::type_state::promote_gc_field_bases(&mut graph, Some(&cc));
        assert_eq!(
            FunctionGraph::concretetype_of(&input),
            crate::codewriter::type_state::ConcreteType::Signed,
            "a Raw owner does not lift the address into the ref bank"
        );

        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = crate::flatten::flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble_with_callcontrol(&mut flat, &regallocs, Some(&cc));
        let keys: Vec<&String> = asm.insns.keys().collect();
        assert!(
            keys.iter().any(|k| k.as_str() == "getfield_raw_i/id>i"),
            "getfield_raw_i missing, got {keys:?}"
        );
        assert!(
            keys.iter().any(|k| k.as_str() == "setfield_raw_i/iid"),
            "setfield_raw_i missing, got {keys:?}"
        );
        assert!(
            keys.iter()
                .all(|k| !k.starts_with("getfield_gc_") && !k.starts_with("setfield_gc_")),
            "raw field access must not emit gc field ops, got {keys:?}"
        );
        let mut offsets = Vec::new();
        for descr in &asm.descrs {
            if let AssemblerDescr::Ready(ready) = descr
                && let crate::jitcode::BhDescr::Field { offset, .. } = ready.as_ref()
            {
                offsets.push(*offset);
            }
        }
        offsets.sort_unstable();
        assert_eq!(offsets, vec![0, 8], "S.a is at 0 and S.b is at 8");
    }

    /// A raw `&mut` of an `Option<(u32, u64)>`-shaped enum reads the payload
    /// at the host offsets. The gc shell still has those fields at `8 + i * 8`.
    #[test]
    fn raw_mut_enum_payload_reads_host_offset_not_sum_shell() {
        use crate::call::{CallControl, StructFieldLayout, StructLayout};
        use crate::model::{FunctionGraph, OpKind, ValueType};
        use majit_ir::descr::ArrayFlag;
        use majit_ir::value::Type;

        let span = || {
            serde_json::json!({
                "data": {
                    "file_id": 0,
                    "beg": {"line": 1, "col": 0},
                    "end": {"line": 1, "col": 1}
                }
            })
        };
        let u32_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U32"}}});
        let u64_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
        let pair_ty = serde_json::json!({
            "Adt": {
                "id": 0,
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        });
        let ref_ty = serde_json::json!({"Ref": ["Erased", pair_ty, "Mut"]});
        let place = |kind: serde_json::Value, ty: &serde_json::Value| serde_json::json!({"kind": kind, "ty": ty});
        let local =
            |index: u64, ty: &serde_json::Value| place(serde_json::json!({"Local": index}), ty);
        let arg = local(1, &ref_ty);
        let deref = place(serde_json::json!({"Projection": [arg, "Deref"]}), &pair_ty);
        let field = |index: u64, ty: &serde_json::Value| {
            place(
                serde_json::json!({"Projection": [deref.clone(), {"Field": [1, index]}]}),
                ty,
            )
        };
        let use_copy = |src: serde_json::Value| serde_json::json!({"Use": [{"Copy": src}, "No"]});
        let assign = |dest: serde_json::Value, rvalue: serde_json::Value| serde_json::json!({"span": span(), "kind": {"Assign": [dest, rvalue]}});
        let item_meta = |path: &[&str]| {
            serde_json::json!({
                "name": path.iter().map(|seg| serde_json::json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
                "span": span(),
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": true
            })
        };
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "rawopt",
                "type_decls": [{
                    "def_id": 0,
                    "item_meta": item_meta(&["rawopt", "RawOptPair"]),
                    "kind": {"Enum": [
                        {"name": "None", "fields": []},
                        {"name": "Some", "fields": [
                            {"name": null, "is_positional": true, "ty": u32_ty},
                            {"name": null, "is_positional": true, "ty": u64_ty}
                        ]}
                    ]},
                    "src": "Normal",
                    "layout": [{
                        "key": "host",
                        "value": {
                            "size": 16,
                            "align": 8,
                            "variant_layouts": [
                                {"field_offsets": []},
                                {"field_offsets": [8, 12]}
                            ],
                            "repr": {"transparent": false}
                        }
                    }]
                }],
                "fun_decls": [{
                    "def_id": 0,
                    "item_meta": item_meta(&["rawopt", "bump"]),
                    "signature": {
                        "is_unsafe": false,
                        "inputs": [ref_ty],
                        "output": u64_ty
                    },
                    "src": "Normal",
                    "body": {
                        "Unstructured": {
                            "span": span(),
                            "locals": {
                                "arg_count": 1,
                                "locals": [
                                    {"index": 0, "name": null, "span": span(), "ty": u64_ty},
                                    {"index": 1, "name": "p", "span": span(), "ty": ref_ty},
                                    {"index": 2, "name": "lo", "span": span(), "ty": u32_ty},
                                    {"index": 3, "name": "hi", "span": span(), "ty": u64_ty}
                                ]
                            },
                            "body": [{
                                "statements": [
                                    assign(local(2, &u32_ty), use_copy(field(0, &u32_ty))),
                                    assign(local(3, &u64_ty), use_copy(field(1, &u64_ty))),
                                    assign(field(0, &u32_ty), use_copy(local(2, &u32_ty))),
                                    assign(field(1, &u64_ty), use_copy(local(3, &u64_ty))),
                                    assign(local(0, &u64_ty), use_copy(local(3, &u64_ty)))
                                ],
                                "terminator": {"span": span(), "kind": "Return"}
                            }]
                        }
                    }
                }],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        let llbc = majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
            .expect("raw enum fixture parses");
        let fd = llbc
            .iter_local_fns()
            .find(|fd| fd.item_meta.name_path().ends_with("::bump"))
            .expect("bump");
        let context = crate::front::mir::LowerContext::new(&llbc);
        let mut graph = crate::front::mir::lower_fun_decl(&context, fd).expect("lower bump");

        let input = graph
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .find_map(|op| match (&op.result, &op.kind) {
                (
                    Some(result),
                    OpKind::Input {
                        name,
                        ty,
                        class_root,
                    },
                ) if name == "p" => {
                    assert_eq!(
                        ty,
                        &ValueType::Int,
                        "&mut RawOptPair of a Raw enum is an address"
                    );
                    assert_eq!(class_root.as_deref(), Some("RawOptPair"));
                    Some(result.clone())
                }
                _ => None,
            })
            .expect("p input");
        let mut payloads = Vec::new();
        let mut read_hi = None;
        for op in graph
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
        {
            assert!(
                crate::model::cast_instance_root(&op.kind).is_none(),
                "a raw address is not cast onto a gc shell"
            );
            match &op.kind {
                OpKind::FieldRead { base, field, .. } | OpKind::FieldWrite { base, field, .. } => {
                    assert_eq!(base, &input);
                    assert_eq!(field.owner_root.as_deref(), Some("RawOptPair::Some"));
                    let expect = match field.name.as_str() {
                        "__pos_0" => Some((1, 0)),
                        "__pos_1" => Some((1, 1)),
                        other => panic!("unexpected field {other}"),
                    };
                    assert_eq!(field.host_index, expect);
                    payloads.push(field.clone());
                    if matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_1")
                    {
                        read_hi = op.result.clone();
                    }
                }
                _ => {}
            }
        }
        assert_eq!(
            payloads.len(),
            4,
            "two payload reads and two payload writes"
        );
        let read_hi = read_hi.expect("hi field read");
        let read_lo = graph
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .find_map(|op| match &op.kind {
                OpKind::FieldRead { field, .. } if field.name == "__pos_0" => op.result.clone(),
                _ => None,
            })
            .expect("lo field read");
        for var in [&input, &read_lo, &read_hi] {
            FunctionGraph::set_concretetype_of_inline(
                var,
                crate::codewriter::type_state::ConcreteType::Signed,
            );
        }

        let enum_id = majit_ir::descr::StructId::from_canonical("RawOptPair");
        let variant_id = majit_ir::descr::StructId::from_canonical("RawOptPair::Some");
        let _guard =
            crate::test_support::register_struct_ids_serialized(std::collections::HashMap::from([
                ("RawOptPair".to_string(), Some(enum_id)),
                ("RawOptPair::Some".to_string(), Some(variant_id)),
            ]));
        let shell_field = |name: &str, offset: usize, size: usize| StructFieldLayout {
            name: name.into(),
            offset,
            size,
            flag: ArrayFlag::Unsigned,
            field_type: Type::Int,
            rank: None,
        };
        let mut cc = CallControl::new();
        cc.set_struct_layout(
            variant_id,
            StructLayout {
                size: 24,
                align: 8,
                gckind: crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw,
                fields: vec![shell_field("__pos_0", 8, 4), shell_field("__pos_1", 16, 8)],
                host: None,
                ll_struct: std::cell::RefCell::new(None),
                ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
            },
        );
        cc.set_struct_layout(
            enum_id,
            StructLayout {
                size: 16,
                align: 8,
                gckind: crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw,
                fields: vec![shell_field("__discriminant", 0, 8)],
                host: Some(crate::front::host_layout::HostLayout {
                    size: 16,
                    align: 8,
                    variant_field_offsets: vec![vec![], vec![0, 4]],
                    tag: None,
                }),
                ll_struct: std::cell::RefCell::new(None),
                ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
            },
        );
        let descr_offset = |field: &crate::model::FieldDescriptor, is_gc: bool| match fielddescrof(
            field,
            &ValueType::Int,
            Some(&cc),
            is_gc,
        ) {
            crate::jitcode::BhDescr::Field { offset, .. } => offset,
            other => panic!("expected a field descr, got {other:?}"),
        };
        let mut shell_offsets: Vec<usize> =
            payloads.iter().map(|f| descr_offset(f, true)).collect();
        let mut host_offsets: Vec<usize> =
            payloads.iter().map(|f| descr_offset(f, false)).collect();
        shell_offsets.sort_unstable();
        host_offsets.sort_unstable();
        assert_eq!(
            shell_offsets,
            vec![8, 8, 16, 16],
            "a gc base keeps the explicit sum shell"
        );
        assert_eq!(
            host_offsets,
            vec![0, 0, 4, 4],
            "a raw base reads variant_field_offsets on the enum root"
        );

        crate::codewriter::type_state::promote_gc_field_bases(&mut graph, Some(&cc));
        assert_eq!(
            FunctionGraph::concretetype_of(&input),
            crate::codewriter::type_state::ConcreteType::Signed,
            "a Raw owner does not lift the address into the ref bank"
        );
        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = crate::flatten::flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble_with_callcontrol(&mut flat, &regallocs, Some(&cc));
        let keys: Vec<&String> = asm.insns.keys().collect();
        assert!(
            keys.iter().any(|k| k.as_str() == "getfield_raw_i/id>i"),
            "getfield_raw_i missing, got {keys:?}"
        );
        assert!(
            keys.iter().any(|k| k.as_str() == "setfield_raw_i/iid"),
            "setfield_raw_i missing, got {keys:?}"
        );
        let mut assembled = Vec::new();
        for descr in &asm.descrs {
            if let AssemblerDescr::Ready(ready) = descr
                && let crate::jitcode::BhDescr::Field { offset, .. } = ready.as_ref()
            {
                assembled.push(*offset);
            }
        }
        assembled.sort_unstable();
        // Identical field ops share one interned descr.
        assert_eq!(assembled, vec![0, 4]);
    }

    /// `match *p` on a niche enum behind a raw pointer decodes the host
    /// tag to the variant index, and that index is what `SwitchInt` lists.
    #[test]
    fn raw_pointer_niche_discriminant_decodes_to_switch_targets() {
        use crate::model::OpKind;
        use majit_charon_reader::ullbc::{TagEncoding, TagLayout};

        let span = || {
            serde_json::json!({
                "data": {
                    "file_id": 0,
                    "beg": {"line": 1, "col": 0},
                    "end": {"line": 1, "col": 1}
                }
            })
        };
        let u8_ty = serde_json::json!({"Scalar": {"Integer": {"Unsigned": "U8"}}});
        let i64_ty = serde_json::json!({"Scalar": {"Integer": {"Signed": "I64"}}});
        let enum_ty = serde_json::json!({
            "Adt": {
                "id": 1,
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        });
        let ref_ty = serde_json::json!({"Ref": ["Erased", enum_ty, "Mut"]});
        let place = |kind: serde_json::Value, ty: &serde_json::Value| serde_json::json!({"kind": kind, "ty": ty});
        let local =
            |index: u64, ty: &serde_json::Value| place(serde_json::json!({"Local": index}), ty);
        let arg = local(1, &ref_ty);
        let deref = place(serde_json::json!({"Projection": [arg, "Deref"]}), &enum_ty);
        let assign = |dest: serde_json::Value, rvalue: serde_json::Value| serde_json::json!({"span": span(), "kind": {"Assign": [dest, rvalue]}});
        let ret = || {
            serde_json::json!({
                "statements": [],
                "terminator": {"span": span(), "kind": "Return"}
            })
        };
        let scalar = |text: &str| serde_json::json!({"Scalar": {"Unsigned": ["U8", text]}});
        let item_meta = |path: &[&str]| {
            serde_json::json!({
                "name": path.iter().map(|seg| serde_json::json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
                "span": span(),
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": true
            })
        };
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "rawniche",
                "type_decls": [null, {
                    "def_id": 1,
                    "item_meta": item_meta(&["rawniche", "E"]),
                    "kind": {"Enum": [
                        {"name": "A", "fields": []},
                        {"name": "Pay", "fields": [
                            {"name": null, "is_positional": true, "ty": u8_ty}
                        ]},
                        {"name": "B", "fields": []}
                    ]},
                    "src": "Normal",
                    "layout": [{
                        "key": "host",
                        "value": {
                            "size": 1,
                            "align": 1,
                            "discriminator": {"Branch": {
                                "offset": {"chosen": 0},
                                "int_ty": {"Unsigned": "U8"},
                                "children": [
                                    [{"start": {"Unsigned": ["U8", "10"]}, "end": {"Unsigned": ["U8", "10"]}}, {"Known": 0}],
                                    [{"start": {"Unsigned": ["U8", "12"]}, "end": {"Unsigned": ["U8", "12"]}}, {"Known": 2}]
                                ],
                                "fallback": {"Known": 1}
                            }},
                            "variant_layouts": [
                                {"field_offsets": []},
                                {"field_offsets": [0]},
                                {"field_offsets": []}
                            ],
                            "repr": {"transparent": false}
                        }
                    }]
                }],
                "fun_decls": [{
                    "def_id": 0,
                    "item_meta": item_meta(&["rawniche", "classify"]),
                    "signature": {
                        "is_unsafe": false,
                        "inputs": [ref_ty],
                        "output": i64_ty
                    },
                    "src": "Normal",
                    "body": {
                        "Unstructured": {
                            "span": span(),
                            "locals": {
                                "arg_count": 1,
                                "locals": [
                                    {"index": 0, "name": null, "span": span(), "ty": i64_ty},
                                    {"index": 1, "name": "p", "span": span(), "ty": ref_ty},
                                    {"index": 2, "name": "tag", "span": span(), "ty": i64_ty}
                                ]
                            },
                            "body": [
                                {
                                    "statements": [
                                        assign(
                                            local(0, &i64_ty),
                                            serde_json::json!({"Use": [
                                                {"Const": [
                                                    {"Integer": {"Signed": ["I64", "0"]}},
                                                    i64_ty
                                                ]},
                                                "No"
                                            ]})
                                        ),
                                        assign(
                                            local(2, &i64_ty),
                                            serde_json::json!({"Discriminant": deref})
                                        )
                                    ],
                                    "terminator": {
                                        "span": span(),
                                        "kind": {"Switch": {
                                            "discr": {"Copy": local(2, &i64_ty)},
                                            "targets": {"SwitchInt": [
                                                null,
                                                [[scalar("0"), 1], [scalar("1"), 2], [scalar("2"), 3]],
                                                4
                                            ]}
                                        }}
                                    }
                                },
                                ret(),
                                ret(),
                                ret(),
                                ret()
                            ]
                        }
                    }
                }],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        let llbc = majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
            .expect("niche pointer fixture parses");
        let fd = llbc
            .iter_local_fns()
            .find(|fd| fd.item_meta.name_path().ends_with("::classify"))
            .expect("classify");
        let context = crate::front::mir::LowerContext::new(&llbc);
        let graph = crate::front::mir::lower_fun_decl(&context, fd).expect("lower classify");

        let tag = TagLayout {
            offset: 0,
            signed: false,
            bits: 8,
            encoding: TagEncoding::Niche {
                untagged_variant: 1,
                niche_variants: 0..=2,
                niche_start: 10,
            },
        };
        let (discr, mut cases) = switch_int_cases(&graph);
        cases.sort_unstable();
        assert_eq!(
            cases,
            vec![0, 1, 2],
            "SwitchInt targets are the variant indices"
        );
        let producers = op_by_result(&graph);
        assert!(
            producers.values().all(|kind| {
                !matches!(
                    kind,
                    OpKind::FieldRead { field, .. } if field.name == "__discriminant"
                )
            }),
            "a raw niche tag is not the sum-shell __discriminant"
        );
        let raw_load = producers.get(&raw_load_id(&producers, discr.id()));
        assert!(
            matches!(
                raw_load,
                Some(OpKind::RawLoad {
                    itemsize: 1,
                    is_item_signed: false,
                    ..
                })
            ),
            "tag load is one unsigned byte, got {raw_load:?}"
        );
        for raw in [0u128, 10, 11, 12, 13, 255] {
            let decoded = eval_int(&producers, &discr, raw as i64);
            let expect = crate::front::host_layout::host_variant_index(&tag, raw) as i64;
            assert_eq!(decoded, expect, "raw tag {raw} decodes to variant {expect}");
            assert!(
                cases.contains(&decoded),
                "decoded variant {decoded} is not a SwitchInt target {cases:?}"
            );
        }
    }

    fn switch_int_cases(
        graph: &crate::model::FunctionGraph,
    ) -> (crate::flowspace::model::Variable, Vec<i64>) {
        let mut found = None;
        for block in &graph.blocks {
            let Some(crate::model::ExitSwitch::Value(var)) = &block.exitswitch else {
                continue;
            };
            let mut cases = Vec::new();
            for link in &block.exits {
                if let Some(crate::model::ExitCase::Const(
                    crate::flowspace::model::ConstValue::Int(n),
                )) = &link.exitcase
                {
                    cases.push(*n);
                }
            }
            if !cases.is_empty() {
                found = Some((var.clone(), cases));
            }
        }
        found.expect("SwitchInt on the decoded discriminant")
    }

    fn op_by_result(
        graph: &crate::model::FunctionGraph,
    ) -> std::collections::HashMap<u64, crate::model::OpKind> {
        let mut ops = std::collections::HashMap::new();
        for block in &graph.blocks {
            for op in &block.operations {
                if let Some(result) = &op.result {
                    ops.insert(result.id(), op.kind.clone());
                }
            }
        }
        ops
    }

    fn raw_load_id(ops: &std::collections::HashMap<u64, crate::model::OpKind>, id: u64) -> u64 {
        fn walk(
            ops: &std::collections::HashMap<u64, crate::model::OpKind>,
            id: u64,
        ) -> Option<u64> {
            let kind = ops.get(&id)?;
            match kind {
                crate::model::OpKind::RawLoad { .. } => Some(id),
                crate::model::OpKind::BinOp { lhs, rhs, .. } => {
                    walk(ops, lhs.id()).or_else(|| walk(ops, rhs.id()))
                }
                _ => None,
            }
        }
        walk(ops, id).expect("decode reads a raw tag")
    }

    fn eval_int(
        ops: &std::collections::HashMap<u64, crate::model::OpKind>,
        var: &crate::flowspace::model::Variable,
        raw_tag: i64,
    ) -> i64 {
        fn rec(
            ops: &std::collections::HashMap<u64, crate::model::OpKind>,
            id: u64,
            raw_tag: i64,
            seen: &mut Vec<u64>,
        ) -> i64 {
            if seen.contains(&id) {
                panic!("cycle in discriminant decode at {id}");
            }
            seen.push(id);
            let kind = ops.get(&id).unwrap_or_else(|| {
                panic!("discriminant decode reads undefined var {id}");
            });
            let value = match kind {
                crate::model::OpKind::ConstInt(n) => *n,
                crate::model::OpKind::RawLoad { .. } => raw_tag,
                crate::model::OpKind::BinOp { op, lhs, rhs, .. } => {
                    let a = rec(ops, lhs.id(), raw_tag, seen);
                    let b = rec(ops, rhs.id(), raw_tag, seen);
                    match op.as_str() {
                        "sub" => a.wrapping_sub(b),
                        "add" => a.wrapping_add(b),
                        "mul" => a.wrapping_mul(b),
                        "and" => a & b,
                        "eq" => i64::from((a as u64) == (b as u64)),
                        "uint_le" => i64::from((a as u64) <= (b as u64)),
                        other => panic!("unexpected decode op {other}"),
                    }
                }
                other => panic!("unexpected decode producer {other:?}"),
            };
            seen.pop();
            value
        }
        rec(ops, var.id(), raw_tag, &mut Vec::new())
    }

    /// `Tuple<i64>` is one signed word. `jtransform.py _rewrite_raw_malloc`
    /// passes that size as a constant; `int_copy` is in `USE_C_FORM`, so
    /// the byte-sized 8 is an inline `c` immediate rather than a pool slot.
    fn raw_tuple_i64_layout() -> crate::call::StructLayout {
        use crate::call::StructFieldLayout;
        use majit_ir::descr::ArrayFlag;
        use majit_ir::value::Type;

        crate::call::StructLayout {
            size: 8,
            align: 8,
            gckind: crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw,
            fields: vec![StructFieldLayout {
                name: "f".into(),
                offset: 0,
                size: 8,
                flag: ArrayFlag::Signed,
                field_type: Type::Int,
                rank: None,
            }],
            host: None,
            ll_struct: std::cell::RefCell::new(None),
            ll_struct_by_args: std::cell::RefCell::new(std::collections::HashMap::new()),
        }
    }

    fn assemble_raw_malloc_roundtrip(zero: bool) -> (JitCodeBody, Assembler) {
        use crate::call::CallControl;
        use crate::flatten::flatten_graph;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FieldDescriptor, FunctionGraph, LinkArg, OpKind, ValueType};

        const MALLOC: i64 = 0x1111_0001;
        const ZERO: i64 = 0x2222_0002;
        const FREE: i64 = 0x3333_0003;
        let owner = "Tuple<i64>";
        let mut cc = CallControl::new();
        let sid = majit_ir::descr::struct_id_for_name(owner).expect("Tuple<i64> shape id");
        cc.set_struct_layout(sid, raw_tuple_i64_layout());
        cc.register_function_fnaddr(
            crate::parse::CallPath::from_segments([
                "majit_rlib",
                "rffi",
                "ll_raw_malloc_fixedsize",
            ]),
            MALLOC,
        );
        cc.register_function_fnaddr(
            crate::parse::CallPath::from_segments([
                "majit_rlib",
                "rffi",
                "ll_raw_malloc_fixedsize_zero",
            ]),
            ZERO,
        );
        cc.register_function_fnaddr(
            crate::parse::CallPath::from_segments(["majit_rlib", "rffi", "ll_raw_free"]),
            FREE,
        );

        let mut graph = FunctionGraph::new("raw_fixedsize");
        let x = push_input_var(&mut graph, "x", ValueType::Int);
        FunctionGraph::set_concretetype_of_inline(&x, crate::model::ConcreteType::Signed);
        let field = FieldDescriptor::new("f", Some(owner.into()));
        let p = graph
            .push_op_var(
                graph.startblock,
                OpKind::RawMalloc {
                    owner: owner.into(),
                    zero,
                },
                true,
            )
            .expect("raw malloc result");
        let _ = graph.push_op_var(
            graph.startblock,
            OpKind::FieldWrite {
                base: p.clone(),
                field: field.clone(),
                value: LinkArg::Value(x),
                ty: ValueType::Int,
            },
            false,
        );
        let y = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: p.clone(),
                    field,
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .expect("field read result");
        let _ = graph.push_op_var(graph.startblock, OpKind::RawFree { ptr: p }, false);
        graph.set_return(graph.startblock, Some(y.clone()));
        FunctionGraph::set_concretetype_of_inline(&y, crate::model::ConcreteType::Signed);

        let config = GraphTransformConfig::default();
        let mut rewritten = {
            let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
            transformer.transform(&graph).graph
        };
        regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
        let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
        let mut flat = flatten_graph(&rewritten, &mut regallocs);
        let mut asm = Assembler::new();
        let body = asm.assemble_with_callcontrol(&mut flat, &regallocs, Some(&cc));
        (body, asm)
    }

    fn assert_raw_malloc_assembly(
        body: &JitCodeBody,
        asm: &Assembler,
        malloc_addr: i64,
        absent: i64,
    ) {
        const FREE: i64 = 0x3333_0003;
        let keys: Vec<&String> = asm.insns.keys().collect();
        assert!(
            keys.iter()
                .any(|k| k.as_str() == "residual_call_ir_i/iIRd>i"),
            "malloc residual missing, got {keys:?}"
        );
        assert!(
            keys.iter().any(|k| k.as_str() == "setfield_raw_i/iid"),
            "setfield_raw_i missing, got {keys:?}"
        );
        assert!(
            keys.iter().any(|k| k.as_str() == "getfield_raw_i/id>i"),
            "getfield_raw_i missing, got {keys:?}"
        );
        assert!(
            keys.iter().any(|k| k.as_str() == "residual_call_ir_v/iIRd"),
            "raw_free residual missing, got {keys:?}"
        );
        assert!(
            keys.iter().any(|k| k.starts_with("live/")),
            "can-raise malloc must emit live/, got {keys:?}"
        );
        assert!(
            body.constants_i.contains(&malloc_addr),
            "helper address {malloc_addr:#x} missing from {:?}",
            body.constants_i
        );
        assert!(
            body.constants_i.contains(&FREE),
            "ll_raw_free address missing from {:?}",
            body.constants_i
        );
        assert!(
            !body.constants_i.contains(&absent),
            "other helper {absent:#x} leaked into {:?}",
            body.constants_i
        );
        // `assembler.py` `USE_C_FORM`: `int_copy` of a byte-sized constant
        // is `int_copy/c>i`, so the struct size 8 is the inline immediate.
        let size_op = *asm
            .insns
            .get("int_copy/c>i")
            .expect("size ConstInt(8) assembles as int_copy/c>i");
        assert!(
            body.code.windows(2).any(|w| w[0] == size_op && w[1] == 8),
            "struct size 8 is not the int_copy short immediate, code {:?}",
            body.code
        );

        let mut malloc_calls = 0;
        let mut free_calls = 0;
        for descr in asm.snapshot_descrs() {
            let crate::jitcode::BhDescr::Call { calldescr } = descr else {
                continue;
            };
            match calldescr.extra_info.oopspecindex {
                majit_ir::descr::OopSpecIndex::None
                    if calldescr.extra_info.extraeffect
                        == majit_ir::descr::ExtraEffect::CanRaise =>
                {
                    assert!(calldescr.extra_info.check_can_raise(false));
                    malloc_calls += 1;
                }
                majit_ir::descr::OopSpecIndex::RawFree => {
                    assert_eq!(
                        calldescr.extra_info.extraeffect,
                        majit_ir::descr::ExtraEffect::CannotRaise
                    );
                    assert!(!calldescr.extra_info.check_can_raise(false));
                    free_calls += 1;
                }
                other => panic!("unexpected residual oopspec {other:?}"),
            }
        }
        assert_eq!(malloc_calls, 1, "one fixed-size malloc residual");
        assert_eq!(free_calls, 1, "one raw_free residual");
    }

    #[test]
    fn raw_malloc_fixedsize_assembles_residual_setfield_getfield_and_raw_free() {
        let (body, asm) = assemble_raw_malloc_roundtrip(false);
        assert_raw_malloc_assembly(&body, &asm, 0x1111_0001, 0x2222_0002);
    }

    #[test]
    fn raw_malloc_fixedsize_zero_assembles_the_zero_helper() {
        let (body, asm) = assemble_raw_malloc_roundtrip(true);
        assert_raw_malloc_assembly(&body, &asm, 0x2222_0002, 0x1111_0001);
    }

    /// `promote_gc_field_bases` rehomes a Signed GC FieldRead base so
    /// the assembler emits the canonical `/rd>i` key.
    #[test]
    fn promote_then_assemble_emits_canonical_getfield_gc_rd() {
        use crate::flatten::flatten_graph;
        use crate::model::{FieldDescriptor, FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("promoted_getfield");
        let base_var = push_input_var(&mut graph, "obj", ValueType::Int);
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base_var.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .expect("field read has a result");
        graph.set_return(graph.startblock, Some(result.clone()));
        FunctionGraph::set_concretetype_of_inline(
            &base_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(
            &result,
            crate::codewriter::type_state::ConcreteType::Signed,
        );

        crate::codewriter::type_state::promote_gc_field_bases(&mut graph, None);
        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
        assert!(
            asm.insns.contains_key("getfield_gc_i/rd>i"),
            "promoted GC base must assemble as getfield_gc_i/rd>i, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.keys().any(|k| k.contains("/id>")),
            "promoted GC base must not emit an int-base getfield key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.keys().any(|k| k.contains("cast_int_to_ptr")),
            "a Signed base with no int use is stamped in place, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    /// A Signed GC base that also reaches a Signed successor stays in the
    /// int bank. Stamping it makes `insert_renamings` copy a Ref into an
    /// int register, which `encode_regorconst_source` rejects.
    #[test]
    fn promote_then_assemble_casts_a_signed_base_that_flows_to_an_int() {
        use crate::flatten::flatten_graph;
        use crate::model::{FieldDescriptor, FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("promoted_getfield_flows");
        let base_var = push_input_var(&mut graph, "obj", ValueType::Int);
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base_var.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .expect("field read has a result");
        FunctionGraph::set_concretetype_of_inline(
            &base_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(
            &result,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        let (next, args) = graph.create_block_with_arg_vars(1);
        let succ = args[0].clone();
        FunctionGraph::set_concretetype_of_inline(
            &succ,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        graph.set_return(next, Some(succ.clone()));
        graph.set_goto(graph.startblock, next, vec![base_var]);

        crate::codewriter::type_state::promote_gc_field_bases(&mut graph, None);
        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
        assert!(
            asm.insns.keys().any(|k| k.contains("cast_int_to_ptr")),
            "the int-flowing base is cast at the field access, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("getfield_gc_i/rd>i"),
            "the cast result is the GC base, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    /// Same bank split as the successor-flow case, with the int use in
    /// the same block: `int_add` of the Signed base. Stamping the base
    /// makes that operand a Ref register, which
    /// `encode_regorconst_source` rejects.
    #[test]
    fn promote_then_assemble_casts_a_signed_base_used_as_an_int_operand() {
        use crate::flatten::flatten_graph;
        use crate::model::{FieldDescriptor, FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("promoted_getfield_int_add");
        let base_var = push_input_var(&mut graph, "obj", ValueType::Int);
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base_var.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .expect("field read has a result");
        FunctionGraph::set_concretetype_of_inline(
            &base_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(
            &result,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        let sum = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: base_var,
                    rhs: result,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .expect("int_add has a result");
        FunctionGraph::set_concretetype_of_inline(
            &sum,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        graph.set_return(graph.startblock, Some(sum));

        crate::codewriter::type_state::promote_gc_field_bases(&mut graph, None);
        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
        assert!(
            asm.insns.keys().any(|k| k.contains("cast_int_to_ptr")),
            "the int operand's base is cast at the field access, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("getfield_gc_i/rd>i"),
            "the cast result is the GC base, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.keys().any(|k| k.starts_with("int_add/")),
            "the original word is still the int operand, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    /// A GcRef link argument into a Signed inputarg of a block that
    /// itself exits is `insert_renamings`' `int_copy` of a Ref register.
    /// `coerce_cross_bank_links` casts before regalloc so assemble accepts it.
    #[test]
    fn coerce_then_assemble_casts_a_ref_link_into_a_signed_inputarg() {
        use crate::flatten::flatten_graph;
        use crate::model::{FunctionGraph, ValueType};

        let mut graph = FunctionGraph::new("ref_link_to_signed");
        let src = push_input_var(&mut graph, "p", ValueType::Ref(None));
        FunctionGraph::set_concretetype_of_inline(
            &src,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );
        let (next, args) = graph.create_block_with_arg_vars(1);
        let input = args[0].clone();
        FunctionGraph::set_concretetype_of_inline(
            &input,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        graph.set_return(next, Some(input));
        graph.set_goto(graph.startblock, next, vec![src]);

        crate::codewriter::type_state::coerce_cross_bank_links(&mut graph);
        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
        assert!(
            asm.insns.keys().any(|k| k.contains("cast_ptr_to_int")),
            "the ref link argument is cast into the int bank, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    /// The inverse bank split: a Signed source into a GcRef inputarg.
    #[test]
    fn coerce_then_assemble_casts_a_signed_link_into_a_ref_inputarg() {
        use crate::flatten::flatten_graph;
        use crate::model::{FunctionGraph, ValueType};

        let mut graph = FunctionGraph::new("signed_link_to_ref");
        let src = push_input_var(&mut graph, "n", ValueType::Int);
        FunctionGraph::set_concretetype_of_inline(
            &src,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        let (next, args) = graph.create_block_with_arg_vars(1);
        let input = args[0].clone();
        FunctionGraph::set_concretetype_of_inline(
            &input,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );
        graph.set_return(next, Some(input));
        graph.set_goto(graph.startblock, next, vec![src]);

        crate::codewriter::type_state::coerce_cross_bank_links(&mut graph);
        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
        assert!(
            asm.insns.keys().any(|k| k.contains("cast_int_to_ptr")),
            "the signed link argument is cast into the ref bank, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    /// Every upstream vable handler declares its base as `r`. An Int-bank
    /// base would not be visited by the moving-GC root walker.
    #[test]
    #[should_panic(expected = "virtualizable base must use the Ref register bank")]
    fn assemble_rejects_int_bank_virtualizable_base() {
        use crate::flatten::flatten_graph;
        use crate::model::{FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("int_vable_base");
        let base_var = push_input_var(&mut graph, "frame", ValueType::Int);
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::VableFieldRead {
                    base: base_var.clone(),
                    field_index: 0,
                    ty: ValueType::Int,
                },
                true,
            )
            .expect("field read has a result");
        graph.set_return(graph.startblock, Some(result.clone()));
        FunctionGraph::set_concretetype_of_inline(
            &base_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(
            &result,
            crate::codewriter::type_state::ConcreteType::Signed,
        );

        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
    }

    /// `rewrite_op_getarraysize` always emits `arraylen_gc`. A `nolength`
    /// ARRAY has no length word; shipping that opcode lets blackhole
    /// `bh_arraylen_gc` panic at run time (`llmodel.py`).
    #[test]
    #[should_panic(expected = "arraylen_gc requires ArrayDescr.lendescr")]
    fn assemble_arraylen_rejects_a_headerless_descr() {
        use crate::flatten::flatten_graph;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("headerless_arraylen");
        let base_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "block".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let len_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::ArrayLen {
                    base: base_var.clone(),
                    array_type_id: Some("*const i64".into()),
                    nolength: true,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(len_var.clone()));
        FunctionGraph::set_concretetype_of_inline(
            &base_var,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );
        FunctionGraph::set_concretetype_of_inline(
            &len_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let mut rewritten = Transformer::new(&config).transform(&graph).graph;
        regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
        let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
        let mut flat = flatten_graph(&rewritten, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
    }

    /// `arraylen_vable` is a cross-layer contract: the codewriter picks the
    /// key, `insns.rs` assigns its byte, and `MIFrame::read_vable_arraylen`
    /// decodes `1B vable_reg + 2B fdescr + 2B adescr + 1B dest` and asserts
    /// the descr pair is `(VableArray, Array)`.  Three files have to agree on
    /// one wire shape, so pin the whole shape here rather than the opname
    /// alone.
    #[test]
    fn assemble_vable_arraylen_emits_the_rdd_to_i_wire_shape() {
        use crate::flatten::{FlatOp, flatten_graph};
        use crate::jtransform::{GraphTransformConfig, Transformer, VirtualizableFieldDescriptor};
        use crate::model::{FieldDescriptor, FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("vable_arraylen");
        let base_var = push_input_var(&mut graph, "frame", ValueType::Ref(None));
        let array_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base_var.clone(),
                    field: FieldDescriptor::new("locals_stack_w", Some("Frame".into())),
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();
        let len_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::ArrayLen {
                    base: array_var,
                    array_type_id: None,
                    nolength: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(len_var.clone()));
        FunctionGraph::set_concretetype_of_inline(
            &base_var,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );
        // The driver that normally supplies this is
        // `type_state::authoritative_result_types`, which answers `Signed` for
        // `VableArrayLen`; this test drives `Transformer` directly, so state
        // it here rather than depend on a pass it does not run.
        FunctionGraph::set_concretetype_of_inline(
            &len_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig {
            vable_arrays: vec![VirtualizableFieldDescriptor::new_with_arraydescr(
                "locals_stack_w",
                Some("Frame".into()),
                0,
                crate::layout::target_word_size(),
                false,
            )],
            ..Default::default()
        };
        let mut rewritten = Transformer::new(&config).transform(&graph).graph;
        regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
        let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
        let mut flat = flatten_graph(&rewritten, &mut regallocs);
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);

        assert!(
            asm.insns.contains_key("arraylen_vable/rdd>i"),
            "a vable array length must assemble to `arraylen_vable/rdd>i`, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.keys().any(|k| k.starts_with("arraylen_gc")),
            "the raw `arraylen_gc` must be gone, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );

        // The two descrs, in the order the decoder reads them.
        let ready: Vec<&crate::jitcode::BhDescr> = asm
            .descrs
            .iter()
            .filter_map(|d| match d {
                AssemblerDescr::Ready(b) => Some(&**b),
                _ => None,
            })
            .collect();
        let vable_at = ready
            .iter()
            .position(|d| matches!(d, crate::jitcode::BhDescr::VableArray { index: 0 }))
            .unwrap_or_else(|| panic!("no VableArray descr minted, got {ready:?}"));
        let array_at = ready
            .iter()
            .position(|d| matches!(d, crate::jitcode::BhDescr::Array { .. }))
            .unwrap_or_else(|| panic!("no Array descr minted, got {ready:?}"));
        assert!(
            vable_at < array_at,
            "the vable-array descr must precede the array descr, got {ready:?}"
        );
        let crate::jitcode::BhDescr::Array {
            base_size,
            itemsize,
            len_offset,
            is_array_of_pointers,
            ..
        } = ready[array_at]
        else {
            unreachable!("checked by the position above");
        };
        let word = crate::layout::target_word_size();
        assert_eq!((*base_size, *itemsize, *len_offset), (word, word, Some(0)));
        assert!(is_array_of_pointers);

        // The pool order above only says which descr was minted first.  What
        // the decoder follows is the pair of indexes in the instruction's own
        // operand bytes, so read them: `1B opcode + 1B vable_reg + 2B fdescr +
        // 2B adescr + 1B dest`, each index little-endian.  A swapped emission
        // would still leave the pool in this order and still key
        // `arraylen_vable/rdd>i`.
        let op_index = flat
            .insns
            .iter()
            .position(|flat_op| {
                matches!(flat_op, FlatOp::Op(inner)
                    if matches!(inner.kind, OpKind::VableArrayLen { .. }))
            })
            .expect("the flattened graph must carry the VableArrayLen");
        let offset = flat.insns_pos.as_ref().expect("assemble records insns_pos")[op_index];
        let wire = &body.code[offset..offset + 7];
        assert_eq!(
            wire[0], asm.insns["arraylen_vable/rdd>i"],
            "the instruction at its recorded position must be the vable arraylen"
        );
        assert_eq!(
            u16::from_le_bytes([wire[2], wire[3]]) as usize,
            vable_at,
            "the first descr operand must be the vable-array descr"
        );
        assert_eq!(
            u16::from_le_bytes([wire[4], wire[5]]) as usize,
            array_at,
            "the second descr operand must be the array descr"
        );
    }

    /// Safe `Index` / `IndexMut` / `.len()` of a virtualizable `Vec<i64>`
    /// field. The front end's address-of mark is cleared because the field is
    /// the declared array, and the codewriter then emits the `_vable` forms.
    /// `nolength: false` is the bounds check the index implies.
    #[test]
    fn vec_index_of_a_vable_array_field_lowers_to_vable_int_ops() {
        use crate::flatten::flatten_graph;
        use crate::front::mir::release_declared_vable_array_address;
        use crate::jtransform::{GraphTransformConfig, Transformer, VirtualizableFieldDescriptor};
        use crate::model::{FieldDescriptor, FunctionGraph, LinkArg, OpKind, ValueType};
        use crate::virtualizable_decl::register_virtualizable_declarations;

        register_virtualizable_declarations([("Frame".to_string(), vec!["words[*]".to_string()])]);
        struct ClearDecl;
        impl Drop for ClearDecl {
            fn drop(&mut self) {
                register_virtualizable_declarations(std::iter::empty::<(String, Vec<String>)>());
            }
        }
        let _clear = ClearDecl;

        let mut graph = FunctionGraph::new("vec_vable_index");
        let frame = push_input_var(&mut graph, "frame", ValueType::Ref(None));
        let index = push_input_var(&mut graph, "index", ValueType::Int);
        let stored = push_input_var(&mut graph, "stored", ValueType::Int);
        let array = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: frame.clone(),
                    field: FieldDescriptor::new("words", Some("Frame".into()))
                        .with_taken_by_address(true),
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();
        assert!(
            release_declared_vable_array_address(&mut graph, &array),
            "the declared array field's address mark must clear"
        );
        let loaded = graph
            .push_op_var(
                graph.startblock,
                OpKind::ArrayRead {
                    base: array.clone(),
                    index: index.clone(),
                    item_ty: ValueType::Int,
                    array_type_id: Some("[Signed]".to_string()),
                    nolength: false,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayWrite {
                base: array.clone(),
                index: index.clone(),
                value: LinkArg::Value(stored.clone()),
                item_ty: ValueType::Int,
                array_type_id: Some("[Signed]".to_string()),
                nolength: false,
            },
            false,
        );
        let len = graph
            .push_op_var(
                graph.startblock,
                OpKind::ArrayLen {
                    base: array,
                    array_type_id: None,
                    nolength: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(loaded.clone()));
        FunctionGraph::set_concretetype_of_inline(
            &frame,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );
        for var in [&index, &stored, &loaded, &len] {
            FunctionGraph::set_concretetype_of_inline(
                var,
                crate::codewriter::type_state::ConcreteType::Signed,
            );
        }

        let config = GraphTransformConfig {
            vable_arrays: vec![VirtualizableFieldDescriptor::new_with_arraydescr(
                "words",
                Some("Frame".into()),
                0,
                8,
                true,
            )],
            ..Default::default()
        };
        let mut rewritten = Transformer::new(&config).transform(&graph).graph;
        let names: Vec<String> = rewritten
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .map(|op| op_kind_to_opname(&op.kind))
            .collect();
        assert!(
            names.iter().any(|name| name == "getarrayitem_vable_i"),
            "expected getarrayitem_vable_i, got {names:?}"
        );
        assert!(
            names.iter().any(|name| name == "setarrayitem_vable_i"),
            "expected setarrayitem_vable_i, got {names:?}"
        );
        assert!(
            names.iter().any(|name| name == "arraylen_vable"),
            "expected arraylen_vable, got {names:?}"
        );

        regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
        let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
        let mut flat = flatten_graph(&rewritten, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
        for key in [
            "getarrayitem_vable_i/ridd>i",
            "setarrayitem_vable_i/riidd",
            "arraylen_vable/rdd>i",
        ] {
            assert!(
                asm.insns.contains_key(key),
                "{key} missing, got {:?}",
                asm.insns.keys().collect::<Vec<_>>()
            );
        }
    }

    /// `jtransform.py` `rewrite_op_setarrayitem` passes `op.args[2]` through.
    /// A constant store into a virtualizable array stays `LinkArg::Const`;
    /// the assembler encodes that constant as the value slot of
    /// `setarrayitem_vable_*`.
    #[test]
    fn vable_array_write_of_a_constant_encodes_the_pool_slot() {
        use crate::flatten::flatten_graph;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::front::mir::release_declared_vable_array_address;
        use crate::jtransform::{GraphTransformConfig, Transformer, VirtualizableFieldDescriptor};
        use crate::model::{FieldDescriptor, FunctionGraph, LinkArg, OpKind, ValueType};
        use crate::virtualizable_decl::register_virtualizable_declarations;

        register_virtualizable_declarations([("Frame".to_string(), vec!["words[*]".to_string()])]);
        struct ClearDecl;
        impl Drop for ClearDecl {
            fn drop(&mut self) {
                register_virtualizable_declarations(std::iter::empty::<(String, Vec<String>)>());
            }
        }
        let _clear = ClearDecl;

        let mut graph = FunctionGraph::new("vable_array_const_store");
        let frame = push_input_var(&mut graph, "frame", ValueType::Ref(None));
        let index = push_input_var(&mut graph, "index", ValueType::Int);
        let array = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: frame.clone(),
                    field: FieldDescriptor::new("words", Some("Frame".into()))
                        .with_taken_by_address(true),
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();
        assert!(release_declared_vable_array_address(&mut graph, &array));
        let stored = LinkArg::Const(Constant::new(ConstValue::Int(0)));
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayWrite {
                base: array,
                index: index.clone(),
                value: stored,
                item_ty: ValueType::Int,
                array_type_id: Some("[Signed]".to_string()),
                nolength: false,
            },
            false,
        );
        graph.set_return(graph.startblock, None);
        FunctionGraph::set_concretetype_of_inline(
            &frame,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );
        FunctionGraph::set_concretetype_of_inline(
            &index,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        let config = GraphTransformConfig {
            vable_arrays: vec![VirtualizableFieldDescriptor::new_with_arraydescr(
                "words",
                Some("Frame".into()),
                0,
                8,
                true,
            )],
            ..Default::default()
        };
        let mut rewritten = Transformer::new(&config).transform(&graph).graph;
        let write = rewritten
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find(|op| matches!(op.kind, OpKind::VableArrayWrite { .. }));
        let Some(write) = write else {
            panic!("constant vable store must rewrite to VableArrayWrite");
        };
        match &write.kind {
            OpKind::VableArrayWrite { value, .. } => {
                assert!(
                    matches!(value, LinkArg::Const(c) if matches!(c.value, ConstValue::Int(0))),
                    "rewrite_op_setarrayitem must keep the constant, got {value:?}"
                );
            }
            _ => unreachable!(),
        }
        regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
        let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
        let mut flat = flatten_graph(&rewritten, &mut regallocs);
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);
        let key = "setarrayitem_vable_i/riidd";
        let Some(&opnum) = asm.insns.get(key) else {
            panic!(
                "constant setarrayitem_vable_i missing, got {:?}",
                asm.insns.keys().collect::<Vec<_>>()
            );
        };
        let zero_slot = body
            .constants_i
            .iter()
            .position(|value| *value == 0)
            .expect("int constant pool must contain 0");
        let value_byte = body
            .code
            .windows(4)
            .find(|window| window[0] == opnum)
            .map(|window| window[3])
            .expect("write instruction missing");
        let selected = value_byte as usize - body.c_num_regs_i as usize;
        assert_eq!(
            selected, zero_slot,
            "value byte {value_byte} does not select the 0 pool slot {zero_slot}"
        );
    }

    #[test]
    fn assemble_typed_reads_use_canonical_non_v_opnames() {
        use crate::flatten::flatten_graph;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FieldDescriptor, FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("typed_reads");
        let base_var = push_input_var(&mut graph, "obj", ValueType::Ref(None));
        let index_var = push_input_var(&mut graph, "i", ValueType::Int);
        let field_result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base_var.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Unknown,
                    pure: false,
                },
                true,
            )
            .unwrap();
        let array_result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::ArrayRead {
                    base: base_var.clone(),
                    index: index_var.clone(),
                    item_ty: ValueType::Unknown,
                    array_type_id: None,
                    nolength: false,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(array_result_var.clone()));

        // Publish kinds to graph cells before jtransform.  Variable
        // Rc-shares the concretetype cell across clones, so the
        // cloned rewritten graph picks up the same kinds.
        FunctionGraph::set_concretetype_of_inline(
            &base_var,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );
        FunctionGraph::set_concretetype_of_inline(
            &index_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(
            &field_result_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(
            &array_result_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let mut rewritten = Transformer::new(&config).transform(&graph).graph;
        regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
        let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
        let mut flat = flatten_graph(&rewritten, &mut regallocs);

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        assert!(
            asm.insns.contains_key("getfield_gc_i/rd>i"),
            "expected canonical getfield_gc_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("getarrayitem_gc_i/rid>i"),
            "expected canonical getarrayitem_gc_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getfield_gc_v/rd>i"),
            "unexpected getfield_gc_v/rd>i key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getfield_gc_v/id>i"),
            "unexpected getfield_gc_v/id>i key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getarrayitem_gc_v/rid>i"),
            "unexpected getarrayitem_gc_v/rid>i key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getarrayitem_gc_v/iid>i"),
            "unexpected getarrayitem_gc_v/iid>i key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getarrayitem_gc_v/ird>i"),
            "unexpected getarrayitem_gc_v/ird>i key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    /// A `pure: true` ArrayRead assembles to the foldable
    /// `getarrayitem_gc_i_pure` opcode (rlist.py:724
    /// `ll_getitem_foldable_nonneg`); a `pure: false` ArrayRead keeps the
    /// non-pure `getarrayitem_gc_i`.  The `rid>i` argcodes are byte-identical
    /// either way (`blackhole.py` aliases the pure handler) — only
    /// the opname/byte differs.
    fn assemble_arrayread_pure(pure: bool) -> Assembler {
        use crate::flatten::flatten_graph;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("arrayread_pure");
        let base_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "block".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let index_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "i".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let array_result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::ArrayRead {
                    base: base_var.clone(),
                    index: index_var.clone(),
                    item_ty: ValueType::Int,
                    array_type_id: None,
                    nolength: false,
                    pure,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(array_result_var.clone()));
        // `assemble` derives every operand kind from the variable
        // concretetype + regallocs coloring — color the base (GcRef array
        // pointer) and index (Signed) inputs as well as the result, else
        // `lookup_coloring` finds no class for them.
        FunctionGraph::set_concretetype_of_inline(
            &base_var,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );
        FunctionGraph::set_concretetype_of_inline(
            &index_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(
            &array_result_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let mut rewritten = Transformer::new(&config).transform(&graph).graph;
        regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
        let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
        let mut flat = flatten_graph(&rewritten, &mut regallocs);

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
        asm
    }

    #[test]
    fn assemble_pure_arrayread_uses_pure_opcode() {
        let asm = assemble_arrayread_pure(true);
        assert!(
            asm.insns.contains_key("getarrayitem_gc_i_pure/rid>i"),
            "expected foldable getarrayitem_gc_i_pure key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getarrayitem_gc_i/rid>i"),
            "unexpected non-pure getarrayitem_gc_i key for pure ArrayRead: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn assemble_non_pure_arrayread_uses_non_pure_opcode() {
        let asm = assemble_arrayread_pure(false);
        assert!(
            asm.insns.contains_key("getarrayitem_gc_i/rid>i"),
            "expected non-pure getarrayitem_gc_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getarrayitem_gc_i_pure/rid>i"),
            "unexpected pure getarrayitem_gc_i_pure key for non-pure ArrayRead: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn assemble_skips_input_opnames_after_flatten() {
        use crate::flatten::flatten_graph;
        use crate::model::{FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("input_free_bytecode");
        let lhs_var = push_input_var(&mut graph, "lhs", ValueType::Int);
        let rhs_var = push_input_var(&mut graph, "rhs", ValueType::Int);
        let sum_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: lhs_var.clone(),
                    rhs: rhs_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(sum_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&lhs_var, crate::model::ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs_var, crate::model::ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&sum_var, crate::model::ConcreteType::Signed);

        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = flatten_graph(&graph, &mut regallocs);
        assert!(
            !flat.insns.iter().any(
                |op| matches!(op, FlatOp::Op(inner) if matches!(inner.kind, OpKind::Input { .. }))
            ),
            "flatten unexpectedly left input ops: {:?}",
            flat.insns
        );

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        assert!(
            !asm.insns.keys().any(|key| key.starts_with("input_")),
            "unexpected input opcode keys: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    /// `Option<fn>` is-none is `int_eq` of two int-bank addresses
    /// (`jtransform.py` `rewrite_op_ptr_eq` / `_rewrite_nongc_ptrs`).
    /// The null arm is `null_fn` (`ValueType::Int`), same bank as the
    /// function pointer, so the compare is `ii`, never `int_is_/ri`.
    #[test]
    fn option_fn_is_none_lowers_to_int_eq() {
        use crate::flatten::flatten_graph;
        use crate::front::option_is_none::{IsNoneSite, rewire_is_none_call_sites};
        use crate::model::{CallTarget, FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("option_fn_is_none");
        let opt = push_input_var(&mut graph, "opt", ValueType::Int);
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::method("is_none", Some("core::option::Option".into())),
                    args: crate::model::call_args(vec![opt.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result.clone()));
        let rewritten = rewire_is_none_call_sites(
            &mut graph,
            &[IsNoneSite {
                result_var: result.clone(),
                option_owner: "core::option::Option".into(),
                is_some: false,
                niche: true,
                niche_null_cast: None,
                fn_ptr: true,
                scalar_niche: false,
            }],
        );
        assert_eq!(rewritten, 1);

        let null = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find_map(|op| match &op.kind {
                OpKind::Call {
                    target,
                    result_ty,
                    args,
                    ..
                } if target.to_string() == "core::ptr::null_fn"
                    && args.is_empty()
                    && *result_ty == ValueType::Int =>
                {
                    op.result.clone()
                }
                _ => None,
            })
            .expect("null_fn Int");
        let is_none = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find(|op| op.result.as_ref() == Some(&result))
            .expect("is_none producer");
        match &is_none.kind {
            OpKind::BinOp {
                op,
                lhs,
                rhs,
                result_ty,
            } => {
                assert_eq!(op, "is_");
                assert_eq!(lhs, &opt);
                assert_eq!(rhs, &null);
                assert_eq!(*result_ty, ValueType::Int);
            }
            other => panic!("is_none is not is_ on two ints: {other:?}"),
        }

        FunctionGraph::set_concretetype_of_inline(&opt, crate::model::ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&null, crate::model::ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&result, crate::model::ConcreteType::Signed);

        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
        assert!(
            asm.insns.contains_key("int_eq/ii>i"),
            "Option<fn> is-none must be int_eq on two ints, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.keys().any(|key| key.contains("int_is_")),
            "Option<fn> is-none must not emit int_is_, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    #[should_panic(expected = "OpKind::Input must be eliminated before assembly")]
    fn assemble_rejects_input_ops() {
        let mut graph = crate::model::FunctionGraph::new("bad_input");
        let vid_var = graph.alloc_value_var();
        let mut flat = SSARepr {
            name: "bad_input".into(),
            insns: vec![FlatOp::op(crate::model::SpaceOperation {
                result: Some(vid_var),
                kind: crate::model::OpKind::Input {
                    name: "x".into(),
                    ty: crate::model::ValueType::Int,
                    class_root: None,
                },
            })],
            num_blocks: 1,
            insns_pos: None,
        };
        // Empty regallocs — the test panics before any coloring lookup
        // runs (the assembler rejects OpKind::Input outright).
        let regallocs = HashMap::new();

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
    }

    /// `rpython/jit/codewriter/jtransform.py:611` —
    /// `<kind>_guard_value` family opname mapping:
    ///   * `'i'` → `int_guard_value`
    ///   * `'r'` → `ref_guard_value`
    ///   * `'f'` → `float_guard_value`
    ///
    /// Pyre does not exercise the `str_guard_value` mapping
    /// (`jtransform.py:631 promote_string` / `:647 promote_unicode`)
    /// because the `PromoteString` / `PromoteUnicode` rewrite arms
    /// panic before emitting — pyre-object has no `rstr.STR` /
    /// `rstr.UNICODE` GC layout (`rpython/rtyper/lltypesystem/
    /// rstr.py:1226-1246`).
    #[test]
    fn op_kind_to_opname_routes_guard_value_kind_chars() {
        use crate::flowspace::model::Variable;
        use crate::model::OpKind;
        let value_var = Variable::new();
        let opnames = ['i', 'r', 'f'].map(|kc| {
            op_kind_to_opname(&OpKind::GuardValue {
                value: value_var.clone(),
                kind_char: kc,
            })
        });
        assert_eq!(opnames[0], "int_guard_value");
        assert_eq!(opnames[1], "ref_guard_value");
        assert_eq!(opnames[2], "float_guard_value");
    }

    #[test]
    fn embedded_liveness_prefix_shifts_runtime_registered_offsets() {
        let mut asm = Assembler::new();
        assert_eq!(asm._register_liveness_offset(&[2], &[], &[]), 0);
        let original = asm.all_liveness().to_vec();
        let prefix = [0_u8, 0, 0];

        asm.prepend_embedded_liveness(&prefix);

        assert_eq!(asm.all_liveness(), [prefix.as_slice(), &original].concat());
        assert_eq!(asm._register_liveness_offset(&[2], &[], &[]), prefix.len());
        asm.prepend_embedded_liveness(&prefix);
        assert_eq!(asm.all_liveness(), [prefix.as_slice(), &original].concat());
    }

    fn point_struct_callcontrol() -> crate::call::CallControl {
        let mut cc = crate::call::CallControl::new();
        let mut registry = crate::front::StructFieldRegistry::default();
        registry.fields.insert(
            "Point".to_string(),
            vec![
                ("x".to_string(), "i64".to_string()),
                ("p".to_string(), "&PyObject".to_string()),
            ],
        );
        cc.set_struct_fields(registry);
        cc.set_known_struct_names(["Point".to_string()].into_iter().collect());
        cc
    }

    /// `jtransform.py rewrite_op_getinteriorfield` emits
    /// `getinteriorfield_gc_i/rid>i` plus an `InteriorFieldDescr` for the field.
    #[test]
    fn interior_field_read_encodes_getinteriorfield_with_descr() {
        use crate::model::{FieldDescriptor, OpKind, ValueType};

        let cc = point_struct_callcontrol();
        let mut graph = crate::model::FunctionGraph::new("interior_read");
        let base = push_input_var(&mut graph, "arr", ValueType::Ref(None));
        let index = push_input_var(&mut graph, "i", ValueType::Int);
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::InteriorFieldRead {
                    base: base.clone(),
                    index: index.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    item_ty: ValueType::Int,
                    array_type_id: Some("Vec<Point>".into()),
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result.clone()));
        crate::model::FunctionGraph::set_concretetype_of_inline(
            &base,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );
        crate::model::FunctionGraph::set_concretetype_of_inline(
            &index,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        crate::model::FunctionGraph::set_concretetype_of_inline(
            &result,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = crate::flatten::flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble_with_callcontrol(&mut flat, &regallocs, Some(&cc));
        assert!(
            asm.insns.contains_key("getinteriorfield_gc_i/rid>i"),
            "expected getinteriorfield_gc_i/rid>i, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        let descrs = asm.snapshot_descrs();
        assert!(
            descrs.iter().any(|d| matches!(
                d,
                crate::jitcode::BhDescr::InteriorField { field, .. }
                    if matches!(
                        field.as_ref(),
                        crate::jitcode::BhDescr::Field { name, .. } if name == "x" || name.ends_with(".x")
                    )
            )),
            "expected an interior field descr for x, got {descrs:?}"
        );
    }

    /// `jtransform.py rewrite_op_setinteriorfield` emits
    /// `setinteriorfield_gc_r/rird` plus an `InteriorFieldDescr` for the field.
    #[test]
    fn interior_field_write_encodes_setinteriorfield_with_descr() {
        use crate::model::{FieldDescriptor, OpKind, ValueType};

        let cc = point_struct_callcontrol();
        let mut graph = crate::model::FunctionGraph::new("interior_write");
        let base = push_input_var(&mut graph, "arr", ValueType::Ref(None));
        let index = push_input_var(&mut graph, "i", ValueType::Int);
        let value = push_input_var(&mut graph, "v", ValueType::Ref(None));
        graph.push_op_var(
            graph.startblock,
            OpKind::InteriorFieldWrite {
                base: base.clone(),
                index: index.clone(),
                field: FieldDescriptor::new("p", Some("Point".into())),
                value: value.clone(),
                item_ty: ValueType::Ref(None),
                array_type_id: Some("Vec<Point>".into()),
            },
            false,
        );
        graph.set_return(graph.startblock, None);
        crate::model::FunctionGraph::set_concretetype_of_inline(
            &base,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );
        crate::model::FunctionGraph::set_concretetype_of_inline(
            &index,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        crate::model::FunctionGraph::set_concretetype_of_inline(
            &value,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );
        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = crate::flatten::flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble_with_callcontrol(&mut flat, &regallocs, Some(&cc));
        assert!(
            asm.insns.contains_key("setinteriorfield_gc_r/rird"),
            "expected setinteriorfield_gc_r/rird, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        let descrs = asm.snapshot_descrs();
        assert!(
            descrs.iter().any(|d| matches!(
                d,
                crate::jitcode::BhDescr::InteriorField { field, .. }
                    if matches!(
                        field.as_ref(),
                        crate::jitcode::BhDescr::Field { name, .. } if name == "p" || name.ends_with(".p")
                    )
            )),
            "expected an interior field descr for p, got {descrs:?}"
        );
    }

    #[test]
    #[should_panic]
    fn interior_field_read_without_descr_panics() {
        use crate::model::{FieldDescriptor, OpKind, ValueType};

        let cc = point_struct_callcontrol();
        let mut graph = crate::model::FunctionGraph::new("interior_missing");
        let base = push_input_var(&mut graph, "arr", ValueType::Ref(None));
        let index = push_input_var(&mut graph, "i", ValueType::Int);
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::InteriorFieldRead {
                    base: base.clone(),
                    index: index.clone(),
                    field: FieldDescriptor::new("x", Some("Missing".into())),
                    item_ty: ValueType::Int,
                    array_type_id: Some("Vec<Missing>".into()),
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result.clone()));
        crate::model::FunctionGraph::set_concretetype_of_inline(
            &base,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );
        crate::model::FunctionGraph::set_concretetype_of_inline(
            &index,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        crate::model::FunctionGraph::set_concretetype_of_inline(
            &result,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = crate::flatten::flatten_graph(&graph, &mut regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble_with_callcontrol(&mut flat, &regallocs, Some(&cc));
    }
}
