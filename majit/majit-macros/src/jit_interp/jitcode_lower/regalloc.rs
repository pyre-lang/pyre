//! Proc-macro JitCode register allocation.
//!
//! RPython runs `tool/algo/regalloc.py::RegAllocator` on the graph before
//! `flatten.py` emits numbered registers (`codewriter.py::CodeWriter.make_jitcode`).
//! The proc-macro lowerer used to number every temporary monotonically and
//! bake those numbers straight into its future `JitCodeBuilder` calls.  This
//! pass colors the still-symbolic [`OpMeta`] control-flow graph, then rewrites
//! the not-yet-executed builder statements.  In particular, it runs before
//! liveness encoding and before bytecode flattening at macro-expansion time.

use super::*;
use majit_jitcode::tool::algo::color::DependencyGraph;
use quote::{ToTokens, quote};
use std::collections::{BTreeSet, HashMap};
use syn::visit_mut::VisitMut;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct RegisterCounts {
    pub(super) ints: u16,
    pub(super) refs: u16,
    pub(super) floats: u16,
}

impl RegisterCounts {
    fn for_kind(self, kind: BindingKind) -> u16 {
        match kind {
            BindingKind::Int => self.ints,
            BindingKind::Ref => self.refs,
            BindingKind::Float => self.floats,
        }
    }

    fn observe(&mut self, reg: Register) {
        let end = u16::from(reg.index) + 1;
        match reg.kind {
            BindingKind::Int => self.ints = self.ints.max(end),
            BindingKind::Ref => self.refs = self.refs.max(end),
            BindingKind::Float => self.floats = self.floats.max(end),
        }
    }
}

/// RPython `tool/algo/unionfind.py::UnionFind`, narrowed to the symbolic
/// registers owned by this proc-macro lowering path.
///
/// `majit-translate` has the same private helper for real flowspace
/// Variables. Keeping this one local avoids making that implementation part
/// of the cross-crate API merely for the temporary adapter that disappears
/// once proc-macro helpers enter the normal codewriter graph.
struct UnionFind {
    parent: HashMap<Register, Register>,
    weight: HashMap<Register, usize>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: HashMap::new(),
            weight: HashMap::new(),
        }
    }

    fn find_rep(&mut self, reg: Register) -> Register {
        if let std::collections::hash_map::Entry::Vacant(e) = self.parent.entry(reg) {
            e.insert(reg);
            self.weight.insert(reg, 1);
            return reg;
        }
        let mut root = reg;
        while self.parent[&root] != root {
            root = self.parent[&root];
        }
        let mut current = reg;
        while current != root {
            let next = self.parent[&current];
            self.parent.insert(current, root);
            current = next;
        }
        root
    }

    fn union(&mut self, left: Register, right: Register) -> Register {
        let left = self.find_rep(left);
        let right = self.find_rep(right);
        if left == right {
            return left;
        }
        let left_weight = self.weight[&left];
        let right_weight = self.weight[&right];
        let (winner, loser) = if left_weight >= right_weight {
            (left, right)
        } else {
            (right, left)
        };
        self.parent.insert(loser, winner);
        self.weight.remove(&loser);
        self.weight.insert(winner, left_weight + right_weight);
        winner
    }
}

fn successors(ops: &[OpMeta]) -> Vec<Vec<usize>> {
    let labels: HashMap<String, usize> = ops
        .iter()
        .enumerate()
        .filter_map(|(index, op)| match op.control {
            ControlFlowClass::LabelDef => op
                .target_label
                .as_ref()
                .map(|label| (label.to_string(), index)),
            _ => None,
        })
        .collect();
    ops.iter()
        .enumerate()
        .map(|(index, op)| {
            let fallthrough = || (index + 1 < ops.len()).then_some(index + 1);
            match op.control {
                ControlFlowClass::Terminal => Vec::new(),
                ControlFlowClass::UnconditionalJump => op
                    .target_label
                    .as_ref()
                    .and_then(|label| labels.get(&label.to_string()).copied())
                    .into_iter()
                    .collect(),
                ControlFlowClass::ConditionalGuard => {
                    let mut out: Vec<usize> = fallthrough().into_iter().collect();
                    if let Some(target) = op
                        .target_label
                        .as_ref()
                        .and_then(|label| labels.get(&label.to_string()).copied())
                        && !out.contains(&target)
                    {
                        out.push(target);
                    }
                    out
                }
                // `liveness.py _compute_liveness_must_continue` merges the
                // alive set of every TLabel a `-live-` names. The `-live-`
                // ahead of a `switch` names each case target, which are the
                // switch's exits.
                ControlFlowClass::LiveMarker => {
                    let mut out: Vec<usize> = fallthrough().into_iter().collect();
                    for target in op
                        .live_target_labels
                        .iter()
                        .filter_map(|label| labels.get(&label.to_string()).copied())
                    {
                        if !out.contains(&target) {
                            out.push(target);
                        }
                    }
                    out
                }
                ControlFlowClass::Linear | ControlFlowClass::LabelDef => {
                    fallthrough().into_iter().collect()
                }
            }
        })
        .collect()
}

fn live_sets(ops: &[OpMeta]) -> (Vec<BTreeSet<Register>>, Vec<BTreeSet<Register>>) {
    let succ = successors(ops);
    let mut live_in = vec![BTreeSet::new(); ops.len()];
    let mut live_out = vec![BTreeSet::new(); ops.len()];
    loop {
        let mut changed = false;
        for index in (0..ops.len()).rev() {
            let mut out = BTreeSet::new();
            for &next in &succ[index] {
                out.extend(live_in[next].iter().copied());
            }
            let mut input = out.clone();
            for written in &ops[index].writes {
                input.remove(written);
            }
            input.extend(ops[index].reads.iter().copied());
            if out != live_out[index] || input != live_in[index] {
                live_out[index] = out;
                live_in[index] = input;
                changed = true;
            }
        }
        if !changed {
            return (live_in, live_out);
        }
    }
}

fn add_clique(graph: &mut DependencyGraph<Register>, regs: &BTreeSet<Register>) {
    for &reg in regs {
        graph.add_node(reg);
    }
    let regs: Vec<_> = regs.iter().copied().collect();
    for (index, &left) in regs.iter().enumerate() {
        for &right in &regs[index + 1..] {
            if left.kind == right.kind && !graph.has_edge(&left, &right) {
                graph.add_edge(left, right);
            }
        }
    }
}

fn coloring(
    ops: &[OpMeta],
    reserved: RegisterCounts,
    pinned: RegisterCounts,
    return_reg: Option<Register>,
) -> HashMap<Register, Register> {
    let mut owned_ops;
    let ops = if let Some(return_reg) = return_reg {
        owned_ops = ops.to_vec();
        owned_ops.push(OpMeta::terminal(vec![return_reg]));
        owned_ops.as_slice()
    } else {
        ops
    };
    let (live_in, live_out) = live_sets(ops);
    let mut graphs: HashMap<BindingKind, DependencyGraph<Register>> = HashMap::new();
    for kind in [BindingKind::Int, BindingKind::Ref, BindingKind::Float] {
        graphs.insert(kind, DependencyGraph::new());
    }
    // `RegAllocator.make_dependencies` starts every block with all of its
    // inputargs live and makes that set a clique. The proc-macro ABI registers
    // are the entry block's inputargs; seed them explicitly so even an unused
    // parameter retains a distinct caller slot.
    for kind in [BindingKind::Int, BindingKind::Ref, BindingKind::Float] {
        let inputs: BTreeSet<_> = (0..reserved.for_kind(kind))
            .map(|index| Register::new(kind, index))
            .collect();
        add_clique(graphs.get_mut(&kind).unwrap(), &inputs);
    }
    for op in ops {
        for &reg in op.reads.iter().chain(op.writes.iter()) {
            graphs.get_mut(&reg.kind).unwrap().add_node(reg);
        }
    }
    for set in live_in.iter().chain(live_out.iter()) {
        for kind in [BindingKind::Int, BindingKind::Ref, BindingKind::Float] {
            let bank: BTreeSet<_> = set.iter().filter(|reg| reg.kind == kind).copied().collect();
            add_clique(graphs.get_mut(&kind).unwrap(), &bank);
        }
    }
    for (op, out) in ops.iter().zip(&live_out) {
        for &written in &op.writes {
            let graph = graphs.get_mut(&written.kind).unwrap();
            for &alive in out {
                if alive.kind == written.kind
                    && alive != written
                    && !graph.has_edge(&written, &alive)
                {
                    graph.add_edge(written, alive);
                }
            }
        }
    }

    // A pinned register holds state the runtime reads or seeds outside any
    // operation's reads and writes (the portal inputs, the state-field
    // identity slots a guard-failure resume restores). It keeps its number
    // for the whole JitCode, so it interferes with every other register of
    // its bank and never coalesces.
    for kind in [BindingKind::Int, BindingKind::Ref, BindingKind::Float] {
        let graph = graphs.get_mut(&kind).unwrap();
        let pinned_regs: Vec<Register> = (0..pinned.for_kind(kind))
            .map(|index| Register::new(kind, index))
            .collect();
        for &reg in &pinned_regs {
            graph.add_node(reg);
        }
        for &reg in &pinned_regs {
            for other in graph.getnodes() {
                if other != reg && !graph.has_edge(&reg, &other) {
                    graph.add_edge(reg, other);
                }
            }
        }
    }

    // RPython `tool/algo/regalloc.py::RegAllocator.coalesce_variables` walks
    // blocks from the end and coalesces each link argument with the matching
    // target inputarg before coloring. The proc-macro CFG has already
    // flattened those links into typed Move operations, so those moves are
    // exactly the source/target pairs to feed to the same algorithm. This was
    // the missing half of this adapter's claimed pre-flatten allocation: every
    // branch join survived as a runtime `int_copy` even when its Variables did
    // not interfere.
    let originals: BTreeSet<Register> = ops
        .iter()
        .flat_map(|op| op.reads.iter().chain(&op.writes))
        .copied()
        .chain(return_reg)
        .collect();
    let mut unionfind = UnionFind::new();
    for op in ops.iter().rev() {
        if !matches!(
            op.kind,
            OpKind::MoveI | OpKind::MoveR | OpKind::MoveF | OpKind::CastIntToWord
        ) || op.reads.len() != 1
            || op.writes.len() != 1
        {
            continue;
        }
        let source = unionfind.find_rep(op.reads[0]);
        let target = unionfind.find_rep(op.writes[0]);
        if source == target || source.kind != target.kind {
            continue;
        }
        let graph = graphs.get_mut(&source.kind).unwrap();
        if graph.has_edge(&source, &target) {
            continue;
        }
        let representative = unionfind.union(source, target);
        if representative == source {
            graph.coalesce(target, source);
        } else {
            graph.coalesce(source, target);
        }
    }

    let mut result = HashMap::new();
    for kind in [BindingKind::Int, BindingKind::Ref, BindingKind::Float] {
        let fixed = reserved.for_kind(kind);
        let graph = &graphs[&kind];
        let mut representative_colors = graph.find_node_coloring();

        // RPython `flatten.py::GraphFlattener.enforce_input_args` does not
        // reserve the ABI prefix while coloring. It swaps colors afterwards,
        // which lets a temporary reuse a dead input slot while still making
        // caller-visible inputargs dense at 0..N. The former adapter excluded
        // the entire prefix from every temporary.
        for input_index in 0..fixed {
            let input = Register::new(kind, input_index);
            let representative = unionfind.find_rep(input);
            let Some(current_color) = representative_colors.get(&representative).copied() else {
                continue;
            };
            let desired_color = usize::from(input_index);
            if current_color == desired_color {
                continue;
            }
            for color in representative_colors.values_mut() {
                if *color == current_color {
                    *color = desired_color;
                } else if *color == desired_color {
                    *color = current_color;
                }
            }
        }

        for original in originals.iter().filter(|reg| reg.kind == kind) {
            let representative = unionfind.find_rep(*original);
            if let Some(&color) = representative_colors.get(&representative) {
                result.insert(
                    *original,
                    Register {
                        kind,
                        index: u8::try_from(color).expect("JitCode register coloring exceeds u8"),
                    },
                );
            }
        }
    }
    result
}

fn is_builder_receiver(expr: &syn::Expr) -> bool {
    matches!(expr, syn::Expr::Path(path) if path.path.is_ident("__builder"))
}

fn is_aux_builder_method(name: &str) -> bool {
    name.starts_with("ensure_")
        || name.starts_with("register_")
        || name.starts_with("set_")
        || name.starts_with("add_")
        || matches!(
            name,
            "new_label" | "mark_label" | "finalize_liveness" | "finish"
        )
}

/// Rewrites one builder call's register literals, in argument order.
///
/// The pass works on tokens, so a register operand and an ordinary integer
/// constant are both `Lit::Int` and nothing in the syntax tells them apart.
/// What separates them is that `remaining` is seeded with exactly this op's
/// reads and writes and each match spends one: by the time a constant is
/// visited, the registers it could collide with are already spent. That holds
/// only because **every builder method lists its register operands before its
/// constants**. A method taking a constant first would let it consume the slot
/// and leave the real register literal unrewritten. That literal is then
/// visited with its register's slots spent; when coloring moved the register,
/// the literal is reported instead of being left with its old number.
struct LiteralRegisterRewriter<'a> {
    mapping: &'a HashMap<Register, Register>,
    remaining: HashMap<Register, usize>,
    declared: &'a HashMap<Register, usize>,
    method: String,
    /// Every register this rewriter recolored, accumulated across all the
    /// builder calls in one statement so [`rewrite_statement`] can check that
    /// the statement spelled each register the [`OpMeta`] declares.
    recolored: &'a mut BTreeSet<Register>,
}

impl VisitMut for LiteralRegisterRewriter<'_> {
    fn visit_expr_lit_mut(&mut self, expr: &mut syn::ExprLit) {
        let syn::Lit::Int(lit) = &expr.lit else {
            return;
        };
        let Ok(index) = lit.base10_parse::<u8>() else {
            return;
        };
        let candidates: Vec<_> = self
            .remaining
            .iter()
            .filter(|(reg, count)| reg.index == index && **count > 0)
            .map(|(reg, _)| *reg)
            .collect();
        if candidates.is_empty() {
            assert!(
                !self
                    .declared
                    .keys()
                    .any(|reg| reg.index == index && self.mapping[reg].index != index),
                "register literal {index} is spelled more times than the operation \
                 declares it, so a constant in the same `{}` call may have taken \
                 its color",
                self.method
            );
            return;
        }
        let replacement = self.mapping[&candidates[0]];
        assert!(
            candidates
                .iter()
                .all(|candidate| self.mapping[candidate].index == replacement.index),
            "ambiguous cross-bank register literal {index} in one builder call"
        );
        *self.remaining.get_mut(&candidates[0]).unwrap() -= 1;
        self.recolored.insert(candidates[0]);
        // Keep the literal's own type: a register list such as
        // `jit_merge_point`'s green bytes is `u8`.
        let suffix = match lit.suffix() {
            "" => "u16",
            suffix => suffix,
        };
        expr.lit = syn::Lit::Int(syn::LitInt::new(
            &format!("{}{suffix}", replacement.index),
            lit.span(),
        ));
    }
}

/// Argument positions of a builder method that are not registers of the
/// JitCode being built: descriptor, field and array indices, constants,
/// labels and callee indices. `flatten.py` keeps these apart from registers
/// by type (`Constant`, `Descr`, `TLabel` next to `Register`); the proc-macro
/// statements spell both as integer literals, so the rewriter skips them by
/// position.
fn non_register_positions(method: &str) -> &'static [usize] {
    match method {
        "record_binop_i_const" => &[1, 3],
        "goto_if_not_int_const" => &[0, 2, 3],
        "load_const_i_value" | "load_const_r_value" | "load_const_f_value" => &[1],
        "cast_int_to_word" => &[2],
        "switch" => &[1],
        "jit_merge_point" | "loop_header" => &[0],
        "recursive_call_int" => &[0],
        "arraylen_gc" | "new_array" | "new_array_clear" | "vable_arraylen_with_base" => &[2],
        "getarrayitem_gc_i"
        | "getarrayitem_gc_r"
        | "getarrayitem_gc_f"
        | "getarrayitem_gc_i_pure"
        | "getarrayitem_gc_r_pure"
        | "getarrayitem_gc_f_pure"
        | "setarrayitem_gc_i"
        | "setarrayitem_gc_r"
        | "setarrayitem_gc_f"
        | "raw_load_i"
        | "raw_load_f"
        | "raw_store_i"
        | "getarrayitem_raw_i" => &[3],
        "getfield_gc_i" | "getfield_gc_r" | "getfield_gc_f" | "setfield_gc_i" | "setfield_gc_r"
        | "setfield_gc_f" => &[2, 3, 4],
        "new_struct" => &[1, 2, 3, 4, 5],
        "new_with_vtable_struct" => &[1, 2, 3, 4, 5, 6, 7],
        "load_state_field"
        | "load_state_field_ref"
        | "load_state_field_float"
        | "store_state_field"
        | "store_state_field_ref"
        | "store_state_field_float" => &[0],
        "vable_getfield_int_with_base"
        | "vable_getfield_ref_with_base"
        | "vable_getfield_float_with_base"
        | "vable_getarrayitem_int_with_base"
        | "vable_getarrayitem_ref_with_base"
        | "vable_getarrayitem_float_with_base" => &[2],
        "vable_setfield_int_with_base"
        | "vable_setfield_ref_with_base"
        | "vable_setfield_float_with_base"
        | "vable_setarrayitem_int_with_base"
        | "vable_setarrayitem_ref_with_base"
        | "vable_setarrayitem_float_with_base" => &[1],
        _ if method.starts_with("inline_call") => &[0],
        // `(fn_idx, value_reg, typed_args, dst)`: the destination is a register.
        "conditional_call_value_ir_i_typed_args" | "conditional_call_value_ir_r_typed_args" => &[0],
        _ if method.starts_with("residual_call")
            || method.starts_with("call_")
            || method.starts_with("conditional_call")
            || method.starts_with("record_known_result") =>
        {
            &[0, 3]
        }
        _ => &[],
    }
}

/// Visit the caller side of each `(caller_reg, callee_reg)` pair of an
/// `inline_call_*` argument list. The callee register names the sub-JitCode's
/// frame, not this one.
fn visit_inline_call_pairs(rewrite: &mut LiteralRegisterRewriter<'_>, arg: &mut syn::Expr) {
    match arg {
        syn::Expr::Reference(reference) => visit_inline_call_pairs(rewrite, &mut reference.expr),
        syn::Expr::Array(array) => {
            for elem in &mut array.elems {
                visit_inline_call_pairs(rewrite, elem);
            }
        }
        syn::Expr::Tuple(tuple) if tuple.elems.len() == 2 => {
            rewrite.visit_expr_mut(&mut tuple.elems[0]);
        }
        other => rewrite.visit_expr_mut(other),
    }
}

struct BuilderStatementRewriter<'a> {
    mapping: &'a HashMap<Register, Register>,
    expected: HashMap<Register, usize>,
    recolored: BTreeSet<Register>,
}

impl VisitMut for BuilderStatementRewriter<'_> {
    fn visit_expr_method_call_mut(&mut self, call: &mut syn::ExprMethodCall) {
        let method = call.method.to_string();
        if is_builder_receiver(&call.receiver) && !is_aux_builder_method(&method) {
            let skipped = non_register_positions(&method);
            let pairs = method.starts_with("inline_call");
            let mut rewrite = LiteralRegisterRewriter {
                mapping: self.mapping,
                remaining: self.expected.clone(),
                declared: &self.expected,
                method,
                recolored: &mut self.recolored,
            };
            for (position, arg) in call.args.iter_mut().enumerate() {
                if skipped.contains(&position) {
                    continue;
                }
                if pairs {
                    visit_inline_call_pairs(&mut rewrite, arg);
                } else {
                    rewrite.visit_expr_mut(arg);
                }
            }
            return;
        }
        syn::visit_mut::visit_expr_method_call_mut(self, call);
    }
}

fn rewrite_statement(
    statement: &TokenStream,
    meta: &OpMeta,
    mapping: &HashMap<Register, Register>,
) -> TokenStream {
    let mut expected = HashMap::new();
    for &reg in meta.reads.iter().chain(meta.writes.iter()) {
        *expected.entry(reg).or_insert(0) += 1;
    }
    if expected.is_empty() {
        return statement.clone();
    }
    let mut block: syn::Block = syn::parse2(quote!({ #statement }))
        .expect("macro-generated JitCode statement must parse as a Rust block");
    let mut rewriter = BuilderStatementRewriter {
        mapping,
        expected: expected.clone(),
        recolored: BTreeSet::new(),
    };
    rewriter.visit_block_mut(&mut block);
    // Only the arguments of a non-aux `__builder` method call are recolored, so
    // a register an operation declares but spells anywhere else — bound to a
    // local first, say — would silently keep the number the lowerer handed out
    // before coloring, and the emitted call would then read a register nothing
    // ever writes. Every declared register that coloring moved must therefore
    // appear as a literal the rewriter reached. A register that keeps its
    // number (a pinned identity slot an operation reads through its field
    // index) needs no spelling.
    for reg in expected.keys().filter(|reg| mapping[*reg] != **reg) {
        assert!(
            rewriter.recolored.contains(reg),
            "JitCode statement declares {reg:?} but never spells it inside a \
             `__builder` call, so coloring cannot reach it: {statement}"
        );
    }
    block
        .stmts
        .into_iter()
        .map(|stmt| stmt.into_token_stream())
        .collect()
}

/// Color the proc-macro operation graph and rewrite its future builder calls.
/// Returns the compact per-bank register counts and the remapped `return_reg`.
pub(super) fn compact_registers(
    lowerer: &mut Lowerer<'_>,
    reserved: RegisterCounts,
    return_reg: Option<Register>,
) -> (RegisterCounts, Option<Register>) {
    compact_registers_pinned(
        lowerer,
        reserved,
        RegisterCounts::default(),
        return_reg,
        None,
    )
}

/// [`compact_registers`] whose first `pinned` registers of each bank keep
/// their numbers and share a color with nothing. `track`, when given, is a
/// statement index that is moved along as `*_copy` operations are dropped.
pub(super) fn compact_registers_pinned(
    lowerer: &mut Lowerer<'_>,
    reserved: RegisterCounts,
    pinned: RegisterCounts,
    return_reg: Option<Register>,
    track: Option<&mut usize>,
) -> (RegisterCounts, Option<Register>) {
    debug_assert_eq!(lowerer.statements.len(), lowerer.op_metadata.len());
    let mapping = coloring(&lowerer.op_metadata, reserved, pinned, return_reg);
    for (statement, meta) in lowerer.statements.iter_mut().zip(&lowerer.op_metadata) {
        // A `-live-` marker's reads are liveness facts, not operands: the
        // statement is re-emitted from the recolored reads by
        // `rewrite_live_marker_statements_with_triples`.
        if matches!(meta.control, ControlFlowClass::LiveMarker) {
            continue;
        }
        *statement = rewrite_statement(statement, meta, &mapping);
    }
    for meta in &mut lowerer.op_metadata {
        for reg in meta.reads.iter_mut().chain(meta.writes.iter_mut()) {
            if let Some(mapped) = mapping.get(reg) {
                *reg = *mapped;
            }
        }
    }
    // `flatten.py::GraphFlattener.insert_renamings` compares post-regalloc
    // source and destination colors and emits no `*_copy` when they match.
    // Our link moves were materialized before this adapter runs, so perform
    // the same check after rewriting and remove the now-empty links.
    let is_empty_link = |meta: &OpMeta| {
        matches!(meta.kind, OpKind::MoveI | OpKind::MoveR | OpKind::MoveF)
            && meta.reads == meta.writes
    };
    if let Some(index) = track {
        let dropped = lowerer.op_metadata[..*index]
            .iter()
            .filter(|meta| is_empty_link(meta))
            .count();
        *index -= dropped;
    }
    let (statements, op_metadata): (Vec<_>, Vec<_>) = std::mem::take(&mut lowerer.statements)
        .into_iter()
        .zip(std::mem::take(&mut lowerer.op_metadata))
        .filter(|(_, meta)| !is_empty_link(meta))
        .unzip();
    lowerer.statements = statements;
    lowerer.op_metadata = op_metadata;
    let mut counts = reserved;
    for mapped in mapping.values() {
        counts.observe(*mapped);
    }
    let return_reg = return_reg.map(|reg| mapping.get(&reg).copied().unwrap_or(reg));
    (counts, return_reg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_overlapping_temporaries_share_a_color_before_flattening() {
        let mut lowerer = Lowerer::new(None);
        lowerer.emit_op(
            OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(1)]),
            quote! { __builder.load_const_i_value(1u16, 41i64); },
        );
        lowerer.emit_op(
            OpMeta::linear(
                OpKind::BinopI,
                vec![Register::int(1)],
                vec![Register::int(2)],
            ),
            quote! { __builder.int_is_true(2u16, 1u16); },
        );
        lowerer.emit_op(
            OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(3)]),
            quote! { __builder.load_const_i_value(3u16, 7i64); },
        );
        lowerer.emit_op(
            OpMeta::terminal(vec![Register::int(3)]),
            quote! { __builder.int_return(3u16); },
        );

        let (counts, returned) = compact_registers(
            &mut lowerer,
            RegisterCounts {
                ints: 1,
                ..RegisterCounts::default()
            },
            Some(Register::int(3)),
        );
        assert!(counts.ints < 4);
        assert_eq!(u16::from(returned.unwrap().index) + 1, counts.ints);
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(!emitted.contains("3u16"));
    }

    /// `RegAllocator.coalesce_variables` makes a link source and its target
    /// inputarg one Variable when they do not interfere; `flatten.py`
    /// `GraphFlattener.insert_renamings` consequently emits no copy.
    #[test]
    fn noninterfering_link_move_is_coalesced_and_not_emitted() {
        let mut lowerer = Lowerer::new(None);
        lowerer.emit_op(
            OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(1)]),
            quote! { __builder.load_const_i_value(1u16, 41i64); },
        );
        lowerer.emit_op(
            OpMeta::linear(
                OpKind::MoveI,
                vec![Register::int(1)],
                vec![Register::int(2)],
            ),
            quote! { __builder.move_i(2u16, 1u16); },
        );
        lowerer.emit_op(
            OpMeta::terminal(vec![Register::int(2)]),
            quote! { __builder.int_return(2u16); },
        );

        let (counts, returned) = compact_registers(
            &mut lowerer,
            RegisterCounts {
                ints: 1,
                ..RegisterCounts::default()
            },
            Some(Register::int(2)),
        );

        assert_eq!(counts.ints, 1, "a dead ABI input slot is reusable");
        assert_eq!(returned, Some(Register::int(0)));
        assert!(
            !lowerer
                .op_metadata
                .iter()
                .any(|meta| meta.kind == OpKind::MoveI)
        );
        assert!(
            !lowerer
                .statements
                .iter()
                .map(ToString::to_string)
                .collect::<String>()
                .contains("move_i")
        );
    }

    /// A pinned register keeps its number, and no working register takes
    /// its color even where the pinned one is dead.
    #[test]
    fn pinned_registers_keep_their_numbers_and_are_never_shared() {
        let mut lowerer = Lowerer::new(None);
        lowerer.emit_op(
            OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(9)]),
            quote! { __builder.load_const_i_value(9u16, 41i64); },
        );
        lowerer.emit_op(
            OpMeta::terminal(vec![Register::int(9)]),
            quote! { __builder.int_return(9u16); },
        );

        let floor = RegisterCounts {
            ints: 3,
            refs: 3,
            floats: 3,
        };
        let (counts, returned) =
            compact_registers_pinned(&mut lowerer, floor, floor, Some(Register::int(9)), None);

        assert_eq!(returned, Some(Register::int(3)));
        assert_eq!(counts.ints, 4);
    }

    /// A value read only at a `switch` case target stays live across the
    /// `-live-` that names the target, so a temporary written before the
    /// switch cannot take its color.
    #[test]
    fn a_switch_case_target_keeps_its_live_values_live_before_the_switch() {
        let case = proc_macro2::Ident::new("case", proc_macro2::Span::call_site());
        let ops = vec![
            OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(1)]),
            OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(2)]),
            OpMeta::linear(OpKind::Aux, vec![Register::int(2)], vec![]),
            OpMeta::live_marker_with(Vec::new(), vec![case.clone()]),
            OpMeta::linear(OpKind::Aux, vec![], vec![]),
            OpMeta::terminal(Vec::new()),
            OpMeta::label_def(case),
            OpMeta::terminal(vec![Register::int(1)]),
        ];

        let mapping = coloring(
            &ops,
            RegisterCounts::default(),
            RegisterCounts::default(),
            None,
        );

        assert_ne!(mapping[&Register::int(1)], mapping[&Register::int(2)]);
    }

    /// A constant argument placed before a register operand is not a
    /// register: `vable_setfield_int_with_base(vable, field_index, src)`
    /// keeps its field index when `src` is recolored from the same number.
    #[test]
    fn a_constant_argument_is_not_recolored_as_a_register() {
        let mut lowerer = Lowerer::new(None);
        lowerer.emit_op(
            OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(5)]),
            quote! { __builder.load_const_i_value(5u16, 41i64); },
        );
        lowerer.emit_op(
            OpMeta::linear(
                OpKind::Aux,
                vec![Register::ref_(0), Register::int(5)],
                vec![],
            ),
            quote! { __builder.vable_setfield_int_with_base(0u16, 5u16, 5u16); },
        );
        lowerer.emit_op(
            OpMeta::terminal(Vec::new()),
            quote! { __builder.void_return(); },
        );

        compact_registers(
            &mut lowerer,
            RegisterCounts {
                refs: 1,
                ..RegisterCounts::default()
            },
            None,
        );

        let emitted = lowerer.statements[1].to_string();
        assert!(
            emitted.contains("vable_setfield_int_with_base (0u16 , 5u16 , 0u16)"),
            "field index 5 must stay, src must be recolored: {emitted}"
        );
    }

    /// `conditional_call_value_ir_{i,r}_typed_args(fn_idx, value, args, dst)`
    /// carries its destination register in the position the other call
    /// builders use for effect info, so that position is recolored here.
    #[test]
    fn a_conditional_call_value_destination_is_recolored() {
        let mut lowerer = Lowerer::new(None);
        lowerer.emit_op(
            OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(5)]),
            quote! { __builder.load_const_i_value(5u16, 41i64); },
        );
        lowerer.emit_op(
            OpMeta::linear(OpKind::Aux, vec![Register::int(5)], vec![Register::int(7)]),
            quote! { __builder.conditional_call_value_ir_i_typed_args(3u16, 5u16, &[], 7u16); },
        );
        lowerer.emit_op(
            OpMeta::terminal(vec![Register::int(7)]),
            quote! { __builder.int_return(7u16); },
        );

        compact_registers(&mut lowerer, RegisterCounts::default(), None);

        let emitted = lowerer.statements[1].to_string();
        assert!(
            emitted.contains("conditional_call_value_ir_i_typed_args (3u16 , 0u16 , & [] , 0u16)"),
            "fn index 3 must stay, value and dst must be recolored: {emitted}"
        );
    }

    /// `as usize` is a rename on a 64-bit word. When its source dies at the
    /// cast the two coalesce, and the op stays for the 32-bit narrowing.
    #[test]
    fn noninterfering_word_cast_is_coalesced_and_kept() {
        let mut lowerer = Lowerer::new(None);
        lowerer.emit_op(
            OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(1)]),
            quote! { __builder.load_const_i_value(1u16, 41i64); },
        );
        lowerer.emit_op(
            OpMeta::linear(
                OpKind::CastIntToWord,
                vec![Register::int(1)],
                vec![Register::int(2)],
            ),
            quote! { __builder.cast_int_to_word(2u16, 1u16, false); },
        );
        lowerer.emit_op(
            OpMeta::terminal(vec![Register::int(2)]),
            quote! { __builder.int_return(2u16); },
        );

        compact_registers(
            &mut lowerer,
            RegisterCounts {
                ints: 1,
                ..RegisterCounts::default()
            },
            Some(Register::int(2)),
        );

        let cast = lowerer
            .op_metadata
            .iter()
            .find(|meta| meta.kind == OpKind::CastIntToWord)
            .expect("the cast is kept");
        assert_eq!(cast.reads, cast.writes, "source and target coalesced");
    }

    /// A link source that stays live after the assignment interferes with the
    /// target. Upstream `_try_coalesce` leaves that renaming for flattening.
    #[test]
    fn interfering_link_move_remains_a_copy() {
        let mut lowerer = Lowerer::new(None);
        lowerer.emit_op(
            OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(1)]),
            quote! { __builder.load_const_i_value(1u16, 41i64); },
        );
        lowerer.emit_op(
            OpMeta::linear(
                OpKind::MoveI,
                vec![Register::int(1)],
                vec![Register::int(2)],
            ),
            quote! { __builder.move_i(2u16, 1u16); },
        );
        lowerer.emit_op(
            OpMeta::linear(
                OpKind::BinopI,
                Register::ints(&[1, 2]),
                vec![Register::int(3)],
            ),
            quote! { __builder.record_binop_i(3u16, majit_ir::OpCode::IntAdd, 1u16, 2u16); },
        );
        lowerer.emit_op(
            OpMeta::terminal(vec![Register::int(3)]),
            quote! { __builder.int_return(3u16); },
        );

        compact_registers(
            &mut lowerer,
            RegisterCounts {
                ints: 1,
                ..RegisterCounts::default()
            },
            Some(Register::int(3)),
        );

        assert!(
            lowerer
                .op_metadata
                .iter()
                .any(|meta| meta.kind == OpKind::MoveI)
        );
    }

    /// The rewriter reaches only the arguments of a `__builder` call, so an
    /// operation that binds its arguments to a local first would keep its
    /// pre-coloring register numbers and call a register nothing writes.
    #[test]
    #[should_panic(expected = "never spells it inside a")]
    fn a_declared_register_spelled_outside_the_builder_call_is_rejected() {
        let mut lowerer = Lowerer::new(None);
        lowerer.emit_op(
            OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(1)]),
            quote! { __builder.load_const_i_value(1u16, 41i64); },
        );
        lowerer.emit_op(
            OpMeta::linear(OpKind::Call, vec![Register::int(1)], vec![Register::int(2)]),
            quote! {
                let __typed_args = &[majit_metainterp::JitCallArg::int(1u16)];
                __builder.residual_call_int_canonical_via_target(__fn_idx, __typed_args, 2u16);
            },
        );
        lowerer.emit_op(
            OpMeta::terminal(vec![Register::int(2)]),
            quote! { __builder.int_return(2u16); },
        );

        compact_registers(
            &mut lowerer,
            RegisterCounts {
                ints: 1,
                ..RegisterCounts::default()
            },
            Some(Register::int(2)),
        );
    }
}
