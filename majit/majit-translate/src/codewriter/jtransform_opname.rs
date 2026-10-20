//! Opname-dispatch transducer for the codewriter convergence ("Spine B").
//!
//! The production codewriter consumes the rich-`OpKind`
//! `crate::model::FunctionGraph` (`jtransform.rs`'s `Transformer::transform`
//! dispatches on `OpKind::FieldRead`/`OpKind::Call`/…).  The rtyper, however,
//! already lowers the *flowspace* graph to upstream-shaped low-level
//! `SpaceOperation`s (`getfield`/`setfield`/`getarrayitem`/`malloc_varsize`/
//! `int_add`/…) in place, and certain helper graphs (the `ll_str*` family
//! built by `lltypesystem/rstr.rs`) are born ONLY in that opname form — they
//! have no rich-`OpKind` twin and are discarded today.
//!
//! [`lower_graph`] is the convergence transducer: it consumes such a
//! `crate::flowspace::model::FunctionGraph` and emits an equivalent
//! `crate::model::FunctionGraph` (rich `OpKind`), which then re-enters the
//! EXISTING flatten/regalloc/assembler tail unchanged
//! (`CodeWriter::finalize_rewritten_graph_to_jitcode`).  This is the port of
//! `jtransform.py`'s `_rewrite_ops[op.opname]` dispatch
//! (`jtransform.py`), reading each `Variable.concretetype` directly (the
//! upstream `getkind(v.concretetype)` path) rather than through the
//! `value_to_var` bridge the rich-`OpKind` spine uses.
//!
//! Coexistence: a graph's ops are either all rich-`OpKind` (Spine A) or all
//! opname (Spine B); the two never mix within one graph.  The drain loop
//! routes a path to this module only when
//! `CallControl::take_opname_graph` returns its registered flowspace graph.
//! Until a helper is registered via
//! `CallControl::register_opname_helper_graph`, this module is dead code.
//!
//! ## String-helper fusion (S1 scope)
//!
//! The string-repr family stores its character data as an `Array(Char)` /
//! `Array(UniChar)` *inline* in the GC `STR` / `UNICODE` struct, so a
//! `getsubstruct(s, "chars")` yields an interior pointer with no standalone
//! runtime object.  The blackhole interpreter has no interior-pointer model;
//! instead it carries by-name handlers that take the **string object** plus
//! register-shaped operands: `strlen(s)`, `strgetitem(s, i)`,
//! `strsetitem(s, i, c)`, `newstr(n)` (and the `unicode*` / `newunicode`
//! peers).  The transducer therefore *fuses* the `getsubstruct` + array op
//! pair back into the string opcode — `getsubstruct` itself emits nothing and
//! records `chars_array_var → string_var`; a following `getarraysize` /
//! `getarrayitem` / `setarrayitem` on that array re-references the recorded
//! string operand.  This mirrors upstream `s.chars[i]`, where the chars
//! substruct is re-derived at each access from the string in hand.
//!
//! The fusion is **block-local**: it relies on the `getsubstruct` and its
//! consuming array op living in the same block, with the string operand
//! available there.  A helper that hoists the substruct out of a loop and
//! threads the *interior chars pointer* through block Phi inputargs (rather
//! than threading the string itself) cannot be fused this way — the string
//! origin is lost across the Phi.  Such graphs are out of scope for this
//! slice; the transducer fail-loud `expect`s a recorded alias rather than
//! silently miscompiling.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::flowspace::model::Variable as FVar;
use crate::flowspace::model::{
    Block as FlowBlock, ConstValue, FunctionGraph as FlowGraph, Hlvalue,
};
use crate::model::{BlockId, ExitCase, ExitSwitch, Link, LinkArg, OpKind, ValueType};
use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};

/// Block identity key — flowspace blocks are `Rc<RefCell<Block>>` with no
/// inherent id, so identity is the allocation address (RPython compares
/// `Block` objects by Python identity; `iterblocks` already keys its
/// visited-set on `Rc::as_ptr`).
type FlowBlockKey = *const RefCell<FlowBlock>;

/// The blackhole opcode family for a given string-element width.  `str*`
/// handlers operate on `Array(Char)`-backed `STR`; `unicode*` /
/// `newunicode` on `Array(UniChar)`-backed `UNICODE`.
struct StrFamily {
    /// `strlen` / `unicodelen` — array length of the inline chars array.
    len: &'static str,
    /// `strgetitem` / `unicodegetitem` — read one element.
    getitem: &'static str,
    /// `strsetitem` / `unicodesetitem` — write one element (void).
    setitem: &'static str,
    /// `newstr` / `newunicode` — variable-size allocation by length.
    alloc: &'static str,
}

impl StrFamily {
    const STR: StrFamily = StrFamily {
        len: "strlen",
        getitem: "strgetitem",
        setitem: "strsetitem",
        alloc: "newstr",
    };
    const UNICODE: StrFamily = StrFamily {
        len: "unicodelen",
        getitem: "unicodegetitem",
        setitem: "unicodesetitem",
        alloc: "newunicode",
    };
}

/// Lower an rtyper low-level helper graph (opname `SpaceOperation`s) to an
/// equivalent rich-`OpKind` `crate::model::FunctionGraph`.
///
/// The flowspace `Variable`s are reused verbatim as the model graph's
/// operands and block inputargs — they are the same
/// `crate::flowspace::model::Variable` type and carry their `concretetype`
/// cell through unchanged, so no value-bridge round-trip is needed.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn lower_graph(graph: &FlowGraph) -> crate::model::FunctionGraph {
    let mut out = crate::model::FunctionGraph::new(graph.name.clone());
    let family = detect_family(graph);

    // Map each flowspace block to a model `BlockId`.  The three canonical
    // blocks (`startblock`/`returnblock`/`exceptblock`) are pre-seeded onto
    // the model graph's own canonical ids so links into them resolve.
    let mut block_map: HashMap<FlowBlockKey, BlockId> = HashMap::new();
    block_map.insert(Rc::as_ptr(&graph.startblock), out.startblock);
    block_map.insert(Rc::as_ptr(&graph.returnblock), out.returnblock);
    block_map.insert(Rc::as_ptr(&graph.exceptblock), out.exceptblock);

    let flow_blocks = graph.iterblocks();

    // Pass 1 — allocate model blocks for every reachable block and copy each
    // block's inputarg `Variable`s across (the canonical blocks already
    // exist; interior blocks get fresh ids).
    for fb in &flow_blocks {
        let key = Rc::as_ptr(fb);
        let id = *block_map.entry(key).or_insert_with(|| out.create_block());
        let inputargs: Vec<FVar> = fb
            .borrow()
            .inputargs
            .iter()
            .filter_map(hlvalue_as_var)
            .collect();
        out.block_mut(id).inputargs = inputargs;
    }
    // The return/except blocks may not appear in `iterblocks` (e.g. a graph
    // with no raising edge never reaches `exceptblock`); set their inputargs
    // from the flowspace canonical blocks so the model return value carries
    // the helper's concretetype.
    let return_inputargs: Vec<FVar> = graph
        .returnblock
        .borrow()
        .inputargs
        .iter()
        .filter_map(hlvalue_as_var)
        .collect();
    let return_id = out.returnblock;
    out.block_mut(return_id).inputargs = return_inputargs;

    // Pass 2 — transduce each block's operations + control flow.
    for fb in &flow_blocks {
        let id = block_map[&Rc::as_ptr(fb)];
        let fb_ref = fb.borrow();

        // `getsubstruct` aliases are block-local: a chars-array Variable maps
        // back to the string Variable it was derived from in this block.
        let mut chars_alias: HashMap<FVar, FVar> = HashMap::new();
        for op in &fb_ref.operations {
            transduce_op(&mut out, id, op, &mut chars_alias, &family);
        }

        let exitswitch = if fb_ref.canraise() {
            Some(ExitSwitch::LastException)
        } else {
            match &fb_ref.exitswitch {
                Some(Hlvalue::Variable(v)) => Some(ExitSwitch::Value(v.clone())),
                _ => None,
            }
        };

        let exits: Vec<Link> = fb_ref
            .exits
            .iter()
            .map(|link_ref| {
                let link = link_ref.borrow();
                let target = link
                    .target
                    .as_ref()
                    .expect("opname-spine link has no target block");
                let target_id = block_map[&Rc::as_ptr(target)];
                let args: Vec<LinkArg> = link
                    .args
                    .iter()
                    .map(|arg| {
                        linkarg_from_hlvalue(
                            arg.as_ref().expect("opname-spine link arg is undefined"),
                        )
                    })
                    .collect();
                let exitcase = match &link.exitcase {
                    Some(Hlvalue::Constant(c)) => exitcase_from_const(&c.value),
                    _ => None,
                };
                let mut model_link = Link::new_mixed(args, target_id, exitcase);
                // Carry the exception extravars the flowspace producer set (e.g.
                // `rbuilder::close_ovfcheck_block`): a canraise exit's
                // last_exception / last_exc_value are required by flatten's
                // make_exception_link (it asserts they are Some and, for an
                // empty-target handler, unwraps them). Dropping them turns a
                // faithful OverflowError edge into a flatten-time panic.
                model_link.last_exception = link.last_exception.as_ref().map(linkarg_from_hlvalue);
                model_link.last_exc_value = link.last_exc_value.as_ref().map(linkarg_from_hlvalue);
                model_link
            })
            .collect();

        drop(fb_ref);
        if exitswitch.is_some() || !exits.is_empty() {
            out.set_control_flow_metadata(id, exitswitch, exits);
        }
    }

    if let Some(func) = &graph.func {
        if func._jit_unroll_safe_ {
            out.func.unroll_safe = true;
        }
        if let Some(look) = func._jit_look_inside_ {
            out.func.jit_look_inside = Some(look);
        }
    }

    out
}

/// Transduce a single flowspace `SpaceOperation` into the model graph at
/// `block`, materialising any constant operands as `ConstInt`/`ConstBool`
/// ops and fusing `getsubstruct`-derived array accesses into the string
/// blackhole opcodes.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn transduce_op(
    out: &mut crate::model::FunctionGraph,
    block: BlockId,
    op: &crate::flowspace::model::SpaceOperation,
    chars_alias: &mut HashMap<FVar, FVar>,
    family: &StrFamily,
) {
    match op.opname.as_str() {
        // `getsubstruct(s, "chars")` — interior pointer with no runtime
        // object; record the alias and emit nothing.  The string operand is
        // always a Variable in the helper graphs.
        "getsubstruct" => {
            let string_var = expect_var(&op.args[0]);
            let chars_var = expect_var(&op.result);
            chars_alias.insert(chars_var, string_var);
        }
        // `len(s.chars)` → `strlen(s)` / `unicodelen(s)`.
        "getarraysize" => {
            let string_var = resolve_string(chars_alias, &op.args[0]);
            let result = expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::LoweredBlackholeOp {
                    opname: family.len.to_string(),
                    args: vec![string_var],
                },
                result,
            );
        }
        // `s.chars[i]` → `strgetitem(s, i)` / `unicodegetitem(s, i)`.
        "getarrayitem" => {
            let string_var = resolve_string(chars_alias, &op.args[0]);
            let index = materialize(out, block, &op.args[1]);
            let result = expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::LoweredBlackholeOp {
                    opname: family.getitem.to_string(),
                    args: vec![string_var, index],
                },
                result,
            );
        }
        // `s.chars[i] = c` → `strsetitem(s, i, c)` / `unicodesetitem(...)`.
        // Void result.
        "setarrayitem" => {
            let string_var = resolve_string(chars_alias, &op.args[0]);
            let index = materialize(out, block, &op.args[1]);
            let value = materialize(out, block, &op.args[2]);
            out.push_op_var(
                block,
                OpKind::LoweredBlackholeOp {
                    opname: family.setitem.to_string(),
                    args: vec![string_var, index, value],
                },
                false,
            );
        }
        // `malloc_varsize(STR, gc, n)` → `newstr(n)` / `newunicode(n)`.  The
        // struct lltype and gc-flavor operands carry no runtime value for the
        // blackhole allocator, which is keyed by the string family.
        "malloc_varsize" => {
            let size = materialize(out, block, &op.args[2]);
            let result = expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::LoweredBlackholeOp {
                    opname: family.alloc.to_string(),
                    args: vec![size],
                },
                result,
            );
        }
        // `copystrcontent(src, dst, srcstart, dststart, length)` /
        // `copyunicodecontent(...)` — bulk char copy.  Both string operands
        // are whole objects (not chars-array interior pointers), so they
        // bypass the `chars_alias` map.  Void result.
        "copystrcontent" | "copyunicodecontent" => {
            let src = expect_var(&op.args[0]);
            let dst = expect_var(&op.args[1]);
            let srcstart = materialize(out, block, &op.args[2]);
            let dststart = materialize(out, block, &op.args[3]);
            let length = materialize(out, block, &op.args[4]);
            out.push_op_var(
                block,
                OpKind::LoweredBlackholeOp {
                    opname: op.opname.clone(),
                    args: vec![src, dst, srcstart, dststart, length],
                },
                false,
            );
        }
        // `jtransform.py`'s low-level integer op dispatch is arity-sensitive:
        // `int_neg` / `int_invert` / `int_is_{true,zero}` are unary blackhole
        // operations, while the remaining `int_*` family consumed by these
        // helpers is binary.  The old Spine-B catch-all indexed `args[1]` for
        // every `int_*` spelling, so the first signed `ll_int2hex` graph
        // panicked on its orthodox `int_neg(i)` before jtransform could see
        // it.  Preserve the canonical full opname for unary operations; the
        // assembler deliberately passes already-prefixed unary names through.
        name @ ("int_neg" | "int_invert" | "int_is_true" | "int_is_zero" | "int_force_ge_zero"
        | "bool_not" | "uint_is_true" | "uint_invert") => {
            let operand = materialize(out, block, &op.args[0]);
            let result = expect_var(&op.result);
            let result_ty = value_type_of(&result);
            // `Transformer` rename table (`jtransform.py`): Bool/Unsigned
            // aliases share the signed int-bank blackhole implementation.
            let canonical = match name {
                "bool_not" => "int_is_zero",
                "uint_is_true" => "int_is_true",
                "uint_invert" => "int_invert",
                _ => name,
            };
            out.push_op_with_result_var(
                block,
                OpKind::UnaryOp {
                    op: canonical.to_string(),
                    operand,
                    result_ty,
                },
                result,
            );
        }
        // `Transformer`'s generated rename methods.  These old low-level
        // spellings all re-enter `rewrite_operation` under the corresponding
        // signed int-bank opname.  Store the bare rich-graph name because the
        // assembler restores the `int_` prefix.
        name @ ("uint_add" | "uint_sub" | "uint_mul" | "uint_eq" | "uint_ne" | "uint_and"
        | "uint_or" | "uint_lshift" | "uint_xor" | "char_lt" | "char_le" | "char_eq"
        | "char_ne" | "char_gt" | "char_ge" | "unichar_eq" | "unichar_ne" | "adr_add"
        | "int_add_nonneg_ovf") => {
            let lhs = materialize(out, block, &op.args[0]);
            let rhs = materialize(out, block, &op.args[1]);
            let result = expect_var(&op.result);
            let result_ty = value_type_of(&result);
            let canonical = match name {
                "adr_add" => "add",
                "int_add_nonneg_ovf" => "add_ovf",
                _ => name.split_once('_').expect("integer alias has a prefix").1,
            };
            out.push_op_with_result_var(
                block,
                OpKind::BinOp {
                    op: canonical.to_string(),
                    lhs,
                    rhs,
                    result_ty,
                },
                result,
            );
        }
        // Unlike the aliases above, these unsigned operations have distinct
        // blackhole opcodes and are passed through unchanged by
        // `_add_default_ops` in `jtransform.py`.
        name
        @ ("uint_rshift" | "uint_lt" | "uint_le" | "uint_gt" | "uint_ge" | "uint_mul_high") => {
            let lhs = materialize(out, block, &op.args[0]);
            let rhs = materialize(out, block, &op.args[1]);
            let result = expect_var(&op.result);
            let result_ty = value_type_of(&result);
            out.push_op_with_result_var(
                block,
                OpKind::BinOp {
                    op: name.to_string(),
                    lhs,
                    rhs,
                    result_ty,
                },
                result,
            );
        }
        // The sole three-input integer blackhole operation.  Keeping this
        // explicit is load-bearing: the old `int_*` catch-all discarded its
        // third operand by forcing every spelling through `BinOp`.
        "int_between" => {
            let args = op
                .args
                .iter()
                .map(|arg| materialize(out, block, arg))
                .collect();
            let result = expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::LoweredBlackholeOp {
                    opname: "int_between".to_string(),
                    args,
                },
                result,
            );
        }
        // Binary integer operations that survive `Transformer` as direct
        // blackhole operations.  Spell the accepted surface literally: an
        // `int_*` prefix is not proof of binary arity (`int_between` has
        // three operands), nor proof that an operation survives as an opcode
        // (`rewrite_op_int_abs` uses a builtin call).
        name @ ("int_add" | "int_sub" | "int_mul" | "int_and" | "int_or" | "int_xor"
        | "int_rshift" | "int_lshift" | "int_lt" | "int_le" | "int_eq" | "int_ne"
        | "int_gt" | "int_ge" | "int_signext" | "int_add_ovf" | "int_sub_ovf"
        | "int_mul_ovf") => {
            let bare = name.strip_prefix("int_").unwrap().to_string();
            let lhs = materialize(out, block, &op.args[0]);
            let rhs = materialize(out, block, &op.args[1]);
            let result = expect_var(&op.result);
            let result_ty = value_type_of(&result);
            out.push_op_with_result_var(
                block,
                OpKind::BinOp {
                    op: bare,
                    lhs,
                    rhs,
                    result_ty,
                },
                result,
            );
        }
        // These two do not survive as direct blackhole operations.  Feed
        // their bare rich-graph names into the shared `Transformer`, whose
        // `rewrite_operation` routes them through `_do_builtin_call`:
        // `inline_call` when the `_ll_2_int_floordiv` / `_ll_2_int_mod`
        // graph is a candidate, otherwise the residual call.
        name @ ("int_floordiv" | "int_mod") => {
            let bare = name.strip_prefix("int_").unwrap().to_string();
            let lhs = materialize(out, block, &op.args[0]);
            let rhs = materialize(out, block, &op.args[1]);
            let result = expect_var(&op.result);
            let result_ty = value_type_of(&result);
            out.push_op_with_result_var(
                block,
                OpKind::BinOp {
                    op: bare,
                    lhs,
                    rhs,
                    result_ty,
                },
                result,
            );
        }
        // `Transformer`'s rename table maps `keepalive` to `-live-`; the
        // operand list is intentionally discarded because flatten computes
        // the variables live at the marker from control-flow liveness.
        "keepalive" => {
            out.push_op_var(block, OpKind::Live, false);
        }
        // `getfield(s, "field")` → `FieldRead`.  The field name is a Void
        // `ByteStr` constant (`void_field_const`); the owning struct identity
        // is the pointee-struct leaf of the base pointer's `concretetype`
        // (`Ptr(Struct("stringbuilder"))` → `"stringbuilder"`), which keys the
        // layout registry the assembler resolves offsets through.  These are
        // the mutable low-level fields of the builder container
        // (`current_pos`/`current_end`/`total_size`), so the read is not
        // `_pure`.
        "getfield" => {
            let base = expect_var(&op.args[0]);
            let field_name = field_name_of(&op.args[1]);
            let owner_root = owner_root_of(&base);
            let result = expect_var(&op.result);
            let ty = value_type_of(&result);
            out.push_op_with_result_var(
                block,
                OpKind::FieldRead {
                    base,
                    field: crate::model::FieldDescriptor::new(field_name, owner_root),
                    ty,
                    pure: false,
                },
                result,
            );
        }
        // `ptr_ne(p, NULL)` / `ptr_eq(p, NULL)` — a pointer null test.
        // RPython `PtrRepr.rtype_bool` lowers it to the unary
        // `ptr_nonzero` / `ptr_iszero` (`assembler.rs` `"r" => "ptr_nonzero"`);
        // when one operand is the null pointer constant, emit that unary form
        // over the non-null operand.  This is the orthodox shape and avoids
        // materialising a null `Ref` constant (`materialize` handles only
        // int/bool constants).  A compare between two live pointers keeps the
        // binary `eq` / `ne`, which the assembler maps to `ptr_eq` / `ptr_ne`
        // on the `rr` operand shape.
        "ptr_ne" | "ptr_eq" => {
            let is_ne = op.opname == "ptr_ne";
            let result = expect_var(&op.result);
            let result_ty = value_type_of(&result);
            if let Some(null_i) = op.args.iter().position(is_null_const) {
                let operand = expect_var(&op.args[1 - null_i]);
                let unop = if is_ne { "ptr_nonzero" } else { "ptr_iszero" };
                out.push_op_with_result_var(
                    block,
                    OpKind::UnaryOp {
                        op: unop.to_string(),
                        operand,
                        result_ty,
                    },
                    result,
                );
            } else {
                let lhs = expect_var(&op.args[0]);
                let rhs = expect_var(&op.args[1]);
                let bare = if is_ne { "ne" } else { "eq" };
                out.push_op_with_result_var(
                    block,
                    OpKind::BinOp {
                        op: bare.to_string(),
                        lhs,
                        rhs,
                        result_ty,
                    },
                    result,
                );
            }
        }
        // `setfield(s, "field", v)` → `FieldWrite`.  Field name and owner are
        // resolved exactly like `getfield`; the stored value is an
        // `AbstractValue` — a `Variable` (register operand) or an inline
        // `Constant` (e.g. `current_pos = 0`) — carried through the
        // `LinkArg::Value`/`Const` union.  `ty` (the setfield kind i/r/f) is
        // the stored value's kind: `current_buf` is a `Ref`, the size/offset
        // fields `Int`.  Void result.
        "setfield" => {
            let base = expect_var(&op.args[0]);
            let field_name = field_name_of(&op.args[1]);
            let owner_root = owner_root_of(&base);
            let value = linkarg_from_hlvalue(&op.args[2]);
            let ty = hlvalue_value_type(&op.args[2]);
            out.push_op_var(
                block,
                OpKind::FieldWrite {
                    base,
                    field: crate::model::FieldDescriptor::new(field_name, owner_root),
                    value,
                    ty,
                },
                false,
            );
        }
        // `malloc(STRUCT, {'flavor':'gc'})` for a fixed-size GcStruct → `New`.
        // The struct leaf (`"stringbuilder"`) is read off the first operand's
        // `LowLevelType` constant and keys the assembler's size descriptor
        // (`bh_size_spec_from_callcontrol`, `path_hash(owner)`).  A plain
        // GcStruct with no boxed `ob_type`, so `New` (not `NewWithVtable`).
        // The flavor operand must be `{'flavor': 'gc'}`: `jtransform.py
        // rewrite_op_malloc` lowers only the gc flavor to `new(descr)`, so a
        // `raw`/`nonmovable` allocation is rejected rather than silently
        // lowered to a GC `New`.
        "malloc" => {
            assert!(
                is_gc_flavor_const(&op.args[1]),
                "jtransform_opname::lower_graph: malloc flavor operand is not {{'flavor': 'gc'}}: {:?}",
                op.args[1]
            );
            let owner = malloc_struct_owner(&op.args[0]);
            let result = expect_var(&op.result);
            out.push_op_with_result_var(block, OpKind::New { owner }, result);
        }
        // `Transformer` rename table: Bool and Signed share the `i` input
        // bank, so the one backend conversion is `cast_int_to_float`.
        "cast_bool_to_float" => {
            let operand = materialize(out, block, &op.args[0]);
            let result = expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::UnaryOp {
                    op: "cast_int_to_float".to_string(),
                    operand,
                    result_ty: value_type_of(&result),
                },
                result,
            );
        }
        // Word-sized integer casts are identity at JitCode level
        // (`getkind(Signed) == getkind(Unsigned) == getkind(Char) == 'int'`).
        // `Transformer.rewrite_op_cast_primitive` delegates to
        // `rewrite_op_force_cast`, whose `_int_to_int_cast` returns `None`
        // for these same-width cases; the named char cast is also an explicit
        // no-op method.  Route all of them through the rich graph's
        // `same_as`, which the shared jtransform tail drops while installing
        // the result→operand alias.
        name @ ("cast_bool_to_int"
        | "cast_bool_to_uint"
        | "cast_char_to_int"
        | "cast_unichar_to_int"
        | "cast_int_to_char"
        | "cast_int_to_unichar"
        | "cast_int_to_uint"
        | "cast_uint_to_int"
        | "cast_primitive") => {
            let operand = materialize(out, block, &op.args[0]);
            let result = expect_var(&op.result);
            if name == "cast_primitive" {
                let word_int = |ty: Option<&LowLevelType>| {
                    matches!(ty, Some(LowLevelType::Signed | LowLevelType::Unsigned))
                };
                let wide_int = |ty: Option<&LowLevelType>| {
                    matches!(
                        ty,
                        Some(
                            LowLevelType::SignedLongLong
                                | LowLevelType::UnsignedLongLong
                                | LowLevelType::SignedLongLongLong
                                | LowLevelType::UnsignedLongLongLong
                        )
                    )
                };
                let src = operand.concretetype();
                let dst = result.concretetype();
                assert!(
                    (word_int(src.as_ref()) && word_int(dst.as_ref()))
                        || wide_int(src.as_ref())
                        || wide_int(dst.as_ref()),
                    "jtransform_opname::lower_graph: cast_primitive is only ported for word-sized \
                     Signed/Unsigned or a longlong widen"
                );
            }
            let result_ty = value_type_of(&result);
            out.push_op_with_result_var(
                block,
                OpKind::UnaryOp {
                    op: "same_as".to_string(),
                    operand,
                    result_ty,
                },
                result,
            );
        }
        // 32-bit longlong lives in the float bank. `jtransform.py`
        // `rewrite_op_cast_int_to_longlong` and the ulonglong siblings
        // call `llong_from_*` / `ullong_from_*` instead of aliasing the
        // float result onto the int operand.
        name @ ("cast_int_to_longlong"
        | "cast_int_to_ulonglong"
        | "cast_uint_to_longlong"
        | "cast_uint_to_ulonglong") => {
            let operand = materialize(out, block, &op.args[0]);
            let result = expect_var(&op.result);
            let helper = match name {
                "cast_int_to_longlong" => "llong_from_int",
                "cast_uint_to_longlong" => "llong_from_uint",
                "cast_int_to_ulonglong" => "ullong_from_int",
                "cast_uint_to_ulonglong" => "ullong_from_uint",
                _ => unreachable!(),
            };
            // 32-bit longlong is the float bank (`f64` bits). 64-bit stays
            // an int. The registered helper uses the same split. This
            // lowering runs in the host prepass, so the width is the
            // cargo target's, not `size_of::<usize>()`.
            let longlong_is_float = crate::layout::target_word_size() < 8;
            let result_kind = if longlong_is_float { 'f' } else { 'i' };
            let result_ll = if longlong_is_float {
                majit_ir::value::Type::Float
            } else {
                majit_ir::value::Type::Int
            };
            let oopspec = match helper {
                "llong_from_int" | "ullong_from_int" => majit_ir::descr::OopSpecIndex::LlongFromInt,
                _ => majit_ir::descr::OopSpecIndex::LlongFromUint,
            };
            let target = crate::model::CallTarget::function_path([helper]);
            let fnaddr = crate::codewriter::call::symbolic_fnaddr_for_target(&target);
            let path = crate::codewriter::call::symbolic_fnaddr_path_for_target(&target);
            let funcptr = out
                .push_op_var(
                    block,
                    OpKind::ConstFnAddr {
                        value: fnaddr,
                        path,
                        symbolic: true,
                    },
                    true,
                )
                .expect("funcptr const produces a result");
            crate::model::FunctionGraph::set_concretetype_of_inline(
                &funcptr,
                crate::model::ConcreteType::Signed,
            );
            out.push_op_with_result_var(
                block,
                OpKind::CallResidual {
                    funcptr: crate::model::CallFuncPtr::Value(funcptr),
                    descriptor: crate::codewriter::call::CallDescriptor::from_signature(
                        &[majit_ir::value::Type::Int],
                        result_ll,
                        majit_ir::descr::EffectInfo::new(
                            majit_ir::descr::ExtraEffect::ElidableCannotRaise,
                            oopspec,
                        ),
                    ),
                    args_i: vec![operand],
                    args_r: Vec::new(),
                    args_f: Vec::new(),
                    result_kind,
                    indirect_targets: None,
                },
                result,
            );
        }
        // `direct_call(funcptr, arg0, arg1, …)` → `Call`.  The callee is the
        // `_func._name` carried by the leading funcptr constant
        // (`ConstValue::LLPtr` → `_ptr_obj::Func`); it becomes a
        // `CallTarget::FunctionPath` whose single segment matches the helper's
        // registered `CallPath`, so `register_opname_helper_graph` makes it
        // resolve as a *regular* callee rather than a residual synthetic
        // fnaddr.  The trailing operands are the call arguments; `OpKind::Call`
        // takes `Variable`s, so constant arguments (e.g. `ll_min(size, 1280)`)
        // are materialised through the same `ConstInt`/`ConstBool` path the
        // `int_*` arm uses.  The builder helpers' `direct_call`s all return a
        // value (`ll_min` → uint, `mallocfn` → the char buffer), so the result
        // is bound as a result var.
        "direct_call" => {
            let callee = callee_name_from_funcptr(&op.args[0]);
            // `_func._name` is the `::`-joined CallPath
            // (`functionptr(ft, key.segments().join("::"), …)`). Split so
            // a call to `majit_rlib::jit::isconstant` hits the marked
            // oopspec rather than a one-segment residual.
            let target = crate::model::CallTarget::function_path(
                callee.split("::").filter(|s| !s.is_empty()),
            );
            let args: Vec<FVar> = op.args[1..]
                .iter()
                .map(|a| materialize(out, block, a))
                .collect();
            let result = expect_var(&op.result);
            let result_ty = value_type_of(&result);
            out.push_op_with_result_var(
                block,
                OpKind::Call {
                    target,
                    args: crate::model::call_args(args),
                    result_ty,
                },
                result,
            );
        }
        other => panic!("jtransform_opname::lower_graph: unsupported opname {other:?}"),
    }
}

/// Resolve the string Variable backing a chars-array operand via the
/// block-local `getsubstruct` alias map.  Fail-loud if the array was not
/// produced by a same-block `getsubstruct` (e.g. threaded across a Phi —
/// out of scope for this slice).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn resolve_string(chars_alias: &HashMap<FVar, FVar>, arr: &Hlvalue) -> FVar {
    let arr_var = expect_var(arr);
    chars_alias
        .get(&arr_var)
        .cloned()
        .expect("array operand was not a same-block getsubstruct(\"chars\") result")
}

/// Map a flowspace operand to a model operand `Variable`, materialising
/// constants into `ConstInt`/`ConstBool` ops pushed ahead of the consumer.
fn materialize(out: &mut crate::model::FunctionGraph, block: BlockId, hlv: &Hlvalue) -> FVar {
    match hlv {
        Hlvalue::Variable(v) => v.clone(),
        Hlvalue::Constant(c) => {
            let (kind, lltype) = match &c.value {
                ConstValue::Int(n) => (OpKind::ConstInt(*n), LowLevelType::Signed),
                ConstValue::UInt(n) => (OpKind::ConstUInt(*n), LowLevelType::Unsigned),
                ConstValue::Bool(b) => (OpKind::ConstBool(*b), LowLevelType::Bool),
                other => panic!(
                    "jtransform_opname::lower_graph: cannot materialise constant operand {other:?}"
                ),
            };
            let result = out
                .push_op_var(block, kind, true)
                .expect("ConstInt/ConstBool op produces a result var");
            // `push_op_var` mints the result with `ConcreteType::Unknown`;
            // stamp the int-bank kind so regalloc colours it.
            result.set_concretetype(Some(lltype));
            result
        }
    }
}

/// Decode a `direct_call` callee operand — the leading funcptr constant.
/// `ConstValue::LLPtr` wraps an `_ptr` whose underlying object is an
/// `_ptr_obj::Func`; its `_func._name` is the callee identity that keys the
/// `CallPath` a helper is registered under.
fn callee_name_from_funcptr(hlv: &Hlvalue) -> String {
    use crate::translator::rtyper::lltypesystem::lltype::_ptr_obj;
    let Hlvalue::Constant(c) = hlv else {
        panic!("jtransform_opname::lower_graph: direct_call callee operand is not a constant");
    };
    let ConstValue::LLPtr(ptr) = &c.value else {
        panic!(
            "jtransform_opname::lower_graph: direct_call callee constant is not an LLPtr: {:?}",
            c.value
        );
    };
    match ptr._obj0_value() {
        Ok(Some(_ptr_obj::Func(func))) => func._name.clone(),
        other => panic!(
            "jtransform_opname::lower_graph: direct_call callee funcptr does not resolve to a Func object: {other:?}"
        ),
    }
}

/// Decode a `getfield` / `setfield` field-name operand — the Void
/// `ByteStr` constant `void_field_const` builds.
fn field_name_of(hlv: &Hlvalue) -> String {
    match hlv {
        Hlvalue::Constant(c) => match &c.value {
            ConstValue::ByteStr(bytes) => String::from_utf8_lossy(bytes).into_owned(),
            other => panic!(
                "jtransform_opname::lower_graph: field-name operand is not a ByteStr: {other:?}"
            ),
        },
        Hlvalue::Variable(_) => {
            panic!("jtransform_opname::lower_graph: field-name operand is a Variable")
        }
    }
}

/// The owning struct leaf of a field access — the pointee-struct name of
/// the base pointer's `concretetype` (`Ptr(Struct("stringbuilder"))` →
/// `"stringbuilder"`).  `None` when the base is not a pointer-to-struct,
/// in which case the layout layer falls back to its type-string heuristic.
fn owner_root_of(base: &FVar) -> Option<String> {
    match base.concretetype()? {
        LowLevelType::Ptr(ptr) => match ptr.TO {
            PtrTarget::Struct(s) => Some(s._name),
            _ => None,
        },
        _ => None,
    }
}

/// Whether an operand is the null pointer constant (`ConstValue::None`),
/// the second operand of the builder `ll_bool` `ptr_ne(builder, NULL)`.
fn is_null_const(hlv: &Hlvalue) -> bool {
    matches!(hlv, Hlvalue::Constant(c) if c.value == ConstValue::None)
}

/// Map a flowspace link arg (`Variable` or `Constant`) to the model
/// `LinkArg` — model links carry mixed var/const args directly.
fn linkarg_from_hlvalue(hlv: &Hlvalue) -> LinkArg {
    match hlv {
        Hlvalue::Variable(v) => LinkArg::Value(v.clone()),
        Hlvalue::Constant(c) => LinkArg::Const(c.clone()),
    }
}

/// Map a flowspace exitcase constant to the model `ExitCase`.
fn exitcase_from_const(value: &ConstValue) -> Option<ExitCase> {
    match value {
        ConstValue::Bool(b) => Some(ExitCase::Bool(*b)),
        other => Some(ExitCase::Const(other.clone())),
    }
}

/// `Hlvalue::Variable` → its `Variable`; `None` for a constant.
fn hlvalue_as_var(hlv: &Hlvalue) -> Option<FVar> {
    match hlv {
        Hlvalue::Variable(v) => Some(v.clone()),
        Hlvalue::Constant(_) => None,
    }
}

/// Expect an `Hlvalue` to be a `Variable` (the position is always a Variable
/// in the helper graphs the transducer accepts).
fn expect_var(hlv: &Hlvalue) -> FVar {
    match hlv {
        Hlvalue::Variable(v) => v.clone(),
        Hlvalue::Constant(c) => {
            panic!("jtransform_opname::lower_graph: expected a Variable, found constant {c:?}")
        }
    }
}

/// Map a `Variable`'s `concretetype` to the model `ValueType` for `BinOp`'s
/// `result_ty` — the `getkind`-collapsed kind space (`Char`/`Bool`/`Signed`
/// all land in the int bank).
fn value_type_of(var: &FVar) -> ValueType {
    use crate::model::ConcreteType;
    match crate::model::FunctionGraph::concretetype_of(var) {
        ConcreteType::Signed => ValueType::Int,
        ConcreteType::GcRef => ValueType::Ref(None),
        ConcreteType::Float => ValueType::Float,
        ConcreteType::Void => ValueType::Void,
        ConcreteType::Unknown => ValueType::Int,
    }
}

/// The stored-value kind of a `setfield` value operand — a `Variable`
/// (register) or an inline `Constant`.  Drives `FieldWrite::ty` (the
/// setfield `i`/`r`/`f` opcode), so it reads the operand's own
/// `concretetype` rather than the container field's declared type.
fn hlvalue_value_type(hlv: &Hlvalue) -> ValueType {
    match hlv {
        Hlvalue::Variable(v) => value_type_of(v),
        Hlvalue::Constant(c) => match &c.concretetype {
            Some(lltype) => value_type_from_lltype(lltype),
            None => ValueType::Int,
        },
    }
}

/// Collapse an `lltype` to the `getkind` model `ValueType` (`Char`/`Bool`/
/// `Signed`/`Unsigned` → int bank, GC `Ptr` → `Ref`).
fn value_type_from_lltype(lltype: &LowLevelType) -> ValueType {
    use crate::model::ConcreteType;
    match crate::model::getkind(lltype) {
        ConcreteType::Signed => ValueType::Int,
        ConcreteType::GcRef => ValueType::Ref(None),
        ConcreteType::Float => ValueType::Float,
        ConcreteType::Void => ValueType::Void,
        ConcreteType::Unknown => ValueType::Int,
    }
}

/// The struct leaf named by a `malloc` type operand — a Void `LowLevelType`
/// constant wrapping the fixed-size GcStruct (`Struct("stringbuilder")` →
/// `"stringbuilder"`).
fn malloc_struct_owner(hlv: &Hlvalue) -> String {
    match hlv {
        Hlvalue::Constant(c) => match &c.value {
            ConstValue::LowLevelType(lltype) => match &**lltype {
                LowLevelType::Struct(s) => s._name.clone(),
                other => panic!(
                    "jtransform_opname::lower_graph: malloc type operand is not a Struct: {other:?}"
                ),
            },
            other => panic!(
                "jtransform_opname::lower_graph: malloc first operand is not a LowLevelType const: {other:?}"
            ),
        },
        Hlvalue::Variable(_) => {
            panic!("jtransform_opname::lower_graph: malloc first operand is a Variable")
        }
    }
}

/// Whether a `malloc` flavor operand is the `{'flavor': 'gc'}` dict constant
/// (`gc_flavor_const`).  `jtransform.py rewrite_op_malloc` lowers only the gc
/// flavor to `new(descr)`; a `raw`/`nonmovable` allocation has no such form,
/// so the transducer rejects it rather than silently emitting a GC `New`.
fn is_gc_flavor_const(hlv: &Hlvalue) -> bool {
    let Hlvalue::Constant(c) = hlv else {
        return false;
    };
    let Some(items) = c.value.dict_items() else {
        return false;
    };
    items.get(&ConstValue::byte_str("flavor")) == Some(&ConstValue::byte_str("gc"))
}

/// A helper graph operates on a single string width, so the family is a
/// graph-wide property: `unicode*` if any operand carries an
/// `Array(UniChar)` pointer, else `str*`.
fn detect_family(graph: &FlowGraph) -> StrFamily {
    let is_unicode = graph.iterblocks().iter().any(|block| {
        block.borrow().operations.iter().any(|op| {
            op.args
                .iter()
                .chain(std::iter::once(&op.result))
                .any(|hlv| matches!(hlv, Hlvalue::Variable(v) if is_unichar_array_ptr(v)))
        })
    });
    if is_unicode {
        StrFamily::UNICODE
    } else {
        StrFamily::STR
    }
}

/// Whether a `Variable`'s `concretetype` is `Ptr(Array(UniChar))` — the
/// chars-array pointer of a `UNICODE` object.
fn is_unichar_array_ptr(var: &FVar) -> bool {
    match var.concretetype() {
        Some(LowLevelType::Ptr(ptr)) => {
            matches!(&ptr.TO, PtrTarget::Array(arr) if matches!(arr.OF, LowLevelType::UniChar))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::lower_graph;
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph as FlowGraph, GraphFunc, Hlvalue,
        Link, SpaceOperation, c_last_exception,
    };
    use crate::model::{ExitCase, ExitSwitch, LinkArg, OpKind, ValueType};
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    use crate::translator::rtyper::lltypesystem::rbuilder::{
        STRINGBUILDERPTR, build_ll_bool_helper_graph, build_ll_getlength_helper_graph,
    };
    use crate::translator::rtyper::lltypesystem::rstr::{
        STRPTR, chars_array_ptr_lltype_from_strptr, struct_lltype_from_strptr,
    };
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};
    use crate::translator::rtyper::rtyper::{constant_with_lltype, variable_with_lltype};

    fn chars_field() -> Hlvalue {
        constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void)
    }
    fn signed_const(n: i64) -> Hlvalue {
        constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed)
    }
    fn bool_const(b: bool) -> Hlvalue {
        constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool)
    }

    /// Build a single-string-type opname helper graph in the fusion-friendly
    /// shape: the chars substruct is re-derived *block-locally* from a
    /// threaded `STR` operand (never threaded as an interior pointer across a
    /// Phi).  Exercises the full S1 opname set:
    ///
    /// ```text
    /// start(s):
    ///     chars  = getsubstruct(s, "chars")
    ///     len    = getarraysize(chars)
    ///     newstr = malloc_varsize(STR, gc, len)
    ///     cond   = int_lt(0, len)
    ///     if cond -> copy(s, newstr) else -> return(newstr)
    /// copy(s, newstr):
    ///     chars_s  = getsubstruct(s, "chars")
    ///     c        = getarrayitem(chars_s, 0)
    ///     newchars = getsubstruct(newstr, "chars")
    ///     setarrayitem(newchars, 0, c)
    ///     -> return(newstr)
    /// ```
    fn build_fusable_str_helper() -> FlowGraph {
        let strptr = STRPTR.clone();
        let chars_ptr = chars_array_ptr_lltype_from_strptr(&strptr).expect("chars ptr lltype");
        let struct_lltype = struct_lltype_from_strptr(&strptr).expect("struct lltype");

        let s = variable_with_lltype("s", strptr.clone());
        let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
        let return_var = variable_with_lltype("result", strptr.clone());
        let graph = FlowGraph::with_return_var(
            "ll_test_fusable_str_helper",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );

        // copy block threads the STRING pointers (s, newstr), NOT interior
        // chars pointers — so each consumer re-derives its substruct locally.
        let s_c = variable_with_lltype("s", strptr.clone());
        let newstr_c = variable_with_lltype("newstr", strptr.clone());
        let copy_block = Block::shared(vec![
            Hlvalue::Variable(s_c.clone()),
            Hlvalue::Variable(newstr_c.clone()),
        ]);

        // startblock.
        let chars = variable_with_lltype("chars", chars_ptr.clone());
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(s.clone()), chars_field()],
            Hlvalue::Variable(chars.clone()),
        ));
        let len = variable_with_lltype("len", LowLevelType::Signed);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(chars)],
            Hlvalue::Variable(len.clone()),
        ));
        let newstr = variable_with_lltype("newstr", strptr.clone());
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "malloc_varsize",
            vec![
                lowlevel_type_const(struct_lltype),
                gc_flavor_const().expect("gc flavor const"),
                Hlvalue::Variable(len.clone()),
            ],
            Hlvalue::Variable(newstr.clone()),
        ));
        let cond = variable_with_lltype("cond", LowLevelType::Bool);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_lt",
            vec![signed_const(0), Hlvalue::Variable(len)],
            Hlvalue::Variable(cond.clone()),
        ));
        startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(s), Hlvalue::Variable(newstr.clone())],
                Some(copy_block.clone()),
                Some(bool_const(true)),
            )
            .into_ref(),
            Link::new(
                vec![Hlvalue::Variable(newstr)],
                Some(graph.returnblock.clone()),
                Some(bool_const(false)),
            )
            .into_ref(),
        ]);

        // copy block.
        let chars_s = variable_with_lltype("chars_s", chars_ptr.clone());
        copy_block.borrow_mut().operations.push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(s_c), chars_field()],
            Hlvalue::Variable(chars_s.clone()),
        ));
        let c = variable_with_lltype("c", LowLevelType::Char);
        copy_block.borrow_mut().operations.push(SpaceOperation::new(
            "getarrayitem",
            vec![Hlvalue::Variable(chars_s), signed_const(0)],
            Hlvalue::Variable(c.clone()),
        ));
        let newchars = variable_with_lltype("newchars", chars_ptr);
        copy_block.borrow_mut().operations.push(SpaceOperation::new(
            "getsubstruct",
            vec![Hlvalue::Variable(newstr_c.clone()), chars_field()],
            Hlvalue::Variable(newchars.clone()),
        ));
        let set = variable_with_lltype("set", LowLevelType::Void);
        copy_block.borrow_mut().operations.push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(newchars),
                signed_const(0),
                Hlvalue::Variable(c),
            ],
            Hlvalue::Variable(set),
        ));
        copy_block.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(newstr_c)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        graph
    }

    #[test]
    fn lower_graph_fuses_str_helper_to_blackhole_opcodes() {
        let flow = build_fusable_str_helper();
        let model = lower_graph(&flow);

        // start(0) / return(1) / except(2) / copy(3).
        assert_eq!(model.blocks.len(), 4);

        let mut blackhole = Vec::new();
        let mut binops = Vec::new();
        let mut const_ints = 0;
        let mut residual = Vec::new();
        for block in &model.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::LoweredBlackholeOp { opname, .. } => blackhole.push(opname.clone()),
                    OpKind::BinOp { op, .. } => binops.push(op.clone()),
                    OpKind::ConstInt(_) | OpKind::ConstFnAddr { .. } => const_ints += 1,
                    other => residual.push(format!("{other:?}")),
                }
            }
        }
        blackhole.sort();

        // `getsubstruct` emits nothing; `getarraysize`/`getarrayitem`/
        // `setarrayitem`/`malloc_varsize` fuse to the string blackhole ops.
        assert_eq!(
            blackhole,
            vec!["newstr", "strgetitem", "strlen", "strsetitem"]
        );
        // `int_lt` lowers to a bare-named `BinOp`.
        assert_eq!(binops, vec!["lt"]);
        // The constant `0` is materialised once per consumer: `int_lt`,
        // `getarrayitem`, `setarrayitem`.
        assert_eq!(const_ints, 3);
        assert!(residual.is_empty(), "unexpected residual ops: {residual:?}");
    }

    /// The integer fragment in `LLHelpers.ll_int2hex` is not the all-binary
    /// family the original Spine-B shortcut assumed.  Pin the exact
    /// `Transformer` normalization: unary `int_neg` remains unary,
    /// `uint_is_true` and `uint_and` use their signed JitCode spellings, and
    /// logical `uint_rshift` keeps its distinct unsigned spelling.
    #[test]
    fn lower_graph_normalizes_ll_int2hex_integer_ops_like_jtransform() {
        let signed = variable_with_lltype("signed", LowLevelType::Signed);
        let unsigned = variable_with_lltype("unsigned", LowLevelType::Unsigned);
        let start = Block::shared(vec![
            Hlvalue::Variable(signed.clone()),
            Hlvalue::Variable(unsigned.clone()),
        ]);
        let returned = variable_with_lltype("returned", LowLevelType::Unsigned);
        let graph = FlowGraph::with_return_var(
            "ll_int2hex_integer_fragment",
            start.clone(),
            Hlvalue::Variable(returned),
        );

        let negated = variable_with_lltype("negated", LowLevelType::Signed);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_neg",
            vec![Hlvalue::Variable(signed)],
            Hlvalue::Variable(negated.clone()),
        ));
        let nonnegative = variable_with_lltype("nonnegative", LowLevelType::Signed);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_force_ge_zero",
            vec![Hlvalue::Variable(negated)],
            Hlvalue::Variable(nonnegative.clone()),
        ));
        let between = variable_with_lltype("between", LowLevelType::Bool);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_between",
            vec![
                signed_const(0),
                Hlvalue::Variable(nonnegative.clone()),
                signed_const(16),
            ],
            Hlvalue::Variable(between.clone()),
        ));
        let kept = variable_with_lltype("kept", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "keepalive",
            vec![Hlvalue::Variable(between)],
            Hlvalue::Variable(kept),
        ));
        let cast_unsigned = variable_with_lltype("cast_unsigned", LowLevelType::Unsigned);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "cast_primitive",
            vec![Hlvalue::Variable(nonnegative)],
            Hlvalue::Variable(cast_unsigned.clone()),
        ));
        let cast_char = variable_with_lltype("cast_char", LowLevelType::Char);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "cast_int_to_char",
            // `LLHelpers.ll_int2hex` produces its literal sign/prefix/digit
            // characters in exactly this constant-operand shape.
            vec![signed_const(48)],
            Hlvalue::Variable(cast_char.clone()),
        ));
        let truth = variable_with_lltype("truth", LowLevelType::Bool);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "uint_is_true",
            vec![Hlvalue::Variable(unsigned.clone())],
            Hlvalue::Variable(truth),
        ));
        let masked = variable_with_lltype("masked", LowLevelType::Unsigned);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "uint_and",
            vec![
                Hlvalue::Variable(cast_unsigned),
                Hlvalue::Variable(cast_char),
            ],
            Hlvalue::Variable(masked.clone()),
        ));
        let shifted = variable_with_lltype("shifted", LowLevelType::Unsigned);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "uint_rshift",
            vec![Hlvalue::Variable(masked), signed_const(4)],
            Hlvalue::Variable(shifted.clone()),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(shifted)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);
        let mut unary = Vec::new();
        let mut binary = Vec::new();
        let mut lowered = Vec::new();
        let mut live = 0;
        for op in &model.block(model.startblock).operations {
            match &op.kind {
                OpKind::UnaryOp { op, .. } => unary.push(op.as_str()),
                OpKind::BinOp { op, .. } => binary.push(op.as_str()),
                OpKind::LoweredBlackholeOp { opname, .. } => lowered.push(opname.as_str()),
                OpKind::Live => live += 1,
                OpKind::ConstInt(_) | OpKind::ConstUInt(_) | OpKind::ConstFnAddr { .. } => {}
                other => panic!("unexpected lowered integer-fragment op: {other:?}"),
            }
        }
        assert_eq!(
            unary,
            vec![
                "int_neg",
                "int_force_ge_zero",
                "same_as",
                "same_as",
                "int_is_true"
            ]
        );
        assert_eq!(binary, vec!["and", "uint_rshift"]);
        assert_eq!(lowered, vec!["int_between"]);
        assert_eq!(live, 1);

        // Do not stop at the transducer shape: the normalized names must
        // survive the shared Transformer/regalloc/flatten/assembler tail and
        // become an installed JitCode body, just as the real helper does.
        use crate::codewriter::call::CallControl;
        use crate::codewriter::codewriter::CodeWriter;
        use crate::codewriter::jtransform::GraphTransformConfig;
        use crate::parse::CallPath;

        let path = CallPath::from_segments(["ll_int2hex_integer_fragment"]);
        let mut callcontrol = CallControl::new();
        let jitcode = callcontrol.register_opname_helper_graph(path, graph);
        let mut codewriter = CodeWriter::new();
        codewriter.drain_pending_graphs(&mut callcontrol, &GraphTransformConfig::default());
        assert!(
            !jitcode
                .try_body()
                .expect("integer fragment drains to JitCode")
                .code
                .is_empty()
        );
    }

    /// Pin every explicit integer-bank no-op in
    /// `Transformer.rewrite_op_cast_*`.  They are aliases, not backend casts:
    /// Char/UniChar/Bool/Signed/Unsigned all occupy JitCode's `i` bank.
    #[test]
    fn lower_graph_ports_the_complete_integer_bank_noop_cast_family() {
        let boolean = variable_with_lltype("boolean", LowLevelType::Bool);
        let character = variable_with_lltype("character", LowLevelType::Char);
        let unichar = variable_with_lltype("unichar", LowLevelType::UniChar);
        let signed = variable_with_lltype("signed", LowLevelType::Signed);
        let unsigned = variable_with_lltype("unsigned", LowLevelType::Unsigned);
        let start = Block::shared(vec![
            Hlvalue::Variable(boolean.clone()),
            Hlvalue::Variable(character.clone()),
            Hlvalue::Variable(unichar.clone()),
            Hlvalue::Variable(signed.clone()),
            Hlvalue::Variable(unsigned.clone()),
        ]);
        let returned = variable_with_lltype("returned", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "integer_bank_noop_casts",
            start.clone(),
            Hlvalue::Variable(returned),
        );

        let cases = [
            ("cast_bool_to_int", boolean.clone(), LowLevelType::Signed),
            ("cast_bool_to_uint", boolean, LowLevelType::Unsigned),
            ("cast_char_to_int", character, LowLevelType::Signed),
            ("cast_unichar_to_int", unichar, LowLevelType::Signed),
            ("cast_int_to_char", signed.clone(), LowLevelType::Char),
            ("cast_int_to_unichar", signed.clone(), LowLevelType::UniChar),
            ("cast_int_to_uint", signed, LowLevelType::Unsigned),
            ("cast_uint_to_int", unsigned, LowLevelType::Signed),
        ];
        let mut last = None;
        for (index, (opname, operand, result_ty)) in cases.into_iter().enumerate() {
            let result = variable_with_lltype(&format!("cast_{index}"), result_ty);
            start.borrow_mut().operations.push(SpaceOperation::new(
                opname,
                vec![Hlvalue::Variable(operand)],
                Hlvalue::Variable(result.clone()),
            ));
            last = Some(result);
        }
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(last.expect("cast family is non-empty"))],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);
        let aliases: Vec<_> = model
            .block(model.startblock)
            .operations
            .iter()
            .map(|op| match &op.kind {
                OpKind::UnaryOp { op, .. } => op.as_str(),
                other => panic!("integer-bank cast must lower to an alias, got {other:?}"),
            })
            .collect();
        assert_eq!(aliases, vec!["same_as"; 8]);

        use crate::codewriter::call::CallControl;
        use crate::codewriter::codewriter::CodeWriter;
        use crate::codewriter::jtransform::GraphTransformConfig;
        use crate::parse::CallPath;

        let path = CallPath::from_segments(["integer_bank_noop_casts"]);
        let mut callcontrol = CallControl::new();
        let jitcode = callcontrol.register_opname_helper_graph(path, graph);
        let mut codewriter = CodeWriter::new();
        codewriter.drain_pending_graphs(&mut callcontrol, &GraphTransformConfig::default());
        assert!(
            !jitcode
                .try_body()
                .expect("integer-bank alias graph drains to JitCode")
                .code
                .is_empty()
        );
    }

    /// `Transformer.rewrite_op_int_{floordiv,mod} = _do_builtin_call`.
    /// The opname spine must enter the shared rich-graph rewrite as the bare
    /// BinOp names.  With no helper graph registered, that is the
    /// `_ll_2_int_*` residual call; emitting `int_floordiv/ii>i` or
    /// `int_mod/ii>i` directly would name handlers that do not exist in
    /// `BlackholeInterpreter`.
    #[test]
    fn lower_graph_routes_int_divmod_through_the_builtin_call_rewrite() {
        let lhs = variable_with_lltype("lhs", LowLevelType::Signed);
        let rhs = variable_with_lltype("rhs", LowLevelType::Signed);
        let start = Block::shared(vec![
            Hlvalue::Variable(lhs.clone()),
            Hlvalue::Variable(rhs.clone()),
        ]);
        let returned = variable_with_lltype("returned", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "lowlevel_int_divmod",
            start.clone(),
            Hlvalue::Variable(returned),
        );
        let modulo = variable_with_lltype("modulo", LowLevelType::Signed);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_mod",
            vec![Hlvalue::Variable(lhs), Hlvalue::Variable(rhs.clone())],
            Hlvalue::Variable(modulo.clone()),
        ));
        let quotient = variable_with_lltype("quotient", LowLevelType::Signed);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_floordiv",
            vec![Hlvalue::Variable(modulo), Hlvalue::Variable(rhs)],
            Hlvalue::Variable(quotient.clone()),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(quotient)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);
        let binary: Vec<_> = model
            .block(model.startblock)
            .operations
            .iter()
            .map(|op| match &op.kind {
                OpKind::BinOp { op, .. } => op.as_str(),
                other => panic!("int div/mod must enter shared jtransform as BinOp, got {other:?}"),
            })
            .collect();
        assert_eq!(binary, vec!["mod", "floordiv"]);

        use crate::codewriter::call::CallControl;
        use crate::codewriter::codewriter::CodeWriter;
        use crate::codewriter::jtransform::GraphTransformConfig;
        use crate::parse::CallPath;

        let path = CallPath::from_segments(["lowlevel_int_divmod"]);
        let mut callcontrol = CallControl::new();
        let jitcode = callcontrol.register_opname_helper_graph(path, graph);
        let mut codewriter = CodeWriter::new();
        codewriter.drain_pending_graphs(&mut callcontrol, &GraphTransformConfig::default());
        let body = jitcode
            .try_body()
            .expect("int div/mod graph drains through builtin residual calls");
        assert!(!body.code.is_empty());
    }

    #[test]
    fn lower_graph_preserves_bool_branch_control_flow() {
        let flow = build_fusable_str_helper();
        let model = lower_graph(&flow);

        // startblock is a 2-way bool branch on the `int_lt` result.
        let start = model.block(model.startblock);
        assert!(matches!(start.exitswitch, Some(ExitSwitch::Value(_))));
        assert_eq!(start.exits.len(), 2);
        let cases: Vec<&Option<ExitCase>> = start.exits.iter().map(|l| &l.exitcase).collect();
        assert!(cases.contains(&&Some(ExitCase::Bool(true))));
        assert!(cases.contains(&&Some(ExitCase::Bool(false))));

        // The true branch targets the copy block, which holds the read/write
        // string ops; the false branch returns directly.
        let true_link = start
            .exits
            .iter()
            .find(|l| l.exitcase == Some(ExitCase::Bool(true)))
            .expect("true exit");
        let copy_block = model.block(true_link.target);
        let copy_opnames: Vec<&str> = copy_block
            .operations
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::LoweredBlackholeOp { opname, .. } => Some(opname.as_str()),
                _ => None,
            })
            .collect();
        assert!(copy_opnames.contains(&"strgetitem"));
        assert!(copy_opnames.contains(&"strsetitem"));

        let false_link = start
            .exits
            .iter()
            .find(|l| l.exitcase == Some(ExitCase::Bool(false)))
            .expect("false exit");
        assert_eq!(false_link.target, model.returnblock);
    }

    /// End-to-end: register the opname helper through Spine B and drive the
    /// drain loop, proving the transduced graph survives the shared
    /// regalloc/flatten/assemble tail and commits a non-empty `JitCode` body.
    /// This is the test that actually exercises the `LoweredBlackholeOp`
    /// assembler encode + dynamic byte assignment.
    #[test]
    fn lower_graph_drains_to_a_jitcode_body() {
        use crate::codewriter::call::CallControl;
        use crate::codewriter::codewriter::CodeWriter;
        use crate::codewriter::jtransform::GraphTransformConfig;
        use crate::parse::CallPath;

        let flow = build_fusable_str_helper();
        let path = CallPath::from_segments(["ll_test_fusable_str_helper"]);

        let mut callcontrol = CallControl::new();
        let jitcode = callcontrol.register_opname_helper_graph(path, flow);
        assert!(jitcode.try_body().is_none(), "shell starts bodyless");

        let mut codewriter = CodeWriter::new();
        codewriter.drain_pending_graphs(&mut callcontrol, &GraphTransformConfig::default());

        let body = jitcode
            .try_body()
            .expect("Spine-B drain commits a jitcode body");
        assert!(!body.code.is_empty(), "assembled bytecode is non-empty");
        assert_eq!(jitcode.try_index(), Some(0));
    }

    /// A flowspace `ovfcheck` block (trailing `int_add_ovf` + `c_last_exception`
    /// switch, the shape `rbuilder::close_ovfcheck_block` emits for
    /// `ll_grow_by`/`ll_grow_and_append`) carries `last_exception` /
    /// `last_exc_value` extravars on its OverflowError exit. `lower_graph` must
    /// copy them onto the model link: flatten's `make_exception_link` asserts
    /// they are `Some` and, for the empty-target MemoryError handler, unwraps
    /// them — dropping them turns the faithful overflow edge into a panic.
    #[test]
    fn lower_graph_preserves_ovfcheck_exception_extravars() {
        use crate::translator::rtyper::rtyper::exception_args;

        let overflow = exception_args("OverflowError").expect("OverflowError args");
        let mem = exception_args("MemoryError").expect("MemoryError args");

        let a = variable_with_lltype("a", LowLevelType::Signed);
        let b = variable_with_lltype("b", LowLevelType::Signed);
        let start = Block::shared(vec![
            Hlvalue::Variable(a.clone()),
            Hlvalue::Variable(b.clone()),
        ]);
        let ret = variable_with_lltype("result", LowLevelType::Signed);
        let graph =
            FlowGraph::with_return_var("ll_ovf_probe", start.clone(), Hlvalue::Variable(ret));

        let sum = variable_with_lltype("sum", LowLevelType::Signed);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add_ovf",
            vec![Hlvalue::Variable(a), Hlvalue::Variable(b)],
            Hlvalue::Variable(sum.clone()),
        ));

        // Empty-ops handler that raises MemoryError via the exceptblock — the
        // grow_by shape whose bare-reraise unwrap would fire on dropped extravars.
        let handler = Block::shared(vec![]);
        handler.closeblock(vec![
            Link::new(mem, Some(graph.exceptblock.clone()), None).into_ref(),
        ]);

        start.borrow_mut().exitswitch = Some(Hlvalue::Constant(c_last_exception()));
        let normal = Link::new(
            vec![Hlvalue::Variable(sum)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        let mut exc = Link::new(vec![], Some(handler), Some(overflow[0].clone()));
        exc.extravars(Some(overflow[0].clone()), Some(overflow[1].clone()));
        start.closeblock(vec![normal, exc.into_ref()]);

        let model = lower_graph(&graph);

        let mut canraise_blocks = 0;
        for block in &model.blocks {
            if matches!(block.exitswitch, Some(ExitSwitch::LastException)) {
                canraise_blocks += 1;
                let exc_exit = block
                    .exits
                    .iter()
                    .find(|l| l.exitcase.is_some())
                    .expect("canraise block has an exception exit");
                assert!(
                    exc_exit.last_exception.is_some(),
                    "lower_graph dropped last_exception on the OverflowError edge"
                );
                assert!(
                    exc_exit.last_exc_value.is_some(),
                    "lower_graph dropped last_exc_value on the OverflowError edge"
                );
            }
        }
        assert_eq!(canraise_blocks, 1, "exactly one c_last_exception block");
    }

    /// `ll_bool(builder)` drains end-to-end with NO struct layout registered:
    /// its only opname is `ptr_ne(builder, NULL)`, which lowers to the unary
    /// `ptr_nonzero` and assembles to an `int_is_true`-family bytecode with no
    /// field/size descr — so the builder container never needs a layout.  This
    /// is the layer-2 floor: the simplest builder accessor commits a body
    /// without any layer-2a registration.
    #[test]
    fn ll_bool_drains_without_a_struct_layout() {
        use crate::codewriter::call::CallControl;
        use crate::codewriter::codewriter::CodeWriter;
        use crate::codewriter::jtransform::GraphTransformConfig;
        use crate::parse::CallPath;

        let helper = build_ll_bool_helper_graph("ll_bool", STRINGBUILDERPTR.clone())
            .expect("build_ll_bool_helper_graph");
        let flow = std::rc::Rc::try_unwrap(helper.graph)
            .expect("sole owner of the ll_bool helper graph")
            .into_inner();
        let path = CallPath::from_segments(["ll_bool"]);

        let mut callcontrol = CallControl::new();
        let jitcode = callcontrol.register_opname_helper_graph(path, flow);
        let mut codewriter = CodeWriter::new();
        codewriter.drain_pending_graphs(&mut callcontrol, &GraphTransformConfig::default());

        let body = jitcode
            .try_body()
            .expect("ll_bool drains to a jitcode body");
        assert!(!body.code.is_empty(), "assembled bytecode is non-empty");
    }

    /// `ll_getlength(builder)`'s three `getfield`s become `getfield_gc_i`
    /// bytecodes whose FieldDescr the assembler resolves through
    /// `CallControl.struct_fields`; with the synthetic STRINGBUILDER layout
    /// registered (mirroring the resizable `"list"` header registration), the
    /// getfield-bearing graph survives the regalloc/flatten/assemble tail and
    /// commits a body.  This test proves survival-through-assembly, not the
    /// resolved offsets themselves — the offsets (buf@0/pos@8/end@16/total@24/
    /// pieces@32) are verified directly against `fielddescrof` in
    /// `call::tests::rtyper_synthesised_stringbuilder_struct_resolves_its_field_descrs`.
    /// (`fielddescrof` falls back to slot 0 rather than panicking when the
    /// owner is unregistered, so a body-nonempty assertion alone cannot
    /// discriminate the layout — that is the descr test's job.)
    #[test]
    fn ll_getlength_drains_with_the_stringbuilder_layout() {
        use crate::codewriter::call::CallControl;
        use crate::codewriter::codewriter::CodeWriter;
        use crate::codewriter::jtransform::GraphTransformConfig;
        use crate::parse::CallPath;

        let helper = build_ll_getlength_helper_graph("ll_getlength", STRINGBUILDERPTR.clone())
            .expect("build_ll_getlength_helper_graph");
        let flow = std::rc::Rc::try_unwrap(helper.graph)
            .expect("sole owner of the ll_getlength helper graph")
            .into_inner();
        let path = CallPath::from_segments(["ll_getlength"]);

        let mut callcontrol = CallControl::new();
        let mut registry = crate::front::StructFieldRegistry::default();
        registry.fields.insert(
            "stringbuilder".to_string(),
            vec![
                ("current_buf".to_string(), "&()".to_string()),
                ("current_pos".to_string(), "i64".to_string()),
                ("current_end".to_string(), "i64".to_string()),
                ("total_size".to_string(), "i64".to_string()),
                ("extra_pieces".to_string(), "&()".to_string()),
            ],
        );
        callcontrol.set_struct_fields(registry);

        let jitcode = callcontrol.register_opname_helper_graph(path, flow);
        let mut codewriter = CodeWriter::new();
        codewriter.drain_pending_graphs(&mut callcontrol, &GraphTransformConfig::default());

        let body = jitcode
            .try_body()
            .expect("ll_getlength drains to a jitcode body");
        assert!(!body.code.is_empty(), "assembled bytecode is non-empty");
    }

    /// A `direct_call` to a registered opname helper must resolve as a
    /// *regular* callee (it owns a generated JitCode), not a residual call to
    /// a synthetic low-level helper.  Pre-fix the helper was visible only in
    /// `opname_graphs`, so `target_to_path`/`graphs_from` missed it and
    /// `guess_call_kind` returned `Residual`.
    #[test]
    fn registered_opname_helper_resolves_as_regular_callee() {
        use crate::codewriter::call::{CallControl, CallKind};
        use crate::model::{CallTarget, OpKind, SpaceOperation, ValueType};
        use crate::parse::CallPath;

        let flow = build_fusable_str_helper();
        let path = CallPath::from_segments(["ll_test_fusable_str_helper"]);

        let mut callcontrol = CallControl::new();
        callcontrol.register_opname_helper_graph(path.clone(), flow);

        let target = CallTarget::function_path(["ll_test_fusable_str_helper"]);
        assert_eq!(callcontrol.target_to_path(&target), Some(path.clone()));

        let call_op = SpaceOperation {
            result: None,
            kind: OpKind::Call {
                target,
                args: crate::model::call_args(vec![]),
                result_ty: ValueType::Void,
            },
        };
        assert_eq!(callcontrol.graphs_from(&call_op), Some(vec![path]));
        assert_eq!(callcontrol.guess_call_kind(&call_op), CallKind::Regular);
    }

    /// End-to-end: the restructured production `ll_strconcat` helper lowers
    /// cleanly through Spine B.  The two `copystrcontent` ops become void
    /// `LoweredBlackholeOp`s, the source-length reads fuse to `strlen`, and
    /// the cross-Phi `resolve_string` fail-loud is never reached because no
    /// per-char loop (strgetitem/strsetitem) survives.
    #[test]
    fn lower_graph_lowers_strconcat_helper_to_copystrcontent() {
        use crate::translator::rtyper::lltypesystem::rstr::{
            STRPTR, build_ll_strconcat_helper_graph,
        };

        let helper = build_ll_strconcat_helper_graph("ll_strconcat", STRPTR.clone())
            .expect("build strconcat helper");
        let flow = helper.graph.borrow();
        let model = lower_graph(&flow);

        let mut blackhole = Vec::new();
        for block in &model.blocks {
            for op in &block.operations {
                if let OpKind::LoweredBlackholeOp { opname, args } = &op.kind {
                    blackhole.push(opname.clone());
                    if opname == "copystrcontent" {
                        assert_eq!(args.len(), 5, "copystrcontent has 5 operands");
                        assert!(op.result.is_none(), "copystrcontent is void");
                    }
                }
            }
        }

        // Two source-length reads (strlen) + one alloc (newstr) + two copies.
        assert_eq!(
            blackhole.iter().filter(|n| *n == "strlen").count(),
            2,
            "two strlen source-length reads"
        );
        assert_eq!(blackhole.iter().filter(|n| *n == "newstr").count(), 1);
        assert_eq!(
            blackhole.iter().filter(|n| *n == "copystrcontent").count(),
            2,
            "two copystrcontent bulk copies"
        );
        // No per-char string ops: the loop is gone, so resolve_string never runs.
        assert!(
            !blackhole
                .iter()
                .any(|n| n == "strgetitem" || n == "strsetitem"),
            "no per-char string ops survive: {blackhole:?}"
        );

        // start forwards unconditionally into the returnblock.
        let start = model.block(model.startblock);
        assert!(start.exitswitch.is_none(), "start exit is unconditional");
        assert_eq!(start.exits.len(), 1);
        assert_eq!(start.exits[0].target, model.returnblock);
    }

    /// `ll_bool(builder)` is `builder != nullptr(TO)`.  The `ptr_ne(builder,
    /// NULL)` lowers to the orthodox unary null test `ptr_nonzero` rather than
    /// a binary compare that would need a null `Ref` operand materialised.
    #[test]
    fn lower_graph_lowers_ll_bool_to_ptr_nonzero() {
        let helper = build_ll_bool_helper_graph("ll_bool", STRINGBUILDERPTR.clone())
            .expect("build_ll_bool_helper_graph");
        let flow = helper.graph.borrow();
        let model = lower_graph(&flow);

        let mut unops = Vec::new();
        let mut residual = Vec::new();
        for block in &model.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::UnaryOp { op, .. } => unops.push(op.clone()),
                    other => residual.push(format!("{other:?}")),
                }
            }
        }
        assert_eq!(unops, vec!["ptr_nonzero"]);
        assert!(residual.is_empty(), "unexpected residual ops: {residual:?}");
    }

    /// `ll_getlength(builder)` is `total_size - (current_end - current_pos)`.
    /// The three `getfield`s lower to `FieldRead`s owned by the
    /// `"stringbuilder"` struct (mutable, so not `_pure`), and the two
    /// `int_sub`s to bare-named `BinOp("sub")`.
    #[test]
    fn lower_graph_lowers_ll_getlength_fields_and_subs() {
        let helper = build_ll_getlength_helper_graph("ll_getlength", STRINGBUILDERPTR.clone())
            .expect("build_ll_getlength_helper_graph");
        let flow = helper.graph.borrow();
        let model = lower_graph(&flow);

        let mut fields = Vec::new();
        let mut binops = Vec::new();
        let mut residual = Vec::new();
        for block in &model.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::FieldRead {
                        field, ty, pure, ..
                    } => {
                        assert_eq!(field.owner_root.as_deref(), Some("stringbuilder"));
                        assert_eq!(*ty, ValueType::Int);
                        assert!(!pure, "builder length fields are mutable");
                        fields.push(field.name.clone());
                    }
                    OpKind::BinOp { op, .. } => binops.push(op.clone()),
                    other => residual.push(format!("{other:?}")),
                }
            }
        }
        assert_eq!(fields, vec!["current_end", "current_pos", "total_size"]);
        assert_eq!(binops, vec!["sub", "sub"]);
        assert!(residual.is_empty(), "unexpected residual ops: {residual:?}");
    }

    /// The builder-construction op fragment: `malloc(STRINGBUILDER)` →
    /// `New{owner:"stringbuilder"}`, `setfield(b, "current_pos", 0)` →
    /// `FieldWrite` with the `0` kept as an inline `Const`, and
    /// `cast_int_to_uint` → a droppable `UnaryOp`.  Exercised on a synthetic
    /// graph rather than the full `ll_new` because `build_ll_new_helper_graph`'s
    /// unit test wires its two `direct_call` callees as dummy `None` consts (no
    /// real funcptr), which the `direct_call` arm rejects; a faithful `ll_new`
    /// drain needs real funcptr consts (a live rtyper).
    #[test]
    fn lower_graph_lowers_malloc_setfield_and_cast() {
        use crate::translator::rtyper::lltypesystem::rbuilder::STRINGBUILDER;

        let size = variable_with_lltype("size", LowLevelType::Signed);
        let startblock = Block::shared(vec![Hlvalue::Variable(size.clone())]);
        let return_var = variable_with_lltype("b", STRINGBUILDERPTR.clone());
        let graph = FlowGraph::with_return_var(
            "ll_test_builder_ctor_fragment",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );

        let b = variable_with_lltype("b", STRINGBUILDERPTR.clone());
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "malloc",
            vec![
                lowlevel_type_const(STRINGBUILDER.clone()),
                gc_flavor_const().expect("gc flavor const"),
            ],
            Hlvalue::Variable(b.clone()),
        ));
        let void_res = variable_with_lltype("v", LowLevelType::Void);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(b.clone()),
                constant_with_lltype(ConstValue::byte_str("current_pos"), LowLevelType::Void),
                constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
            ],
            Hlvalue::Variable(void_res),
        ));
        let uint_size = variable_with_lltype("uint_size", LowLevelType::Unsigned);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "cast_int_to_uint",
            vec![Hlvalue::Variable(size)],
            Hlvalue::Variable(uint_size),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(b)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);

        let mut news = Vec::new();
        let mut writes = Vec::new();
        let mut casts = Vec::new();
        let mut residual = Vec::new();
        for block in &model.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::New { owner } => news.push(owner.clone()),
                    OpKind::FieldWrite {
                        field, value, ty, ..
                    } => {
                        assert_eq!(field.owner_root.as_deref(), Some("stringbuilder"));
                        assert!(
                            matches!(value, LinkArg::Const(_)),
                            "the `0` stays an inline setfield const"
                        );
                        writes.push((field.name.clone(), ty.clone()));
                    }
                    OpKind::UnaryOp { op, .. } => casts.push(op.clone()),
                    other => residual.push(format!("{other:?}")),
                }
            }
        }
        assert_eq!(news, vec!["stringbuilder"]);
        assert_eq!(writes, vec![("current_pos".to_string(), ValueType::Int)]);
        assert_eq!(casts, vec!["same_as"]);
        assert!(residual.is_empty(), "unexpected residual ops: {residual:?}");
    }

    /// `direct_call(funcptr, size, 1280)` (the `ll_min` shape from `ll_new`:
    /// one Variable arg + one constant arg) → `Call{FunctionPath[callee]}`.
    /// The callee name is read off the leading funcptr constant's
    /// `_func._name`; the `1280` constant argument materialises through the
    /// same `ConstInt` path the `int_*` arm uses because `OpKind::Call` takes
    /// `Variable`s.  The funcptr is a hand-built `._example()` LLPtr whose
    /// `_func._name` is the marker `"<example>"` (a live rtyper carries the
    /// real helper name); this exercises the extraction + materialisation
    /// mechanism without the `None` callee `build_ll_new_helper_graph`'s test
    /// wires in.
    #[test]
    fn lower_graph_lowers_direct_call_to_a_function_path_call() {
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, Ptr, PtrTarget};

        let size = variable_with_lltype("size", LowLevelType::Unsigned);
        let startblock = Block::shared(vec![Hlvalue::Variable(size.clone())]);
        let return_var = variable_with_lltype("out", LowLevelType::Unsigned);
        let graph = FlowGraph::with_return_var(
            "ll_test_direct_call_fragment",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );

        // A funcptr constant `_func._name == "<example>"`.  The concretetype is
        // inert to the name extraction, so a Void spelling suffices (matching
        // the `dummy_funcptr_const` convention for callee consts).
        let func_type = FuncType {
            args: vec![LowLevelType::Unsigned, LowLevelType::Unsigned],
            result: LowLevelType::Unsigned,
        };
        let funcptr = Ptr {
            TO: PtrTarget::Func(func_type),
        }
        ._example();
        let callee = constant_with_lltype(ConstValue::LLPtr(Box::new(funcptr)), LowLevelType::Void);

        let out = variable_with_lltype("out", LowLevelType::Unsigned);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![
                callee,
                Hlvalue::Variable(size),
                constant_with_lltype(ConstValue::UInt(1280), LowLevelType::Unsigned),
            ],
            Hlvalue::Variable(out.clone()),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(out)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);

        let mut calls = Vec::new();
        let mut const_uints = 0;
        let mut residual = Vec::new();
        for block in &model.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::Call { target, args, .. } => calls.push((target.clone(), args.len())),
                    OpKind::ConstUInt(1280) => const_uints += 1,
                    other => residual.push(format!("{other:?}")),
                }
            }
        }
        assert_eq!(
            const_uints, 1,
            "the 1280 constant arg materialises as ConstUInt(1280)"
        );
        assert_eq!(calls.len(), 1);
        let (target, argc) = &calls[0];
        assert_eq!(
            *target,
            crate::model::CallTarget::function_path(["<example>"])
        );
        assert_eq!(*argc, 2, "size + the materialised 1280");
        assert!(residual.is_empty(), "unexpected residual ops: {residual:?}");
    }

    /// A funcptr `_name` is the `::`-joined CallPath. Splitting restores
    /// the marked `["majit_rlib", "jit", "isconstant"]` oopspec path.
    #[test]
    fn lower_graph_splits_funcptr_name_into_callpath_segments() {
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, functionptr};

        let value = variable_with_lltype("value", LowLevelType::Signed);
        let startblock = Block::shared(vec![Hlvalue::Variable(value.clone())]);
        let return_var = variable_with_lltype("out", LowLevelType::Bool);
        let mut graph = FlowGraph::with_return_var(
            "_ll_alloc_and_set_isconstant",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let ft = FuncType {
            args: vec![LowLevelType::Signed],
            result: LowLevelType::Bool,
        };
        let funcptr = functionptr(
            ft,
            "majit_rlib::jit::isconstant",
            None,
            Some("majit_rlib::jit::isconstant".to_string()),
        );
        let callee = constant_with_lltype(ConstValue::LLPtr(Box::new(funcptr)), LowLevelType::Void);
        let out = variable_with_lltype("out", LowLevelType::Bool);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![callee, Hlvalue::Variable(value)],
            Hlvalue::Variable(out.clone()),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(out)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let mut func = GraphFunc::new(
            "_ll_alloc_and_set_isconstant",
            Constant::new(ConstValue::Dict(Default::default())),
        );
        func._jit_unroll_safe_ = true;
        graph.func = Some(func);

        let model = lower_graph(&graph);
        assert!(model.func.unroll_safe);
        let calls: Vec<_> = model
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .filter_map(|op| match &op.kind {
                OpKind::Call { target, .. } => Some(target.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            calls,
            vec![crate::model::CallTarget::function_path([
                "majit_rlib",
                "jit",
                "isconstant"
            ])]
        );
    }
}
