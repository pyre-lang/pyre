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
//! `crate::model::FunctionGraph` (rich `OpKind`). Leftover `Call` from
//! `direct_call` (this transducer has no CallControl) is classified by
//! `Transformer.rewrite_op_direct_call` (`jtransform.py`) before the
//! shared flatten/regalloc/assembler tail
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
//!
//! `Transformer.rewrite_op_malloc_varsize` / `rewrite_op_getarrayitem` /
//! `rewrite_op_setarrayitem` / `rewrite_op_getarraysize` (`jtransform.py`)
//! rewrite a standalone `GcArray` (a list `items` block) to `new_array` /
//! `getarrayitem_gc_*` / `setarrayitem_gc_*` / `arraylen_gc`.  The
//! transducer distinguishes that ARRAY from a STR/UNICODE `chars` interior
//! array by the lltype on the op (`malloc_varsize`'s first-arg Constant
//! ARRAY, or the array operand's `concretetype`), not by the helper name.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::flowspace::model::Variable as FVar;
use crate::flowspace::model::{
    Block as FlowBlock, ConstValue, FunctionGraph as FlowGraph, Hlvalue,
};
use crate::model::{
    BlockId, CallTarget, ExitCase, ExitSwitch, FN_CONST_HEAD, Link, LinkArg, OpKind, ValueType,
};
use crate::translator::backendopt::constfold::WE_ARE_JITTED_TAG_ID;
use crate::translator::rtyper::lltypesystem::lltype::{Array, GcKind, LowLevelType, PtrTarget};
use crate::translator::rtyper::lltypesystem::rstr::{STR, UNICODE};
use majit_jitcode::codewriter::jtransform::{
    LIST_FLOAT_ITEMS_ARRAY, LIST_INT_ITEMS_ARRAY, LIST_OBJ_ITEMS_ARRAY,
};

/// Block identity key — flowspace blocks are `Rc<RefCell<Block>>` with no
/// inherent id, so identity is the allocation address (RPython compares
/// `Block` objects by Python identity; `iterblocks` already keys its
/// visited-set on `Rc::as_ptr`).
type FlowBlockKey = *const RefCell<FlowBlock>;

/// The blackhole opcode family for a given string-element width.  `str*`
/// handlers operate on `Array(Char)`-backed `STR`; `unicode*` /
/// `newunicode` on `Array(UniChar)`-backed `UNICODE`.
#[derive(Clone, Copy)]
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

        let we_are_jitted_switch = matches!(
            &fb_ref.exitswitch,
            Some(Hlvalue::Constant(c))
                if matches!(c.value, ConstValue::SpecTag(tag) if tag == WE_ARE_JITTED_TAG_ID)
        );

        let mut exits: Vec<Link> = fb_ref
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

        let exitswitch = if fb_ref.canraise() {
            Some(ExitSwitch::LastException)
        } else if we_are_jitted_switch {
            // `rlib/jit.py we_are_jitted` / `rbuilder.py`: `jtransform`
            // folds the SpecTag to true on the JIT path
            // (`fold_we_are_jitted_calls` then `fold_constant_exitswitch`).
            exits.retain(|link| matches!(link.exitcase, Some(ExitCase::Bool(true))));
            assert!(
                !exits.is_empty(),
                "jtransform_opname::lower_graph: we_are_jitted SpecTag has no true link"
            );
            for link in &mut exits {
                link.exitcase = None;
            }
            None
        } else {
            match &fb_ref.exitswitch {
                Some(Hlvalue::Variable(v)) => Some(ExitSwitch::Value(v.clone())),
                _ => None,
            }
        };

        drop(fb_ref);
        if exitswitch.is_some() || !exits.is_empty() {
            out.set_control_flow_metadata(id, exitswitch, exits);
        }
    }

    if let Some(func) = &graph.func {
        if func._jit_unroll_safe_ {
            out.func.unroll_safe = true;
        }
        if func._elidable_function_ {
            out.func.elidable = true;
        }
        if let Some(look) = func._jit_look_inside_ {
            out.func.jit_look_inside = Some(look);
        }
        out.func.oopspec = func.oopspec.clone();
        if func.oopspec.is_some() {
            out.func.oopspec_argnames = startblock_oopspec_argnames(graph);
        }
    }

    out
}

/// Bare startblock inputarg names for `FuncEffects.oopspec_argnames`.
/// Same source as `effect_graph_from_rtyped_helper`'s `OpKind::Input { name }`
/// (`Variable::name_prefix` when renamed); strip the intern trailing `_`
/// so the names match harvested `#[oopspec]` argnames (no numbering suffix).
fn startblock_oopspec_argnames(graph: &FlowGraph) -> Vec<String> {
    graph
        .startblock
        .borrow()
        .inputargs
        .iter()
        .filter_map(|hlv| match hlv {
            Hlvalue::Variable(v) => {
                let name = if v.renamed() {
                    v.name_prefix()
                } else {
                    "arg".to_string()
                };
                Some(name.strip_suffix('_').unwrap_or(&name).to_string())
            }
            Hlvalue::Constant(_) => None,
        })
        .collect()
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
        // `len(s.chars)` → `strlen(s)` / `unicodelen(s)`.  A standalone
        // `GcArray` is `arraylen_gc` (`rewrite_op_getarraysize`).
        "getarraysize" => {
            if let Some(array) = gc_array_of_operand(&op.args[0]) {
                let base = expect_var(&op.args[0]);
                let result = expect_var(&op.result);
                out.push_op_with_result_var(
                    block,
                    OpKind::ArrayLen {
                        base,
                        array_type_id: array_type_id_of(&array),
                        nolength: array_nolength(&array),
                    },
                    result,
                );
            } else {
                let string_var = resolve_string(chars_alias, &op.args[0]);
                let result = expect_var(&op.result);
                out.push_op_with_result_var(
                    block,
                    OpKind::LoweredBlackholeOp {
                        opname: family.len.to_string(),
                        args: vec![string_var.into()],
                    },
                    result,
                );
            }
        }
        // `s.chars[i]` → `strgetitem(s, i)` / `unicodegetitem(s, i)`.  A
        // standalone `GcArray` is `getarrayitem_gc_<kind>`
        // (`rewrite_op_getarrayitem`).  Prefix: `_array_of_voids` then
        // `FixedSizeArray` (`rewrite_op_getarrayitem`).
        "getarrayitem" => {
            if let Some(array_ty) = array_type_of_operand(&op.args[0]) {
                if array_of_voids(&array_ty) {
                    // `_array_of_voids(ARRAY)` returns [].  getsubstruct is
                    // the other "emit nothing" case, and it still records an
                    // alias; a Void-item getarrayitem result is unused.
                    // debug_assert!: the result Variable must stay unconsumed.
                    debug_assert!(
                        matches!(
                            expect_var(&op.result).concretetype().as_ref(),
                            Some(LowLevelType::Void)
                        ),
                        "getarrayitem of a Void-item ARRAY has a Void result that must not be consumed"
                    );
                } else if is_fixed_size_array(&array_ty) {
                    panic_fixed_size_array_unsupported(&out.name, &array_ty);
                } else {
                    emit_getarrayitem(out, block, op, chars_alias, family);
                }
            } else {
                emit_getarrayitem(out, block, op, chars_alias, family);
            }
        }
        // `s.chars[i] = c` → `strsetitem(s, i, c)` / `unicodesetitem(...)`.
        // Void result.  A standalone `GcArray` is `setarrayitem_gc_<kind>`
        // (`rewrite_op_setarrayitem`).  Prefix: `_array_of_voids` then
        // `FixedSizeArray` (`rewrite_op_setarrayitem`).
        "setarrayitem" => {
            if let Some(array_ty) = array_type_of_operand(&op.args[0]) {
                if array_of_voids(&array_ty) {
                    // `_array_of_voids(ARRAY)` returns [].  Void result; no
                    // op is emitted.  debug_assert!: unused afterwards.
                    debug_assert!(
                        matches!(
                            expect_var(&op.result).concretetype().as_ref(),
                            Some(LowLevelType::Void) | None
                        ),
                        "setarrayitem of a Void-item ARRAY has a Void result that must not be consumed"
                    );
                } else if is_fixed_size_array(&array_ty) {
                    panic_fixed_size_array_unsupported(&out.name, &array_ty);
                } else {
                    emit_setarrayitem(out, block, op, chars_alias, family);
                }
            } else {
                emit_setarrayitem(out, block, op, chars_alias, family);
            }
        }
        // `rewrite_op_malloc_varsize`: flavor `raw` first (these helpers are
        // gc), then `op.args[0].value == rstr.STR` → `newstr`,
        // `== rstr.UNICODE` → `newunicode`, else ARRAY → `new_array` /
        // `new_array_clear`.
        "malloc_varsize" => {
            let malloc_ty = malloc_type_lltype(&op.args[0]);
            if let Some(alloc_family) = str_family_from_lltype(malloc_ty) {
                let size = materialize(out, block, &op.args[2]);
                let result = expect_var(&op.result);
                out.push_op_with_result_var(
                    block,
                    OpKind::LoweredBlackholeOp {
                        opname: alloc_family.alloc.to_string(),
                        args: vec![size.into()],
                    },
                    result,
                );
            } else if let Some(array) = gc_array_from_lltype(malloc_ty) {
                let size = materialize(out, block, &op.args[2]);
                let result = expect_var(&op.result);
                let item_ty = array_item_value_type(&array.OF);
                let array_type_id = array_type_id_of(&array);
                let kind = if array_malloc_clears(&array, &op.args[1]) {
                    OpKind::NewArrayClear {
                        length: size,
                        item_ty,
                        array_type_id,
                    }
                } else {
                    OpKind::NewArray {
                        length: size,
                        item_ty,
                        array_type_id,
                    }
                };
                out.push_op_with_result_var(block, kind, result);
            } else {
                panic!(
                    "jtransform_opname::lower_graph: malloc_varsize type is not STR, UNICODE, or GcArray: {malloc_ty:?}"
                );
            }
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
                    args: vec![
                        src.into(),
                        dst.into(),
                        srcstart.into(),
                        dststart.into(),
                        length.into(),
                    ],
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
            let args =
                crate::model::call_args(op.args.iter().map(|arg| materialize(out, block, arg)));
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
        // `jtransform.py Transformer.rewrite_op_jit_record_exact_value`:
        // int → `record_exact_value_i`, ref → `record_exact_value_r`.
        // Void result, same `push_op_var(..., false)` shape as `keepalive`
        // / `copystrcontent`. Operands pass through unchanged upstream
        // (`LinkArg::Const` keeps the original ConstValue for
        // `Assembler.emit_const`).
        "jit_record_exact_value" => {
            let kind0 = hlvalue_getkind(&op.args[0]);
            let kind1 = hlvalue_getkind(&op.args[1]);
            let opname = if kind0 == crate::model::ConcreteType::Signed {
                assert_eq!(
                    kind1,
                    crate::model::ConcreteType::Signed,
                    "record_exact_value int pair"
                );
                "record_exact_value_i"
            } else {
                assert_eq!(
                    kind0,
                    crate::model::ConcreteType::GcRef,
                    "record_exact_value ref pair"
                );
                assert_eq!(
                    kind1,
                    crate::model::ConcreteType::GcRef,
                    "record_exact_value ref pair"
                );
                "record_exact_value_r"
            };
            out.push_op_var(
                block,
                OpKind::LoweredBlackholeOp {
                    opname: opname.to_string(),
                    args: vec![
                        record_exact_value_linkarg(&op.args[0]),
                        record_exact_value_linkarg(&op.args[1]),
                    ],
                },
                false,
            );
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
        // `jtransform.py Transformer.rewrite_op_ptr_nonzero` /
        // `rewrite_op_ptr_iszero` alias `_rewrite_cmp_ptrs`.  A GC pointer
        // stays the unary `ptr_nonzero` / `ptr_iszero` (blackhole
        // `bhimpl_ptr_nonzero`); a raw pointer becomes `int_is_true` /
        // `int_is_zero`.  The GC form is the same `UnaryOp` the `ptr_ne` /
        // `ptr_eq` NULL arm emits, so `optimize_goto_if_not` folds the
        // exitswitch into `goto_if_not_ptr_nonzero` /
        // `goto_if_not_ptr_iszero`.
        "ptr_nonzero" | "ptr_iszero" => {
            let operand = expect_var(&op.args[0]);
            let result = expect_var(&op.result);
            let result_ty = value_type_of(&result);
            let unop = if lltype_is_gc_ptr(&operand) {
                op.opname.as_str()
            } else if op.opname == "ptr_nonzero" {
                "int_is_true"
            } else {
                "int_is_zero"
            };
            out.push_op_with_result_var(
                block,
                OpKind::UnaryOp {
                    op: unop.to_string(),
                    operand,
                    result_ty,
                },
                result,
            );
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
            // `ll_get_raw_address_of_string` is a FunctionPath Call. This
            // transducer has no CallControl; `rewrite_op_direct_call`
            // mints the `dont_look_inside` residual before `guess_call_kind`
            // (`codewriter::get_raw_address`).
            let args: Vec<FVar> = non_void_args(&op.args[1..])
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
        // `rlib/jit.py conditional_call` inlines to `jit_conditional_call`
        // on the `we_are_jitted()` true arm. `jtransform.py`
        // `Transformer.rewrite_op_jit_conditional_call` /
        // `_rewrite_op_cond_call` rewrites the interpreter-facing
        // `majit_rlib::jit::conditional_callN` Call (`oopspec
        // jit.conditional_call`) to `OpKind::ConditionalCall`.
        "jit_conditional_call" => {
            assert!(
                op.args.len() >= 2,
                "jtransform_opname::lower_graph: jit_conditional_call needs cond and fnptr"
            );
            let cond = materialize(out, block, &op.args[0]);
            let callee = callee_name_from_funcptr(&op.args[1]);
            let extra_args: Vec<FVar> = non_void_args(&op.args[2..])
                .map(|a| materialize(out, block, a))
                .collect();
            let result = expect_var(&op.result);
            emit_conditional_call_oopspec_named(
                out,
                block,
                cond,
                &callee,
                extra_args,
                result,
                "conditional_call",
            );
        }
        // `rlib/jit.py conditional_call_elidable` inlines to
        // `jit_conditional_call_value` on the `we_are_jitted()` true arm.
        // `jtransform.py Transformer.rewrite_op_jit_conditional_call_value`
        // / `_rewrite_op_cond_call` rewrites the interpreter-facing
        // `majit_rlib::jit::conditional_call_elidableN` Call (`oopspec
        // jit.conditional_call_value`) to `OpKind::ConditionalCallValue`.
        "jit_conditional_call_value" => {
            assert!(
                op.args.len() >= 2,
                "jtransform_opname::lower_graph: jit_conditional_call_value needs value and fnptr"
            );
            let value = materialize(out, block, &op.args[0]);
            let callee = callee_name_from_funcptr(&op.args[1]);
            let extra_args: Vec<FVar> = non_void_args(&op.args[2..])
                .map(|a| materialize(out, block, a))
                .collect();
            let result = expect_var(&op.result);
            emit_conditional_call_oopspec_named(
                out,
                block,
                value,
                &callee,
                extra_args,
                result,
                "conditional_call_elidable",
            );
        }
        // `jtransform.py Transformer.rewrite_op_debug_assert` returns [].
        "debug_assert" => {}
        other => panic!("jtransform_opname::lower_graph: unsupported opname {other:?}"),
    }
}

/// Interpreter-facing `oopspec("jit.conditional_call")` Call.
///
/// `_rewrite_op_cond_call` reads args[0] as the condition, args[1] as
/// the function (`fn_const_target_of` of a 0-arg `__fn_const` Call),
/// and args[2:] as the callee arguments. The helper leaf is
/// `conditional_callN` (`is_jit_conditional_call_path`).
pub(crate) fn emit_conditional_call_oopspec(
    out: &mut crate::model::FunctionGraph,
    block: BlockId,
    cond: FVar,
    callee_name: &str,
    extra_args: Vec<FVar>,
    result: FVar,
) {
    emit_conditional_call_oopspec_named(
        out,
        block,
        cond,
        callee_name,
        extra_args,
        result,
        "conditional_call",
    )
}

/// Interpreter-facing `oopspec("jit.conditional_call_value")` Call.
///
/// `_rewrite_op_cond_call` reads args[0] as the value, args[1] as
/// the function (`fn_const_target_of` of a 0-arg `__fn_const` Call),
/// and args[2:] as the callee arguments. The helper leaf is
/// `conditional_call_elidableN`.
pub(crate) fn emit_conditional_call_value_oopspec(
    out: &mut crate::model::FunctionGraph,
    block: BlockId,
    value: FVar,
    callee_name: &str,
    extra_args: Vec<FVar>,
    result: FVar,
) {
    emit_conditional_call_oopspec_named(
        out,
        block,
        value,
        callee_name,
        extra_args,
        result,
        "conditional_call_elidable",
    )
}

/// Interpreter-facing `oopspec("jit.conditional_call")` /
/// `oopspec("jit.conditional_call_value")` Call.
///
/// `_rewrite_op_cond_call` reads args[0] as the condition/value, args[1] as
/// the function (`fn_const_target_of` of a 0-arg `__fn_const` Call),
/// and args[2:] as the callee arguments. The helper leaf is
/// `{helper_prefix}{n}`: `conditional_callN` (`is_jit_conditional_call_path`)
/// or `conditional_call_elidableN`. `Transformer._rewrite_op_cond_call`
/// rejects floats and more than 4 function arguments.
fn emit_conditional_call_oopspec_named(
    out: &mut crate::model::FunctionGraph,
    block: BlockId,
    cond: FVar,
    callee_name: &str,
    extra_args: Vec<FVar>,
    result: FVar,
    helper_prefix: &str,
) {
    let n = extra_args.len();
    let has_float = std::iter::once(&cond).chain(extra_args.iter()).any(|v| {
        v.concretetype()
            .is_some_and(|ty| crate::model::getkind(&ty) == crate::model::ConcreteType::Float)
    });
    if n > 4 || has_float {
        panic!("Conditional call does not support floats or more than 4 arguments");
    }
    let fn_target = CallTarget::function_path(
        std::iter::once(FN_CONST_HEAD).chain(callee_name.split("::").filter(|s| !s.is_empty())),
    );
    let fn_var = out
        .push_op_var(
            block,
            OpKind::Call {
                target: fn_target,
                args: crate::model::call_args(Vec::new()),
                result_ty: ValueType::Int,
            },
            true,
        )
        .expect("__fn_const produces a result var");
    fn_var.set_concretetype(Some(LowLevelType::Signed));
    let helper = format!("{helper_prefix}{n}");
    let target = CallTarget::function_path(["majit_rlib", "jit", helper.as_str()]);
    let mut args = Vec::with_capacity(2 + extra_args.len());
    args.push(cond);
    args.push(fn_var);
    args.extend(extra_args);
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

/// The ARRAY lltype constant `malloc_varsize` takes as `op.args[0]`
/// (`rewrite_op_malloc_varsize` `op.args[0].value`).
fn malloc_type_lltype(hlv: &Hlvalue) -> &LowLevelType {
    match hlv {
        Hlvalue::Constant(c) => match &c.value {
            ConstValue::LowLevelType(lltype) => lltype.as_ref(),
            other => panic!(
                "jtransform_opname::lower_graph: malloc_varsize type operand is not a LowLevelType const: {other:?}"
            ),
        },
        Hlvalue::Variable(_) => {
            panic!("jtransform_opname::lower_graph: malloc_varsize type operand is a Variable")
        }
    }
}

/// `GcArray` carried by `v.concretetype` / `concretetype.TO`
/// (`rewrite_op_getarrayitem` `ARRAY = op.args[0].concretetype.TO`).
fn gc_array_of_operand(hlv: &Hlvalue) -> Option<Array> {
    let var = hlvalue_as_var(hlv)?;
    gc_array_from_lltype(&var.concretetype()?)
}

fn gc_array_from_lltype(ty: &LowLevelType) -> Option<Array> {
    match ty {
        LowLevelType::ForwardReference(fwd) => gc_array_from_lltype(&fwd.resolved()?),
        LowLevelType::Array(arr) if arr._gckind == GcKind::Gc => Some(arr.as_ref().clone()),
        LowLevelType::Ptr(ptr) => match &ptr.TO {
            PtrTarget::Array(arr) if arr._gckind == GcKind::Gc => Some(arr.clone()),
            PtrTarget::ForwardReference(fwd) => gc_array_from_lltype(&fwd.resolved()?),
            _ => None,
        },
        LowLevelType::InteriorPtr(ip) => gc_array_from_lltype(ip.TO.as_ref()),
        _ => None,
    }
}

/// `ARRAY = op.args[0].concretetype.TO` (`rewrite_op_getarrayitem` /
/// `rewrite_op_setarrayitem`).  `malloc_type_lltype` produces the TYPE
/// constant itself, not a Ptr; this helper is the operand-side Ptr.TO.
fn array_type_of_operand(hlv: &Hlvalue) -> Option<LowLevelType> {
    let var = hlvalue_as_var(hlv)?;
    ptr_to(&var.concretetype()?)
}

fn ptr_to(ty: &LowLevelType) -> Option<LowLevelType> {
    match ty {
        LowLevelType::ForwardReference(fwd) => ptr_to(&fwd.resolved()?),
        LowLevelType::Ptr(ptr) => Some(LowLevelType::from(ptr.TO.clone())),
        LowLevelType::InteriorPtr(ip) => Some((*ip.TO).clone()),
        _ => None,
    }
}

/// `Transformer._array_of_voids(ARRAY)`: `ARRAY.OF == lltype.Void`.
fn array_of_voids(array: &LowLevelType) -> bool {
    match array {
        LowLevelType::ForwardReference(fwd) => fwd.resolved().as_ref().is_some_and(array_of_voids),
        LowLevelType::Array(arr) => matches!(arr.OF, LowLevelType::Void),
        LowLevelType::FixedSizeArray(arr) => matches!(arr.OF, LowLevelType::Void),
        _ => false,
    }
}

/// `isinstance(ARRAY, lltype.FixedSizeArray)` (`rewrite_op_getarrayitem`).
fn is_fixed_size_array(array: &LowLevelType) -> bool {
    match array {
        LowLevelType::ForwardReference(fwd) => {
            fwd.resolved().as_ref().is_some_and(is_fixed_size_array)
        }
        LowLevelType::FixedSizeArray(_) => true,
        _ => false,
    }
}

fn panic_fixed_size_array_unsupported(graph: &str, array: &LowLevelType) -> ! {
    panic!(
        "NotImplementedError: {graph:?} uses {array:?}, which is not supported by the JIT codewriter"
    )
}

/// STR vs UNICODE from the malloc type: lltype equality with
/// `rstr.STR` / `rstr.UNICODE` (`rewrite_op_malloc_varsize`
/// `op.args[0].value == rstr.STR`).  The producer (`struct_lltype_from_strptr`)
/// passes the TYPE constant itself, not a Ptr.
fn str_family_from_lltype(ty: &LowLevelType) -> Option<StrFamily> {
    if *ty == *STR {
        Some(StrFamily::STR)
    } else if *ty == *UNICODE {
        Some(StrFamily::UNICODE)
    } else {
        None
    }
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn emit_getarrayitem(
    out: &mut crate::model::FunctionGraph,
    block: BlockId,
    op: &crate::flowspace::model::SpaceOperation,
    chars_alias: &HashMap<FVar, FVar>,
    family: &StrFamily,
) {
    if let Some(array) = gc_array_of_operand(&op.args[0]) {
        let base = expect_var(&op.args[0]);
        let index = materialize(out, block, &op.args[1]);
        let result = expect_var(&op.result);
        out.push_op_with_result_var(
            block,
            OpKind::ArrayRead {
                base,
                index,
                item_ty: value_type_of(&result),
                array_type_id: array_type_id_of(&array),
                nolength: array_nolength(&array),
                pure: array._immutable_field(),
            },
            result,
        );
    } else {
        let string_var = resolve_string(chars_alias, &op.args[0]);
        let index = materialize(out, block, &op.args[1]);
        let result = expect_var(&op.result);
        out.push_op_with_result_var(
            block,
            OpKind::LoweredBlackholeOp {
                opname: family.getitem.to_string(),
                args: vec![string_var.into(), index.into()],
            },
            result,
        );
    }
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn emit_setarrayitem(
    out: &mut crate::model::FunctionGraph,
    block: BlockId,
    op: &crate::flowspace::model::SpaceOperation,
    chars_alias: &HashMap<FVar, FVar>,
    family: &StrFamily,
) {
    if let Some(array) = gc_array_of_operand(&op.args[0]) {
        let base = expect_var(&op.args[0]);
        let index = materialize(out, block, &op.args[1]);
        let value = linkarg_from_hlvalue(&op.args[2]);
        out.push_op_var(
            block,
            OpKind::ArrayWrite {
                base,
                index,
                value,
                item_ty: hlvalue_value_type(&op.args[2]),
                array_type_id: array_type_id_of(&array),
                nolength: array_nolength(&array),
            },
            false,
        );
    } else {
        let string_var = resolve_string(chars_alias, &op.args[0]);
        let index = materialize(out, block, &op.args[1]);
        let value = materialize(out, block, &op.args[2]);
        out.push_op_var(
            block,
            OpKind::LoweredBlackholeOp {
                opname: family.setitem.to_string(),
                args: vec![string_var.into(), index.into(), value.into()],
            },
            false,
        );
    }
}

/// `cpu.arraydescrof(ARRAY)` identity.  `GcArray(Signed)` / `GcArray(Float)`
/// share the list-items ARRAY the main Transformer already names
/// (`LIST_INT_ITEMS_ARRAY` / `LIST_FLOAT_ITEMS_ARRAY`); a GC-pointer OF
/// is `LIST_OBJ_ITEMS_ARRAY`.
fn array_type_id_of(arr: &Array) -> Option<String> {
    match &arr.OF {
        LowLevelType::Signed | LowLevelType::Unsigned => Some(LIST_INT_ITEMS_ARRAY.to_string()),
        LowLevelType::Float => Some(LIST_FLOAT_ITEMS_ARRAY.to_string()),
        LowLevelType::Ptr(p) if p._needsgc() => Some(LIST_OBJ_ITEMS_ARRAY.to_string()),
        LowLevelType::Struct(s) => Some(format!("GcArray({})", s._name)),
        other => Some(format!("GcArray({})", other.short_name())),
    }
}

fn array_nolength(arr: &Array) -> bool {
    matches!(arr._hints.get("nolength"), Some(ConstValue::Bool(true)))
}

fn array_item_value_type(of: &LowLevelType) -> ValueType {
    match of {
        LowLevelType::Void => ValueType::Void,
        LowLevelType::Struct(_) => ValueType::Ref(None),
        other => value_type_from_lltype(other),
    }
}

/// `rewrite_op_malloc_varsize`: `new_array_clear` when `zero` is set, or
/// when `ARRAY.OF` is a GC pointer or a Struct; otherwise `new_array`.
fn array_malloc_clears(arr: &Array, flags: &Hlvalue) -> bool {
    if malloc_flags_zero(flags) {
        return true;
    }
    match &arr.OF {
        LowLevelType::Ptr(p) if p._needsgc() => true,
        LowLevelType::Struct(_) => true,
        _ => false,
    }
}

fn malloc_flags_zero(hlv: &Hlvalue) -> bool {
    let Hlvalue::Constant(c) = hlv else {
        return false;
    };
    let Some(items) = c.value.dict_items() else {
        return false;
    };
    matches!(
        items.get(&ConstValue::byte_str("zero")),
        Some(ConstValue::Bool(true))
    )
}

/// `jit_record_exact_value` operands pass through as [`LinkArg`], the
/// same Variable-or-Constant carrier `FieldWrite.value` uses, so a
/// Constant reaches `Assembler.emit_const` with its original
/// `ConstValue`.
fn record_exact_value_linkarg(hlv: &Hlvalue) -> LinkArg {
    match hlv {
        Hlvalue::Variable(v) => LinkArg::Value(v.clone()),
        Hlvalue::Constant(c) => LinkArg::Const(c.clone()),
    }
}

/// Runtime call operands, dropping Void. Void is compile-time only.
///
/// `jtransform.py Transformer.make_three_lists` skips `kind == 'void'`
/// (`add_in_correct_list`); `call.py CallControl.getcalldescr` builds
/// `NON_VOID_ARGS` the same way. `_rewrite_op_cond_call` routes extra
/// args through `rewrite_call` → `make_three_lists`.
fn non_void_args(args: &[Hlvalue]) -> impl Iterator<Item = &Hlvalue> {
    args.iter().filter(|hlv| match hlv {
        Hlvalue::Variable(v) => !matches!(v.concretetype(), Some(LowLevelType::Void)),
        Hlvalue::Constant(c) => !matches!(c.concretetype, Some(LowLevelType::Void)),
    })
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

/// `rgc.lltype_is_gc` / `jtransform.py Transformer._is_gc`:
/// `v.concretetype.TO._gckind == 'gc'`.
fn lltype_is_gc_ptr(var: &FVar) -> bool {
    match var.concretetype() {
        Some(LowLevelType::Ptr(ptr)) => ptr.TO._gckind() == GcKind::Gc,
        Some(LowLevelType::InteriorPtr(ip)) => ip.TO._gckind() == GcKind::Gc,
        _ => false,
    }
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

/// `history.py getkind` of a flowspace operand's `concretetype`.
fn hlvalue_getkind(hlv: &Hlvalue) -> crate::model::ConcreteType {
    let ty = match hlv {
        Hlvalue::Variable(v) => v
            .concretetype()
            .expect("jit_record_exact_value operand has concretetype"),
        Hlvalue::Constant(c) => c
            .concretetype
            .clone()
            .expect("jit_record_exact_value constant operand has concretetype"),
    };
    crate::model::getkind(&ty)
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
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph as FlowGraph, GraphFunc, Hlvalue,
        Link, SpaceOperation, c_last_exception,
    };
    use crate::model::{ExitCase, ExitSwitch, LinkArg, OpKind, ValueType};
    use crate::translator::rtyper::lltypesystem::lltype::{
        Array, LowLevelType, Ptr, PtrTarget, Struct,
    };
    use crate::translator::rtyper::lltypesystem::rbuilder::{
        STRINGBUILDERPTR, build_ll_bool_helper_graph, build_ll_getlength_helper_graph,
    };
    use crate::translator::rtyper::lltypesystem::rstr::{
        STRPTR, build_ll_search_helper_graph, chars_array_ptr_lltype_from_strptr,
        direct_call_funcobjs, graph_ref_for_funcptr_id, struct_lltype_from_strptr,
    };
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};
    use crate::translator::rtyper::rtyper::{
        RPythonTyper, constant_with_lltype, variable_with_lltype,
    };
    use majit_jitcode::codewriter::jtransform::LIST_INT_ITEMS_ARRAY;

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

    /// `jtransform.py rewrite_op_jit_record_exact_value`:
    /// `jit_record_exact_value(int_lt_result, Constant(True))` →
    /// `record_exact_value_i`.
    #[test]
    fn lower_graph_lowers_jit_record_exact_value_int_to_record_exact_value_i() {
        let cond = variable_with_lltype("cond", LowLevelType::Bool);
        let start = Block::shared(vec![Hlvalue::Variable(cond.clone())]);
        let graph = FlowGraph::with_return_var(
            "ll_record_exact_value_int",
            start.clone(),
            Hlvalue::Variable(cond.clone()),
        );
        let void_res = variable_with_lltype("v", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "jit_record_exact_value",
            vec![Hlvalue::Variable(cond.clone()), bool_const(true)],
            Hlvalue::Variable(void_res),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(cond.clone())],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);
        let ops = &model.block(model.startblock).operations;
        assert!(
            ops.iter()
                .all(|op| !matches!(op.kind, OpKind::ConstBool(_))),
            "Constant stays an operand, no ConstBool define: {ops:?}"
        );
        let rec = ops
            .iter()
            .find(|op| {
                matches!(
                    &op.kind,
                    OpKind::LoweredBlackholeOp { opname, .. } if opname == "record_exact_value_i"
                )
            })
            .unwrap_or_else(|| panic!("expected record_exact_value_i, got {ops:?}"));
        assert!(rec.result.is_none(), "record_exact_value_i is void");
        let OpKind::LoweredBlackholeOp { args, .. } = &rec.kind else {
            unreachable!();
        };
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], cond);
        assert!(
            matches!(
                &args[1],
                LinkArg::Const(c) if matches!(c.value, ConstValue::Bool(true))
            ),
            "const operand is LinkArg::Const: {args:?}"
        );
    }

    fn objectptr_const(null: bool) -> Hlvalue {
        use crate::translator::rtyper::lltypesystem::lltype::{MallocFlavor, malloc, nullptr};
        use crate::translator::rtyper::rclass::{OBJECT, OBJECTPTR};
        let ptr = if null {
            nullptr(OBJECT.clone()).expect("nullptr(OBJECT)")
        } else {
            malloc(OBJECT.clone(), None, MallocFlavor::Gc, true).expect("malloc OBJECT")
        };
        constant_with_lltype(ConstValue::LLPtr(Box::new(ptr)), OBJECTPTR.clone())
    }

    fn lower_graph_lowers_jit_record_exact_value_ref_const(const_arg: Hlvalue, expect_null: bool) {
        use crate::translator::rtyper::rclass::OBJECTPTR;
        let value = variable_with_lltype("p", OBJECTPTR.clone());
        let start = Block::shared(vec![Hlvalue::Variable(value.clone())]);
        let graph = FlowGraph::with_return_var(
            "ll_record_exact_value_ref",
            start.clone(),
            Hlvalue::Variable(value.clone()),
        );
        let void_res = variable_with_lltype("v", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "jit_record_exact_value",
            vec![Hlvalue::Variable(value.clone()), const_arg],
            Hlvalue::Variable(void_res),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(value.clone())],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);
        let ops = &model.block(model.startblock).operations;
        assert!(
            ops.iter().all(|op| !matches!(
                op.kind,
                OpKind::ConstRefNull | OpKind::ConstStr(_) | OpKind::ConstRefAddr(_)
            )),
            "ref Constant stays an operand, no Const* define: {ops:?}"
        );
        let rec = ops
            .iter()
            .find(|op| {
                matches!(
                    &op.kind,
                    OpKind::LoweredBlackholeOp { opname, .. } if opname == "record_exact_value_r"
                )
            })
            .unwrap_or_else(|| panic!("expected record_exact_value_r, got {ops:?}"));
        assert!(rec.result.is_none(), "record_exact_value_r is void");
        let OpKind::LoweredBlackholeOp { args, .. } = &rec.kind else {
            unreachable!();
        };
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], value);
        let LinkArg::Const(c) = &args[1] else {
            panic!("const operand is LinkArg::Const: {args:?}");
        };
        if expect_null {
            assert!(
                matches!(&c.value, ConstValue::LLPtr(p) if !p.nonzero())
                    || matches!(c.value, ConstValue::None),
                "null const operand: {c:?}"
            );
        } else {
            assert!(
                matches!(&c.value, ConstValue::LLPtr(_)),
                "prebuilt Ptr const operand: {c:?}"
            );
        }
    }

    #[test]
    fn lower_graph_lowers_jit_record_exact_value_null_ref_to_record_exact_value_r() {
        use crate::translator::rtyper::rclass::OBJECTPTR;
        lower_graph_lowers_jit_record_exact_value_ref_const(
            constant_with_lltype(ConstValue::None, OBJECTPTR.clone()),
            true,
        );
        lower_graph_lowers_jit_record_exact_value_ref_const(objectptr_const(true), true);
    }

    fn strptr_const() -> Hlvalue {
        use crate::translator::rtyper::lltypesystem::rstr::{STRPTR, const_str_cache_llstr};
        constant_with_lltype(
            ConstValue::LLPtr(Box::new(
                const_str_cache_llstr(b"hello").expect("cache llstr"),
            )),
            STRPTR.clone(),
        )
    }

    #[test]
    fn lower_graph_lowers_jit_record_exact_value_constptr_to_record_exact_value_r() {
        lower_graph_lowers_jit_record_exact_value_ref_const(strptr_const(), false);
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

    fn build_ptr_null_test_graph(opname: &str, ptr_lltype: LowLevelType) -> FlowGraph {
        let p = variable_with_lltype("p", ptr_lltype);
        let startblock = Block::shared(vec![Hlvalue::Variable(p.clone())]);
        let return_var = variable_with_lltype("result", LowLevelType::Bool);
        let graph = FlowGraph::with_return_var(
            format!("ll_test_{opname}"),
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let nz = variable_with_lltype("nz", LowLevelType::Bool);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            vec![Hlvalue::Variable(p)],
            Hlvalue::Variable(nz.clone()),
        ));
        startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz));
        startblock.closeblock(vec![
            Link::new(
                vec![bool_const(true)],
                Some(graph.returnblock.clone()),
                Some(bool_const(true)),
            )
            .into_ref(),
            Link::new(
                vec![bool_const(false)],
                Some(graph.returnblock.clone()),
                Some(bool_const(false)),
            )
            .into_ref(),
        ]);
        graph
    }

    fn startblock_unops(model: &crate::model::FunctionGraph) -> Vec<String> {
        model
            .block(model.startblock)
            .operations
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::UnaryOp { op, .. } => Some(op.clone()),
                _ => None,
            })
            .collect()
    }

    /// Rtyped `ptr_nonzero(p: Ptr(STR))` / `ptr_iszero` stay the unary
    /// null test (`jtransform.py Transformer._rewrite_cmp_ptrs` GC
    /// branch); the bool exitswitch is the same shape `optimize_goto_if_not`
    /// folds for `ll_bool`.
    #[test]
    fn lower_graph_lowers_ptr_nonzero_and_ptr_iszero_on_str() {
        let nonzero = lower_graph(&build_ptr_null_test_graph("ptr_nonzero", STRPTR.clone()));
        assert_eq!(startblock_unops(&nonzero), vec!["ptr_nonzero"]);
        let start = nonzero.block(nonzero.startblock);
        assert!(matches!(start.exitswitch, Some(ExitSwitch::Value(_))));
        assert_eq!(start.exits.len(), 2);
        let cases: Vec<&Option<ExitCase>> = start.exits.iter().map(|l| &l.exitcase).collect();
        assert!(cases.contains(&&Some(ExitCase::Bool(true))));
        assert!(cases.contains(&&Some(ExitCase::Bool(false))));

        let iszero = lower_graph(&build_ptr_null_test_graph("ptr_iszero", STRPTR.clone()));
        assert_eq!(startblock_unops(&iszero), vec!["ptr_iszero"]);
        let start = iszero.block(iszero.startblock);
        assert!(matches!(start.exitswitch, Some(ExitSwitch::Value(_))));
        assert_eq!(start.exits.len(), 2);
    }

    /// `_rewrite_cmp_ptrs` maps a raw (non-GC) pointer's `ptr_nonzero` /
    /// `ptr_iszero` to `int_is_true` / `int_is_zero`.
    #[test]
    fn lower_graph_rewrites_raw_ptr_nonzero_to_int_is_true() {
        let raw = LowLevelType::Ptr(Box::new(
            crate::translator::rtyper::lltypesystem::lltype::NONGCREF.clone(),
        ));
        let nonzero = lower_graph(&build_ptr_null_test_graph("ptr_nonzero", raw.clone()));
        assert_eq!(startblock_unops(&nonzero), vec!["int_is_true"]);
        let iszero = lower_graph(&build_ptr_null_test_graph("ptr_iszero", raw));
        assert_eq!(startblock_unops(&iszero), vec!["int_is_zero"]);
    }

    /// `LLHelpers.ll_search` `assert value is not None`: `ptr_nonzero`
    /// then the False arm raises AssertionError on the graph exceptblock
    /// (`exception_args`).  `lower_graph` keeps that raise link.
    #[test]
    fn lower_graph_keeps_ptr_nonzero_assertionerror_except_link() {
        use crate::translator::rtyper::rtyper::exception_args;

        let p = variable_with_lltype("p", STRPTR.clone());
        let startblock = Block::shared(vec![Hlvalue::Variable(p.clone())]);
        let return_var = variable_with_lltype("result", STRPTR.clone());
        let graph = FlowGraph::with_return_var(
            "ll_test_assert_ptr_not_none",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let nz = variable_with_lltype("nz", LowLevelType::Bool);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "ptr_nonzero",
            vec![Hlvalue::Variable(p.clone())],
            Hlvalue::Variable(nz.clone()),
        ));
        startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(nz));
        let exc_args = exception_args("AssertionError").expect("AssertionError args");
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(p)],
                Some(graph.returnblock.clone()),
                Some(bool_const(true)),
            )
            .into_ref(),
            Link::new(
                exc_args,
                Some(graph.exceptblock.clone()),
                Some(bool_const(false)),
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);
        assert_eq!(startblock_unops(&model), vec!["ptr_nonzero"]);
        let start = model.block(model.startblock);
        let false_link = start
            .exits
            .iter()
            .find(|l| l.exitcase == Some(ExitCase::Bool(false)))
            .expect("false exit");
        assert_eq!(false_link.target, model.exceptblock);
        assert_eq!(false_link.args.len(), 2);
        match &false_link.args[0] {
            LinkArg::Const(c) => match &c.value {
                ConstValue::HostObject(h) => {
                    assert_eq!(h.qualname(), "AssertionError");
                }
                other => panic!("except class is not a HostObject: {other:?}"),
            },
            other => panic!("except class is not a Const: {other:?}"),
        }
    }

    /// `build_ll_search_helper_graph` start / check-other blocks are
    /// `ptr_nonzero` with False → exceptblock AssertionError.  Both raise
    /// links survive `lower_graph`.
    #[test]
    fn lower_graph_lowers_ll_search_assertionerror_raise() {
        let helper = build_ll_search_helper_graph("ll_search", STRPTR.clone())
            .expect("build_ll_search_helper_graph");
        let flow = helper.graph.borrow();
        let model = lower_graph(&flow);

        let unops: Vec<String> = model
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::UnaryOp { op, .. } => Some(op.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(unops, vec!["ptr_nonzero", "ptr_nonzero"]);

        let assertionerror_exits = model
            .blocks
            .iter()
            .flat_map(|block| &block.exits)
            .filter(|link| {
                link.target == model.exceptblock
                    && matches!(
                        link.args.first(),
                        Some(LinkArg::Const(c))
                            if matches!(
                                &c.value,
                                ConstValue::HostObject(h) if h.qualname() == "AssertionError"
                            )
                    )
            })
            .count();
        assert_eq!(assertionerror_exits, 2);
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

    /// Void-typed `direct_call` operands are compile-time only
    /// (`jtransform.py Transformer.make_three_lists`,
    /// `call.py CallControl.getcalldescr`) and must not be materialised.
    /// `lowlevel_type_const` is the `T` argument `ll_dict_lookup` receives.
    #[test]
    fn lower_graph_skips_void_typed_direct_call_constant_args() {
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, Ptr, PtrTarget};

        let size = variable_with_lltype("size", LowLevelType::Unsigned);
        let startblock = Block::shared(vec![Hlvalue::Variable(size.clone())]);
        let return_var = variable_with_lltype("out", LowLevelType::Unsigned);
        let graph = FlowGraph::with_return_var(
            "ll_test_direct_call_void_t",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );

        let func_type = FuncType {
            args: vec![
                LowLevelType::Unsigned,
                LowLevelType::Unsigned,
                LowLevelType::Void,
            ],
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
                lowlevel_type_const(LowLevelType::Unsigned),
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
        assert_eq!(*argc, 2, "size + the materialised 1280; Void T is dropped");
        assert!(residual.is_empty(), "unexpected residual ops: {residual:?}");
    }

    /// `build_ll_call_lookup_function_helper_graph` calls `ll_dict_lookup`
    /// with a Void `T` (`lowlevel_type_const(TYPE_LONG)`). Lowering drops
    /// that operand rather than materialising it.
    #[test]
    fn lower_graph_lowers_ll_call_lookup_function_dropping_void_t() {
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, functionptr};
        use crate::translator::rtyper::lltypesystem::rordereddict::build_ll_call_lookup_function_helper_graph;

        let dict_ptr = STRPTR.clone();
        let key_lltype = LowLevelType::Signed;
        let ensure_ft = FuncType {
            args: vec![dict_ptr.clone()],
            result: LowLevelType::Void,
        };
        let ensure_fn = functionptr(
            ensure_ft,
            "ll_ensure_indexes",
            None,
            Some("ll_ensure_indexes".to_string()),
        );
        let lookup_ft = FuncType {
            args: vec![
                dict_ptr.clone(),
                key_lltype.clone(),
                LowLevelType::Signed,
                LowLevelType::Signed,
                LowLevelType::Void,
            ],
            result: LowLevelType::Signed,
        };
        let lookup_fn = functionptr(
            lookup_ft,
            "ll_dict_lookup",
            None,
            Some("ll_dict_lookup".to_string()),
        );
        let helper = build_ll_call_lookup_function_helper_graph(
            "ll_call_lookup_function",
            dict_ptr,
            key_lltype,
            Constant::with_concretetype(ConstValue::LLPtr(Box::new(ensure_fn)), LowLevelType::Void),
            Constant::with_concretetype(ConstValue::LLPtr(Box::new(lookup_fn)), LowLevelType::Void),
        )
        .expect("build_ll_call_lookup_function_helper_graph");

        let model = lower_graph(&helper.graph.borrow());
        let mut calls = Vec::new();
        for block in &model.blocks {
            for op in &block.operations {
                if let OpKind::Call { target, args, .. } = &op.kind {
                    calls.push((target.clone(), args.len()));
                }
            }
        }
        assert_eq!(
            calls,
            vec![
                (
                    crate::model::CallTarget::function_path(["ll_ensure_indexes"]),
                    1
                ),
                (
                    crate::model::CallTarget::function_path(["ll_dict_lookup"]),
                    4
                ),
            ],
            "lookup keeps d, key, hash, flag and drops Void T"
        );
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

    /// `rlib/jit.py elidable` stamps `func._elidable_function_`; Spine B
    /// copies it onto `FuncEffects.elidable` for `call.py getcalldescr`.
    #[test]
    fn lower_graph_copies_elidable_function_for_ll_find_char() {
        let value = variable_with_lltype("value", LowLevelType::Signed);
        let startblock = Block::shared(vec![Hlvalue::Variable(value.clone())]);
        let mut graph = FlowGraph::with_return_var(
            "ll_find_char",
            startblock.clone(),
            Hlvalue::Variable(value.clone()),
        );
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(value)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let mut func = GraphFunc::new(
            "ll_find_char",
            Constant::new(ConstValue::Dict(Default::default())),
        );
        func._elidable_function_ = true;
        graph.func = Some(func);

        let model = lower_graph(&graph);
        assert!(model.func.elidable);
    }

    fn gcarray_signed() -> LowLevelType {
        LowLevelType::Array(Box::new(Array::gc(LowLevelType::Signed)))
    }

    fn gcarray_signed_ptr() -> LowLevelType {
        LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(Array::gc(LowLevelType::Signed)),
        }))
    }

    fn gcarray_void() -> LowLevelType {
        LowLevelType::Array(Box::new(Array::gc(LowLevelType::Void)))
    }

    fn gcarray_void_ptr() -> LowLevelType {
        LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(Array::gc(LowLevelType::Void)),
        }))
    }

    /// `rewrite_op_malloc_varsize` compares `op.args[0].value == rstr.STR`.
    /// A GcStruct that only looks like STR (`chars: Array(Char)` under a
    /// different name) is not STR.
    #[test]
    #[should_panic(expected = "str_lookalike")]
    fn malloc_varsize_of_str_lookalike_is_not_newstr() {
        let lookalike = LowLevelType::Struct(Box::new(Struct::gc(
            "str_lookalike",
            vec![
                ("hash".into(), LowLevelType::Signed),
                (
                    "chars".into(),
                    LowLevelType::Array(Box::new(Array::new(LowLevelType::Char))),
                ),
            ],
        )));
        let n = variable_with_lltype("n", LowLevelType::Signed);
        let startblock = Block::shared(vec![Hlvalue::Variable(n.clone())]);
        let return_var = variable_with_lltype("s", STRPTR.clone());
        let graph = FlowGraph::with_return_var(
            "ll_test_malloc_varsize_str_lookalike",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let s = variable_with_lltype("s", STRPTR.clone());
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "malloc_varsize",
            vec![
                lowlevel_type_const(lookalike),
                gc_flavor_const().expect("gc flavor const"),
                Hlvalue::Variable(n),
            ],
            Hlvalue::Variable(s.clone()),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(s)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let _model = lower_graph(&graph);
    }

    /// `rewrite_op_getarrayitem`: `_array_of_voids(ARRAY)` returns [].
    #[test]
    fn getarrayitem_of_void_array_emits_nothing() {
        let array_ptr = gcarray_void_ptr();
        let arr = variable_with_lltype("arr", array_ptr);
        let i = variable_with_lltype("i", LowLevelType::Signed);
        let startblock = Block::shared(vec![
            Hlvalue::Variable(arr.clone()),
            Hlvalue::Variable(i.clone()),
        ]);
        let return_var = variable_with_lltype("n", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "ll_test_void_getarrayitem",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let x = variable_with_lltype("x", LowLevelType::Void);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getarrayitem",
            vec![Hlvalue::Variable(arr), Hlvalue::Variable(i.clone())],
            Hlvalue::Variable(x),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(i)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);
        for block in &model.blocks {
            for op in &block.operations {
                panic!(
                    "void-item getarrayitem must emit nothing, got {:?}",
                    op.kind
                );
            }
        }
    }

    /// `rewrite_op_malloc_varsize`: a Void-item `GcArray` still allocates
    /// (`new_array`); `_array_of_voids` is getarrayitem/setarrayitem only.
    #[test]
    fn malloc_varsize_of_void_array_is_new_array() {
        let n = variable_with_lltype("n", LowLevelType::Signed);
        let startblock = Block::shared(vec![Hlvalue::Variable(n.clone())]);
        let array_ptr = gcarray_void_ptr();
        let return_var = variable_with_lltype("a", array_ptr.clone());
        let graph = FlowGraph::with_return_var(
            "ll_test_malloc_varsize_gcarray_void",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let a = variable_with_lltype("a", array_ptr);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "malloc_varsize",
            vec![
                lowlevel_type_const(gcarray_void()),
                gc_flavor_const().expect("gc flavor const"),
                Hlvalue::Variable(n.clone()),
            ],
            Hlvalue::Variable(a.clone()),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(a.clone())],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);
        let mut news = Vec::new();
        for block in &model.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::NewArray {
                        length,
                        item_ty,
                        array_type_id,
                    } => {
                        assert_eq!(length, &n);
                        assert_eq!(*item_ty, ValueType::Void);
                        assert_eq!(array_type_id.as_deref(), Some("GcArray(Void)"));
                        assert_eq!(op.result.as_ref(), Some(&a));
                        news.push("new_array");
                    }
                    other => {
                        panic!("void-item malloc_varsize must lower to NewArray, got {other:?}")
                    }
                }
            }
        }
        assert_eq!(news, vec!["new_array"]);
    }

    /// `rewrite_op_malloc_varsize`: `malloc_varsize(GcArray(Signed), {flavor: gc}, n)`
    /// is `new_array`, not `newstr`.
    #[test]
    fn lower_graph_malloc_varsize_gcarray_signed_is_new_array() {
        let n = variable_with_lltype("n", LowLevelType::Signed);
        let startblock = Block::shared(vec![Hlvalue::Variable(n.clone())]);
        let array_ptr = gcarray_signed_ptr();
        let return_var = variable_with_lltype("a", array_ptr.clone());
        let graph = FlowGraph::with_return_var(
            "ll_test_malloc_varsize_gcarray",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let a = variable_with_lltype("a", array_ptr);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "malloc_varsize",
            vec![
                lowlevel_type_const(gcarray_signed()),
                gc_flavor_const().expect("gc flavor const"),
                Hlvalue::Variable(n),
            ],
            Hlvalue::Variable(a.clone()),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(a)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);
        let mut news = Vec::new();
        let mut residual = Vec::new();
        for block in &model.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::NewArray {
                        item_ty,
                        array_type_id,
                        ..
                    } => {
                        assert_eq!(*item_ty, ValueType::Int);
                        assert_eq!(array_type_id.as_deref(), Some(LIST_INT_ITEMS_ARRAY));
                        news.push("new_array");
                    }
                    OpKind::NewArrayClear { .. } => news.push("new_array_clear"),
                    OpKind::LoweredBlackholeOp { opname, .. } => residual.push(opname.clone()),
                    other => residual.push(format!("{other:?}")),
                }
            }
        }
        assert_eq!(news, vec!["new_array"]);
        assert!(residual.is_empty(), "unexpected residual ops: {residual:?}");
    }

    /// `rewrite_op_getarraysize` / `rewrite_op_getarrayitem` /
    /// `rewrite_op_setarrayitem` on a `GcArray(Signed)` are the array ops,
    /// not the string `chars` fusion.
    #[test]
    fn lower_graph_gcarray_item_ops_are_array_ops() {
        let array_ptr = gcarray_signed_ptr();
        let arr = variable_with_lltype("arr", array_ptr);
        let i = variable_with_lltype("i", LowLevelType::Signed);
        let v = variable_with_lltype("v", LowLevelType::Signed);
        let startblock = Block::shared(vec![
            Hlvalue::Variable(arr.clone()),
            Hlvalue::Variable(i.clone()),
            Hlvalue::Variable(v.clone()),
        ]);
        let return_var = variable_with_lltype("n", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "ll_test_gcarray_item_ops",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let n = variable_with_lltype("n", LowLevelType::Signed);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(arr.clone())],
            Hlvalue::Variable(n.clone()),
        ));
        let x = variable_with_lltype("x", LowLevelType::Signed);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getarrayitem",
            vec![Hlvalue::Variable(arr.clone()), Hlvalue::Variable(i.clone())],
            Hlvalue::Variable(x),
        ));
        let set = variable_with_lltype("set", LowLevelType::Void);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(arr),
                Hlvalue::Variable(i),
                Hlvalue::Variable(v),
            ],
            Hlvalue::Variable(set),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(n)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);
        let mut kinds = Vec::new();
        for block in &model.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::ArrayLen {
                        array_type_id,
                        nolength,
                        ..
                    } => {
                        assert_eq!(array_type_id.as_deref(), Some(LIST_INT_ITEMS_ARRAY));
                        assert!(!*nolength);
                        kinds.push("arraylen_gc");
                    }
                    OpKind::ArrayRead {
                        item_ty,
                        array_type_id,
                        nolength,
                        pure,
                        ..
                    } => {
                        assert_eq!(*item_ty, ValueType::Int);
                        assert_eq!(array_type_id.as_deref(), Some(LIST_INT_ITEMS_ARRAY));
                        assert!(!*nolength);
                        assert!(!*pure);
                        kinds.push("getarrayitem_gc");
                    }
                    OpKind::ArrayWrite {
                        item_ty,
                        array_type_id,
                        nolength,
                        ..
                    } => {
                        assert_eq!(*item_ty, ValueType::Int);
                        assert_eq!(array_type_id.as_deref(), Some(LIST_INT_ITEMS_ARRAY));
                        assert!(!*nolength);
                        kinds.push("setarrayitem_gc");
                    }
                    other => panic!("unexpected op: {other:?}"),
                }
            }
        }
        assert_eq!(
            kinds,
            vec!["arraylen_gc", "getarrayitem_gc", "setarrayitem_gc"]
        );
    }

    /// `ll_newlist` / `ll_append` graphs `_twoway_preprocess` uses (the
    /// `_search` helper chain) lower without treating the items `GcArray`
    /// as a string allocation.
    #[test]
    fn lower_graph_lowers_twoway_list_helpers() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let _search = crate::rlib::rstring::build_search_helper_graph(
            &rtyper,
            "_search",
            STRPTR.clone(),
            false,
        )
        .expect("_search");
        let prep = crate::rlib::rstring::build_twoway_preprocess_helper_graph(
            &rtyper,
            "_twoway_preprocess",
            STRPTR.clone(),
        )
        .expect("_twoway_preprocess");
        let mut saw_newlist = false;
        let mut saw_append = false;
        for func in direct_call_funcobjs(&prep.graph.borrow()) {
            let name = func._name.as_str();
            if name != "ll_newlist" && name != "ll_append" {
                continue;
            }
            let graph_id = func.graph.expect("list helper funcptr carries graph");
            let callee = graph_ref_for_funcptr_id(&rtyper, graph_id)
                .expect("list helper graph lives on translator.graphs");
            let model = lower_graph(&callee.borrow());
            let has_newstr = model.blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::LoweredBlackholeOp { opname, .. } if opname == "newstr"
                    )
                })
            });
            assert!(
                !has_newstr,
                "{name} must not lower malloc_varsize to newstr"
            );
            if name == "ll_newlist" {
                assert!(
                    model.blocks.iter().any(|block| {
                        block
                            .operations
                            .iter()
                            .any(|op| matches!(op.kind, OpKind::NewArray { .. }))
                    }),
                    "ll_newlist malloc_varsize of GcArray(Signed) is new_array"
                );
                saw_newlist = true;
            } else {
                saw_append = true;
            }
        }
        assert!(saw_newlist, "_twoway_preprocess calls ll_newlist");
        assert!(saw_append, "_twoway_preprocess calls ll_append");
    }

    /// `_ll_list_resize_ge` inlines `rlib/jit.py conditional_call` to a
    /// `jit_conditional_call` llop. `lower_graph` emits the interpreter-facing
    /// `majit_rlib::jit::conditional_callN` Call `_rewrite_op_cond_call` rewrites.
    #[test]
    fn lower_graph_lowers_jit_conditional_call_in_ll_list_resize_ge() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        crate::rlib::rstring::build_twoway_preprocess_helper_graph(
            &rtyper,
            "_twoway_preprocess",
            STRPTR.clone(),
        )
        .expect("_twoway_preprocess");
        let graphs = ann.translator.graphs.borrow();
        let ge = graphs
            .iter()
            .find(|g| g.borrow().name == "_ll_list_resize_ge")
            .expect("_twoway_preprocess mints _ll_list_resize_ge");
        let model = lower_graph(&ge.borrow());
        let cond_calls: Vec<_> = model
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call { target, args, .. } => {
                    let segs = target.path_segments()?;
                    if segs == ["majit_rlib", "jit", "conditional_call3"] {
                        Some(args.len())
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            cond_calls,
            vec![5],
            "one conditional_call3 Call (cond, fn, l, newsize, True), got {cond_calls:?}"
        );
        let has_fn_const = model.blocks.iter().any(|block| {
            block.operations.iter().any(|op| match &op.kind {
                OpKind::Call { target, args, .. } => {
                    target
                        .path_segments()
                        .is_some_and(|segs| segs.first() == Some(&crate::model::FN_CONST_HEAD))
                        && args.is_empty()
                }
                _ => false,
            })
        });
        assert!(has_fn_const, "function arg is a 0-arg __fn_const Call");
    }

    /// `ll_strhash` inlines `rlib/jit.py conditional_call_elidable` to a
    /// `jit_conditional_call_value` llop. `lower_graph` emits the interpreter-facing
    /// `majit_rlib::jit::conditional_call_elidable1` Call `_rewrite_op_cond_call` rewrites.
    #[test]
    fn lower_graph_lowers_jit_conditional_call_value_in_ll_strhash() {
        use crate::translator::rtyper::rmodel::Repr;
        use crate::translator::rtyper::rstr::string_repr;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let llfn = string_repr()
            .get_ll_hash_function(&rtyper)
            .expect("ll_strhash helper")
            .expect("Some");
        let graph = llfn.graph.as_ref().expect("ll_strhash graph");
        let model = lower_graph(&graph.graph.borrow());
        let cond_calls: Vec<_> = model
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call { target, args, .. } => {
                    let segs = target.path_segments()?;
                    if segs == ["majit_rlib", "jit", "conditional_call_elidable1"] {
                        assert!(
                            op.result.is_some(),
                            "conditional_call_elidable1 result var is the op result"
                        );
                        Some(args.len())
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            cond_calls,
            vec![3],
            "one conditional_call_elidable1 Call (value, fn, s), got {cond_calls:?}"
        );
        let has_fn_const = model.blocks.iter().any(|block| {
            block.operations.iter().any(|op| match &op.kind {
                OpKind::Call { target, args, .. } => {
                    target
                        .path_segments()
                        .is_some_and(|segs| segs.first() == Some(&crate::model::FN_CONST_HEAD))
                        && args.is_empty()
                }
                _ => false,
            })
        });
        assert!(has_fn_const, "function arg is a 0-arg __fn_const Call");
    }

    fn lower_jit_conditional_call_value_with_extra(n: usize) -> crate::model::FunctionGraph {
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, functionptr};

        let value = variable_with_lltype("h", LowLevelType::Signed);
        let extra: Vec<_> = (0..n)
            .map(|i| variable_with_lltype(&format!("a{i}"), LowLevelType::Signed))
            .collect();
        let mut inputargs = vec![Hlvalue::Variable(value.clone())];
        inputargs.extend(extra.iter().cloned().map(Hlvalue::Variable));
        let startblock = Block::shared(inputargs);
        let return_var = variable_with_lltype("out", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "ll_test_conditional_call_value",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let ft = FuncType {
            args: vec![LowLevelType::Signed; n],
            result: LowLevelType::Signed,
        };
        let funcptr = functionptr(ft, "compute", None, Some("compute".to_string()));
        let callee = constant_with_lltype(ConstValue::LLPtr(Box::new(funcptr)), LowLevelType::Void);
        let out = variable_with_lltype("out", LowLevelType::Signed);
        let mut args = vec![Hlvalue::Variable(value), callee];
        args.extend(extra.into_iter().map(Hlvalue::Variable));
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "jit_conditional_call_value",
            args,
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
        lower_graph(&graph)
    }

    fn cond_elidable_call_argc(model: &crate::model::FunctionGraph, leaf: &str) -> Vec<usize> {
        model
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call { target, args, .. } => {
                    let segs = target.path_segments()?;
                    if segs == ["majit_rlib", "jit", leaf] {
                        Some(args.len())
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect()
    }

    /// `Transformer._rewrite_op_cond_call` accepts 0 extra arguments
    /// (`conditional_call_elidable0`).
    #[test]
    fn lower_graph_lowers_jit_conditional_call_value_arity0() {
        let model = lower_jit_conditional_call_value_with_extra(0);
        assert_eq!(
            cond_elidable_call_argc(&model, "conditional_call_elidable0"),
            vec![2],
            "one conditional_call_elidable0 Call (value, fn)"
        );
    }

    /// `Transformer._rewrite_op_cond_call` accepts 2 extra arguments
    /// (`conditional_call_elidable2`).
    #[test]
    fn lower_graph_lowers_jit_conditional_call_value_arity2() {
        let model = lower_jit_conditional_call_value_with_extra(2);
        assert_eq!(
            cond_elidable_call_argc(&model, "conditional_call_elidable2"),
            vec![4],
            "one conditional_call_elidable2 Call (value, fn, a0, a1)"
        );
    }

    /// `Transformer._rewrite_op_cond_call` rejects more than 4 function
    /// arguments.
    #[test]
    #[should_panic(expected = "Conditional call does not support floats or more than 4 arguments")]
    fn lower_graph_jit_conditional_call_value_arity5_is_rejected() {
        let _ = lower_jit_conditional_call_value_with_extra(5);
    }
}
