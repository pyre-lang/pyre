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
//! `getarrayitem_gc_*` / `setarrayitem_gc_*` / `arraylen_gc`, and a raw
//! `ARRAY` (`rffi.CArray`, `_gckind == 'raw'`) to `raw_malloc_varsize` /
//! `getarrayitem_raw_*` / `setarrayitem_raw_*` (`_rewrite_raw_malloc`).
//! The transducer distinguishes those ARRAYs from a STR/UNICODE `chars`
//! interior array by the lltype on the op (`malloc_varsize`'s first-arg
//! Constant ARRAY, or the array operand's `concretetype`), not by the
//! helper name.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::flowspace::model::Variable as FVar;
use crate::flowspace::model::{
    Block as FlowBlock, ConstValue, Constant, FunctionGraph as FlowGraph, GraphRef, Hlvalue,
    HostObject,
};
use crate::model::{
    BlockId, CallTarget, ExitCase, ExitSwitch, FN_CONST_HEAD, Link, LinkArg, OpKind, ValueType,
};
use crate::parse::CallPath;
use crate::translator::backendopt::constfold::{MALLOC_ZERO_FILLED_TAG_ID, WE_ARE_JITTED_TAG_ID};
use crate::translator::rtyper::lltypesystem::lltype::{
    _address, _ptr, Array, ForwardReference, GcKind, LowLevelType, LowLevelValue, Ptr, PtrTarget,
    Struct,
};
use crate::translator::rtyper::lltypesystem::rbytearray::BYTEARRAYPTR;
use crate::translator::rtyper::lltypesystem::rffi::CCHARP;
use crate::translator::rtyper::lltypesystem::rstr::{
    STR, STRPTR, UNICODE, UNICODEPTR, prebuilt_str_bytes_and_hash,
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
pub fn lower_graph(graph: &FlowGraph) -> crate::model::FunctionGraph {
    lower_graph_with_callees(graph, &|_| None, None)
}

/// Lower `graph`, resolving each `direct_call` / `jit_conditional_call`
/// callee by graph identity when `callee_path` returns a [`CallPath`].
///
/// `call.py graphs_from` looks up `funcobj.graph` in
/// `CallControl.jitcodes = {graph: jitcode}`; a miss falls back to
/// `_func._name` (the `ll_*` helper path through
/// `register_opname_helper_graph`).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn lower_graph_with_callees(
    graph: &FlowGraph,
    callee_path: &dyn Fn(&GraphRef) -> Option<CallPath>,
    carrier_class: Option<&HostObject>,
) -> crate::model::FunctionGraph {
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
        // `Transformer.optimize_block` `renamings` / `do_rename`'s Void
        // branch (`jtransform.py`): block-local exactly like upstream's
        // `renamings`.
        let mut void_renamings: HashMap<FVar, Hlvalue> = HashMap::new();
        for op in &fb_ref.operations {
            if matches!(
                op.opname.as_str(),
                "same_as" | "cast_pointer" | "likely" | "unlikely" | "cast_opaque_ptr"
            ) {
                if let Hlvalue::Variable(result) = &op.result {
                    if matches!(result.concretetype(), Some(LowLevelType::Void)) {
                        void_renamings.insert(result.clone(), void_none_hlvalue());
                        continue;
                    }
                }
            }
            if let Some(renamed) = rename_void_op_args(op, &void_renamings) {
                transduce_op(
                    &mut out,
                    id,
                    &renamed,
                    &mut chars_alias,
                    &family,
                    callee_path,
                );
            } else {
                transduce_op(&mut out, id, op, &mut chars_alias, &family, callee_path);
            }
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
                let target_inputargs = out.block(target_id).inputargs.clone();
                let graph_name = out.name.clone();
                let args: Vec<LinkArg> = link
                    .args
                    .iter()
                    .enumerate()
                    .map(|(i, arg)| {
                        // `Transformer._do_renaming_on_link` (`jtransform.py`).
                        let hlv = arg.as_ref().expect("opname-spine link arg is undefined");
                        linkarg_from_hlvalue_matching_target(
                            rename_void_hlvalue(hlv, &void_renamings),
                            target_inputargs.get(i),
                            &graph_name,
                        )
                    })
                    .collect();
                let exitcase = match &link.exitcase {
                    Some(Hlvalue::Constant(c)) => exitcase_from_const(&c.value, carrier_class),
                    _ => None,
                };
                let mut model_link = Link::new_mixed(args, target_id, exitcase);
                // Carry the exception extravars the flowspace producer set (e.g.
                // `rbuilder::close_ovfcheck_block`): a canraise exit's
                // last_exception / last_exc_value are required by flatten's
                // make_exception_link (it asserts they are Some and, for an
                // empty-target handler, unwraps them). Dropping them turns a
                // faithful OverflowError edge into a flatten-time panic.
                model_link.last_exception = link
                    .last_exception
                    .as_ref()
                    .map(|hlv| linkarg_from_hlvalue(rename_void_hlvalue(hlv, &void_renamings)));
                model_link.last_exc_value = link
                    .last_exc_value
                    .as_ref()
                    .map(|hlv| linkarg_from_hlvalue(rename_void_hlvalue(hlv, &void_renamings)));
                // `flatten.py insert_exits` reads `link.llexitcase`.
                // `RPythonTyper._convert_link` already stored the low-level
                // constant on the flowspace link; copy it. Do not recompute.
                model_link.llexitcase = match &link.llexitcase {
                    Some(Hlvalue::Constant(c)) => Some(c.value.clone()),
                    None => model_link.llexitcase,
                    Some(other) => panic!(
                        "jtransform_opname::lower_graph: llexitcase is not a Constant: {other:?}"
                    ),
                };
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
            match fb_ref
                .exitswitch
                .as_ref()
                .map(|hlv| rename_void_hlvalue(hlv, &void_renamings))
            {
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
pub(crate) fn startblock_oopspec_argnames(graph: &FlowGraph) -> Vec<String> {
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
    callee_path: &dyn Fn(&GraphRef) -> Option<CallPath>,
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
                        array: Some(array.clone()),
                        nolength: array_nolength(&array),
                        gckind: crate::model::ArrayGcKind::Gc,
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
        // `Transformer.rewrite_op_getarrayitem`: `_array_of_voids` then
        // `FixedSizeArray`, then a standalone ARRAY.  `ARRAY._gckind ==
        // 'gc'` is `getarrayitem_gc_<kind>` (`pure` from
        // `_immutable_field`); `ARRAY._gckind == 'raw'` is
        // `getarrayitem_raw_<kind>` with `pure` always false (never
        // `_pure`) and `getarrayitem_raw_r` refused.  A STR/UNICODE
        // `chars` interior pointer fuses to `strgetitem` /
        // `unicodegetitem`.
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
        // `Transformer.rewrite_op_setarrayitem`: `_array_of_voids` then
        // `FixedSizeArray`, then a standalone ARRAY.
        // `setarrayitem_<gckind>_<kind>`; `setarrayitem_raw_r` is
        // refused.  A STR/UNICODE `chars` interior pointer fuses to
        // `strsetitem` / `unicodesetitem`.
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
        // `Transformer.rewrite_op_malloc_varsize`: flavor `raw` first
        // (`_rewrite_raw_malloc` / `raw_malloc_varsize`), then
        // `op.args[0].value == rstr.STR` → `newstr`, `== rstr.UNICODE`
        // → `newunicode`, else ARRAY → `new_array` / `new_array_clear`.
        "malloc_varsize" => {
            let malloc_ty = malloc_type_lltype(&op.args[0]);
            if crate::codewriter::call::effect_malloc_flavor_is_raw(&op.args[1]) {
                let (zero, track_allocation, add_memory_pressure, extra_flags) =
                    crate::codewriter::call::effect_malloc_raw_flags(&op.args[1]);
                let owner = crate::codewriter::call::effect_malloc_owner(&op.args[0]);
                let size = materialize(out, block, &op.args[2]);
                let result = expect_var(&op.result);
                out.push_op_with_result_var(
                    block,
                    OpKind::raw_malloc_with_flags(
                        owner,
                        Some(malloc_ty.clone()),
                        zero,
                        track_allocation,
                        add_memory_pressure,
                        extra_flags,
                        Some(size),
                    ),
                    result,
                );
            } else if let Some(alloc_family) = str_family_from_lltype(malloc_ty) {
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
                        array: Some(array.clone()),
                    }
                } else {
                    OpKind::NewArray {
                        length: size,
                        item_ty,
                        array_type_id,
                        array: Some(array.clone()),
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
        // `bhimpl_float_neg` / `bhimpl_float_abs` via `_add_default_ops`.
        // Keep the full opname, matching `int_neg`: the assembler passes
        // already-prefixed unary names through, and the shared Transformer
        // emits the same `UnaryOp` for a float-kind `neg` / `abs`.
        name @ ("float_neg" | "float_abs") => {
            let operand = materialize(out, block, &op.args[0]);
            let result = expect_var(&op.result);
            let result_ty = value_type_of(&result);
            out.push_op_with_result_var(
                block,
                OpKind::UnaryOp {
                    op: name.to_string(),
                    operand,
                    result_ty,
                },
                result,
            );
        }
        // `Transformer.rewrite_op_float_is_true`: `float_ne` against
        // `Constant(0.0, lltype.Float)`, then `rewrite_operation`.
        "float_is_true" => {
            let operand = materialize(out, block, &op.args[0]);
            let zero = materialize(
                out,
                block,
                &Hlvalue::Constant(Constant::with_concretetype(
                    ConstValue::Float(0.0_f64.to_bits()),
                    LowLevelType::Float,
                )),
            );
            let result = expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::BinOp {
                    op: "float_ne".to_string(),
                    lhs: operand,
                    rhs: zero,
                    result_ty: value_type_of(&result),
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
        // Float arithmetic and compares that survive as blackhole ops
        // (`bhimpl_float_*`, `_add_default_ops` `keep_operation_unchanged`).
        // `rewrite_op_float_add` / `rewrite_op_float_mul` /
        // `rewrite_op_float_{lt,le,gt,ge}` are `_rewrite_symmetric`;
        // `float_sub` / `float_truediv` / `float_eq` / `float_ne` have no
        // rewrite.  Keep the full llop name: the assembler preserves
        // `float_*` `BinOp` labels, and the shared Transformer emits the
        // same spelling after its float-kind rewrite.
        name @ ("float_add" | "float_sub" | "float_mul" | "float_truediv" | "float_lt"
        | "float_le" | "float_eq" | "float_ne" | "float_gt" | "float_ge") => {
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
        // `jtransform.py` `('keepalive', '-live-')`: keep the operands as
        // forced live values (`liveness.py` `-live-` args).
        "keepalive" => {
            let forced: Vec<_> = non_void_args(&op.args)
                .map(|arg| materialize(out, block, arg))
                .collect();
            out.block_mut(block)
                .operations
                .push(crate::model::SpaceOperation {
                    result: None,
                    kind: OpKind::Live { forced },
                });
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
                        linkarg_from_hlvalue_matching_target(
                            &op.args[0],
                            hlvalue_as_var(&op.args[1]).as_ref(),
                            &out.name,
                        ),
                        linkarg_from_hlvalue_matching_target(
                            &op.args[1],
                            hlvalue_as_var(&op.args[0]).as_ref(),
                            &out.name,
                        ),
                    ],
                },
                false,
            );
        }
        // `getfield(s, "field")` → `FieldRead`.  The field name is a Void
        // `ByteStr` constant (`void_field_const`); the owning struct identity
        // is the pointee-struct leaf of the base pointer's `concretetype`
        // (`Ptr(Struct("stringbuilder"))` → `"stringbuilder"`), which keys the
        // layout registry the assembler resolves offsets through.  An rtyper
        // tuple's identity is the lltype spelling (`tuple{n}<s0,…>`), not the
        // bare `tuple{n}` leaf.  `rewrite_op_getfield` sets `_pure` from
        // `STRUCT._immutable_field(name)`; a struct with the `immutable`
        // hint (`TUPLE_TYPE`) is `getfield_gc_*_pure`.
        "getfield" => {
            // `Transformer.rewrite_op_getfield` returns for a Void result.
            if hlvalue_is_void(&op.result) {
                return;
            }
            let field_name = field_name_of(&op.args[1]);
            // `Transformer.handle_getfield_typeptr`: a Constant receiver
            // folds to `same_as` of `op.args[0].value.typeptr`.
            if field_name == "typeptr"
                && let Hlvalue::Constant(c) = &op.args[0]
            {
                let ConstValue::LLPtr(ptr) = &c.value else {
                    panic!(
                        "jtransform_opname::lower_graph: Constant typeptr receiver is not an LLPtr: {:?}",
                        c.value
                    );
                };
                let typeptr = constant_receiver_typeptr(ptr);
                let result = expect_var(&op.result);
                let result_ty = result.concretetype().expect(
                    "jtransform_opname::lower_graph: typeptr getfield result has concretetype",
                );
                let cls =
                    Constant::with_concretetype(ConstValue::LLPtr(Box::new(typeptr)), result_ty);
                let op = crate::codewriter::jtransform::materialize_const_result_copy(result, &cls);
                out.block_mut(block).operations.push(op);
                return;
            }
            let field = field_descriptor_of_hlvalue(&op.args[0], &field_name, "getfield");
            let pure = match &op.args[0] {
                Hlvalue::Variable(base) => {
                    owner_struct_of(base).is_some_and(|s| s._immutable_field(&field_name))
                }
                Hlvalue::Constant(c) => owner_struct_of_lltype(c.concretetype.as_ref().unwrap())
                    .is_some_and(|s| s._immutable_field(&field_name)),
            };
            let base = materialize(out, block, &op.args[0]);
            let result = expect_var(&op.result);
            let ty = value_type_of(&result);
            out.push_op_with_result_var(
                block,
                OpKind::FieldRead {
                    base,
                    field,
                    ty,
                    pure,
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
                let operand = materialize(out, block, &op.args[1 - null_i]);
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
                let lhs = materialize(out, block, &op.args[0]);
                let rhs = materialize(out, block, &op.args[1]);
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
            let operand = materialize(out, block, &op.args[0]);
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
            // `Transformer.rewrite_op_setfield` returns for a Void value.
            if hlvalue_is_void(&op.args[2]) {
                return;
            }
            let field_name = field_name_of(&op.args[1]);
            let field = field_descriptor_of_hlvalue(&op.args[0], &field_name, "setfield");
            let base = materialize(out, block, &op.args[0]);
            let value = linkarg_from_hlvalue(&op.args[2]);
            let ty = hlvalue_value_type(&op.args[2]);
            out.push_op_var(
                block,
                OpKind::FieldWrite {
                    base,
                    field,
                    value,
                    ty,
                },
                false,
            );
        }
        // `Transformer.rewrite_op_malloc`: flavor `raw` is
        // `_rewrite_raw_malloc` / `raw_malloc_fixedsize`; flavor `gc`
        // is `new(descr)` (`New`).  The struct leaf (`"stringbuilder"`)
        // is read off the first operand's `LowLevelType` constant and
        // keys the assembler's size descriptor
        // (`bh_size_spec_from_callcontrol`, `path_hash(owner)`).  A
        // plain GcStruct with no boxed `ob_type`, so `New` (not
        // `NewWithVtable`).  Any other flavor is refused.
        "malloc" => {
            if crate::codewriter::call::effect_malloc_flavor_is_raw(&op.args[1]) {
                let (zero, track_allocation, add_memory_pressure, extra_flags) =
                    crate::codewriter::call::effect_malloc_raw_flags(&op.args[1]);
                let owner = crate::codewriter::call::effect_malloc_owner(&op.args[0]);
                let malloc_ty = crate::codewriter::call::effect_malloc_type(&op.args[0]);
                let result = expect_var(&op.result);
                out.push_op_with_result_var(
                    block,
                    OpKind::raw_malloc_with_flags(
                        owner,
                        Some(malloc_ty),
                        zero,
                        track_allocation,
                        add_memory_pressure,
                        extra_flags,
                        None,
                    ),
                    result,
                );
            } else {
                assert!(
                    is_gc_flavor_const(&op.args[1]),
                    "jtransform_opname::lower_graph: malloc flavor operand is not {{'flavor': 'gc'}}: {:?}",
                    op.args[1]
                );
                let owner = malloc_struct_owner(&op.args[0]);
                let result = expect_var(&op.result);
                out.push_op_with_result_var(block, OpKind::New { owner }, result);
            }
        }
        // `jtransform.py rewrite_op_gc_identityhash = _do_builtin_call`.
        // `Transformer._do_builtin_call` matches this `LoweredBlackholeOp`
        // and emits `direct_call(_ll_1_gc_identityhash, x)` (`support.py`).
        "gc_identityhash" => {
            assert_eq!(
                op.args.len(),
                1,
                "jtransform_opname::lower_graph: gc_identityhash takes one argument"
            );
            let args =
                crate::model::call_args(op.args.iter().map(|arg| materialize(out, block, arg)));
            let result = expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::LoweredBlackholeOp {
                    opname: "gc_identityhash".to_string(),
                    args,
                },
                result,
            );
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
        // Float-bank unaries that survive as blackhole ops.  `cast_int_to_float`
        // / `cast_float_to_int` / `cast_float_to_singlefloat` /
        // `cast_singlefloat_to_float` have no `rewrite_op_*` and pass through
        // `_add_default_ops`.  `rewrite_op_convert_float_bytes_to_longlong`
        // / `rewrite_op_convert_longlong_bytes_to_float` are `_noop_rewrite`.
        // `rewrite_op_cast_uint_to_float` / `rewrite_op_cast_float_to_uint`
        // are `_do_builtin_call`; emit the same `UnaryOp` the shared
        // Transformer residualizes, matching `int_floordiv`.
        name @ ("cast_int_to_float"
        | "cast_float_to_int"
        | "cast_uint_to_float"
        | "cast_float_to_uint"
        | "cast_float_to_singlefloat"
        | "cast_singlefloat_to_float"
        | "convert_float_bytes_to_longlong"
        | "convert_longlong_bytes_to_float") => {
            let operand = materialize(out, block, &op.args[0]);
            let result = expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::UnaryOp {
                    op: name.to_string(),
                    operand,
                    result_ty: value_type_of(&result),
                },
                result,
            );
        }
        // Word-sized integer casts are identity at JitCode level
        // (`getkind(Signed) == getkind(Unsigned) == getkind(Char) == 'int'`).
        // The named char/bool/uint casts are explicit no-op methods.
        // Route them through the rich graph's `same_as`, which the shared
        // jtransform tail drops while installing the result→operand alias.
        name @ ("cast_bool_to_int"
        | "cast_bool_to_uint"
        | "cast_char_to_int"
        | "cast_unichar_to_int"
        | "cast_int_to_char"
        | "cast_int_to_unichar"
        | "cast_int_to_uint"
        | "cast_uint_to_int") => {
            let operand = materialize(out, block, &op.args[0]);
            let result = expect_var(&op.result);
            let _ = name;
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
            // an int. `rewrite_op_cast_int_to_longlong` residualizes to
            // `llong_from_*`; the result kind follows `is_64_bit`
            // (`longlong.py`): Signed width 8 is int, otherwise float.
            let longlong_is_float = !crate::codewriter::longlong::is_64_bit();
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
        // `direct_call(funcptr, arg0, arg1, …)` → `Call`.  The callee is
        // `funcobj.graph` when `callee_path` knows that graph (`call.py
        // graphs_from`); otherwise `_func._name` on the leading funcptr
        // (`ConstValue::LLPtr` → `_ptr_obj::Func`), split on `::` so
        // `majit_rlib::jit::isconstant` and `register_opname_helper_graph`
        // hit a regular CallPath rather than a residual fnaddr.  The
        // trailing operands are the call arguments; `OpKind::Call` takes
        // `Variable`s, so constant arguments (e.g. `ll_min(size, 1280)`)
        // are materialised through the same `ConstInt`/`ConstBool` path the
        // `int_*` arm uses.  The builder helpers' `direct_call`s all return a
        // value (`ll_min` → uint, `mallocfn` → the char buffer), so the result
        // is bound as a result var.
        "direct_call" => {
            let func = callee_func_from_funcptr(&op.args[0]);
            // `call.py graphs_from` resolves `funcobj.graph` by identity
            // (`CallControl.jitcodes = {graph: jitcode}`), never by
            // `_func._name`.
            let target = match func.graph_ref().and_then(|g| callee_path(&g)) {
                Some(path) => CallTarget::from_call_path(&path),
                None => CallTarget::from_call_path(&callpath_from_func_name(&func._name)),
            };
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
            check_cond_call_operands(&op.args);
            let cond = materialize(out, block, &op.args[0]);
            let func = callee_func_from_funcptr(&op.args[1]);
            // `call.py graphs_from` (`funcobj.graph`), same as `direct_call`.
            let callee = match func.graph_ref().and_then(|g| callee_path(&g)) {
                Some(path) => CallTarget::from_call_path(&path).with_resolved_path(path),
                None => {
                    let name = callee_name_from_funcptr(&op.args[1]);
                    CallTarget::function_path(name.split("::").filter(|s| !s.is_empty()))
                }
            };
            let extra_args: Vec<FVar> = non_void_args(&op.args[2..])
                .map(|a| materialize(out, block, a))
                .collect();
            let result = expect_var(&op.result);
            emit_conditional_call_oopspec_named(
                out,
                block,
                cond,
                callee,
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
            check_cond_call_operands(&op.args);
            let value = materialize(out, block, &op.args[0]);
            let func = callee_func_from_funcptr(&op.args[1]);
            // `call.py graphs_from` (`funcobj.graph`), same as `direct_call`.
            let callee = match func.graph_ref().and_then(|g| callee_path(&g)) {
                Some(path) => CallTarget::from_call_path(&path).with_resolved_path(path),
                None => {
                    let name = callee_name_from_funcptr(&op.args[1]);
                    CallTarget::function_path(name.split("::").filter(|s| !s.is_empty()))
                }
            };
            let extra_args: Vec<FVar> = non_void_args(&op.args[2..])
                .map(|a| materialize(out, block, a))
                .collect();
            let result = expect_var(&op.result);
            emit_conditional_call_oopspec_named(
                out,
                block,
                value,
                callee,
                extra_args,
                result,
                "conditional_call_elidable",
            );
        }
        // `jtransform.py Transformer.rewrite_op_debug_assert` returns [].
        "debug_assert" => {}
        // `ForceCastEntry` (`ll2ctypes.py`) emits `force_cast`.
        // `Transformer.rewrite_op_cast_primitive` delegates to
        // `rewrite_op_force_cast`. The rich graph keeps the original
        // opname (`name.to_string()`) so `rewrite_operation` can
        // dispatch it; a typed cast never survives that rewrite
        // (assembler `int_force_cast` would be unsupported).
        name @ ("force_cast" | "cast_primitive") => {
            let operand = materialize(out, block, &op.args[0]);
            let result = expect_var(&op.result);
            let result_ty = value_type_of(&result);
            out.push_op_with_result_var(
                block,
                OpKind::UnaryOp {
                    op: name.to_string(),
                    operand,
                    result_ty,
                },
                result,
            );
        }
        // `Transformer.rewrite_op_same_as`, `rewrite_op_cast_pointer`, and
        // `rewrite_op_cast_opaque_ptr` (`jtransform.py`) return None: the
        // result is renamed to the argument, which is what the shared
        // `UnaryOp same_as` handling in `Transformer` does.
        "same_as" | "cast_pointer" | "likely" | "unlikely" | "cast_opaque_ptr" => {
            // `Transformer.rewrite_op_same_as` returns None, so `optimize_block`
            // renames the result. `do_rename` makes a Void result
            // `Constant(None, lltype.Void)` and allocates no register.
            if hlvalue_is_void(&op.args[0]) || hlvalue_is_void(&op.result) {
                return;
            }
            let result = expect_var(&op.result);
            if let Hlvalue::Constant(c) = &op.args[0] {
                assert_rtyped_link_constant_kind(&out.name, c, &result);
            }
            let operand = materialize(out, block, &op.args[0]);
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
        // `rewrite_op_cast_ptr_to_int` keeps a GC pointer as the llop
        // and returns None (identity) otherwise. `cast_int_to_ptr` has
        // no rewrite (`bhimpl_cast_int_to_ptr` / `_add_default_ops`).
        // `rewrite_op_cast_ptr_to_adr` raises on GC and is identity on
        // raw; emit the UnaryOp the shared Transformer already accepts.
        name @ ("cast_ptr_to_int" | "cast_int_to_ptr" | "cast_ptr_to_adr") => {
            let operand = materialize(out, block, &op.args[0]);
            let result = expect_var(&op.result);
            let opname = if name == "cast_ptr_to_int" && !lltype_is_gc_ptr(&operand) {
                "same_as"
            } else {
                name
            };
            out.push_op_with_result_var(
                block,
                OpKind::UnaryOp {
                    op: opname.to_string(),
                    operand,
                    result_ty: value_type_of(&result),
                },
                result,
            );
        }
        // `jtransform.py Transformer.rewrite_op_getinteriorfield` /
        // `rewrite_op_setinteriorfield` / `rewrite_op_getinteriorarraysize`.
        "getinteriorfield" => emit_getinteriorfield(out, block, op),
        "setinteriorfield" => emit_setinteriorfield(out, block, op),
        "getinteriorarraysize" => emit_getinteriorarraysize(out, block, op),
        // `jtransform.py Transformer.rewrite_op_raw_load` /
        // `rewrite_op_raw_store`: `raw_{load,store}_<kind>` plus
        // `arraydescrof(rffi.CArray(T))`. Kind `'r'` is refused.
        name @ ("raw_load" | "raw_store") => emit_raw_load_or_store(out, block, op, name),
        // `jtransform.py Transformer.rewrite_op_direct_ptradd`: CCHARP
        // is one `int_add`; otherwise scale the shift by
        // `llmemory.sizeof(TO.OF)` then `int_add`.
        "direct_ptradd" => emit_direct_ptradd(out, block, op),
        // `jtransform.py Transformer.rewrite_op_indirect_call`: emit
        // the same `OpKind::IndirectCall` the shared Transformer
        // residualizes / `handle_regular_indirect_call`s. Trailing
        // Void graphs stay `None` here (`transduce_op` has no
        // CallControl to map `ConstValue::Graphs` ids to CallPath).
        "indirect_call" => {
            assert!(
                !op.args.is_empty(),
                "jtransform_opname::lower_graph: indirect_call needs a funcptr"
            );
            let funcptr = materialize(out, block, &op.args[0]);
            let args: Vec<FVar> = non_void_args(&op.args[1..])
                .map(|a| materialize(out, block, a))
                .collect();
            let result = expect_var(&op.result);
            let result_ty = value_type_of(&result);
            out.push_op_with_result_var(
                block,
                OpKind::IndirectCall {
                    funcptr,
                    args,
                    graphs: None,
                    family_key: None,
                    result_ty,
                },
                result,
            );
        }
        // `jtransform.py Transformer.rewrite_op_jit_force_virtualizable`:
        // emit the leftover Call the shared Transformer already
        // dispatches (`is_jit_force_virtualizable_target`). Filing
        // `vable_flags` / `get_vinfo` needs CallControl.
        "jit_force_virtualizable" => {
            let args: Vec<LinkArg> = op.args.iter().map(linkarg_from_hlvalue).collect();
            let result = expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::Call {
                    target: CallTarget::function_path(["jit_force_virtualizable"]),
                    args,
                    result_ty: value_type_of(&result),
                },
                result,
            );
        }
        // `jtransform.py Transformer.rewrite_op_jit_force_quasi_immutable`:
        // `-live-` + `jit_force_quasi_immutable(v_inst, fielddescr)`.
        // Applied here because the Transformer Call arm needs
        // `callcontrol.callee_llop`.
        "jit_force_quasi_immutable" => {
            assert_eq!(
                op.args.len(),
                2,
                "jtransform_opname::lower_graph: jit_force_quasi_immutable needs v_inst and fieldname"
            );
            let field_name = field_name_of(&op.args[1]);
            let mutate_field =
                field_descriptor_of_hlvalue(&op.args[0], &field_name, "jit_force_quasi_immutable");
            out.block_mut(block)
                .operations
                .push(crate::model::SpaceOperation {
                    result: None,
                    kind: OpKind::live(),
                });
            out.push_op_var(
                block,
                OpKind::JitForceQuasiImmutable {
                    base: linkarg_from_hlvalue(&op.args[0]),
                    mutate_field,
                },
                false,
            );
        }
        // `jtransform.py Transformer.rewrite_op_free`: flavor `'raw'`,
        // leftover flags `UnsupportedMallocFlags`, then `raw_free`.
        "free" => {
            assert!(
                op.args.len() >= 2,
                "jtransform_opname::lower_graph: free needs ptr and flags"
            );
            assert!(
                crate::codewriter::call::effect_malloc_flavor_is_raw(&op.args[1]),
                "jtransform_opname::lower_graph: free flavor is not raw"
            );
            let extra = free_leftover_flags(&op.args[1]);
            if !extra.is_empty() {
                panic!("UnsupportedMallocFlags({extra:?})");
            }
            let ptr = materialize(out, block, &op.args[0]);
            out.push_op_var(block, OpKind::RawFree { ptr }, false);
        }
        // `jtransform.py Transformer.rewrite_op_gc_add_memory_pressure
        // = _do_builtin_call`. Leftover Call so
        // `rewrite_op_direct_call` residualizes it.
        "gc_add_memory_pressure" => {
            emit_leftover_named_call(out, block, op, "gc_add_memory_pressure");
        }
        // `jtransform.py Transformer.rewrite_op_threadlocalref_get`:
        // residual `OS_THREADLOCALREF_GET`. Front / rtyper emit the
        // 2-arg BinOp (offset, loop-invariant flag) the Transformer
        // already residualizes.
        "threadlocalref_get" => {
            assert!(
                op.args.len() >= 2,
                "jtransform_opname::lower_graph: threadlocalref_get needs offset and flag"
            );
            let lhs = materialize(out, block, &op.args[0]);
            let rhs = materialize(out, block, &op.args[1]);
            let result = expect_var(&op.result);
            out.push_op_with_result_var(
                block,
                OpKind::BinOp {
                    op: "threadlocalref_get".to_string(),
                    lhs,
                    rhs,
                    result_ty: value_type_of(&result),
                },
                result,
            );
        }
        other => panic!("jtransform_opname::lower_graph: unsupported opname {other:?}"),
    }
}

/// `Transformer._rewrite_op_cond_call`: on the operation's own operands
/// (`[cond_or_value, funcptr, *args]`, Void included), reject any float
/// operand or more than 4 function arguments.
pub(crate) fn check_cond_call_operands(args: &[Hlvalue]) {
    let have_floats = args.iter().any(|arg| {
        let ty = match arg {
            Hlvalue::Variable(v) => v.concretetype(),
            Hlvalue::Constant(c) => c.concretetype.clone(),
        };
        ty.is_some_and(|ty| crate::model::getkind(&ty) == crate::model::ConcreteType::Float)
    });
    if args.len() > 4 + 2 || have_floats {
        panic!("Conditional call does not support floats or more than 4 arguments");
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
        CallTarget::function_path(callee_name.split("::").filter(|s| !s.is_empty())),
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
        CallTarget::function_path(callee_name.split("::").filter(|s| !s.is_empty())),
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
/// or `conditional_call_elidableN`. The arity/float check is
/// `check_cond_call_operands`, run by the callers on the original operands.
fn emit_conditional_call_oopspec_named(
    out: &mut crate::model::FunctionGraph,
    block: BlockId,
    cond: FVar,
    callee: CallTarget,
    extra_args: Vec<FVar>,
    result: FVar,
    helper_prefix: &str,
) {
    let n = extra_args.len();
    debug_assert!(
        n <= 4,
        "the helper leaf `{helper_prefix}{{n}}` only exists for 0..=4"
    );
    let fn_target = match callee {
        CallTarget::FunctionPath {
            mut segments,
            resolved_path,
            fun_decl_id,
            specialization,
        } => {
            segments.insert(0, FN_CONST_HEAD.to_string());
            CallTarget::FunctionPath {
                segments,
                resolved_path,
                fun_decl_id,
                specialization,
            }
        }
        other => panic!(
            "jtransform_opname::lower_graph: jit_conditional_call callee is not a FunctionPath: {other:?}"
        ),
    };
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

/// Raw `Array` carried by `v.concretetype` / `concretetype.TO`
/// (`Transformer.rewrite_op_getarrayitem` `ARRAY._gckind == 'raw'`).
fn raw_array_of_operand(hlv: &Hlvalue) -> Option<Array> {
    let var = hlvalue_as_var(hlv)?;
    raw_array_from_lltype(&var.concretetype()?)
}

fn raw_array_from_lltype(ty: &LowLevelType) -> Option<Array> {
    match ty {
        LowLevelType::ForwardReference(fwd) => raw_array_from_lltype(&fwd.resolved()?),
        LowLevelType::Array(arr) if arr._gckind == GcKind::Raw => Some(arr.as_ref().clone()),
        LowLevelType::Ptr(ptr) => match &ptr.TO {
            PtrTarget::Array(arr) if arr._gckind == GcKind::Raw => Some(arr.clone()),
            PtrTarget::ForwardReference(fwd) => raw_array_from_lltype(&fwd.resolved()?),
            _ => None,
        },
        LowLevelType::InteriorPtr(ip) => raw_array_from_lltype(ip.TO.as_ref()),
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

/// `v.concretetype == rstr.STRPTR` (`Ptr(STR)` / `rpy_string`).
/// `StringRepr.convert_const` stamps this lltype; UNICODE is
/// `UNICODEPTR` and stays out.
fn is_str_ptr(p: &Ptr) -> bool {
    LowLevelType::Ptr(Box::new(p.clone())) == *STRPTR
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
                array: Some(array.clone()),
                nolength: array_nolength(&array),
                gckind: crate::model::ArrayGcKind::Gc,
                pure: array._immutable_field(),
            },
            result,
        );
    } else if let Some(array) = raw_array_of_operand(&op.args[0])
        .filter(|_| !chars_alias.contains_key(&expect_var(&op.args[0])))
    {
        // `Transformer.rewrite_op_getarrayitem`: a standalone raw ARRAY
        // never uses `_pure`; a reference result is
        // `getarrayitem_raw_r not supported`.  STR/UNICODE `chars` is
        // also `GcKind::Raw`; a same-block `getsubstruct` alias stays
        // on the fusion arm below.
        let base = expect_var(&op.args[0]);
        let index = materialize(out, block, &op.args[1]);
        let result = expect_var(&op.result);
        let item_ty = value_type_of(&result);
        if matches!(item_ty, ValueType::Ref(_)) {
            panic!("getarrayitem_raw_r not supported");
        }
        out.push_op_with_result_var(
            block,
            OpKind::ArrayRead {
                base,
                index,
                item_ty,
                array_type_id: array_type_id_of(&array),
                array: Some(array.clone()),
                nolength: array_nolength(&array),
                gckind: crate::model::ArrayGcKind::Raw,
                pure: false,
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
                nolength: array_nolength(&array),
                array_type_id: array_type_id_of(&array),
                array: Some(array.clone()),
                gckind: crate::model::ArrayGcKind::Gc,
            },
            false,
        );
    } else if let Some(array) = raw_array_of_operand(&op.args[0])
        .filter(|_| !chars_alias.contains_key(&expect_var(&op.args[0])))
    {
        // `Transformer.rewrite_op_setarrayitem`: a standalone raw ARRAY
        // of references is `setarrayitem_raw_r not supported`.
        // STR/UNICODE `chars` is also `GcKind::Raw`; a same-block
        // `getsubstruct` alias stays on the fusion arm below.
        let item_ty = hlvalue_value_type(&op.args[2]);
        if matches!(item_ty, ValueType::Ref(_)) {
            panic!("setarrayitem_raw_r not supported");
        }
        let base = expect_var(&op.args[0]);
        let index = materialize(out, block, &op.args[1]);
        let value = linkarg_from_hlvalue(&op.args[2]);
        out.push_op_var(
            block,
            OpKind::ArrayWrite {
                base,
                index,
                value,
                item_ty,
                array_type_id: array_type_id_of(&array),
                array: Some(array.clone()),
                nolength: array_nolength(&array),
                gckind: crate::model::ArrayGcKind::Raw,
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

/// `descr.py get_array_descr` cache key: one spelling per ARRAY lltype.
fn array_type_id_of(arr: &Array) -> Option<String> {
    Some(crate::codewriter::call::array_type_id_of_lltype(arr))
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

/// `Transformer.make_three_lists` takes a Constant operand of any kind
/// (`getkind(v.concretetype)`); `assembler.py emit_const` pools it by kind.
pub(crate) fn materialize_constant(
    out: &mut crate::model::FunctionGraph,
    block: BlockId,
    c: &Constant,
    who: &str,
) -> FVar {
    let fail = || -> ! {
        panic!(
            "{who}: cannot materialise constant operand {:?} of type {:?}",
            c.value, c.concretetype
        )
    };
    // `Assembler.emit_const` kind `'int'`: Number / RInt is
    // `cast_primitive(Signed, value)` → `ConstInt`.
    let (kind, lltype) = if let Some((n, lltype)) =
        crate::codewriter::assembler::signed_int_from_number_const(
            &c.value,
            c.concretetype.as_ref(),
        ) {
        (OpKind::ConstInt(n), lltype)
    } else {
        let kind = match c.concretetype.as_ref() {
            Some(LowLevelType::Signed) => match &c.value {
                ConstValue::Int(n) => OpKind::ConstInt(*n),
                // `rstr.py` `new_malloc`: `int_is_true(Constant(malloc_zero_filled, Signed))`.
                // `constfold` leaves the `CDefinedIntSymbolic` alone;
                // `rewrite_op_int_is_true` folds it once this carrier
                // reaches `transform_graph`.
                ConstValue::SpecTag(tag) if *tag == MALLOC_ZERO_FILLED_TAG_ID => {
                    OpKind::ConstSymbolic {
                        tag: *tag,
                        ty: ValueType::Int,
                    }
                }
                _ => return materialize_constant_copy(out, block, c),
            },
            Some(LowLevelType::SignedLongLong) => match &c.value {
                ConstValue::Int(n) => OpKind::ConstInt(*n),
                _ => return materialize_constant_copy(out, block, c),
            },
            Some(LowLevelType::Unsigned | LowLevelType::UnsignedLongLong) => match &c.value {
                ConstValue::UInt(n) => OpKind::ConstUInt(*n),
                ConstValue::Int(n) if *n >= 0 => OpKind::ConstUInt(*n as u64),
                _ => return materialize_constant_copy(out, block, c),
            },
            Some(LowLevelType::Bool) => match &c.value {
                ConstValue::Bool(b) => OpKind::ConstBool(*b),
                _ => return materialize_constant_copy(out, block, c),
            },
            Some(LowLevelType::Char) => match &c.value {
                ConstValue::ByteStr(bytes) if bytes.len() == 1 => {
                    OpKind::ConstInt(i64::from(bytes[0]))
                }
                _ => return materialize_constant_copy(out, block, c),
            },
            Some(LowLevelType::UniChar) => match &c.value {
                ConstValue::UniStr(s) if s.chars().count() == 1 => {
                    OpKind::ConstInt(s.chars().next().expect("one UniChar") as i64)
                }
                _ => return materialize_constant_copy(out, block, c),
            },
            Some(LowLevelType::Float) => match &c.value {
                ConstValue::Float(bits) => OpKind::ConstFloat(*bits),
                _ => return materialize_constant_copy(out, block, c),
            },
            Some(LowLevelType::SingleFloat) => match &c.value {
                ConstValue::Float(bits) => {
                    OpKind::ConstSingleFloat((f64::from_bits(*bits) as f32).to_bits())
                }
                _ => return materialize_constant_copy(out, block, c),
            },
            Some(LowLevelType::Ptr(p)) => match &c.value {
                ConstValue::LLPtr(ptr) if matches!(ptr._obj0_value(), Ok(None)) => {
                    if p._needsgc() {
                        OpKind::ConstRefNull
                    } else {
                        OpKind::ConstInt(0)
                    }
                }
                // Adapter `const_ref_addr_define_hlvalue`: a GCREF `_ptr`
                // whose object is `_ptr_obj::IntCast(a)`. Spine A carries
                // this as `OpKind::ConstRefAddr(a)` (`Assembler.emit_const`
                // kind `'ref'`).
                ConstValue::LLPtr(ptr)
                    if ptr.nonzero()
                        && let Ok(Some(
                            crate::translator::rtyper::lltypesystem::lltype::_ptr_obj::IntCast(a),
                        )) = ptr._obj0_value() =>
                {
                    OpKind::ConstRefAddr(a)
                }
                // `Assembler.emit_const` kind `'ref'`: a prebuilt
                // `Ptr(STR)` from `StringRepr.convert_const` is stored
                // in `constants_r`. The rich-graph carrier is
                // `OpKind::ConstStr` of the `rpy_string.chars` bytes
                // (`prebuilt_str_bytes_and_hash`). UNICODE stays
                // unmaterialised.
                ConstValue::LLPtr(ptr) if is_str_ptr(p) => match prebuilt_str_bytes_and_hash(ptr) {
                    Some((bytes, _)) => OpKind::ConstStr(bytes),
                    None => fail(),
                },
                _ => return materialize_constant_copy(out, block, c),
            },
            // `Assembler.emit_const` (`assembler.py`): Address → int via
            // `adr2int`. A catalogued address is relocated
            // (`ConstIRelocKind::StaticAddr`); otherwise the assembler
            // keeps the raw bits — the same split Spine A's
            // `ConstRefAddr` arm does. `_address::Null` is 0.
            Some(LowLevelType::Address) => match &c.value {
                ConstValue::LLAddress(_address::IntCast(n)) => OpKind::ConstRefAddr(*n),
                ConstValue::LLAddress(_address::Null) => OpKind::ConstInt(0),
                _ => return materialize_constant_copy(out, block, c),
            },
            None => match &c.value {
                ConstValue::Int(n) => OpKind::ConstInt(*n),
                ConstValue::UInt(n) => OpKind::ConstUInt(*n),
                ConstValue::Bool(b) => OpKind::ConstBool(*b),
                _ => return materialize_constant_copy(out, block, c),
            },
            Some(LowLevelType::Void) => fail(),
            Some(_) => return materialize_constant_copy(out, block, c),
        };
        let lltype = match c.concretetype.clone() {
            Some(ty) => ty,
            None => match &c.value {
                ConstValue::Int(_) => LowLevelType::Signed,
                ConstValue::UInt(_) => LowLevelType::Unsigned,
                ConstValue::Bool(_) => LowLevelType::Bool,
                _ => fail(),
            },
        };
        (kind, lltype)
    };
    let result = out
        .push_op_var(block, kind, true)
        .expect("constant op produces a result var");
    // `push_op_var` mints the result with `ConcreteType::Unknown`;
    // stamp the Constant's concretetype so regalloc colours it.
    result.set_concretetype(Some(lltype));
    result
}

/// Kinds with no dedicated `Const*` form (a symbolic `AddressOffset`, a
/// prebuilt GC aggregate): `{kind}_copy(Constant)`, as
/// `flatten.py insert_renamings`. Dedicated forms come first because
/// `assembler.py Assembler.emit_const` pools the operand without an
/// instruction, and a `{kind}_copy` adds one.
fn materialize_constant_copy(
    out: &mut crate::model::FunctionGraph,
    block: BlockId,
    c: &Constant,
) -> FVar {
    let var = out.alloc_value_var();
    if let Some(ty) = c.concretetype.clone() {
        var.set_concretetype(Some(ty));
    }
    let op = crate::codewriter::jtransform::materialize_const_result_copy(var.clone(), c);
    out.block_mut(block).operations.push(op);
    var
}

/// Map a flowspace operand to a model operand `Variable`, materialising
/// constants into typed `Const*` ops pushed ahead of the consumer.
fn materialize(out: &mut crate::model::FunctionGraph, block: BlockId, hlv: &Hlvalue) -> FVar {
    match hlv {
        Hlvalue::Variable(v) => v.clone(),
        Hlvalue::Constant(c) => {
            materialize_constant(out, block, c, "jtransform_opname::lower_graph")
        }
    }
}

/// `_func._name` split used when `graphs_from` has no subject path
/// (`call.py` falls back to the funcobj name). Shared with the opname
/// spine's oopspec walk so `mark_oopspec` keys the same CallPath the
/// leftover `direct_call` Call will guess.
pub(crate) fn callpath_from_func_name(name: &str) -> CallPath {
    CallPath::from_segments(name.split("::").filter(|s| !s.is_empty()))
}

/// Decode a `direct_call` callee operand — the leading funcptr constant.
/// `ConstValue::LLPtr` wraps an `_ptr` whose underlying object is an
/// `_ptr_obj::Func`; its `_func._name` is the callee identity that keys the
/// `CallPath` a helper is registered under.
fn callee_name_from_funcptr(hlv: &Hlvalue) -> String {
    callee_func_from_funcptr(hlv)._name
}

/// Same decode as [`callee_name_from_funcptr`], returning the `_func`
/// so Spine B can resolve by `graph_ref()` (`call.py graphs_from`
/// `funcobj.graph`).
pub(crate) fn callee_func_from_funcptr(
    hlv: &Hlvalue,
) -> crate::translator::rtyper::lltypesystem::lltype::_func {
    let Hlvalue::Constant(c) = hlv else {
        panic!("jtransform_opname::lower_graph: direct_call callee operand is not a constant");
    };
    use crate::translator::rtyper::lltypesystem::lltype::_ptr_obj;
    let ConstValue::LLPtr(ptr) = &c.value else {
        panic!(
            "jtransform_opname::lower_graph: direct_call callee constant is not an LLPtr: {:?}",
            c.value
        );
    };
    match ptr._obj0_value() {
        Ok(Some(_ptr_obj::Func(func))) => func,
        other => panic!(
            "jtransform_opname::lower_graph: direct_call callee funcptr does not resolve to a Func object: {other:?}"
        ),
    }
}

/// `_func._name` of a non-null funcptr `ConstValue::LLPtr`. Shared with
/// `Assembler.emit_const_i_from_const_allow_short` (`assembler.rs`).
pub(crate) fn callee_name_from_funcptr_value(value: &ConstValue) -> String {
    use crate::translator::rtyper::lltypesystem::lltype::_ptr_obj;
    let ConstValue::LLPtr(ptr) = value else {
        panic!(
            "jtransform_opname::lower_graph: direct_call callee constant is not an LLPtr: {:?}",
            value
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

/// Owning `Struct` of a field-access base — `jtransform.py
/// rewrite_op_getfield`: `STRUCT = v_inst.concretetype.TO`.  A resolved
/// `ForwardReference` is unwrapped the way `LowLevelType::short_name` /
/// `Hash` unwrap `LowLevelType::ForwardReference` (`lltype.rs`).
fn owner_struct_of(base: &FVar) -> Option<Struct> {
    match base.concretetype()? {
        LowLevelType::Ptr(ptr) => owner_struct_of_ptr_target(&ptr.TO),
        _ => None,
    }
}

fn owner_struct_of_ptr_target(to: &PtrTarget) -> Option<Struct> {
    match to {
        PtrTarget::Struct(s) => Some(s.clone()),
        PtrTarget::ForwardReference(fr) => resolved_struct_of(fr),
        _ => None,
    }
}

fn resolved_struct_of(fr: &ForwardReference) -> Option<Struct> {
    match fr.resolved() {
        Some(LowLevelType::Struct(s)) => Some(*s),
        Some(_) => None,
        None => panic!(
            "jtransform_opname::owner_struct_of: ForwardReference is unresolved \
             (cpu.fielddescrof needs STRUCT)"
        ),
    }
}

fn owner_struct_of_lltype(ty: &LowLevelType) -> Option<Struct> {
    match ty {
        LowLevelType::Ptr(ptr) => owner_struct_of_ptr_target(&ptr.TO),
        LowLevelType::Struct(s) => Some((**s).clone()),
        LowLevelType::ForwardReference(fr) => resolved_struct_of(fr),
        _ => None,
    }
}

/// The owning struct leaf of a field access — the pointee-struct name of
/// the base pointer's `concretetype` (`Ptr(Struct("stringbuilder"))` →
/// `"stringbuilder"`).  An rtyper tuple uses the lltype spelling
/// (`tuple{n}<s0,…>`) rather than the bare `tuple{n}` leaf.  `None` when
/// the base is not a pointer-to-struct, in which case the layout layer
/// falls back to its type-string heuristic.
fn owner_root_of(base: &FVar) -> Option<String> {
    owner_struct_of(base).as_ref().map(struct_owner_spelling)
}

/// STRUCT identity used as a FieldRead/New owner. An rtyper tuple's
/// identity is `ll_tuple_owner_spelling`; any other struct keeps `_name`.
/// `tuple_struct_items` matching with no item spelling is a lowering gap.
fn struct_owner_spelling(s: &Struct) -> String {
    match crate::translator::rtyper::rtuple::tuple_struct_items(s) {
        None => s._name.clone(),
        Some(items) => {
            if let Some(sp) = crate::codewriter::call::ll_tuple_owner_spelling(s) {
                return sp;
            }
            for lt in &items {
                if crate::codewriter::call::lltype_item_spelling(lt).is_none() {
                    panic!("rtyper tuple item lltype {lt:?} has no spelling");
                }
            }
            panic!("rtyper tuple item lltype has no spelling");
        }
    }
}

/// `jtransform.py is_typeptr_getset`: `op.args[0].concretetype.TO._hints.get('typeptr')`.
fn owner_typeptr_hint_of(base: &FVar) -> bool {
    owner_struct_of(base)
        .is_some_and(|s| matches!(s._hints.get("typeptr"), Some(ConstValue::Bool(true))))
}

/// `FieldDescriptor` for a non-Void `getfield` / `setfield`.  Upstream
/// `rewrite_op_getfield` always has the STRUCT (`cpu.fielddescrof(STRUCT,
/// c_fieldname.value)`); a `Ptr` base without one is refused.  `_hints
/// ['fieldnames']` maps the rtyper's mangled `inst_<attr>` ll name to the
/// Rust layout attribute name.
fn field_descriptor_of(
    base: &FVar,
    ll_field_name: &str,
    opname: &str,
) -> crate::model::FieldDescriptor {
    let owner = owner_struct_of(base);
    if matches!(base.concretetype(), Some(LowLevelType::Ptr(_))) && owner.is_none() {
        panic!(
            "jtransform_opname::lower_graph: {opname} on {ll_field_name} has no STRUCT owner \
             (cpu.fielddescrof(STRUCT, name))"
        );
    }
    let field_name = owner
        .as_ref()
        .map(|s| field_name_for_struct(s, ll_field_name))
        .unwrap_or_else(|| ll_field_name.to_string());
    let owner_root = owner_root_of(base);
    let owner_id = owner_root
        .as_deref()
        .and_then(majit_ir::descr::struct_id_for_name);
    crate::model::FieldDescriptor::new(field_name, owner_root)
        .with_owner_typeptr_hint(owner_typeptr_hint_of(base))
        .with_owner_id(owner_id)
}

fn field_name_for_struct(owner: &Struct, ll_name: &str) -> String {
    if let Some(ConstValue::Dict(map)) = owner._hints.get("fieldnames") {
        if let Some(mapped) = map.get(&ConstValue::byte_str(ll_name)) {
            if let Some(bytes) = mapped.as_byte_str() {
                return String::from_utf8_lossy(bytes).into_owned();
            }
        }
    }
    ll_name.to_string()
}

fn field_descriptor_of_hlvalue(
    inst: &Hlvalue,
    ll_field_name: &str,
    opname: &str,
) -> crate::model::FieldDescriptor {
    match inst {
        Hlvalue::Variable(v) => field_descriptor_of(v, ll_field_name, opname),
        Hlvalue::Constant(c) => {
            let owner = c.concretetype.as_ref().and_then(owner_struct_of_lltype);
            if matches!(c.concretetype, Some(LowLevelType::Ptr(_))) && owner.is_none() {
                panic!(
                    "jtransform_opname::lower_graph: {opname} on {ll_field_name} has no STRUCT owner \
                     (cpu.fielddescrof(STRUCT, name))"
                );
            }
            let field_name = owner
                .as_ref()
                .map(|s| field_name_for_struct(s, ll_field_name))
                .unwrap_or_else(|| ll_field_name.to_string());
            let owner_root = owner.as_ref().map(struct_owner_spelling);
            let owner_id = owner_root
                .as_deref()
                .and_then(majit_ir::descr::struct_id_for_name);
            crate::model::FieldDescriptor::new(field_name, owner_root).with_owner_id(owner_id)
        }
    }
}

/// `Transformer.handle_getfield_typeptr`: `cls = op.args[0].value.typeptr`.
/// Walk `super` the way `rclass` `getattr(inst, 'typeptr')` does.
fn constant_receiver_typeptr(ptr: &_ptr) -> _ptr {
    let mut current = ptr.clone();
    loop {
        match current.getattr("typeptr") {
            Ok(LowLevelValue::Ptr(typeptr)) => return *typeptr,
            Ok(other) => {
                panic!("jtransform_opname::lower_graph: typeptr field is not a Ptr: {other:?}")
            }
            Err(_) => match current.getattr("super") {
                Ok(LowLevelValue::Ptr(super_ptr)) => current = *super_ptr,
                other => panic!(
                    "jtransform_opname::lower_graph: Constant typeptr receiver has no typeptr field ({other:?})"
                ),
            },
        }
    }
}

fn hlvalue_is_void(hlv: &Hlvalue) -> bool {
    match hlv {
        Hlvalue::Variable(v) => matches!(v.concretetype(), Some(LowLevelType::Void)),
        Hlvalue::Constant(c) => matches!(c.concretetype, Some(LowLevelType::Void)),
    }
}

fn var_is_lltype(var: &FVar, ty: &LowLevelType) -> bool {
    var.concretetype().as_ref() == Some(ty)
}

fn interior_field_descriptor(array: &Array, field_name: &str) -> crate::model::FieldDescriptor {
    let LowLevelType::Struct(s) = &array.OF else {
        panic!(
            "jtransform_opname::lower_graph: getinteriorfield ARRAY.OF is not a Struct: {:?}",
            array.OF
        );
    };
    let name = field_name_for_struct(s, field_name);
    let owner_root = struct_owner_spelling(s);
    let owner_id = majit_ir::descr::struct_id_for_name(&owner_root);
    crate::model::FieldDescriptor::new(name, Some(owner_root)).with_owner_id(owner_id)
}

fn chars_array_of_string_like(ty: &LowLevelType) -> Option<Array> {
    let s = owner_struct_of_lltype(ty)?;
    match s.getattr_field_type("chars") {
        Some(LowLevelType::Array(arr)) => Some(*arr),
        _ => None,
    }
}

fn raw_item_descr(ty: &LowLevelType, opname: &str) -> (ValueType, usize, bool) {
    let item_ty = value_type_from_lltype(ty);
    assert!(
        !matches!(item_ty, ValueType::Ref(_)),
        "{opname}_r not supported"
    );
    let (itemsize, unsigned) = crate::translator::rtyper::lltypesystem::rffi::size_and_sign(ty)
        .unwrap_or_else(|err| panic!("jtransform_opname::lower_graph: {opname} {err}"));
    let is_item_signed = !matches!(item_ty, ValueType::Float) && !unsigned;
    (item_ty, itemsize, is_item_signed)
}

fn ptradd_item_bytes(ty: &LowLevelType) -> Option<i64> {
    let of = match ty {
        LowLevelType::Ptr(ptr) => match &ptr.TO {
            PtrTarget::Array(arr) => arr.OF.clone(),
            PtrTarget::ForwardReference(fwd) => match fwd.resolved()? {
                LowLevelType::Array(arr) => arr.OF.clone(),
                _ => return None,
            },
            _ => return None,
        },
        _ => return None,
    };
    match crate::translator::rtyper::lltypesystem::rffi::sizeof(&of) {
        Ok(ConstValue::Int(n)) if n >= 0 => Some(n),
        _ => None,
    }
}

fn free_leftover_flags(flags: &Hlvalue) -> Vec<String> {
    let Hlvalue::Constant(c) = flags else {
        return Vec::new();
    };
    let ConstValue::Dict(map) = &c.value else {
        return Vec::new();
    };
    let mut leftover = Vec::new();
    for (key, _) in map.iter() {
        let name = match key {
            ConstValue::ByteStr(b) => String::from_utf8_lossy(b).into_owned(),
            ConstValue::UniStr(s) => s.clone(),
            other => {
                leftover.push(format!("{other:?}"));
                continue;
            }
        };
        match name.as_str() {
            "flavor" | "track_allocation" => {}
            other => leftover.push(other.to_string()),
        }
    }
    leftover
}

fn emit_leftover_named_call(
    out: &mut crate::model::FunctionGraph,
    block: BlockId,
    op: &crate::flowspace::model::SpaceOperation,
    name: &'static str,
) {
    let args: Vec<FVar> = non_void_args(&op.args)
        .map(|a| materialize(out, block, a))
        .collect();
    let result = expect_var(&op.result);
    let result_ty = value_type_of(&result);
    out.push_op_with_result_var(
        block,
        OpKind::Call {
            target: CallTarget::function_path([name]),
            args: crate::model::call_args(args),
            result_ty,
        },
        result,
    );
}

/// `jtransform.py Transformer.rewrite_op_getinteriorfield`.
fn emit_getinteriorfield(
    out: &mut crate::model::FunctionGraph,
    block: BlockId,
    op: &crate::flowspace::model::SpaceOperation,
) {
    // `Transformer.rewrite_op_getinteriorfield` returns for a Void result.
    if hlvalue_is_void(&op.result) {
        return;
    }
    assert_eq!(
        op.args.len(),
        3,
        "jtransform_opname::lower_graph: getinteriorfield needs 3 args"
    );
    let base = expect_var(&op.args[0]);
    if var_is_lltype(&base, &STRPTR) {
        let index = materialize(out, block, &op.args[2]);
        let result = expect_var(&op.result);
        out.push_op_with_result_var(
            block,
            OpKind::LoweredBlackholeOp {
                opname: "strgetitem".to_string(),
                args: vec![base.into(), index.into()],
            },
            result,
        );
        return;
    }
    if var_is_lltype(&base, &UNICODEPTR) {
        let index = materialize(out, block, &op.args[2]);
        let result = expect_var(&op.result);
        out.push_op_with_result_var(
            block,
            OpKind::LoweredBlackholeOp {
                opname: "unicodegetitem".to_string(),
                args: vec![base.into(), index.into()],
            },
            result,
        );
        return;
    }
    if var_is_lltype(&base, &BYTEARRAYPTR) {
        let index = materialize(out, block, &op.args[2]);
        let result = expect_var(&op.result);
        let array = chars_array_of_string_like(&BYTEARRAYPTR).expect("BYTEARRAY chars Array");
        out.push_op_with_result_var(
            block,
            OpKind::ArrayRead {
                base,
                index,
                item_ty: value_type_of(&result),
                array_type_id: array_type_id_of(&array),
                array: Some(array.clone()),
                nolength: array_nolength(&array),
                gckind: crate::model::ArrayGcKind::Gc,
                pure: array._immutable_field(),
            },
            result,
        );
        return;
    }
    let Some(array) = gc_array_of_operand(&op.args[0]) else {
        panic!(
            "jtransform_opname::lower_graph: getinteriorfield not supported on {:?}",
            base.concretetype()
        );
    };
    let index = materialize(out, block, &op.args[1]);
    let field = interior_field_descriptor(&array, &field_name_of(&op.args[2]));
    let result = expect_var(&op.result);
    out.push_op_with_result_var(
        block,
        OpKind::InteriorFieldRead {
            base,
            index,
            field,
            item_ty: value_type_of(&result),
            array_type_id: array_type_id_of(&array),
            array: Some(array),
        },
        result,
    );
}

/// `jtransform.py Transformer.rewrite_op_setinteriorfield`.
fn emit_setinteriorfield(
    out: &mut crate::model::FunctionGraph,
    block: BlockId,
    op: &crate::flowspace::model::SpaceOperation,
) {
    assert_eq!(
        op.args.len(),
        4,
        "jtransform_opname::lower_graph: setinteriorfield needs 4 args"
    );
    let base = expect_var(&op.args[0]);
    if var_is_lltype(&base, &STRPTR) {
        let index = materialize(out, block, &op.args[2]);
        let value = materialize(out, block, &op.args[3]);
        out.push_op_var(
            block,
            OpKind::LoweredBlackholeOp {
                opname: "strsetitem".to_string(),
                args: vec![base.into(), index.into(), value.into()],
            },
            false,
        );
        return;
    }
    if var_is_lltype(&base, &UNICODEPTR) {
        let index = materialize(out, block, &op.args[2]);
        let value = materialize(out, block, &op.args[3]);
        out.push_op_var(
            block,
            OpKind::LoweredBlackholeOp {
                opname: "unicodesetitem".to_string(),
                args: vec![base.into(), index.into(), value.into()],
            },
            false,
        );
        return;
    }
    if var_is_lltype(&base, &BYTEARRAYPTR) {
        let index = materialize(out, block, &op.args[2]);
        let value = linkarg_from_hlvalue(&op.args[3]);
        let array = chars_array_of_string_like(&BYTEARRAYPTR).expect("BYTEARRAY chars Array");
        out.push_op_var(
            block,
            OpKind::ArrayWrite {
                base,
                index,
                value,
                item_ty: hlvalue_value_type(&op.args[3]),
                array_type_id: array_type_id_of(&array),
                array: Some(array.clone()),
                nolength: array_nolength(&array),
                gckind: crate::model::ArrayGcKind::Gc,
            },
            false,
        );
        return;
    }
    if hlvalue_is_void(&op.args[3]) {
        return;
    }
    let Some(array) = gc_array_of_operand(&op.args[0]) else {
        panic!(
            "jtransform_opname::lower_graph: setinteriorfield not supported on {:?}",
            base.concretetype()
        );
    };
    let index = materialize(out, block, &op.args[1]);
    let field = interior_field_descriptor(&array, &field_name_of(&op.args[2]));
    let value = materialize(out, block, &op.args[3]);
    out.push_op_var(
        block,
        OpKind::InteriorFieldWrite {
            base,
            index,
            field,
            value,
            item_ty: hlvalue_value_type(&op.args[3]),
            array_type_id: array_type_id_of(&array),
            array: Some(array),
        },
        false,
    );
}

/// `jtransform.py Transformer.rewrite_op_getinteriorarraysize`.
fn emit_getinteriorarraysize(
    out: &mut crate::model::FunctionGraph,
    block: BlockId,
    op: &crate::flowspace::model::SpaceOperation,
) {
    assert_eq!(
        op.args.len(),
        2,
        "jtransform_opname::lower_graph: getinteriorarraysize needs 2 args"
    );
    assert_eq!(
        field_name_of(&op.args[1]),
        "chars",
        "jtransform_opname::lower_graph: getinteriorarraysize field is not chars"
    );
    let base = expect_var(&op.args[0]);
    let result = expect_var(&op.result);
    if var_is_lltype(&base, &STRPTR) {
        out.push_op_with_result_var(
            block,
            OpKind::LoweredBlackholeOp {
                opname: "strlen".to_string(),
                args: vec![base.into()],
            },
            result,
        );
        return;
    }
    if var_is_lltype(&base, &UNICODEPTR) {
        out.push_op_with_result_var(
            block,
            OpKind::LoweredBlackholeOp {
                opname: "unicodelen".to_string(),
                args: vec![base.into()],
            },
            result,
        );
        return;
    }
    if var_is_lltype(&base, &BYTEARRAYPTR) {
        let array = chars_array_of_string_like(&BYTEARRAYPTR).expect("BYTEARRAY chars Array");
        out.push_op_with_result_var(
            block,
            OpKind::ArrayLen {
                base,
                array_type_id: array_type_id_of(&array),
                array: Some(array.clone()),
                nolength: array_nolength(&array),
                gckind: crate::model::ArrayGcKind::Gc,
            },
            result,
        );
        return;
    }
    panic!(
        "jtransform_opname::lower_graph: getinteriorarraysize supported type {:?}",
        base.concretetype()
    );
}

/// `jtransform.py Transformer.rewrite_op_raw_load` / `rewrite_op_raw_store`.
fn emit_raw_load_or_store(
    out: &mut crate::model::FunctionGraph,
    block: BlockId,
    op: &crate::flowspace::model::SpaceOperation,
    name: &str,
) {
    let base = materialize(out, block, &op.args[0]);
    let offset = materialize(out, block, &op.args[1]);
    if name == "raw_load" {
        let result = expect_var(&op.result);
        let ty = result
            .concretetype()
            .expect("raw_load result has concretetype");
        let (item_ty, itemsize, is_item_signed) = raw_item_descr(&ty, name);
        out.push_op_with_result_var(
            block,
            OpKind::RawLoad {
                base,
                offset,
                item_ty,
                itemsize,
                is_item_signed,
            },
            result,
        );
    } else {
        let value = materialize(out, block, &op.args[2]);
        let ty = value
            .concretetype()
            .expect("raw_store value has concretetype");
        let (item_ty, itemsize, is_item_signed) = raw_item_descr(&ty, name);
        out.push_op_var(
            block,
            OpKind::RawStore {
                base,
                offset,
                value,
                item_ty,
                itemsize,
                is_item_signed,
            },
            false,
        );
    }
}

/// `jtransform.py Transformer.rewrite_op_direct_ptradd`.
fn emit_direct_ptradd(
    out: &mut crate::model::FunctionGraph,
    block: BlockId,
    op: &crate::flowspace::model::SpaceOperation,
) {
    assert!(
        op.args.len() >= 2,
        "jtransform_opname::lower_graph: direct_ptradd needs ptr and shift"
    );
    let ptr = materialize(out, block, &op.args[0]);
    let mut shift = materialize(out, block, &op.args[1]);
    let result = expect_var(&op.result);
    let ptr_ty = ptr.concretetype();
    let is_ccharp = ptr_ty.as_ref().is_some_and(|ty| ty == &*CCHARP);
    if !is_ccharp {
        let Some(by) = ptr_ty.as_ref().and_then(ptradd_item_bytes) else {
            let args = vec![ptr, shift];
            out.push_op_with_result_var(
                block,
                OpKind::Call {
                    target: CallTarget::function_path(["lltype", "direct_ptradd"]),
                    args: crate::model::call_args(args),
                    result_ty: value_type_of(&result),
                },
                result,
            );
            return;
        };
        let c_by = out
            .push_op_var(block, OpKind::ConstInt(by), true)
            .expect("sizeof const produces a result");
        c_by.set_concretetype(Some(LowLevelType::Signed));
        let prod = out
            .push_op_var(
                block,
                OpKind::BinOp {
                    op: "int_mul".to_string(),
                    lhs: shift,
                    rhs: c_by,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .expect("int_mul produces a result");
        prod.set_concretetype(Some(LowLevelType::Signed));
        shift = prod;
    }
    out.push_op_with_result_var(
        block,
        OpKind::BinOp {
            op: "int_add".to_string(),
            lhs: ptr,
            rhs: shift,
            result_ty: ValueType::Int,
        },
        result,
    );
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

/// `Constant(None, lltype.Void)` — `Transformer.optimize_block.do_rename`
/// (`jtransform.py`) Void branch.
fn void_none_hlvalue() -> Hlvalue {
    Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ))
}

/// `Transformer._do_renaming` (`jtransform.py`): clone the op with renamed
/// args only when some arg is in the block-local Void `renamings`.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn rename_void_op_args(
    op: &crate::flowspace::model::SpaceOperation,
    void_renamings: &HashMap<FVar, Hlvalue>,
) -> Option<crate::flowspace::model::SpaceOperation> {
    if !op.args.iter().any(|arg| match arg {
        Hlvalue::Variable(v) => void_renamings.contains_key(v),
        _ => false,
    }) {
        return None;
    }
    let mut renamed = op.clone();
    for arg in &mut renamed.args {
        if let Hlvalue::Variable(v) = arg {
            if let Some(mapped) = void_renamings.get(v) {
                *arg = mapped.clone();
            }
        }
    }
    Some(renamed)
}

/// `Transformer._do_renaming_on_link` (`jtransform.py`): map a Variable
/// through the block-local Void `renamings`.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn rename_void_hlvalue<'a>(
    hlv: &'a Hlvalue,
    void_renamings: &'a HashMap<FVar, Hlvalue>,
) -> &'a Hlvalue {
    match hlv {
        Hlvalue::Variable(v) => void_renamings.get(v).unwrap_or(hlv),
        _ => hlv,
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

/// `rtyper.py insert_link_conversions` rewrites each Constant link arg
/// with `inputconst(r_a2, a1.value)` into the dest inputarg's repr, so
/// `getkind` of the constant equals the target's. A mismatch is an
/// ill-typed rtyped graph.
fn linkarg_from_hlvalue_matching_target(
    hlv: &Hlvalue,
    target: Option<&FVar>,
    graph_name: &str,
) -> LinkArg {
    if let (Hlvalue::Constant(c), Some(target)) = (hlv, target) {
        assert_rtyped_link_constant_kind(graph_name, c, target);
    }
    linkarg_from_hlvalue(hlv)
}

fn history_kind_name(k: crate::model::ConcreteType) -> &'static str {
    match k {
        crate::model::ConcreteType::Signed => "int",
        crate::model::ConcreteType::GcRef => "ref",
        crate::model::ConcreteType::Float => "float",
        crate::model::ConcreteType::Void => "void",
        crate::model::ConcreteType::Unknown => "unknown",
    }
}

fn assert_rtyped_link_constant_kind(graph_name: &str, c: &Constant, target: &FVar) {
    let Some(const_ty) = c.concretetype.as_ref() else {
        return;
    };
    let Some(target_ty) = target.concretetype() else {
        return;
    };
    let k1 = crate::model::getkind(const_ty);
    let k2 = crate::model::getkind(&target_ty);
    if k1 == crate::model::ConcreteType::Void || k2 == crate::model::ConcreteType::Void {
        return;
    }
    if k1 != k2 {
        panic!(
            "ill-typed rtyped graph {graph_name}: link constant kind {} != target kind {}",
            history_kind_name(k1),
            history_kind_name(k2),
        );
    }
}

/// Map a flowspace exitcase constant to the model `ExitCase`.
/// Inverse of `flowspace_adapter::exitcase_to_hlvalue`: the carrier class
/// HostObject is `ExitCase::ErrorCarrier`, compared by identity.
fn exitcase_from_const(value: &ConstValue, carrier_class: Option<&HostObject>) -> Option<ExitCase> {
    match value {
        ConstValue::Bool(b) => Some(ExitCase::Bool(*b)),
        ConstValue::HostObject(obj) if carrier_class.is_some_and(|carrier| carrier == obj) => {
            Some(ExitCase::ErrorCarrier)
        }
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
            ConstValue::LowLevelType(lltype) => owner_struct_of_lltype(lltype)
                .as_ref()
                .map(struct_owner_spelling)
                .unwrap_or_else(|| {
                    panic!(
                        "jtransform_opname::lower_graph: malloc type operand is not a Struct: {lltype:?}"
                    )
                }),
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
/// (`gc_flavor_const`).  `Transformer.rewrite_op_malloc` lowers the gc
/// flavor to `new(descr)`.
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
    use super::{lower_graph, lower_graph_with_callees, materialize_constant};
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::flowspace::model::{
        Block, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph as FlowGraph, GraphFunc,
        Hlvalue, Link, SpaceOperation, Variable, c_last_exception,
    };
    use crate::model::{ExitCase, ExitSwitch, LinkArg, OpKind, ValueType};
    use crate::translator::rtyper::lltypesystem::lltype::{
        Array, ForwardReference, LowLevelType, NONGCREF, Ptr, PtrTarget, Struct,
    };
    use crate::translator::rtyper::lltypesystem::rbuilder::{
        STRINGBUILDERPTR, build_ll_bool_helper_graph, build_ll_getlength_helper_graph,
    };
    use crate::translator::rtyper::lltypesystem::rbytearray::BYTEARRAYPTR;
    use crate::translator::rtyper::lltypesystem::rffi::{CArray, CArrayPtr, CCHARP};
    use crate::translator::rtyper::lltypesystem::rstr::{
        STRPTR, UNICODEPTR, build_ll_mallocstr_helper_graph, build_ll_search_helper_graph,
        chars_array_ptr_lltype_from_strptr, direct_call_funcobjs, graph_ref_for_funcptr_id,
        struct_lltype_from_strptr,
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
                OpKind::Live { .. } => live += 1,
                OpKind::ConstInt(_) | OpKind::ConstUInt(_) | OpKind::ConstFnAddr { .. } => {}
                other => panic!("unexpected lowered integer-fragment op: {other:?}"),
            }
        }
        assert_eq!(
            unary,
            vec![
                "int_neg",
                "int_force_ge_zero",
                "cast_primitive",
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

    /// `Transformer.rewrite_op_cast_opaque_ptr` (`jtransform.py`) returns
    /// None: the result is aliased to `op.args[0]`, the same `same_as`
    /// rewrite as `rewrite_op_same_as` / `rewrite_op_cast_pointer`.
    #[test]
    fn lower_graph_aliases_cast_opaque_ptr_to_same_as() {
        let p = variable_with_lltype("p", STRPTR.clone());
        let start = Block::shared(vec![Hlvalue::Variable(p.clone())]);
        let returned = variable_with_lltype("returned", STRPTR.clone());
        let graph = FlowGraph::with_return_var(
            "cast_opaque_ptr_alias",
            start.clone(),
            Hlvalue::Variable(returned),
        );
        let casted = variable_with_lltype("casted", STRPTR.clone());
        start.borrow_mut().operations.push(SpaceOperation::new(
            "cast_opaque_ptr",
            vec![Hlvalue::Variable(p.clone())],
            Hlvalue::Variable(casted.clone()),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(casted.clone())],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);
        assert_eq!(startblock_unops(&model), vec!["same_as".to_string()]);
        let op = &model.block(model.startblock).operations[0];
        let OpKind::UnaryOp { operand, .. } = &op.kind else {
            panic!("cast_opaque_ptr must lower to UnaryOp same_as, got {op:?}");
        };
        assert_eq!(op.result.as_ref(), Some(&casted));
        assert_eq!(operand, &p);
        assert_eq!(
            model.block(model.startblock).exits[0].args,
            vec![LinkArg::Value(casted)]
        );
    }

    fn assert_void_lowering_emits_nothing(graph: &FlowGraph) {
        let model = lower_graph(graph);
        for block in &model.blocks {
            for op in &block.operations {
                let is_const = matches!(
                    op.kind,
                    OpKind::ConstInt(_)
                        | OpKind::ConstUInt(_)
                        | OpKind::ConstInt128(_)
                        | OpKind::ConstUInt128(_)
                        | OpKind::ConstBool(_)
                        | OpKind::ConstFloat(_)
                        | OpKind::ConstSingleFloat(_)
                        | OpKind::ConstStr(_)
                        | OpKind::ConstInternedStr(_)
                        | OpKind::ConstFnAddr { .. }
                        | OpKind::ConstVtable { .. }
                );
                let is_copy = match &op.kind {
                    OpKind::LoweredBlackholeOp { opname, .. } => opname.ends_with("_copy"),
                    OpKind::UnaryOp { op, .. } => op.ends_with("_copy") || op == "same_as",
                    _ => false,
                };
                panic!(
                    "void operand must emit nothing and no Const*/_copy, got {:?} (const={is_const}, copy={is_copy})",
                    op.kind
                );
            }
        }
    }

    fn void_field_struct_ptr() -> LowLevelType {
        let s = Struct::gc("VoidField", vec![("empty".into(), LowLevelType::Void)]);
        LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(s),
        }))
    }

    /// `Transformer.rewrite_op_same_as` / `do_rename`: Void is
    /// `Constant(None, lltype.Void)` and allocates no register.
    #[test]
    fn void_same_as_emits_nothing() {
        let n = variable_with_lltype("n", LowLevelType::Signed);
        let start = Block::shared(vec![Hlvalue::Variable(n.clone())]);
        let graph =
            FlowGraph::with_return_var("void_same_as", start.clone(), Hlvalue::Variable(n.clone()));
        let void_res = variable_with_lltype("v", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "same_as",
            vec![constant_with_lltype(ConstValue::None, LowLevelType::Void)],
            Hlvalue::Variable(void_res),
        ));
        close_return(&start, &graph, n);
        assert_void_lowering_emits_nothing(&graph);
    }

    /// `Transformer.rewrite_op_getfield` returns for a Void result.
    #[test]
    fn void_getfield_result_emits_nothing() {
        let v = variable_with_lltype("v", void_field_struct_ptr());
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FlowGraph::with_return_var(
            "void_getfield",
            start.clone(),
            Hlvalue::Variable(v.clone()),
        );
        let void_res = variable_with_lltype("empty", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(v.clone()),
                constant_with_lltype(ConstValue::byte_str("empty"), LowLevelType::Void),
            ],
            Hlvalue::Variable(void_res),
        ));
        close_return(&start, &graph, v);
        assert_void_lowering_emits_nothing(&graph);
    }

    /// `Transformer.rewrite_op_setfield` returns for a Void stored value.
    #[test]
    fn void_setfield_value_emits_nothing() {
        let v = variable_with_lltype("v", void_field_struct_ptr());
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FlowGraph::with_return_var(
            "void_setfield",
            start.clone(),
            Hlvalue::Variable(v.clone()),
        );
        let void_res = variable_with_lltype("set", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(v.clone()),
                constant_with_lltype(ConstValue::byte_str("empty"), LowLevelType::Void),
                constant_with_lltype(ConstValue::None, LowLevelType::Void),
            ],
            Hlvalue::Variable(void_res),
        ));
        close_return(&start, &graph, v);
        assert_void_lowering_emits_nothing(&graph);
    }

    /// `Transformer.rewrite_op_getinteriorfield` returns for a Void result.
    #[test]
    fn void_getinteriorfield_result_emits_nothing() {
        let entry = Struct::new("VOIDENTRY", vec![("empty".into(), LowLevelType::Void)]);
        let array_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(Array::gc(LowLevelType::Struct(Box::new(entry)))),
        }));
        let arr = variable_with_lltype("arr", array_ptr);
        let index = variable_with_lltype("i", LowLevelType::Signed);
        let start = Block::shared(vec![
            Hlvalue::Variable(arr.clone()),
            Hlvalue::Variable(index.clone()),
        ]);
        let graph = FlowGraph::with_return_var(
            "void_getinteriorfield",
            start.clone(),
            Hlvalue::Variable(arr.clone()),
        );
        let void_res = variable_with_lltype("empty", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "getinteriorfield",
            vec![
                Hlvalue::Variable(arr.clone()),
                Hlvalue::Variable(index),
                constant_with_lltype(ConstValue::byte_str("empty"), LowLevelType::Void),
            ],
            Hlvalue::Variable(void_res),
        ));
        close_return(&start, &graph, arr);
        assert_void_lowering_emits_nothing(&graph);
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
    fn lower_graph_float_binop_and_compare_arms() {
        let lhs = variable_with_lltype("lhs", LowLevelType::Float);
        let rhs = variable_with_lltype("rhs", LowLevelType::Float);
        let start = Block::shared(vec![
            Hlvalue::Variable(lhs.clone()),
            Hlvalue::Variable(rhs.clone()),
        ]);
        let returned = variable_with_lltype("returned", LowLevelType::Bool);
        let graph = FlowGraph::with_return_var(
            "float_binop_and_compare",
            start.clone(),
            Hlvalue::Variable(returned),
        );
        let cases = [
            ("float_add", LowLevelType::Float),
            ("float_sub", LowLevelType::Float),
            ("float_mul", LowLevelType::Float),
            ("float_truediv", LowLevelType::Float),
            ("float_lt", LowLevelType::Bool),
            ("float_eq", LowLevelType::Bool),
        ];
        let mut last = None;
        for (index, (opname, result_ty)) in cases.into_iter().enumerate() {
            let result = variable_with_lltype(&format!("f_{index}"), result_ty);
            start.borrow_mut().operations.push(SpaceOperation::new(
                opname,
                vec![
                    Hlvalue::Variable(lhs.clone()),
                    Hlvalue::Variable(rhs.clone()),
                ],
                Hlvalue::Variable(result.clone()),
            ));
            last = Some(result);
        }
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(last.expect("float family is non-empty"))],
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
                other => panic!("float binop/compare must lower to BinOp, got {other:?}"),
            })
            .collect();
        assert_eq!(
            binary,
            vec![
                "float_add",
                "float_sub",
                "float_mul",
                "float_truediv",
                "float_lt",
                "float_eq"
            ]
        );
    }

    #[test]
    fn lower_graph_int_float_cast_arms() {
        let signed = variable_with_lltype("signed", LowLevelType::Signed);
        let float = variable_with_lltype("float", LowLevelType::Float);
        let unsigned = variable_with_lltype("unsigned", LowLevelType::Unsigned);
        let start = Block::shared(vec![
            Hlvalue::Variable(signed.clone()),
            Hlvalue::Variable(float.clone()),
            Hlvalue::Variable(unsigned.clone()),
        ]);
        let returned = variable_with_lltype("returned", LowLevelType::Float);
        let graph = FlowGraph::with_return_var(
            "int_float_casts",
            start.clone(),
            Hlvalue::Variable(returned),
        );
        let cases = [
            (
                "cast_int_to_float",
                Hlvalue::Variable(signed),
                LowLevelType::Float,
            ),
            (
                "cast_float_to_int",
                Hlvalue::Variable(float),
                LowLevelType::Signed,
            ),
            (
                "cast_uint_to_float",
                Hlvalue::Variable(unsigned),
                LowLevelType::Float,
            ),
        ];
        let mut last = None;
        for (index, (opname, operand, result_ty)) in cases.into_iter().enumerate() {
            let result = variable_with_lltype(&format!("cast_{index}"), result_ty);
            start.borrow_mut().operations.push(SpaceOperation::new(
                opname,
                vec![operand],
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
        let unary: Vec<_> = model
            .block(model.startblock)
            .operations
            .iter()
            .map(|op| match &op.kind {
                OpKind::UnaryOp { op, .. } => op.as_str(),
                other => panic!("int/float cast must lower to UnaryOp, got {other:?}"),
            })
            .collect();
        assert_eq!(
            unary,
            vec![
                "cast_int_to_float",
                "cast_float_to_int",
                "cast_uint_to_float"
            ]
        );
    }

    /// `Transformer.rewrite_op_float_is_true`: `float_ne` against 0.0.
    #[test]
    fn lower_graph_rewrites_float_is_true_to_float_ne() {
        let operand = variable_with_lltype("operand", LowLevelType::Float);
        let start = Block::shared(vec![Hlvalue::Variable(operand.clone())]);
        let returned = variable_with_lltype("returned", LowLevelType::Bool);
        let graph = FlowGraph::with_return_var(
            "float_is_true_rewrite",
            start.clone(),
            Hlvalue::Variable(returned),
        );
        let truth = variable_with_lltype("truth", LowLevelType::Bool);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "float_is_true",
            vec![Hlvalue::Variable(operand.clone())],
            Hlvalue::Variable(truth.clone()),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(truth.clone())],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);
        let ops = &model.block(model.startblock).operations;
        assert_eq!(ops.len(), 2, "ConstFloat(0.0) then float_ne; ops={ops:?}");
        match &ops[0].kind {
            OpKind::ConstFloat(bits) => assert_eq!(*bits, 0.0_f64.to_bits()),
            other => panic!("float_is_true must materialise 0.0, got {other:?}"),
        }
        match &ops[1].kind {
            OpKind::BinOp { op, lhs, rhs, .. } => {
                assert_eq!(op, "float_ne");
                assert_eq!(lhs, &operand);
                assert_eq!(rhs, ops[0].result.as_ref().expect("zero result"));
            }
            other => panic!("float_is_true must lower to float_ne, got {other:?}"),
        }
        assert_eq!(ops[1].result.as_ref(), Some(&truth));
    }

    /// `Transformer.rewrite_op_cast_uint_to_float = _do_builtin_call`.
    /// Feed the shared Transformer the same `UnaryOp` the front spine
    /// residualizes; with no helper graph that is the `_ll_1_cast_*`
    /// residual call.
    #[test]
    fn lower_graph_routes_cast_uint_to_float_through_the_builtin_call_rewrite() {
        let operand = variable_with_lltype("operand", LowLevelType::Unsigned);
        let start = Block::shared(vec![Hlvalue::Variable(operand.clone())]);
        let returned = variable_with_lltype("returned", LowLevelType::Float);
        let graph = FlowGraph::with_return_var(
            "lowlevel_cast_uint_to_float",
            start.clone(),
            Hlvalue::Variable(returned),
        );
        let casted = variable_with_lltype("casted", LowLevelType::Float);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "cast_uint_to_float",
            vec![Hlvalue::Variable(operand)],
            Hlvalue::Variable(casted.clone()),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(casted)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);
        assert_eq!(
            startblock_unops(&model),
            vec!["cast_uint_to_float".to_string()]
        );

        use crate::codewriter::call::CallControl;
        use crate::codewriter::codewriter::CodeWriter;
        use crate::codewriter::jtransform::GraphTransformConfig;
        use crate::parse::CallPath;

        let path = CallPath::from_segments(["lowlevel_cast_uint_to_float"]);
        let mut callcontrol = CallControl::new();
        let jitcode = callcontrol.register_opname_helper_graph(path, graph);
        let mut codewriter = CodeWriter::new();
        codewriter.drain_pending_graphs(&mut callcontrol, &GraphTransformConfig::default());
        let body = jitcode
            .try_body()
            .expect("cast_uint_to_float drains through builtin residual call");
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

    /// `RPythonTyper._convert_link` stores the low-level class constant on
    /// the flowspace link. `lower_graph` copies that `llexitcase` onto the
    /// model link (`flatten.py insert_exits`).
    #[test]
    fn lowered_typed_exception_link_carries_the_rtyper_llexitcase() {
        use crate::translator::rtyper::rtyper::exception_args;

        let overflow = exception_args("OverflowError").expect("OverflowError args");
        let a = variable_with_lltype("a", LowLevelType::Signed);
        let b = variable_with_lltype("b", LowLevelType::Signed);
        let start = Block::shared(vec![
            Hlvalue::Variable(a.clone()),
            Hlvalue::Variable(b.clone()),
        ]);
        let ret = variable_with_lltype("result", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "ll_typed_exc_llexitcase",
            start.clone(),
            Hlvalue::Variable(ret),
        );
        let sum = variable_with_lltype("sum", LowLevelType::Signed);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add_ovf",
            vec![Hlvalue::Variable(a), Hlvalue::Variable(b)],
            Hlvalue::Variable(sum.clone()),
        ));
        start.borrow_mut().exitswitch = Some(Hlvalue::Constant(c_last_exception()));
        let rtyper_llexitcase = ConstValue::Int(0x51);
        let handler = Block::shared(vec![]);
        handler.closeblock(vec![
            Link::new(overflow.clone(), Some(graph.exceptblock.clone()), None).into_ref(),
        ]);
        let normal = Link::new(
            vec![Hlvalue::Variable(sum)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        let mut exc = Link::new(vec![], Some(handler), Some(overflow[0].clone()));
        exc.extravars(Some(overflow[0].clone()), Some(overflow[1].clone()));
        exc.llexitcase = Some(Hlvalue::Constant(Constant::new(rtyper_llexitcase.clone())));
        start.closeblock(vec![normal, exc.into_ref()]);

        let model = lower_graph(&graph);
        let start = model.block(model.startblock);
        let exc_exit = start
            .exits
            .iter()
            .find(|l| l.exitcase.is_some())
            .expect("typed OverflowError exit");
        assert_eq!(
            exc_exit.llexitcase.as_ref(),
            Some(&rtyper_llexitcase),
            "model link must carry the rtyper llexitcase, not recompute it"
        );
    }

    /// Inverse of `flowspace_adapter::exitcase_to_hlvalue`: the bookkeeper's
    /// carrier class HostObject lowers to `ExitCase::ErrorCarrier` by identity.
    #[test]
    fn carrier_class_exitcase_lowers_to_the_error_carrier_case() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::flowspace::model::HostObject;
        use std::rc::Rc;

        let bk = Rc::new(Bookkeeper::new());
        bk.set_exception_carrier("error::PyErrorObject");
        let carrier = bk
            .exception_carrier_class()
            .expect("bookkeeper mints the carrier class");
        let a = variable_with_lltype("a", LowLevelType::Signed);
        let start = Block::shared(vec![Hlvalue::Variable(a.clone())]);
        let ret = variable_with_lltype("result", LowLevelType::Signed);
        let graph =
            FlowGraph::with_return_var("ll_carrier_catch", start.clone(), Hlvalue::Variable(ret));
        start.borrow_mut().exitswitch = Some(Hlvalue::Constant(c_last_exception()));
        let handler = Block::shared(vec![]);
        handler.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(a.clone())],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let normal = Link::new(
            vec![Hlvalue::Variable(a.clone())],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        let mut exc = Link::new(
            vec![],
            Some(handler),
            Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                carrier.clone(),
            )))),
        );
        exc.extravars(None, None);
        start.closeblock(vec![normal, exc.into_ref()]);

        let model = lower_graph_with_callees(&graph, &|_| None, Some(&carrier));
        let start = model.block(model.startblock);
        let exc_exit = start
            .exits
            .iter()
            .find(|l| l.exitcase.is_some())
            .expect("carrier exception exit");
        assert_eq!(exc_exit.exitcase, Some(ExitCase::ErrorCarrier));

        let other = HostObject::new_class("error::PyErrorObject", Vec::new());
        let start_other = Block::shared(vec![Hlvalue::Variable(a.clone())]);
        let graph_other = FlowGraph::with_return_var(
            "ll_carrier_catch_other",
            start_other.clone(),
            Hlvalue::Variable(variable_with_lltype("result", LowLevelType::Signed)),
        );
        start_other.borrow_mut().exitswitch = Some(Hlvalue::Constant(c_last_exception()));
        let handler_other = Block::shared(vec![]);
        handler_other.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(a.clone())],
                Some(graph_other.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let normal_other = Link::new(
            vec![Hlvalue::Variable(a)],
            Some(graph_other.returnblock.clone()),
            None,
        )
        .into_ref();
        let mut exc_other = Link::new(
            vec![],
            Some(handler_other),
            Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                other,
            )))),
        );
        exc_other.extravars(None, None);
        start_other.closeblock(vec![normal_other, exc_other.into_ref()]);
        let model_other = lower_graph_with_callees(&graph_other, &|_| None, Some(&carrier));
        let other_exit = model_other
            .block(model_other.startblock)
            .exits
            .iter()
            .find(|l| l.exitcase.is_some())
            .expect("same-name host exit");
        assert_ne!(other_exit.exitcase, Some(ExitCase::ErrorCarrier));
        assert!(matches!(
            other_exit.exitcase,
            Some(ExitCase::Const(ConstValue::HostObject(_)))
        ));
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
    /// (`fielddescrof` panics when `getattr(STRUCT, fieldname)` misses, so
    /// a body-nonempty assertion alone cannot discriminate the layout —
    /// that is the descr test's job.)
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
    /// cleanly through Spine B.  The two `copy_string_contents` `direct_call`s
    /// become `OpKind::Call`s, the source-length reads fuse to `strlen`, and
    /// the cross-Phi `resolve_string` fail-loud is never reached because no
    /// per-char loop (strgetitem/strsetitem) survives.
    #[test]
    fn lower_graph_lowers_strconcat_helper_copy_calls() {
        use crate::model::CallTarget;
        use crate::translator::rtyper::lltypesystem::rstr::{
            STRPTR, build_ll_strconcat_helper_graph,
        };

        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let rtyper = crate::translator::rtyper::rtyper::RPythonTyper::new(&ann);
        let helper = build_ll_strconcat_helper_graph(&rtyper, "ll_strconcat", STRPTR.clone())
            .expect("build strconcat helper");
        let flow = helper.graph.borrow();
        let model = lower_graph(&flow);

        let mut blackhole = Vec::new();
        let mut copy_calls = 0;
        for block in &model.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::LoweredBlackholeOp { opname, .. } => {
                        blackhole.push(opname.clone());
                    }
                    OpKind::Call { target, .. }
                        if *target == CallTarget::function_path(["copy_string_contents"]) =>
                    {
                        copy_calls += 1;
                    }
                    _ => {}
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
            0,
            "no copystrcontent blackhole op"
        );
        assert_eq!(copy_calls, 2, "two copy_string_contents calls");
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

    fn funcptr_named_with_graph(
        name: &str,
    ) -> (
        crate::translator::rtyper::lltypesystem::lltype::_ptr,
        crate::flowspace::model::GraphRef,
    ) {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FuncType, functionptr_for_graph_with_type,
        };
        let start = Block::shared(vec![]);
        let graph = std::rc::Rc::new(std::cell::RefCell::new(FlowGraph::new(name, start)));
        let ft = FuncType {
            args: vec![],
            result: LowLevelType::Signed,
        };
        let ptr = functionptr_for_graph_with_type(&graph, ft);
        (ptr, graph)
    }

    fn flow_graph_direct_calling(
        funcptr: crate::translator::rtyper::lltypesystem::lltype::_ptr,
    ) -> FlowGraph {
        let startblock = Block::shared(vec![]);
        let return_var = variable_with_lltype("out", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "caller_direct_call",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let callee = constant_with_lltype(ConstValue::LLPtr(Box::new(funcptr)), LowLevelType::Void);
        let out = variable_with_lltype("out", LowLevelType::Signed);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![callee],
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
        graph
    }

    fn lowered_direct_call_target(model: &crate::model::FunctionGraph) -> crate::model::CallTarget {
        model
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find_map(|op| match &op.kind {
                OpKind::Call { target, .. } => Some(target.clone()),
                _ => None,
            })
            .expect("direct_call lowers to Call")
    }

    /// `call.py graphs_from` resolves by `funcobj.graph`, so a funcptr whose
    /// `_name` is the rustc-style `a::<Impl>::m` still becomes the subject's
    /// CallPath when the resolver maps that graph.
    #[test]
    fn lower_graph_direct_call_target_is_the_callee_graph_path() {
        let (funcptr, callee_graph) = funcptr_named_with_graph("a::<Impl>::m");
        let flow = flow_graph_direct_calling(funcptr);
        let key = crate::flowspace::model::GraphKey::of(&callee_graph);
        let resolved = crate::parse::CallPath::from_segments(["T", "m"]);
        let model = lower_graph_with_callees(
            &flow,
            &|g| (crate::flowspace::model::GraphKey::of(g) == key).then(|| resolved.clone()),
            None,
        );
        assert_eq!(
            lowered_direct_call_target(&model),
            crate::model::CallTarget::from_call_path(&resolved)
        );
    }

    /// A subject path's `GraphSpecialization` is copied onto the CallTarget
    /// (`CallTarget::from_call_path`).
    #[test]
    fn lower_graph_direct_call_keeps_callee_specialization() {
        let (funcptr, callee_graph) = funcptr_named_with_graph("a::<Impl>::m");
        let flow = flow_graph_direct_calling(funcptr);
        let key = crate::flowspace::model::GraphKey::of(&callee_graph);
        let resolved = crate::parse::CallPath {
            segments: vec!["T".into(), "m".into()],
            specialization: crate::parse::GraphSpecialization::AccessDirect,
        };
        let model = lower_graph_with_callees(
            &flow,
            &|g| (crate::flowspace::model::GraphKey::of(g) == key).then(|| resolved.clone()),
            None,
        );
        assert_eq!(
            lowered_direct_call_target(&model),
            crate::model::CallTarget::from_call_path(&resolved)
        );
    }

    /// `ll_*` helpers are not two-phase subjects: resolver `None` keeps
    /// `_func._name` segments.
    #[test]
    fn lower_graph_direct_call_without_subject_graph_uses_func_name() {
        let (funcptr, _callee_graph) = funcptr_named_with_graph("a::<Impl>::m");
        let flow = flow_graph_direct_calling(funcptr);
        let model = lower_graph_with_callees(&flow, &|_| None, None);
        assert_eq!(
            lowered_direct_call_target(&model),
            crate::model::CallTarget::function_path(["a", "<Impl>", "m"])
        );
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

    /// `build_ll_call_lookup_function_helper_graph` calls each width's
    /// `ll_dict_lookup__T` with a Void `T` (`lowlevel_type_const(TYPE_*)`).
    /// Lowering drops that operand rather than materialising it. The else
    /// arm is `ll_dict_create_initial_index(d)`.
    #[test]
    fn lower_graph_lowers_ll_call_lookup_function_dropping_void_t() {
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, functionptr};
        use crate::translator::rtyper::lltypesystem::rordereddict::{
            build_ll_call_lookup_function_helper_graph, index_width_rows,
        };

        let dict_ptr = STRPTR.clone();
        let key_lltype = LowLevelType::Signed;
        let create_ft = FuncType {
            args: vec![dict_ptr.clone()],
            result: LowLevelType::Void,
        };
        let create_fn = functionptr(
            create_ft,
            "ll_dict_create_initial_index",
            None,
            Some("ll_dict_create_initial_index".to_string()),
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
        let lookup_fns = index_width_rows()
            .into_iter()
            .map(|(func, t, _)| {
                let name = format!("ll_dict_lookup__{}", t.short_name());
                let lookup_fn = functionptr(lookup_ft.clone(), &name, None, Some(name.clone()));
                (
                    func,
                    t,
                    Constant::with_concretetype(
                        ConstValue::LLPtr(Box::new(lookup_fn)),
                        LowLevelType::Void,
                    ),
                )
            })
            .collect();
        let helper = build_ll_call_lookup_function_helper_graph(
            "ll_call_lookup_function",
            dict_ptr,
            key_lltype,
            lookup_fns,
            Constant::with_concretetype(ConstValue::LLPtr(Box::new(create_fn)), LowLevelType::Void),
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
        let widths = index_width_rows();
        assert_eq!(
            calls.len(),
            widths.len() + 1,
            "one lookup per width plus create_initial_index, got {calls:?}"
        );
        for (_, t, _) in &widths {
            let name = format!("ll_dict_lookup__{}", t.short_name());
            assert!(
                calls.iter().any(|(target, argc)| {
                    *target == crate::model::CallTarget::function_path([name.as_str()])
                        && *argc == 4
                }),
                "lookup keeps d, key, hash, flag and drops Void T ({name}), got {calls:?}"
            );
        }
        assert!(
            calls.iter().any(|(target, argc)| {
                *target == crate::model::CallTarget::function_path(["ll_dict_create_initial_index"])
                    && *argc == 1
            }),
            "else arm is ll_dict_create_initial_index(d), got {calls:?}"
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

    /// `mallocstr` (`rstr.py` `new_malloc`) guards `hash = 0` on
    /// `int_is_true(Constant(malloc_zero_filled, Signed))`.
    /// `lower_graph` keeps that `int_is_true`, the `ConstSymbolic`
    /// operand, and the branch. `transform_graph` runs
    /// `rewrite_op_int_is_true` then `fold_constant_exitswitch`, so the
    /// store, the symbolic, and the residual branch are gone.
    #[test]
    fn lower_graph_folds_mallocstr_hash_store_away() {
        use crate::codewriter::jtransform::{GraphTransformConfig, transform_graph};

        let helper =
            build_ll_mallocstr_helper_graph("mallocstr", STRPTR.clone()).expect("mallocstr helper");
        let model = lower_graph(&helper.graph.borrow());
        assert_mallocstr_lowered_keeps_int_is_true_branch(&model);
        let transformed = transform_graph(&model, &GraphTransformConfig::default());
        assert_mallocstr_fold_dropped_hash_store(&transformed.graph);
    }

    fn assert_mallocstr_lowered_keeps_int_is_true_branch(model: &crate::model::FunctionGraph) {
        use crate::translator::backendopt::constfold::MALLOC_ZERO_FILLED_TAG_ID;
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

        let mut symbolic = None;
        let mut truth = None;
        let mut saw_hash = false;
        let mut saw_branch = false;
        let mut saw_newstr = false;
        for block in &model.blocks {
            if let Some(ExitSwitch::Value(switch)) = &block.exitswitch {
                assert_eq!(
                    block.exits.len(),
                    2,
                    "malloc_zero_filled branch has two links"
                );
                let cases: Vec<_> = block
                    .exits
                    .iter()
                    .map(|link| link.exitcase.clone())
                    .collect();
                assert!(
                    cases.contains(&Some(ExitCase::Bool(true)))
                        && cases.contains(&Some(ExitCase::Bool(false))),
                    "true link joins, false link stores hash"
                );
                assert!(
                    truth.is_none(),
                    "mallocstr has one malloc_zero_filled branch"
                );
                truth = Some(switch.clone());
                saw_branch = true;
            }
            for op in &block.operations {
                if let OpKind::ConstSymbolic {
                    tag,
                    ty: ValueType::Int,
                } = &op.kind
                    && *tag == MALLOC_ZERO_FILLED_TAG_ID
                {
                    let result = op.result.clone().expect("ConstSymbolic result");
                    assert_eq!(result.concretetype(), Some(LowLevelType::Signed));
                    symbolic = Some(result);
                }
                if matches!(
                    &op.kind,
                    OpKind::FieldWrite { field, .. } if field.name == "hash"
                ) {
                    saw_hash = true;
                }
                if matches!(
                    &op.kind,
                    OpKind::LoweredBlackholeOp { opname, .. } if opname == "newstr"
                ) {
                    saw_newstr = true;
                }
            }
        }
        let symbolic = symbolic.expect("lower_graph carries malloc_zero_filled as ConstSymbolic");
        let truth = truth.expect("lower_graph keeps the malloc_zero_filled branch");
        let truth_op = model
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .find(|op| op.result.as_ref() == Some(&truth))
            .expect("int_is_true defines the exitswitch");
        assert!(
            matches!(
                &truth_op.kind,
                OpKind::UnaryOp { op, operand, result_ty }
                    if op == "int_is_true"
                        && operand == &symbolic
                        && *result_ty == ValueType::Int
            ),
            "int_is_true reads the Signed ConstSymbolic"
        );
        assert_eq!(truth.concretetype(), Some(LowLevelType::Bool));
        assert!(saw_newstr, "malloc_varsize of STR lowers to newstr");
        assert!(
            saw_hash,
            "lower_graph keeps the hash store on the false arm"
        );
        assert!(saw_branch);
    }

    fn assert_mallocstr_fold_dropped_hash_store(model: &crate::model::FunctionGraph) {
        let mut saw_newstr = false;
        for block in &model.blocks {
            assert!(
                block.exitswitch.is_none(),
                "folded mallocstr keeps no symbolic exitswitch"
            );
            assert!(
                block.exits.len() <= 1,
                "folded mallocstr keeps no residual branch"
            );
            for op in &block.operations {
                assert!(
                    !matches!(op.kind, OpKind::ConstSymbolic { .. }),
                    "folded mallocstr keeps no ConstSymbolic"
                );
                assert!(
                    !matches!(
                        &op.kind,
                        OpKind::FieldWrite { field, .. } if field.name == "hash"
                    ),
                    "folded mallocstr does not setfield hash"
                );
                if matches!(
                    &op.kind,
                    OpKind::LoweredBlackholeOp { opname, .. } if opname == "newstr"
                ) {
                    saw_newstr = true;
                }
            }
        }
        assert!(saw_newstr, "malloc_varsize of STR lowers to newstr");
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
                        array: _,
                    } => {
                        assert_eq!(length, &n);
                        assert_eq!(*item_ty, ValueType::Void);
                        assert_eq!(array_type_id.as_deref(), Some("GcArray<()>"));
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
                        array: _,
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
                        array: _,
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
                        array: _,
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
                        array: _,
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

    /// Flags dict `{flavor: raw, ...}` in the same shape as `gc_flavor_const`.
    fn raw_flavor_const(extra: &[(&str, ConstValue)]) -> Hlvalue {
        use indexmap::IndexMap;
        let mut flags =
            IndexMap::from([(ConstValue::byte_str("flavor"), ConstValue::byte_str("raw"))]);
        for (key, value) in extra {
            flags.insert(ConstValue::byte_str(*key), value.clone());
        }
        constant_with_lltype(
            ConstValue::Dict(std::sync::Arc::new(flags)),
            LowLevelType::Void,
        )
    }

    /// `Transformer.rewrite_op_getarrayitem` / `rewrite_op_setarrayitem`:
    /// a standalone raw `CArray(Signed)` is `getarrayitem_raw_*` /
    /// `setarrayitem_raw_*`, never the STR/UNICODE `chars` fusion.
    #[test]
    fn lower_graph_lowers_raw_array_getarrayitem_and_setarrayitem() {
        let array_ptr = CArrayPtr(LowLevelType::Signed);
        let p = variable_with_lltype("p", array_ptr);
        let i = variable_with_lltype("i", LowLevelType::Signed);
        let startblock = Block::shared(vec![
            Hlvalue::Variable(p.clone()),
            Hlvalue::Variable(i.clone()),
        ]);
        let return_var = variable_with_lltype("x", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "ll_test_raw_array_item_ops",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let x = variable_with_lltype("x", LowLevelType::Signed);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getarrayitem",
            vec![Hlvalue::Variable(p.clone()), Hlvalue::Variable(i.clone())],
            Hlvalue::Variable(x.clone()),
        ));
        let set = variable_with_lltype("set", LowLevelType::Void);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(p),
                Hlvalue::Variable(i),
                Hlvalue::Variable(x.clone()),
            ],
            Hlvalue::Variable(set),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(x)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);
        let mut reads = 0;
        let mut writes = 0;
        for block in &model.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::ArrayRead {
                        item_ty,
                        array_type_id,
                        array: _,
                        nolength,
                        pure,
                        ..
                    } => {
                        assert_eq!(*item_ty, ValueType::Int);
                        assert_eq!(array_type_id.as_deref(), Some("Vec<isize>"));
                        assert!(*nolength);
                        assert!(!*pure);
                        reads += 1;
                    }
                    OpKind::ArrayWrite {
                        item_ty,
                        array_type_id,
                        array: _,
                        nolength,
                        ..
                    } => {
                        assert_eq!(*item_ty, ValueType::Int);
                        assert_eq!(array_type_id.as_deref(), Some("Vec<isize>"));
                        assert!(*nolength);
                        writes += 1;
                    }
                    OpKind::LoweredBlackholeOp { opname, .. } => {
                        panic!("raw array must not fuse to {opname}")
                    }
                    other => panic!("unexpected op: {other:?}"),
                }
            }
        }
        assert_eq!(reads, 1);
        assert_eq!(writes, 1);
    }

    /// `Transformer.rewrite_op_malloc_varsize` flavor `raw` first:
    /// `_rewrite_raw_malloc` / `raw_malloc_varsize`.
    #[test]
    fn lower_graph_lowers_raw_malloc_varsize_to_raw_malloc() {
        let n = variable_with_lltype("n", LowLevelType::Signed);
        let array_ty = LowLevelType::Array(Box::new(CArray(LowLevelType::Char)));
        let array_ptr = CArrayPtr(LowLevelType::Char);

        let lower = |flags: Hlvalue| {
            let startblock = Block::shared(vec![Hlvalue::Variable(n.clone())]);
            let return_var = variable_with_lltype("p", array_ptr.clone());
            let graph = FlowGraph::with_return_var(
                "ll_test_raw_malloc_varsize",
                startblock.clone(),
                Hlvalue::Variable(return_var),
            );
            let p = variable_with_lltype("p", array_ptr.clone());
            startblock.borrow_mut().operations.push(SpaceOperation::new(
                "malloc_varsize",
                vec![
                    lowlevel_type_const(array_ty.clone()),
                    flags,
                    Hlvalue::Variable(n.clone()),
                ],
                Hlvalue::Variable(p.clone()),
            ));
            startblock.closeblock(vec![
                Link::new(
                    vec![Hlvalue::Variable(p)],
                    Some(graph.returnblock.clone()),
                    None,
                )
                .into_ref(),
            ]);
            lower_graph(&graph)
        };

        let assert_raw = |model: crate::model::FunctionGraph, expect_zero: bool| {
            let mut found = false;
            for block in &model.blocks {
                for op in &block.operations {
                    match &op.kind {
                        OpKind::RawMalloc {
                            TYPE,
                            length,
                            zero,
                            track_allocation,
                            ..
                        } => {
                            assert!(TYPE.is_some(), "TYPE: Some(..), got {TYPE:?}");
                            assert!(length.is_some(), "length: Some(_), got {length:?}");
                            assert_eq!(*zero, expect_zero);
                            assert!(*track_allocation);
                            found = true;
                        }
                        other => {
                            panic!("raw malloc_varsize must lower to RawMalloc, got {other:?}")
                        }
                    }
                }
            }
            assert!(found, "missing RawMalloc");
        };

        assert_raw(lower(raw_flavor_const(&[])), false);
        assert_raw(
            lower(raw_flavor_const(&[("zero", ConstValue::Bool(true))])),
            true,
        );
    }

    /// `Transformer.rewrite_op_malloc` flavor `raw`:
    /// `_rewrite_raw_malloc` / `raw_malloc_fixedsize`.
    #[test]
    fn lower_graph_lowers_raw_malloc_fixedsize_to_raw_malloc() {
        let raw_struct = LowLevelType::Struct(Box::new(Struct::new(
            "raw_tuple",
            vec![("f".into(), LowLevelType::Signed)],
        )));
        let struct_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(match &raw_struct {
                LowLevelType::Struct(s) => s.as_ref().clone(),
                _ => unreachable!(),
            }),
        }));
        let startblock = Block::shared(vec![]);
        let return_var = variable_with_lltype("p", struct_ptr.clone());
        let graph = FlowGraph::with_return_var(
            "ll_test_raw_malloc_fixedsize",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let p = variable_with_lltype("p", struct_ptr);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "malloc",
            vec![lowlevel_type_const(raw_struct), raw_flavor_const(&[])],
            Hlvalue::Variable(p.clone()),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(p)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let model = lower_graph(&graph);
        let mut found = false;
        for block in &model.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::RawMalloc {
                        TYPE,
                        length,
                        zero,
                        track_allocation,
                        ..
                    } => {
                        assert!(TYPE.is_some());
                        assert!(length.is_none(), "fixed-size raw malloc has length: None");
                        assert!(!*zero);
                        assert!(*track_allocation);
                        found = true;
                    }
                    other => panic!("raw malloc must lower to RawMalloc, got {other:?}"),
                }
            }
        }
        assert!(found, "missing RawMalloc");
    }

    /// `Transformer.rewrite_op_getarrayitem`: a raw ARRAY of GC pointers
    /// is `getarrayitem_raw_r not supported`.
    #[test]
    #[should_panic(expected = "getarrayitem_raw_r not supported")]
    fn lower_graph_getarrayitem_raw_r_is_not_supported() {
        let array_ptr = CArrayPtr(STRPTR.clone());
        let p = variable_with_lltype("p", array_ptr);
        let i = variable_with_lltype("i", LowLevelType::Signed);
        let startblock = Block::shared(vec![
            Hlvalue::Variable(p.clone()),
            Hlvalue::Variable(i.clone()),
        ]);
        let return_var = variable_with_lltype("x", STRPTR.clone());
        let graph = FlowGraph::with_return_var(
            "ll_test_raw_array_ref_getitem",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let x = variable_with_lltype("x", STRPTR.clone());
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getarrayitem",
            vec![Hlvalue::Variable(p), Hlvalue::Variable(i)],
            Hlvalue::Variable(x.clone()),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(x)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let _model = lower_graph(&graph);
    }

    /// `ll_newlist` / `ll_append` graphs `_twoway_preprocess` uses (the
    /// `_search` helper chain) lower without treating the items `GcArray`
    /// as a string allocation.
    #[test]
    fn lower_graph_lowers_twoway_list_helpers() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
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

    /// `Transformer._rewrite_op_cond_call` on
    /// `jit_conditional_call(cond, f, *args)`.
    fn lower_jit_conditional_call(
        extra: Vec<crate::flowspace::model::Variable>,
    ) -> crate::model::FunctionGraph {
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, functionptr};

        let cond = variable_with_lltype("cond", LowLevelType::Bool);
        let mut inputargs = vec![Hlvalue::Variable(cond.clone())];
        inputargs.extend(extra.iter().cloned().map(Hlvalue::Variable));
        let startblock = Block::shared(inputargs);
        let return_var = variable_with_lltype("out", LowLevelType::Void);
        let graph = FlowGraph::with_return_var(
            "ll_test_conditional_call",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let ft = FuncType {
            args: extra
                .iter()
                .map(|v| v.concretetype().unwrap_or(LowLevelType::Void))
                .collect(),
            result: LowLevelType::Void,
        };
        let funcptr = functionptr(ft, "compute", None, Some("compute".to_string()));
        let callee = constant_with_lltype(ConstValue::LLPtr(Box::new(funcptr)), LowLevelType::Void);
        let out = variable_with_lltype("out", LowLevelType::Void);
        let mut args = vec![Hlvalue::Variable(cond), callee];
        args.extend(extra.into_iter().map(Hlvalue::Variable));
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "jit_conditional_call",
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

    /// `_rewrite_op_cond_call` counts Void operands: five function
    /// arguments of which one is Void are still `len(op.args) > 4 + 2`.
    #[test]
    #[should_panic(expected = "Conditional call does not support floats or more than 4 arguments")]
    fn cond_call_rejects_five_args_even_if_one_is_void() {
        let extra: Vec<_> = (0..4)
            .map(|i| variable_with_lltype(&format!("a{i}"), LowLevelType::Signed))
            .chain(std::iter::once(variable_with_lltype(
                "v",
                LowLevelType::Void,
            )))
            .collect();
        let _ = lower_jit_conditional_call(extra);
    }

    /// Four function arguments and no Void operand lower to
    /// `conditional_call4`.
    #[test]
    fn cond_call_accepts_four_args_plus_none_void() {
        let extra: Vec<_> = (0..4)
            .map(|i| variable_with_lltype(&format!("a{i}"), LowLevelType::Signed))
            .collect();
        let model = lower_jit_conditional_call(extra);
        assert_eq!(
            cond_elidable_call_argc(&model, "conditional_call4"),
            vec![6],
            "one conditional_call4 Call (cond, fn, a0, a1, a2, a3)"
        );
    }

    /// `_rewrite_op_cond_call` rejects a float operand on the original
    /// `jit_conditional_call_value` value, before Void removal.
    #[test]
    #[should_panic(expected = "Conditional call does not support floats or more than 4 arguments")]
    fn cond_call_value_rejects_a_float_value() {
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, functionptr};

        let value = variable_with_lltype("h", LowLevelType::Float);
        let startblock = Block::shared(vec![Hlvalue::Variable(value.clone())]);
        let return_var = variable_with_lltype("out", LowLevelType::Float);
        let graph = FlowGraph::with_return_var(
            "ll_test_conditional_call_value_float",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let ft = FuncType {
            args: vec![],
            result: LowLevelType::Float,
        };
        let funcptr = functionptr(ft, "compute", None, Some("compute".to_string()));
        let callee = constant_with_lltype(ConstValue::LLPtr(Box::new(funcptr)), LowLevelType::Void);
        let out = variable_with_lltype("out", LowLevelType::Float);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "jit_conditional_call_value",
            vec![Hlvalue::Variable(value), callee],
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
        let _ = lower_graph(&graph);
    }

    fn nullptr_gc_struct() -> (Constant, LowLevelType) {
        use crate::translator::rtyper::lltypesystem::lltype::nullptr;
        let container = LowLevelType::Struct(Box::new(Struct::gc("S", vec![])));
        let ptr_ty = LowLevelType::Ptr(Box::new(
            Ptr::from_container_type(container.clone()).expect("Ptr(GcStruct)"),
        ));
        let null = nullptr(container).expect("nullptr(GcStruct)");
        (
            Constant::with_concretetype(ConstValue::LLPtr(Box::new(null)), ptr_ty.clone()),
            ptr_ty,
        )
    }

    #[test]
    fn materialize_constant_null_gc_ptr_is_const_ref_null() {
        let (c, ptr_ty) = nullptr_gc_struct();
        let mut out = crate::model::FunctionGraph::new("materialize_constant_null_gc_ptr");
        let start = out.startblock;
        let result = materialize_constant(&mut out, start, &c, "test");
        let ops = &out.block(out.startblock).operations;
        assert_eq!(ops.len(), 1);
        assert!(
            matches!(ops[0].kind, OpKind::ConstRefNull),
            "null GC Ptr Constant becomes ConstRefNull: {:?}",
            ops[0].kind
        );
        assert_eq!(result.concretetype(), Some(ptr_ty));
    }

    #[test]
    fn materialize_constant_char_is_const_int() {
        let c = Constant::with_concretetype(ConstValue::byte_str([b'a']), LowLevelType::Char);
        let mut out = crate::model::FunctionGraph::new("materialize_constant_char");
        let start = out.startblock;
        let result = materialize_constant(&mut out, start, &c, "test");
        let ops = &out.block(out.startblock).operations;
        assert_eq!(ops.len(), 1);
        assert!(
            matches!(ops[0].kind, OpKind::ConstInt(97)),
            "Char b'a' becomes ConstInt(97): {:?}",
            ops[0].kind
        );
        assert_eq!(result.concretetype(), Some(LowLevelType::Char));
    }

    #[test]
    fn materialize_constant_float_is_const_float() {
        let c = Constant::with_concretetype(ConstValue::float(1.5), LowLevelType::Float);
        let mut out = crate::model::FunctionGraph::new("materialize_constant_float");
        let start = out.startblock;
        let result = materialize_constant(&mut out, start, &c, "test");
        let ops = &out.block(out.startblock).operations;
        assert_eq!(ops.len(), 1);
        assert!(
            matches!(ops[0].kind, OpKind::ConstFloat(bits) if bits == 1.5f64.to_bits()),
            "Float 1.5 becomes ConstFloat: {:?}",
            ops[0].kind
        );
        assert_eq!(result.concretetype(), Some(LowLevelType::Float));
    }

    fn r_uchar_const(n: i128) -> (Constant, LowLevelType) {
        use crate::translator::rtyper::lltypesystem::rffi;
        let uchar = (*rffi::UCHAR).clone();
        let class = match &uchar {
            LowLevelType::Number(num) => num._type,
            other => panic!("UCHAR is a Number, got {other:?}"),
        };
        (
            Constant::with_concretetype(ConstValue::rint(class, n), uchar.clone()),
            uchar,
        )
    }

    /// `Assembler.emit_const` kind `'int'`: `cast_primitive(Signed, r_uchar(200))`
    /// is `ConstInt(200)` and keeps the Number concretetype.
    #[test]
    fn materialize_constant_rint_uchar_is_const_int() {
        let (c, uchar) = r_uchar_const(200);
        let mut out = crate::model::FunctionGraph::new("materialize_constant_rint_uchar");
        let start = out.startblock;
        let result = materialize_constant(&mut out, start, &c, "test");
        let ops = &out.block(out.startblock).operations;
        assert_eq!(ops.len(), 1);
        assert!(
            matches!(ops[0].kind, OpKind::ConstInt(200)),
            "r_uchar(200) becomes ConstInt(200): {:?}",
            ops[0].kind
        );
        assert_eq!(result.concretetype(), Some(uchar));
    }

    /// Without a concretetype, the Number lltype is `build_number` of the
    /// RInt class (`UCHAR` is already interned by rffi setup).
    #[test]
    fn materialize_constant_rint_without_concretetype_uses_build_number() {
        use crate::translator::rtyper::lltypesystem::lltype::build_number;
        use crate::translator::rtyper::lltypesystem::rffi;
        let uchar = (*rffi::UCHAR).clone();
        let class = match &uchar {
            LowLevelType::Number(num) => num._type,
            other => panic!("UCHAR is a Number, got {other:?}"),
        };
        let c = Constant::new(ConstValue::rint(class, 200));
        let mut out = crate::model::FunctionGraph::new("materialize_constant_rint_no_lltype");
        let start = out.startblock;
        let result = materialize_constant(&mut out, start, &c, "test");
        let ops = &out.block(out.startblock).operations;
        assert_eq!(ops.len(), 1);
        assert!(
            matches!(ops[0].kind, OpKind::ConstInt(200)),
            "r_uchar(200) becomes ConstInt(200): {:?}",
            ops[0].kind
        );
        let expected = build_number(None, crate::annotator::model::KnownType::RInt(class))
            .expect("UCHAR is interned");
        assert_eq!(result.concretetype(), Some(expected));
    }

    /// `Assembler.emit_const` kind `'ref'`: `StringRepr.convert_const`
    /// prebuilt `Ptr(STR)` (`const_str_cache_llstr`) is `ConstStr` of
    /// the `rpy_string.chars` bytes.
    #[test]
    fn lower_graph_materializes_prebuilt_str_ptr_as_const_str() {
        let start = Block::shared(vec![]);
        let returned = variable_with_lltype("returned", STRPTR.clone());
        let graph = FlowGraph::with_return_var(
            "prebuilt_str_const",
            start.clone(),
            Hlvalue::Variable(returned.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "same_as",
            vec![strptr_const()],
            Hlvalue::Variable(returned.clone()),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(returned.clone())],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let model = lower_graph(&graph);
        let ops = &model.block(model.startblock).operations;
        assert!(
            ops.iter()
                .any(|op| matches!(&op.kind, OpKind::ConstStr(b) if b == b"hello")),
            "prebuilt Ptr(STR) becomes ConstStr(b\"hello\"): {ops:?}"
        );
    }

    fn gcref_static_address_const(addr: i64) -> (Constant, LowLevelType) {
        use crate::translator::rtyper::lltypesystem::lltype::{_ptr, _ptr_obj, GCREF};
        let LowLevelType::Ptr(gcref_t) = GCREF.clone() else {
            panic!("GCREF must be a Ptr");
        };
        let p = _ptr::new_with_solid(*gcref_t, Ok(Some(_ptr_obj::IntCast(addr))), true);
        (
            Constant::with_concretetype(ConstValue::LLPtr(Box::new(p)), GCREF.clone()),
            GCREF.clone(),
        )
    }

    #[test]
    fn materialize_constant_gcref_static_address_is_const_ref_addr() {
        let (c, ptr_ty) = gcref_static_address_const(0x1000);
        let mut out = crate::model::FunctionGraph::new("materialize_constant_gcref_static");
        let start = out.startblock;
        let result = materialize_constant(&mut out, start, &c, "test");
        let ops = &out.block(out.startblock).operations;
        assert_eq!(ops.len(), 1);
        assert!(
            matches!(ops[0].kind, OpKind::ConstRefAddr(0x1000)),
            "GCREF IntCast static address becomes ConstRefAddr: {:?}",
            ops[0].kind
        );
        assert_eq!(result.concretetype(), Some(ptr_ty));
    }

    /// `Assembler.emit_const` Address → `adr2int`; Spine A carries
    /// `cast_int_to_adr(n)` (`_address::IntCast(n)`) as `ConstRefAddr(n)`.
    /// `_address::Null` stays `ConstInt(0)`.
    #[test]
    fn materialize_constant_address_intcast_is_const_ref_addr() {
        use crate::translator::rtyper::lltypesystem::lltype::_address;
        let c = Constant::with_concretetype(
            ConstValue::LLAddress(_address::IntCast(42)),
            LowLevelType::Address,
        );
        let mut out = crate::model::FunctionGraph::new("materialize_constant_address_intcast");
        let start = out.startblock;
        let result = materialize_constant(&mut out, start, &c, "test");
        let ops = &out.block(out.startblock).operations;
        assert_eq!(ops.len(), 1);
        assert!(
            matches!(ops[0].kind, OpKind::ConstRefAddr(42)),
            "Address IntCast(42) becomes ConstRefAddr(42): {:?}",
            ops[0].kind
        );
        assert_eq!(result.concretetype(), Some(LowLevelType::Address));

        let null = Constant::with_concretetype(
            ConstValue::LLAddress(_address::Null),
            LowLevelType::Address,
        );
        let mut out = crate::model::FunctionGraph::new("materialize_constant_address_null");
        let start = out.startblock;
        let result = materialize_constant(&mut out, start, &null, "test");
        let ops = &out.block(out.startblock).operations;
        assert_eq!(ops.len(), 1);
        assert!(
            matches!(ops[0].kind, OpKind::ConstInt(0)),
            "Address Null becomes ConstInt(0): {:?}",
            ops[0].kind
        );
        assert_eq!(result.concretetype(), Some(LowLevelType::Address));
    }

    #[test]
    #[should_panic(expected = "ill-typed rtyped graph")]
    fn lower_graph_link_constant_kind_mismatch_panics() {
        let start = Block::shared(vec![]);
        let returned = variable_with_lltype("returned", STRPTR.clone());
        let graph = FlowGraph::with_return_var(
            "link_kind_mismatch",
            start.clone(),
            Hlvalue::Variable(returned),
        );
        start.closeblock(vec![
            Link::new(
                vec![signed_const(42)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let _ = lower_graph(&graph);
    }

    /// `rewrite_op_cast_primitive` / `rewrite_op_force_cast`: a Number
    /// constant operand materialises as `ConstInt` carrying 200.
    #[test]
    fn lower_graph_cast_primitive_rint_uchar_operand_carries_200() {
        for opname in ["force_cast", "cast_primitive"] {
            let (c, _) = r_uchar_const(200);
            let startblock = Block::shared(vec![]);
            let out = variable_with_lltype("out", LowLevelType::Signed);
            let graph = FlowGraph::with_return_var(
                format!("ll_test_{opname}_rint_uchar"),
                startblock.clone(),
                Hlvalue::Variable(out.clone()),
            );
            startblock.borrow_mut().operations.push(SpaceOperation::new(
                opname,
                vec![Hlvalue::Constant(c)],
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
            let ops = &model.block(model.startblock).operations;
            let const_var = ops
                .iter()
                .find(|op| matches!(op.kind, OpKind::ConstInt(200)))
                .and_then(|op| op.result.clone())
                .unwrap_or_else(|| {
                    panic!("{opname}: r_uchar(200) did not materialise as ConstInt(200): {ops:?}")
                });
            assert!(
                ops.iter().any(|op| matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, operand, .. }
                        if op == opname && operand == &const_var
                )),
                "{opname} operand is the ConstInt(200) var: {ops:?}"
            );
        }
    }

    /// `rewrite_op_direct_call` materialises a Number constant arg as
    /// `ConstInt(200)` (`OpKind::Call` takes Variables).
    #[test]
    fn lower_graph_direct_call_rint_constant_arg_carries_200() {
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, Ptr, PtrTarget};

        let (c, uchar) = r_uchar_const(200);
        let startblock = Block::shared(vec![]);
        let return_var = variable_with_lltype("out", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "ll_test_direct_call_rint_arg",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let func_type = FuncType {
            args: vec![uchar],
            result: LowLevelType::Signed,
        };
        let funcptr = Ptr {
            TO: PtrTarget::Func(func_type),
        }
        ._example();
        let callee = constant_with_lltype(ConstValue::LLPtr(Box::new(funcptr)), LowLevelType::Void);
        let out = variable_with_lltype("out", LowLevelType::Signed);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![callee, Hlvalue::Constant(c)],
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
        let ops = &model.block(model.startblock).operations;
        let const_var = ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::ConstInt(200)))
            .and_then(|op| op.result.clone())
            .unwrap_or_else(|| panic!("RInt arg did not materialise as ConstInt(200): {ops:?}"));
        let call_args = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::Call { args, .. } => Some(args),
                _ => None,
            })
            .expect("direct_call lowers to Call");
        assert_eq!(
            call_args.first(),
            Some(&LinkArg::Value(const_var)),
            "Call arg is the ConstInt(200) var: {call_args:?}"
        );
    }

    /// `jit_conditional_call_value` value operand may be a null GC-pointer
    /// Constant (`Transformer.make_three_lists` accepts a ref-kind Constant).
    #[test]
    fn cond_call_value_accepts_a_null_pointer_cached_value() {
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, functionptr};

        let (value_c, ptr_ty) = nullptr_gc_struct();
        let value = Hlvalue::Constant(value_c);
        let startblock = Block::shared(vec![]);
        let return_var = variable_with_lltype("out", ptr_ty.clone());
        let graph = FlowGraph::with_return_var(
            "ll_test_conditional_call_value_null",
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let ft = FuncType {
            args: vec![],
            result: ptr_ty.clone(),
        };
        let funcptr = functionptr(ft, "compute", None, Some("compute".to_string()));
        let callee = constant_with_lltype(ConstValue::LLPtr(Box::new(funcptr)), LowLevelType::Void);
        let out = variable_with_lltype("out", ptr_ty);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "jit_conditional_call_value",
            vec![value, callee],
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
        let ops = &model.block(model.startblock).operations;
        let null_var = ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::ConstRefNull))
            .and_then(|op| op.result.clone())
            .expect("cached value materialises as ConstRefNull");
        let call_args = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::Call { target, args, .. } => {
                    let segs = target.path_segments()?;
                    (segs == ["majit_rlib", "jit", "conditional_call_elidable0"]).then_some(args)
                }
                _ => None,
            })
            .expect("conditional_call_elidable0 Call");
        assert_eq!(
            call_args.first(),
            Some(&LinkArg::Value(null_var)),
            "first argument is the ConstRefNull result: {call_args:?}"
        );
    }

    fn transform_lowered(model: crate::model::FunctionGraph) -> crate::model::FunctionGraph {
        use crate::codewriter::call::CallControl;
        use crate::codewriter::jtransform::{GraphTransformConfig, Transformer};

        let mut cc = CallControl::new();
        Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(&mut cc)
            .transform(&model)
            .graph
    }

    fn close_return(startblock: &BlockRef, graph: &FlowGraph, value: Variable) {
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(value)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
    }

    /// `jtransform.py handle_getfield_typeptr`: `getfield(v, 'typeptr')` on
    /// `Ptr(rclass.OBJECT)` is `-live-` + `guard_class(v)`, not a FieldRead.
    /// `rclass.OBJECT` has no CallControl row; the STRUCT `_hints['typeptr']`
    /// is what `is_typeptr_getset` reads.
    #[test]
    fn lower_graph_object_typeptr_getfield_becomes_guard_class() {
        use crate::translator::rtyper::rclass::{CLASSTYPE, OBJECTPTR};

        let v = variable_with_lltype("v", OBJECTPTR.clone());
        let startblock = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let cls = variable_with_lltype("cls", CLASSTYPE.clone());
        let graph = FlowGraph::with_return_var(
            "object_typeptr_getfield",
            startblock.clone(),
            Hlvalue::Variable(cls.clone()),
        );
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(v.clone()),
                constant_with_lltype(ConstValue::byte_str("typeptr"), LowLevelType::Void),
            ],
            Hlvalue::Variable(cls.clone()),
        ));
        close_return(&startblock, &graph, cls);

        let transformed = transform_lowered(lower_graph(&graph));
        let ops: Vec<_> = transformed
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect();
        let saw_handle = ops.windows(2).any(|window| {
            matches!(&window[0].kind, OpKind::Live { forced } if forced.is_empty())
                && matches!(&window[1].kind, OpKind::GuardClass { base } if *base == v)
        });
        assert!(
            saw_handle,
            "handle_getfield_typeptr is -live- + GuardClass {{ base: v }}; ops={ops:?}"
        );
        assert!(
            ops.iter()
                .all(|op| !matches!(op.kind, OpKind::FieldRead { .. })),
            "OBJECT typeptr getfield must not stay FieldRead: {ops:?}"
        );
    }

    /// `jtransform.py rewrite_op_setfield`: `is_typeptr_getset` drops the
    /// store (`# done by 'new'`).
    #[test]
    fn lower_graph_object_typeptr_setfield_is_dropped() {
        use crate::translator::rtyper::rclass::{CLASSTYPE, OBJECTPTR};

        let v = variable_with_lltype("v", OBJECTPTR.clone());
        let c = variable_with_lltype("c", CLASSTYPE.clone());
        let startblock = Block::shared(vec![
            Hlvalue::Variable(v.clone()),
            Hlvalue::Variable(c.clone()),
        ]);
        let graph = FlowGraph::with_return_var(
            "object_typeptr_setfield",
            startblock.clone(),
            Hlvalue::Variable(v.clone()),
        );
        let void_res = variable_with_lltype("set", LowLevelType::Void);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(v.clone()),
                constant_with_lltype(ConstValue::byte_str("typeptr"), LowLevelType::Void),
                Hlvalue::Variable(c),
            ],
            Hlvalue::Variable(void_res),
        ));
        close_return(&startblock, &graph, v);

        let transformed = transform_lowered(lower_graph(&graph));
        let ops: Vec<_> = transformed
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect();
        assert!(
            ops.is_empty(),
            "is_typeptr_getset setfield emits nothing; ops={ops:?}"
        );
    }

    /// A `GcStruct` named field `typeptr` without `_hints['typeptr']` is an
    /// ordinary load. `is_typeptr_getset` requires the hint.
    #[test]
    fn lower_graph_unhinted_typeptr_field_stays_field_read() {
        let s = Struct::gc("S", vec![("typeptr".into(), LowLevelType::Signed)]);
        let ptr_ty = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(s),
        }));
        let v = variable_with_lltype("v", ptr_ty);
        let startblock = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let result = variable_with_lltype("tp", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "unhinted_typeptr_getfield",
            startblock.clone(),
            Hlvalue::Variable(result.clone()),
        );
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(v),
                constant_with_lltype(ConstValue::byte_str("typeptr"), LowLevelType::Void),
            ],
            Hlvalue::Variable(result.clone()),
        ));
        close_return(&startblock, &graph, result);

        let transformed = transform_lowered(lower_graph(&graph));
        let ops: Vec<_> = transformed
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect();
        let field = ops.iter().find_map(|op| match &op.kind {
            OpKind::FieldRead { field, .. } => Some(field),
            _ => None,
        });
        let field = field.unwrap_or_else(|| panic!("unhinted typeptr stays FieldRead: {ops:?}"));
        assert_eq!(field.name, "typeptr");
        assert!(
            !field.owner_typeptr_hint,
            "GcStruct S does not carry the typeptr hint"
        );
        assert!(
            ops.iter()
                .all(|op| !matches!(op.kind, OpKind::GuardClass { .. })),
            "unhinted typeptr is not GuardClass: {ops:?}"
        );
    }

    /// `Ptr(ForwardReference(<resolved GcStruct m::W_X>))` with
    /// `_hints['fieldnames']` mapping `inst_name` → `name`.  Models
    /// `InstanceRepr.object_type` after `_setup_repr.become`.
    fn w_x_instance_ptr() -> LowLevelType {
        use std::sync::Arc;
        let mut fieldnames = indexmap::IndexMap::new();
        fieldnames.insert(
            ConstValue::byte_str("inst_name"),
            ConstValue::byte_str("name"),
        );
        let body = Struct::gc_with_hints(
            "m::W_X",
            vec![("inst_name".into(), LowLevelType::Signed)],
            vec![("fieldnames".into(), ConstValue::Dict(Arc::new(fieldnames)))],
        );
        let fwd = ForwardReference::gc();
        fwd.r#become(LowLevelType::Struct(Box::new(body)))
            .expect("ForwardReference.become GcStruct");
        LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::ForwardReference(fwd),
        }))
    }

    /// `jtransform.py rewrite_op_getfield`: `STRUCT = v_inst.concretetype.TO`
    /// after `ForwardReference.become`; `cpu.fielddescrof(STRUCT, name)`
    /// uses the `fieldnames` hint so the descriptor is `name` on `m::W_X`.
    #[test]
    fn lower_graph_getfield_through_forward_reference_keeps_struct_owner() {
        let ptr_ty = w_x_instance_ptr();
        let v = variable_with_lltype("v", ptr_ty);
        let startblock = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let result = variable_with_lltype("name", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "getfield_forward_reference",
            startblock.clone(),
            Hlvalue::Variable(result.clone()),
        );
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(v),
                constant_with_lltype(ConstValue::byte_str("inst_name"), LowLevelType::Void),
            ],
            Hlvalue::Variable(result.clone()),
        ));
        close_return(&startblock, &graph, result);

        let model = lower_graph(&graph);
        let field = model
            .block(model.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::FieldRead { field, .. } => Some(field),
                _ => None,
            })
            .expect("getfield lowers to FieldRead");
        assert_eq!(field.name, "name");
        assert_eq!(field.owner_root.as_deref(), Some("m::W_X"));
    }

    #[test]
    fn lower_graph_setfield_through_forward_reference_keeps_struct_owner() {
        let ptr_ty = w_x_instance_ptr();
        let v = variable_with_lltype("v", ptr_ty);
        let startblock = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FlowGraph::with_return_var(
            "setfield_forward_reference",
            startblock.clone(),
            Hlvalue::Variable(v.clone()),
        );
        let void_res = variable_with_lltype("set", LowLevelType::Void);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(v.clone()),
                constant_with_lltype(ConstValue::byte_str("inst_name"), LowLevelType::Void),
                constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
            ],
            Hlvalue::Variable(void_res),
        ));
        close_return(&startblock, &graph, v);

        let model = lower_graph(&graph);
        let field = model
            .block(model.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::FieldWrite { field, .. } => Some(field),
                _ => None,
            })
            .expect("setfield lowers to FieldWrite");
        assert_eq!(field.name, "name");
        assert_eq!(field.owner_root.as_deref(), Some("m::W_X"));
    }

    #[test]
    #[should_panic(expected = "has no STRUCT owner")]
    fn lower_graph_getfield_without_struct_owner_panics() {
        let ptr_ty = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(Array::gc(LowLevelType::Signed)),
        }));
        let v = variable_with_lltype("v", ptr_ty);
        let startblock = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let result = variable_with_lltype("x", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "getfield_without_struct_owner",
            startblock.clone(),
            Hlvalue::Variable(result.clone()),
        );
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(v),
                constant_with_lltype(ConstValue::byte_str("item"), LowLevelType::Void),
            ],
            Hlvalue::Variable(result.clone()),
        ));
        close_return(&startblock, &graph, result);
        let _ = lower_graph(&graph);
    }

    /// `TUPLE_TYPE([Ptr(gc), Signed])` mallocs and `getfield(item0)` under
    /// the lltype spelling `tuple2<GCREF,isize>`, not the bare `tuple2`
    /// leaf. `rewrite_op_getfield` / `STRUCT._immutable_field` makes the
    /// read `_pure`.
    #[test]
    fn lower_graph_tuple_getfield_owner_is_the_lltype_spelling() {
        use crate::translator::rtyper::lltypesystem::lltype::GCREF;
        use crate::translator::rtyper::rtuple::TUPLE_TYPE;

        let tuple_ty = TUPLE_TYPE(&[GCREF.clone(), LowLevelType::Signed]);
        let t = variable_with_lltype("t", tuple_ty.clone());
        let startblock = Block::shared(vec![Hlvalue::Variable(t.clone())]);
        let result = variable_with_lltype("item0", GCREF.clone());
        let graph = FlowGraph::with_return_var(
            "tuple_getfield_owner",
            startblock.clone(),
            Hlvalue::Variable(result.clone()),
        );
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(t),
                constant_with_lltype(ConstValue::byte_str("item0"), LowLevelType::Void),
            ],
            Hlvalue::Variable(result.clone()),
        ));
        close_return(&startblock, &graph, result);

        let model = lower_graph(&graph);
        let (owner, name, pure, owner_id) = model
            .block(model.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::FieldRead { field, pure, .. } => Some((
                    field.owner_root.clone(),
                    field.name.clone(),
                    *pure,
                    field.owner_id,
                )),
                _ => None,
            })
            .expect("getfield lowers to FieldRead");
        assert_eq!(owner.as_deref(), Some("tuple2<GCREF,isize>"));
        assert_eq!(name, "item0");
        assert!(pure, "TUPLE_TYPE immutable hint is getfield_gc_*_pure");
        assert_eq!(
            owner_id,
            majit_ir::descr::struct_id_for_name("tuple2<GCREF,isize>")
        );
    }

    /// `rewrite_op_cast_ptr_to_int` keeps a GC pointer and is identity
    /// (`same_as`) on raw. `cast_int_to_ptr` has no rewrite (`bhimpl`).
    /// `rewrite_op_cast_ptr_to_adr` is identity on raw.
    #[test]
    fn lower_graph_cast_ptr_int_arms() {
        let gc = variable_with_lltype("gc", STRPTR.clone());
        let raw = variable_with_lltype("raw", LowLevelType::Ptr(Box::new(NONGCREF.clone())));
        let signed = variable_with_lltype("signed", LowLevelType::Signed);
        let start = Block::shared(vec![
            Hlvalue::Variable(gc.clone()),
            Hlvalue::Variable(raw.clone()),
            Hlvalue::Variable(signed.clone()),
        ]);
        let returned = variable_with_lltype("returned", LowLevelType::Address);
        let graph = FlowGraph::with_return_var(
            "cast_ptr_int_arms",
            start.clone(),
            Hlvalue::Variable(returned),
        );
        let gc_int = variable_with_lltype("gc_int", LowLevelType::Signed);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "cast_ptr_to_int",
            vec![Hlvalue::Variable(gc)],
            Hlvalue::Variable(gc_int.clone()),
        ));
        let raw_int = variable_with_lltype("raw_int", LowLevelType::Signed);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "cast_ptr_to_int",
            vec![Hlvalue::Variable(raw.clone())],
            Hlvalue::Variable(raw_int),
        ));
        let as_ptr = variable_with_lltype("as_ptr", LowLevelType::Ptr(Box::new(NONGCREF.clone())));
        start.borrow_mut().operations.push(SpaceOperation::new(
            "cast_int_to_ptr",
            vec![Hlvalue::Variable(signed)],
            Hlvalue::Variable(as_ptr),
        ));
        let adr = variable_with_lltype("adr", LowLevelType::Address);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "cast_ptr_to_adr",
            vec![Hlvalue::Variable(raw)],
            Hlvalue::Variable(adr.clone()),
        ));
        close_return(&start, &graph, adr);

        let model = lower_graph(&graph);
        let unary: Vec<_> = model
            .block(model.startblock)
            .operations
            .iter()
            .map(|op| match &op.kind {
                OpKind::UnaryOp { op, .. } => op.as_str(),
                other => panic!("cast_ptr_* must lower to UnaryOp, got {other:?}"),
            })
            .collect();
        assert_eq!(
            unary,
            vec![
                "cast_ptr_to_int",
                "same_as",
                "cast_int_to_ptr",
                "cast_ptr_to_adr"
            ]
        );
        assert_eq!(
            model.block(model.startblock).operations[0].result.as_ref(),
            Some(&gc_int)
        );
    }

    fn entry_gcarray_ptr() -> LowLevelType {
        let entry = Struct::new(
            "ENTRY",
            vec![
                ("v".into(), LowLevelType::Signed),
                ("k".into(), LowLevelType::Signed),
            ],
        );
        LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(Array::gc(LowLevelType::Struct(Box::new(entry)))),
        }))
    }

    fn startblock_kinds(graph: &FlowGraph) -> Vec<OpKind> {
        let model = lower_graph(graph);
        model
            .block(model.startblock)
            .operations
            .iter()
            .map(|op| op.kind.clone())
            .collect()
    }

    /// `rewrite_op_getinteriorfield` / `rewrite_op_setinteriorfield` /
    /// `rewrite_op_getinteriorarraysize`.
    #[test]
    fn lower_graph_interior_field_arms() {
        let index = variable_with_lltype("i", LowLevelType::Signed);
        let ch = variable_with_lltype("ch", LowLevelType::Char);
        let unich = variable_with_lltype("unich", LowLevelType::UniChar);

        let s = variable_with_lltype("s", STRPTR.clone());
        let start = Block::shared(vec![
            Hlvalue::Variable(s.clone()),
            Hlvalue::Variable(index.clone()),
        ]);
        let got = variable_with_lltype("got", LowLevelType::Char);
        let graph = FlowGraph::with_return_var(
            "interior_str_get",
            start.clone(),
            Hlvalue::Variable(got.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "getinteriorfield",
            vec![
                Hlvalue::Variable(s.clone()),
                chars_field(),
                Hlvalue::Variable(index.clone()),
            ],
            Hlvalue::Variable(got.clone()),
        ));
        close_return(&start, &graph, got);
        match startblock_kinds(&graph).as_slice() {
            [OpKind::LoweredBlackholeOp { opname, args }] => {
                assert_eq!(opname, "strgetitem");
                assert_eq!(args.len(), 2);
            }
            other => panic!("STR getinteriorfield → strgetitem, got {other:?}"),
        }

        let u = variable_with_lltype("u", UNICODEPTR.clone());
        let start = Block::shared(vec![
            Hlvalue::Variable(u.clone()),
            Hlvalue::Variable(index.clone()),
        ]);
        let got = variable_with_lltype("got", LowLevelType::UniChar);
        let graph = FlowGraph::with_return_var(
            "interior_unicode_get",
            start.clone(),
            Hlvalue::Variable(got.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "getinteriorfield",
            vec![
                Hlvalue::Variable(u.clone()),
                chars_field(),
                Hlvalue::Variable(index.clone()),
            ],
            Hlvalue::Variable(got.clone()),
        ));
        close_return(&start, &graph, got);
        match startblock_kinds(&graph).as_slice() {
            [OpKind::LoweredBlackholeOp { opname, .. }] => {
                assert_eq!(opname, "unicodegetitem");
            }
            other => panic!("UNICODE getinteriorfield → unicodegetitem, got {other:?}"),
        }

        let ba = variable_with_lltype("ba", BYTEARRAYPTR.clone());
        let start = Block::shared(vec![
            Hlvalue::Variable(ba.clone()),
            Hlvalue::Variable(index.clone()),
        ]);
        let got = variable_with_lltype("got", LowLevelType::Char);
        let graph = FlowGraph::with_return_var(
            "interior_bytearray_get",
            start.clone(),
            Hlvalue::Variable(got.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "getinteriorfield",
            vec![
                Hlvalue::Variable(ba.clone()),
                chars_field(),
                Hlvalue::Variable(index.clone()),
            ],
            Hlvalue::Variable(got.clone()),
        ));
        close_return(&start, &graph, got);
        match startblock_kinds(&graph).as_slice() {
            [
                OpKind::ArrayRead {
                    gckind: crate::model::ArrayGcKind::Gc,
                    ..
                },
            ] => {}
            other => panic!("BYTEARRAY getinteriorfield → ArrayRead gc, got {other:?}"),
        }

        let entries = variable_with_lltype("entries", entry_gcarray_ptr());
        let start = Block::shared(vec![
            Hlvalue::Variable(entries.clone()),
            Hlvalue::Variable(index.clone()),
        ]);
        let got = variable_with_lltype("got", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "interior_dict_get",
            start.clone(),
            Hlvalue::Variable(got.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "getinteriorfield",
            vec![
                Hlvalue::Variable(entries.clone()),
                Hlvalue::Variable(index.clone()),
                constant_with_lltype(ConstValue::byte_str("v"), LowLevelType::Void),
            ],
            Hlvalue::Variable(got.clone()),
        ));
        close_return(&start, &graph, got);
        match startblock_kinds(&graph).as_slice() {
            [OpKind::InteriorFieldRead { field, .. }] => {
                assert_eq!(field.name, "v");
                assert_eq!(field.owner_root.as_deref(), Some("ENTRY"));
            }
            other => panic!("GcArray-of-Struct → InteriorFieldRead, got {other:?}"),
        }

        let start = Block::shared(vec![
            Hlvalue::Variable(entries.clone()),
            Hlvalue::Variable(index.clone()),
        ]);
        let graph = FlowGraph::with_return_var(
            "interior_dict_get_void",
            start.clone(),
            Hlvalue::Variable(entries.clone()),
        );
        let void_res = variable_with_lltype("void", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "getinteriorfield",
            vec![
                Hlvalue::Variable(entries.clone()),
                Hlvalue::Variable(index.clone()),
                constant_with_lltype(ConstValue::byte_str("v"), LowLevelType::Void),
            ],
            Hlvalue::Variable(void_res),
        ));
        close_return(&start, &graph, entries.clone());
        assert!(
            startblock_kinds(&graph).is_empty(),
            "Void getinteriorfield is dropped"
        );

        let start = Block::shared(vec![
            Hlvalue::Variable(s.clone()),
            Hlvalue::Variable(index.clone()),
            Hlvalue::Variable(ch.clone()),
        ]);
        let graph = FlowGraph::with_return_var(
            "interior_str_set",
            start.clone(),
            Hlvalue::Variable(s.clone()),
        );
        let void_res = variable_with_lltype("set", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "setinteriorfield",
            vec![
                Hlvalue::Variable(s.clone()),
                chars_field(),
                Hlvalue::Variable(index.clone()),
                Hlvalue::Variable(ch),
            ],
            Hlvalue::Variable(void_res),
        ));
        close_return(&start, &graph, s.clone());
        match startblock_kinds(&graph).as_slice() {
            [OpKind::LoweredBlackholeOp { opname, .. }] => {
                assert_eq!(opname, "strsetitem");
            }
            other => panic!("STR setinteriorfield → strsetitem, got {other:?}"),
        }

        let start = Block::shared(vec![
            Hlvalue::Variable(u.clone()),
            Hlvalue::Variable(index.clone()),
            Hlvalue::Variable(unich.clone()),
        ]);
        let graph = FlowGraph::with_return_var(
            "interior_unicode_set",
            start.clone(),
            Hlvalue::Variable(u.clone()),
        );
        let void_res = variable_with_lltype("set", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "setinteriorfield",
            vec![
                Hlvalue::Variable(u.clone()),
                chars_field(),
                Hlvalue::Variable(index.clone()),
                Hlvalue::Variable(unich),
            ],
            Hlvalue::Variable(void_res),
        ));
        close_return(&start, &graph, u.clone());
        match startblock_kinds(&graph).as_slice() {
            [OpKind::LoweredBlackholeOp { opname, .. }] => {
                assert_eq!(opname, "unicodesetitem");
            }
            other => panic!("UNICODE setinteriorfield → unicodesetitem, got {other:?}"),
        }

        let start = Block::shared(vec![
            Hlvalue::Variable(ba.clone()),
            Hlvalue::Variable(index.clone()),
        ]);
        let graph = FlowGraph::with_return_var(
            "interior_bytearray_set",
            start.clone(),
            Hlvalue::Variable(ba.clone()),
        );
        let void_res = variable_with_lltype("set", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "setinteriorfield",
            vec![
                Hlvalue::Variable(ba.clone()),
                chars_field(),
                Hlvalue::Variable(index.clone()),
                signed_const(b'x' as i64),
            ],
            Hlvalue::Variable(void_res),
        ));
        close_return(&start, &graph, ba.clone());
        match startblock_kinds(&graph).as_slice() {
            [
                OpKind::ArrayWrite {
                    gckind: crate::model::ArrayGcKind::Gc,
                    ..
                },
            ] => {}
            other => panic!("BYTEARRAY setinteriorfield → ArrayWrite gc, got {other:?}"),
        }

        let start = Block::shared(vec![
            Hlvalue::Variable(entries.clone()),
            Hlvalue::Variable(index.clone()),
        ]);
        let graph = FlowGraph::with_return_var(
            "interior_dict_set",
            start.clone(),
            Hlvalue::Variable(entries.clone()),
        );
        let void_res = variable_with_lltype("set", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "setinteriorfield",
            vec![
                Hlvalue::Variable(entries.clone()),
                Hlvalue::Variable(index.clone()),
                constant_with_lltype(ConstValue::byte_str("v"), LowLevelType::Void),
                signed_const(1),
            ],
            Hlvalue::Variable(void_res),
        ));
        close_return(&start, &graph, entries.clone());
        let field = startblock_kinds(&graph)
            .into_iter()
            .find_map(|kind| match kind {
                OpKind::InteriorFieldWrite { field, .. } => Some(field),
                _ => None,
            });
        let field = field.unwrap_or_else(|| panic!("GcArray-of-Struct set → InteriorFieldWrite"));
        assert_eq!(field.name, "v");
        assert_eq!(field.owner_root.as_deref(), Some("ENTRY"));

        let start = Block::shared(vec![Hlvalue::Variable(s.clone())]);
        let n = variable_with_lltype("n", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "interior_str_len",
            start.clone(),
            Hlvalue::Variable(n.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "getinteriorarraysize",
            vec![Hlvalue::Variable(s), chars_field()],
            Hlvalue::Variable(n.clone()),
        ));
        close_return(&start, &graph, n);
        match startblock_kinds(&graph).as_slice() {
            [OpKind::LoweredBlackholeOp { opname, .. }] => assert_eq!(opname, "strlen"),
            other => panic!("STR getinteriorarraysize → strlen, got {other:?}"),
        }

        let start = Block::shared(vec![Hlvalue::Variable(u.clone())]);
        let n = variable_with_lltype("n", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "interior_unicode_len",
            start.clone(),
            Hlvalue::Variable(n.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "getinteriorarraysize",
            vec![Hlvalue::Variable(u), chars_field()],
            Hlvalue::Variable(n.clone()),
        ));
        close_return(&start, &graph, n);
        match startblock_kinds(&graph).as_slice() {
            [OpKind::LoweredBlackholeOp { opname, .. }] => assert_eq!(opname, "unicodelen"),
            other => panic!("UNICODE getinteriorarraysize → unicodelen, got {other:?}"),
        }

        let start = Block::shared(vec![Hlvalue::Variable(ba.clone())]);
        let n = variable_with_lltype("n", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "interior_bytearray_len",
            start.clone(),
            Hlvalue::Variable(n.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "getinteriorarraysize",
            vec![Hlvalue::Variable(ba), chars_field()],
            Hlvalue::Variable(n.clone()),
        ));
        close_return(&start, &graph, n);
        match startblock_kinds(&graph).as_slice() {
            [
                OpKind::ArrayLen {
                    gckind: crate::model::ArrayGcKind::Gc,
                    ..
                },
            ] => {}
            other => panic!("BYTEARRAY getinteriorarraysize → ArrayLen gc, got {other:?}"),
        }
    }

    /// `rewrite_op_raw_load` / `rewrite_op_raw_store`: `raw_{load,store}_<kind>`
    /// plus `arraydescrof(rffi.CArray(T))`. Kind `'r'` is refused.
    #[test]
    fn lower_graph_raw_load_store_arms() {
        let addr = variable_with_lltype("addr", LowLevelType::Address);
        let offset = variable_with_lltype("off", LowLevelType::Signed);
        let start = Block::shared(vec![
            Hlvalue::Variable(addr.clone()),
            Hlvalue::Variable(offset.clone()),
        ]);
        let loaded = variable_with_lltype("loaded", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "raw_load_store",
            start.clone(),
            Hlvalue::Variable(loaded.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "raw_load",
            vec![
                Hlvalue::Variable(addr.clone()),
                Hlvalue::Variable(offset.clone()),
            ],
            Hlvalue::Variable(loaded.clone()),
        ));
        let stored = variable_with_lltype("stored", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "raw_store",
            vec![
                Hlvalue::Variable(addr),
                Hlvalue::Variable(offset),
                Hlvalue::Variable(loaded.clone()),
            ],
            Hlvalue::Variable(stored),
        ));
        close_return(&start, &graph, loaded);

        let word = crate::layout::target_signed_size();
        let kinds = startblock_kinds(&graph);
        match kinds.as_slice() {
            [
                OpKind::RawLoad {
                    item_ty,
                    itemsize,
                    is_item_signed,
                    ..
                },
                OpKind::RawStore {
                    item_ty: store_ty,
                    itemsize: store_size,
                    is_item_signed: store_signed,
                    ..
                },
            ] => {
                assert_eq!(*item_ty, ValueType::Int);
                assert_eq!(*itemsize, word);
                assert!(*is_item_signed);
                assert_eq!(*store_ty, ValueType::Int);
                assert_eq!(*store_size, word);
                assert!(*store_signed);
            }
            other => panic!("raw_load/store → RawLoad then RawStore, got {other:?}"),
        }
    }

    /// `rewrite_op_direct_ptradd`: CCHARP is one `int_add`; otherwise
    /// `int_mul` by `sizeof(TO.OF)` then `int_add`.
    #[test]
    fn lower_graph_direct_ptradd_arm() {
        let p = variable_with_lltype("p", CCHARP.clone());
        let n = variable_with_lltype("n", LowLevelType::Signed);
        let start = Block::shared(vec![
            Hlvalue::Variable(p.clone()),
            Hlvalue::Variable(n.clone()),
        ]);
        let q = variable_with_lltype("q", CCHARP.clone());
        let graph = FlowGraph::with_return_var(
            "direct_ptradd_ccharp",
            start.clone(),
            Hlvalue::Variable(q.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "direct_ptradd",
            vec![Hlvalue::Variable(p.clone()), Hlvalue::Variable(n.clone())],
            Hlvalue::Variable(q.clone()),
        ));
        close_return(&start, &graph, q);
        match startblock_kinds(&graph).as_slice() {
            [OpKind::BinOp { op, lhs, rhs, .. }] => {
                assert_eq!(op, "int_add");
                assert_eq!(lhs, &p);
                assert_eq!(rhs, &n);
            }
            other => panic!("CCHARP direct_ptradd → int_add, got {other:?}"),
        }

        let signedp = CArrayPtr(LowLevelType::Signed);
        let p = variable_with_lltype("p", signedp.clone());
        let n = variable_with_lltype("n", LowLevelType::Signed);
        let start = Block::shared(vec![
            Hlvalue::Variable(p.clone()),
            Hlvalue::Variable(n.clone()),
        ]);
        let q = variable_with_lltype("q", signedp);
        let graph = FlowGraph::with_return_var(
            "direct_ptradd_signed",
            start.clone(),
            Hlvalue::Variable(q.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "direct_ptradd",
            vec![Hlvalue::Variable(p.clone()), Hlvalue::Variable(n.clone())],
            Hlvalue::Variable(q.clone()),
        ));
        close_return(&start, &graph, q);
        let model = lower_graph(&graph);
        let ops = &model.block(model.startblock).operations;
        match ops.as_slice() {
            [scale, mul, add] => match (&scale.kind, &mul.kind, &add.kind) {
                (
                    OpKind::ConstInt(by),
                    OpKind::BinOp {
                        op: mul_op, lhs, ..
                    },
                    OpKind::BinOp {
                        op: add_op,
                        lhs: add_lhs,
                        rhs: add_rhs,
                        ..
                    },
                ) => {
                    assert_eq!(*by, crate::layout::target_signed_size() as i64);
                    assert_eq!(mul_op, "int_mul");
                    assert_eq!(lhs, &n);
                    assert_eq!(add_op, "int_add");
                    assert_eq!(add_lhs, &p);
                    assert_eq!(add_rhs, mul.result.as_ref().expect("int_mul result"));
                }
                other => panic!("CArray(Signed) direct_ptradd → int_mul + int_add, got {other:?}"),
            },
            other => panic!("CArray(Signed) direct_ptradd → int_mul + int_add, got {other:?}"),
        }
    }

    /// `rewrite_op_indirect_call` emits `OpKind::IndirectCall`. Trailing
    /// Void graphs stay `None` (`transduce_op` has no CallControl).
    #[test]
    fn lower_graph_indirect_call_arm() {
        use crate::translator::rtyper::lltypesystem::lltype::FuncType;

        let fnptr_ty = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Signed,
            }),
        }));
        let f = variable_with_lltype("f", fnptr_ty);
        let x = variable_with_lltype("x", LowLevelType::Signed);
        let start = Block::shared(vec![
            Hlvalue::Variable(f.clone()),
            Hlvalue::Variable(x.clone()),
        ]);
        let y = variable_with_lltype("y", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "indirect_call_arm",
            start.clone(),
            Hlvalue::Variable(y.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "indirect_call",
            vec![
                Hlvalue::Variable(f.clone()),
                Hlvalue::Variable(x.clone()),
                constant_with_lltype(ConstValue::None, LowLevelType::Void),
            ],
            Hlvalue::Variable(y.clone()),
        ));
        close_return(&start, &graph, y);
        match startblock_kinds(&graph).as_slice() {
            [
                OpKind::IndirectCall {
                    funcptr,
                    args,
                    graphs,
                    family_key,
                    ..
                },
            ] => {
                assert_eq!(funcptr, &f);
                assert_eq!(args.as_slice(), std::slice::from_ref(&x));
                assert!(graphs.is_none());
                assert!(family_key.is_none());
            }
            other => panic!("indirect_call → IndirectCall graphs None, got {other:?}"),
        }
    }

    /// Leftover Call so `rewrite_op_jit_force_virtualizable` can file
    /// `vable_flags` / `get_vinfo`. Do not drain: empty CallControl asserts.
    #[test]
    fn lower_graph_jit_force_virtualizable_arm() {
        let v = variable_with_lltype("v", STRPTR.clone());
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FlowGraph::with_return_var(
            "jit_force_virtualizable_arm",
            start.clone(),
            Hlvalue::Variable(v.clone()),
        );
        let void_res = variable_with_lltype("force", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "jit_force_virtualizable",
            vec![
                Hlvalue::Variable(v.clone()),
                constant_with_lltype(ConstValue::byte_str(""), LowLevelType::Void),
                constant_with_lltype(
                    ConstValue::Dict(std::sync::Arc::new(indexmap::IndexMap::new())),
                    LowLevelType::Void,
                ),
            ],
            Hlvalue::Variable(void_res),
        ));
        close_return(&start, &graph, v);
        match startblock_kinds(&graph).as_slice() {
            [OpKind::Call { target, args, .. }] => {
                assert_eq!(
                    target.path_segments().as_deref(),
                    Some(["jit_force_virtualizable"].as_slice())
                );
                assert_eq!(args.len(), 3, "hook_access_field [vinst, cname, cflags]");
            }
            other => panic!("jit_force_virtualizable leftover Call, got {other:?}"),
        }
    }

    /// `rewrite_op_jit_force_quasi_immutable`: `-live-` +
    /// `jit_force_quasi_immutable(v_inst, fielddescr)`.
    #[test]
    fn lower_graph_jit_force_quasi_immutable_arm() {
        let inst = variable_with_lltype("inst", STRINGBUILDERPTR.clone());
        let start = Block::shared(vec![Hlvalue::Variable(inst.clone())]);
        let graph = FlowGraph::with_return_var(
            "jit_force_quasi_immutable_arm",
            start.clone(),
            Hlvalue::Variable(inst.clone()),
        );
        let void_res = variable_with_lltype("force", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "jit_force_quasi_immutable",
            vec![
                Hlvalue::Variable(inst.clone()),
                constant_with_lltype(ConstValue::byte_str("current_pos"), LowLevelType::Void),
            ],
            Hlvalue::Variable(void_res),
        ));
        close_return(&start, &graph, inst);
        match startblock_kinds(&graph).as_slice() {
            [
                OpKind::Live { .. },
                OpKind::JitForceQuasiImmutable { mutate_field, .. },
            ] => {
                assert_eq!(mutate_field.name, "current_pos");
                assert_eq!(mutate_field.owner_root.as_deref(), Some("stringbuilder"));
            }
            other => panic!("quasi-immutable → Live + JitForceQuasiImmutable, got {other:?}"),
        }
    }

    /// `rewrite_op_free`: flavor `'raw'`, leftover flags
    /// `UnsupportedMallocFlags`, then `raw_free`.
    #[test]
    fn lower_graph_free_arm() {
        let p = variable_with_lltype("p", CCHARP.clone());
        let start = Block::shared(vec![Hlvalue::Variable(p.clone())]);
        let graph =
            FlowGraph::with_return_var("free_arm", start.clone(), Hlvalue::Variable(p.clone()));
        let void_res = variable_with_lltype("freed", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "free",
            vec![Hlvalue::Variable(p.clone()), raw_flavor_const(&[])],
            Hlvalue::Variable(void_res),
        ));
        close_return(&start, &graph, p.clone());
        match startblock_kinds(&graph).as_slice() {
            [OpKind::RawFree { ptr }] => assert_eq!(ptr, &p),
            other => panic!("free → RawFree, got {other:?}"),
        }
    }

    /// `rewrite_op_gc_add_memory_pressure = _do_builtin_call`. Leftover
    /// Call so `rewrite_op_direct_call` residualizes it.
    #[test]
    fn lower_graph_gc_add_memory_pressure_arm() {
        let n = variable_with_lltype("n", LowLevelType::Signed);
        let start = Block::shared(vec![Hlvalue::Variable(n.clone())]);
        let graph = FlowGraph::with_return_var(
            "gc_add_memory_pressure_arm",
            start.clone(),
            Hlvalue::Variable(n.clone()),
        );
        let void_res = variable_with_lltype("pressure", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "gc_add_memory_pressure",
            vec![Hlvalue::Variable(n.clone())],
            Hlvalue::Variable(void_res),
        ));
        close_return(&start, &graph, n);
        match startblock_kinds(&graph).as_slice() {
            [OpKind::Call { target, .. }] => {
                assert_eq!(
                    target.path_segments().as_deref(),
                    Some(["gc_add_memory_pressure"].as_slice())
                );
            }
            other => panic!("gc_add_memory_pressure leftover Call, got {other:?}"),
        }
    }

    #[test]
    fn lower_graph_gc_identityhash_arm() {
        let p = variable_with_lltype("p", STRPTR.clone());
        let start = Block::shared(vec![Hlvalue::Variable(p.clone())]);
        let h = variable_with_lltype("h", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "gc_identityhash_arm",
            start.clone(),
            Hlvalue::Variable(h.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "gc_identityhash",
            vec![Hlvalue::Variable(p.clone())],
            Hlvalue::Variable(h.clone()),
        ));
        close_return(&start, &graph, h);
        match startblock_kinds(&graph).as_slice() {
            [OpKind::LoweredBlackholeOp { opname, args }] => {
                assert_eq!(opname, "gc_identityhash");
                assert_eq!(args.len(), 1);
            }
            other => panic!("gc_identityhash → LoweredBlackholeOp, got {other:?}"),
        }
    }

    /// Front / rtyper emit the 2-arg BinOp (offset, loop-invariant flag)
    /// `rewrite_op_threadlocalref_get` residualizes.
    #[test]
    fn lower_graph_threadlocalref_get_arm() {
        let offset = variable_with_lltype("offset", LowLevelType::Signed);
        let flag = variable_with_lltype("flag", LowLevelType::Bool);
        let start = Block::shared(vec![
            Hlvalue::Variable(offset.clone()),
            Hlvalue::Variable(flag.clone()),
        ]);
        let got = variable_with_lltype("got", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "threadlocalref_get_arm",
            start.clone(),
            Hlvalue::Variable(got.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "threadlocalref_get",
            vec![
                Hlvalue::Variable(offset.clone()),
                Hlvalue::Variable(flag.clone()),
            ],
            Hlvalue::Variable(got.clone()),
        ));
        close_return(&start, &graph, got);
        match startblock_kinds(&graph).as_slice() {
            [OpKind::BinOp { op, lhs, rhs, .. }] => {
                assert_eq!(op, "threadlocalref_get");
                assert_eq!(lhs, &offset);
                assert_eq!(rhs, &flag);
            }
            other => panic!("threadlocalref_get → BinOp, got {other:?}"),
        }
    }

    fn op_is_const_or_copy(op: &crate::model::SpaceOperation) -> bool {
        match &op.kind {
            OpKind::LoweredBlackholeOp { opname, .. }
                if matches!(opname.as_str(), "int_copy" | "ref_copy" | "float_copy") =>
            {
                true
            }
            OpKind::ConstInt(_)
            | OpKind::ConstUInt(_)
            | OpKind::ConstBool(_)
            | OpKind::ConstFloat(_)
            | OpKind::ConstSingleFloat(_)
            | OpKind::ConstRefNull
            | OpKind::ConstStr(_)
            | OpKind::ConstRefAddr(_)
            | OpKind::ConstNone
            | OpKind::ConstVtable { .. }
            | OpKind::ConstSymbolic { .. } => true,
            _ => false,
        }
    }

    fn prebuilt_gc_struct_with_signed_x() -> (Hlvalue, LowLevelType) {
        use crate::translator::rtyper::lltypesystem::lltype::{MallocFlavor, malloc};
        let container = LowLevelType::Struct(Box::new(Struct::gc(
            "PrebuiltX",
            vec![("x".into(), LowLevelType::Signed)],
        )));
        let ptr =
            malloc(container.clone(), None, MallocFlavor::Gc, true).expect("malloc PrebuiltX");
        let ptr_ty = LowLevelType::Ptr(Box::new(
            Ptr::from_container_type(container).expect("Ptr(PrebuiltX)"),
        ));
        (
            constant_with_lltype(ConstValue::LLPtr(Box::new(ptr)), ptr_ty.clone()),
            ptr_ty,
        )
    }

    /// `Transformer.add_in_correct_list` returns early for `kind == 'void'`.
    #[test]
    fn keepalive_drops_void_operands() {
        let v = variable_with_lltype("p", STRPTR.clone());
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FlowGraph::with_return_var(
            "keepalive_drops_void",
            start.clone(),
            Hlvalue::Variable(v.clone()),
        );
        let void_res = variable_with_lltype("k", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "keepalive",
            vec![
                Hlvalue::Variable(v.clone()),
                constant_with_lltype(ConstValue::None, LowLevelType::Void),
            ],
            Hlvalue::Variable(void_res),
        ));
        close_return(&start, &graph, v.clone());
        let model = lower_graph(&graph);
        let ops = &model.block(model.startblock).operations;
        let live = ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::Live { .. }))
            .unwrap_or_else(|| panic!("keepalive lowers to Live: {ops:?}"));
        let OpKind::Live { forced } = &live.kind else {
            unreachable!();
        };
        assert_eq!(forced.as_slice(), std::slice::from_ref(&v));
        assert!(
            ops.iter().all(|op| !op_is_const_or_copy(op)),
            "Void keepalive operand emits no Const op: {ops:?}"
        );
    }

    #[test]
    fn prebuilt_object_getfield_receiver_is_materialized() {
        let (receiver, ptr_ty) = prebuilt_gc_struct_with_signed_x();
        let start = Block::shared(vec![]);
        let result = variable_with_lltype("x", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "prebuilt_object_getfield",
            start.clone(),
            Hlvalue::Variable(result.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![
                receiver,
                constant_with_lltype(ConstValue::byte_str("x"), LowLevelType::Void),
            ],
            Hlvalue::Variable(result.clone()),
        ));
        close_return(&start, &graph, result);
        let model = lower_graph(&graph);
        let ops = &model.block(model.startblock).operations;
        let base = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::FieldRead { base, .. } => Some(base.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("getfield lowers to FieldRead: {ops:?}"));
        assert_eq!(base.concretetype(), Some(ptr_ty));
        assert!(
            ops.iter()
                .any(|op| op.result.as_ref() == Some(&base) && op_is_const_or_copy(op)),
            "FieldRead base is defined by a const/copy op: {ops:?}"
        );
    }

    #[test]
    fn setfield_on_a_prebuilt_receiver_is_materialized() {
        let (receiver, ptr_ty) = prebuilt_gc_struct_with_signed_x();
        let dummy = variable_with_lltype("p", ptr_ty.clone());
        let start = Block::shared(vec![Hlvalue::Variable(dummy.clone())]);
        let graph = FlowGraph::with_return_var(
            "prebuilt_object_setfield",
            start.clone(),
            Hlvalue::Variable(dummy.clone()),
        );
        let void_res = variable_with_lltype("set", LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "setfield",
            vec![
                receiver,
                constant_with_lltype(ConstValue::byte_str("x"), LowLevelType::Void),
                signed_const(1),
            ],
            Hlvalue::Variable(void_res),
        ));
        close_return(&start, &graph, dummy);
        let model = lower_graph(&graph);
        let ops = &model.block(model.startblock).operations;
        let base = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::FieldWrite { base, .. } => Some(base.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("setfield lowers to FieldWrite: {ops:?}"));
        assert_eq!(base.concretetype(), Some(ptr_ty));
        assert!(
            ops.iter()
                .any(|op| op.result.as_ref() == Some(&base) && op_is_const_or_copy(op)),
            "FieldWrite base is defined by a const/copy op: {ops:?}"
        );
    }

    #[test]
    fn pointer_compare_accepts_a_prebuilt_constant_operand() {
        let (rhs, ptr_ty) = prebuilt_gc_struct_with_signed_x();
        let lhs = variable_with_lltype("p", ptr_ty);
        let start = Block::shared(vec![Hlvalue::Variable(lhs.clone())]);
        let result = variable_with_lltype("eq", LowLevelType::Bool);
        let graph = FlowGraph::with_return_var(
            "ptr_eq_prebuilt_const",
            start.clone(),
            Hlvalue::Variable(result.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "ptr_eq",
            vec![Hlvalue::Variable(lhs.clone()), rhs],
            Hlvalue::Variable(result.clone()),
        ));
        close_return(&start, &graph, result.clone());
        let model = lower_graph(&graph);
        let ops = &model.block(model.startblock).operations;
        let bin = ops
            .iter()
            .find(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "eq"))
            .unwrap_or_else(|| panic!("ptr_eq lowers to binary eq: {ops:?}"));
        let OpKind::BinOp {
            lhs: got_lhs,
            rhs: got_rhs,
            ..
        } = &bin.kind
        else {
            unreachable!();
        };
        assert_eq!(got_lhs, &lhs);
        assert_ne!(got_rhs, &lhs);
        assert!(
            ops.iter()
                .any(|op| op.result.as_ref() == Some(got_rhs) && op_is_const_or_copy(op)),
            "ptr_eq rhs is a materialized var: {ops:?}"
        );
        assert_eq!(bin.result.as_ref(), Some(&result));
    }

    #[test]
    fn constant_receiver_typeptr_read_folds_to_the_class_constant() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            LowLevelValue, MallocFlavor, malloc,
        };
        let vtable_container = LowLevelType::Struct(Box::new(Struct::new(
            "vtable",
            vec![("subclassrange_min".into(), LowLevelType::Signed)],
        )));
        let vtable =
            malloc(vtable_container.clone(), None, MallocFlavor::Raw, true).expect("malloc vtable");
        let vtable_ptr_ty = LowLevelType::Ptr(Box::new(
            Ptr::from_container_type(vtable_container).expect("Ptr(vtable)"),
        ));
        let object_container = LowLevelType::Struct(Box::new(Struct::gc(
            "object",
            vec![("typeptr".into(), vtable_ptr_ty.clone())],
        )));
        let mut inst =
            malloc(object_container.clone(), None, MallocFlavor::Gc, true).expect("malloc object");
        inst.setattr("typeptr", LowLevelValue::Ptr(Box::new(vtable)))
            .expect("set typeptr");
        let inst_ptr_ty = LowLevelType::Ptr(Box::new(
            Ptr::from_container_type(object_container).expect("Ptr(object)"),
        ));
        let receiver = constant_with_lltype(ConstValue::LLPtr(Box::new(inst)), inst_ptr_ty);
        let start = Block::shared(vec![]);
        let cls = variable_with_lltype("cls", vtable_ptr_ty);
        let graph = FlowGraph::with_return_var(
            "constant_receiver_typeptr",
            start.clone(),
            Hlvalue::Variable(cls.clone()),
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![
                receiver,
                constant_with_lltype(ConstValue::byte_str("typeptr"), LowLevelType::Void),
            ],
            Hlvalue::Variable(cls.clone()),
        ));
        close_return(&start, &graph, cls.clone());
        let model = lower_graph(&graph);
        let ops = &model.block(model.startblock).operations;
        assert!(
            ops.iter()
                .all(|op| !matches!(op.kind, OpKind::FieldRead { .. })),
            "Constant typeptr getfield emits no FieldRead: {ops:?}"
        );
        let defining = ops
            .iter()
            .find(|op| op.result.as_ref() == Some(&cls))
            .unwrap_or_else(|| panic!("typeptr result is defined: {ops:?}"));
        match &defining.kind {
            OpKind::LoweredBlackholeOp { args, .. } => {
                assert!(
                    matches!(
                        args.as_slice(),
                        [LinkArg::Const(c)] if matches!(&c.value, ConstValue::LLPtr(p) if p.nonzero())
                    ),
                    "typeptr fold carries the vtable constant: {args:?}"
                );
            }
            other => panic!("typeptr fold defines the result with a const/copy: {other:?}"),
        }
    }

    #[test]
    fn address_offset_constant_reaches_the_integer_pool_unresolved() {
        use crate::translator::rtyper::lltypesystem::llmemory::{AddressOffset, ItemOffset};
        let v = variable_with_lltype("n", LowLevelType::Signed);
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let result = variable_with_lltype("sum", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "address_offset_int_add",
            start.clone(),
            Hlvalue::Variable(result.clone()),
        );
        let offset = constant_with_lltype(
            ConstValue::AddressOffset(AddressOffset::ItemOffset(ItemOffset {
                TYPE: LowLevelType::Char,
                repeat: 1,
            })),
            LowLevelType::Signed,
        );
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(v.clone()), offset],
            Hlvalue::Variable(result.clone()),
        ));
        close_return(&start, &graph, result);
        let model = lower_graph(&graph);
        let ops = &model.block(model.startblock).operations;
        let copy = ops
            .iter()
            .find(|op| {
                matches!(
                    &op.kind,
                    OpKind::LoweredBlackholeOp { opname, args }
                        if opname == "int_copy"
                            && matches!(
                                args.as_slice(),
                                [LinkArg::Const(c)]
                                    if matches!(c.value, ConstValue::AddressOffset(_))
                            )
                )
            })
            .unwrap_or_else(|| {
                panic!(
                    "AddressOffset Constant becomes int_copy of the unresolved Constant: {ops:?}"
                )
            });
        assert!(
            !matches!(copy.kind, OpKind::ConstInt(_)),
            "AddressOffset is not a resolved ConstInt: {ops:?}"
        );
    }

    #[test]
    fn leftover_constant_kind_falls_back_to_a_typed_copy() {
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, Ptr, PtrTarget};

        let (receiver, ptr_ty) = prebuilt_gc_struct_with_signed_x();
        let leftover = match &receiver {
            Hlvalue::Constant(c) => c.clone(),
            other => panic!("prebuilt GcStruct is a Constant, got {other:?}"),
        };
        let start = Block::shared(vec![]);
        let result = variable_with_lltype("out", LowLevelType::Signed);
        let graph = FlowGraph::with_return_var(
            "leftover_const_copy",
            start.clone(),
            Hlvalue::Variable(result.clone()),
        );
        let func_type = FuncType {
            args: vec![ptr_ty],
            result: LowLevelType::Signed,
        };
        let funcptr = Ptr {
            TO: PtrTarget::Func(func_type),
        }
        ._example();
        let callee = constant_with_lltype(ConstValue::LLPtr(Box::new(funcptr)), LowLevelType::Void);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![callee, receiver],
            Hlvalue::Variable(result.clone()),
        ));
        close_return(&start, &graph, result);
        let model = lower_graph(&graph);
        let ops = &model.block(model.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::LoweredBlackholeOp { opname, args }
                    if opname == "ref_copy"
                        && matches!(
                            args.as_slice(),
                            [LinkArg::Const(c)] if c == &leftover
                        )
            )),
            "leftover GcStruct Constant becomes ref_copy of that Constant: {ops:?}"
        );
    }
}
