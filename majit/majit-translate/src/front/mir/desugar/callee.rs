//! Callee classification helpers (fn-ptr family, registered paths, dest-ref tracing).

use super::*;

/// Which PBC family an `OpKind::IndirectCall` through a bare function
/// pointer carries — the `c_graphs` argument `FunctionReprBase.call`
/// appends at `rpbc.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::front::mir) enum FnPtrFamily {
    /// The generated builtin-wrapper family.  `Some([])` is the
    /// pre-CallControl marker `rpbc::lower_indirect_calls` fills once the
    /// registered graphs and the linker-resolved wrapper addresses are
    /// available.
    BuiltinWrapper,
    /// The candidate set is not recoverable from the LLBC: the pointer is
    /// installed at runtime (a host-registered callback, a settable hook),
    /// so the addresses that reach the site are not a closed set a static
    /// walk of the artifact can enumerate.  Carries `None` —
    /// `graphs_from` then answers `None` (`call.py`) and
    /// `guess_call_kind` answers `residual` (`call.py`), so
    /// `rewrite_op_indirect_call` emits `handle_residual_call`
    /// (`jtransform.py`).  `None` is also what keeps the family
    /// analyzers conservative: `Some([])` reads as "empty family" and
    /// collapses canraise / can_invalidate / forces_virtualizable to their
    /// bottom result, which is unsound for a callee this side cannot see.
    Unknown,
}

/// Crate-stripped paths of the prebuilt eval hook in this artefact.
///
/// `register_eval_override`'s `FnDef` argument, then every other `FnDef`
/// assigned to that parameter's type (the plain evaluator). Callers in
/// another crate see the registrar as an external `FunDecl`; the
/// translator unions the lists before lowering.
pub(crate) fn discover_eval_hook_graphs(llbc: &Llbc) -> Vec<String> {
    let Some(eval_ty) = eval_fn_type_id(llbc) else {
        return Vec::new();
    };
    let registrar_ids = registrar_fun_ids(llbc);
    let mut overrides = Vec::new();
    let mut plain = Vec::new();
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let mut bound = vec![None; body.locals.locals.len()];
        let forward_blocks = forward_reachable_mask(llbc, &body);
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            for stmt in &bb.statements {
                let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                    continue;
                };
                let Some(fun) = rvalue_fundef(llbc, rvalue) else {
                    continue;
                };
                if let PlaceKind::Local(index) = place.kind
                    && (index as usize) < bound.len()
                {
                    bound[index as usize] = Some(fun);
                }
                if tyref_dedup_id(&place.ty) == Some(eval_ty) {
                    push_registered_fun_path(llbc, fun, &mut plain);
                }
            }
            let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
                continue;
            };
            let CallFunc::Regular(reg) = &call.func else {
                continue;
            };
            let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
                continue;
            };
            if !registrar_ids.contains(id) {
                continue;
            }
            let Some(arg) = call.args.first() else {
                continue;
            };
            if let Some(fun) = operand_bound_fundef(llbc, arg, &bound) {
                push_registered_fun_path(llbc, fun, &mut overrides);
            }
        }
    }
    for path in plain {
        if !overrides.contains(&path) {
            overrides.push(path);
        }
    }
    overrides
}

/// `register_eval_override`'s parameter type: the `EvalFn` pointer.
///
/// Resolved once per artefact, from every `FunDecl` including an external
/// one. Call sites compare that cached dedup id.
pub(in crate::front::mir) fn eval_fn_type_id(llbc: &Llbc) -> Option<u64> {
    llbc.eval_fn_type_id()
}

pub(in crate::front::mir) fn registrar_fun_ids(llbc: &Llbc) -> Vec<u64> {
    llbc.iter_fun_decls()
        .filter(|fd| {
            fd.item_meta
                .name_path()
                .ends_with("::call::register_eval_override")
        })
        .map(|fd| fd.def_id)
        .collect()
}

pub(in crate::front::mir) fn tyref_dedup_id(ty: &TyRef) -> Option<u64> {
    match ty {
        TyRef::Dedup { id } => Some(*id),
        TyRef::Inline { value: (id, _) } => Some(*id),
        TyRef::Other(_) => None,
    }
}

pub(in crate::front::mir) fn push_registered_fun_path(llbc: &Llbc, id: u64, out: &mut Vec<String>) {
    let Some(fd) = llbc.fn_by_id(id) else {
        return;
    };
    let key = registered_path_for_fun_decl(llbc, fd).canonical_key();
    if !key.is_empty() && !out.contains(&key) {
        out.push(key);
    }
}

pub(in crate::front::mir) fn rvalue_fundef(llbc: &Llbc, rvalue: &Rvalue) -> Option<u64> {
    match rvalue {
        Rvalue::Use(operand, _) | Rvalue::UnaryOp(_, operand) | Rvalue::Cast(_, operand, _) => {
            operand_fundef(llbc, operand)
        }
        _ => None,
    }
}

pub(in crate::front::mir) fn operand_fundef(llbc: &Llbc, operand: &Operand) -> Option<u64> {
    let Operand::Const(value) = operand else {
        return None;
    };
    const_fn_def_regular_id(llbc, value)
}

pub(in crate::front::mir) fn operand_bound_fundef(
    llbc: &Llbc,
    operand: &Operand,
    bound: &[Option<u64>],
) -> Option<u64> {
    if let Some(fun) = operand_fundef(llbc, operand) {
        return Some(fun);
    }
    let (Operand::Copy(place) | Operand::Move(place)) = operand else {
        return None;
    };
    let PlaceKind::Local(index) = place.kind else {
        return None;
    };
    bound.get(index as usize).copied().flatten()
}

/// `get_eval_fn`'s result, or any other value of `register_eval_override`'s
/// parameter type: the prebuilt family published on `llbc`.
pub(in crate::front::mir) fn eval_hook_graphs_for_call(
    llbc: &Llbc,
    body: &Unstructured,
    operand: &Operand,
) -> Option<Vec<crate::parse::CallPath>> {
    let paths = llbc.eval_hook_graphs();
    if paths.is_empty() {
        return None;
    }
    if !operand_is_eval_fn(operand, llbc) && !operand_produced_by_get_eval_fn(body, llbc, operand) {
        return None;
    }
    Some(
        paths
            .iter()
            .map(|path| crate::parse::CallPath::from_segments(path.split("::")))
            .collect(),
    )
}

pub(in crate::front::mir) fn operand_is_eval_fn(operand: &Operand, llbc: &Llbc) -> bool {
    let ty = match operand {
        Operand::Copy(place) | Operand::Move(place) => &place.ty,
        Operand::Const(_) => return false,
    };
    match (tyref_dedup_id(ty), eval_fn_type_id(llbc)) {
        (Some(operand_ty), Some(eval_ty)) => operand_ty == eval_ty,
        _ => false,
    }
}

/// The local was produced by a call whose callee returns
/// `register_eval_override`'s parameter type.
pub(in crate::front::mir) fn operand_produced_by_get_eval_fn(
    body: &Unstructured,
    llbc: &Llbc,
    operand: &Operand,
) -> bool {
    let Some(eval_ty) = eval_fn_type_id(llbc) else {
        return false;
    };
    let (Operand::Copy(place) | Operand::Move(place)) = operand else {
        return false;
    };
    let PlaceKind::Local(local) = place.kind else {
        return false;
    };
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            continue;
        };
        let PlaceKind::Local(dest) = call.dest.kind else {
            continue;
        };
        if dest != local {
            continue;
        }
        let CallFunc::Regular(reg) = &call.func else {
            continue;
        };
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            continue;
        };
        let Some(fd) = llbc.fn_by_id(*id) else {
            continue;
        };
        if tyref_dedup_id(&fd.signature.output) == Some(eval_ty) {
            return true;
        }
    }
    false
}

/// Classify a `CallFunc::Dynamic` operand that resolves to a bare
/// function pointer.  `None` means the operand is not a plain fn pointer
/// (a `dyn` fat pointer), or is one this side declines to name a family
/// for, and leaves the caller on its `__dyn_call` fallback.
pub(in crate::front::mir) fn operand_fn_ptr_family(
    operand: &Operand,
    llbc: &Llbc,
) -> Option<FnPtrFamily> {
    let ty = match operand {
        Operand::Copy(place) | Operand::Move(place) => &place.ty,
        Operand::Const(_) => return None,
    };
    let signature = tyref_node(ty, llbc)
        .and_then(|node| strip_ty_wrappers(node, llbc))
        .and_then(|node| node.get("FnPtr"))?
        .get("skip_binder")
        .and_then(serde_json::Value::as_object)?;
    // A missing or non-boolean `is_unsafe` reads as unsafe.
    let is_safe = signature
        .get("is_unsafe")
        .and_then(serde_json::Value::as_bool)
        == Some(false);
    fn_ptr_family_for(
        is_safe,
        fn_ptr_signature_is_builtin_code_fn(signature, llbc),
        fnptr_indirect_enabled(),
    )
}

/// The family decision itself, separated from the LLBC digging above so it
/// can be exercised without an extracted artefact — the fixture corpus
/// declares exactly one fn-pointer alias and it is safe, so a corpus-driven
/// test cannot reach the `unsafe` rows at all.
///
/// `is_builtin_shape` is [`fn_ptr_signature_is_builtin_code_fn`], which
/// matches on signature SHAPE. An `unsafe fn` of that shape would otherwise
/// be handed the `__majit_wrap_*` family: a positive claim about a callee set
/// it is not a member of. Only a safe pointer may name that family.
///
/// An `unsafe fn` pointer is otherwise the same value with the same ABI as a
/// safe one, so it still carries the unknown family — `None` graphs is an
/// honest residual, not a claim about the callee.
pub(in crate::front::mir) fn fn_ptr_family_for(
    is_safe: bool,
    is_builtin_shape: bool,
    fnptr_indirect: bool,
) -> Option<FnPtrFamily> {
    if is_builtin_shape {
        return is_safe.then_some(FnPtrFamily::BuiltinWrapper);
    }
    fnptr_indirect.then_some(FnPtrFamily::Unknown)
}

/// Whether an fn-pointer signature is the exact source signature of
/// `gateway::BuiltinCodeFn`: `fn(&[PyObjectRef]) -> Result<PyObjectRef,
/// PyError>`.  That family's members are the `__majit_wrap_*` graphs
/// `CallControl::builtin_wrapper_indirect_graphs` enumerates, so a call
/// through it carries the deferred-family marker instead of `None`.
pub(in crate::front::mir) fn fn_ptr_signature_is_builtin_code_fn(
    signature: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
) -> bool {
    let Some(inputs) = signature
        .get("inputs")
        .and_then(serde_json::Value::as_array)
        .filter(|inputs| inputs.len() == 1)
    else {
        return false;
    };
    let input = charon_type_value_to_ast_string(&inputs[0], llbc, 0);
    let output = signature
        .get("output")
        .map(|value| charon_type_value_to_ast_string(value, llbc, 0))
        .unwrap_or_default();
    builtin_code_fn_shape(&input, &output)
}

/// Whether a function declaration has the `gateway::BuiltinCodeFn` type
/// itself — a safe `fn(&[PyObjectRef]) -> Result<PyObjectRef, PyError>`.
/// Every graph an indirect call through that pointer type can reach is such
/// a declaration, so its `FUNC.RESULT` is the pointer type's `RESULT`.
pub(in crate::front::mir) fn fun_decl_is_builtin_code_fn(fd: &FunDecl, llbc: &Llbc) -> bool {
    let [input] = fd.signature.inputs.as_slice() else {
        return false;
    };
    !fd.signature.is_unsafe
        && builtin_code_fn_shape(
            &tyref_to_ast_string(input, llbc),
            &tyref_to_ast_string(&fd.signature.output, llbc),
        )
}

/// The `BuiltinCodeFn` shape test on rendered input / output type strings,
/// shared by the fn-pointer operand and the function-declaration sides.
pub(in crate::front::mir) fn builtin_code_fn_shape(input: &str, output: &str) -> bool {
    input.starts_with('[')
        && input.contains("PyObject")
        && output.starts_with("Result<")
        && output.contains("PyObject")
        && output.contains("PyError")
}

// Helpers

/// Collect the MIR locals bound by a scalar [`Rvalue::BinaryOp`]
/// anywhere in `body`.  See [`Lowering::binop_result_locals`] for why a
/// single function-wide set is sound (a local's MIR type is fixed, so a
/// local bound by `BinaryOp` is a `*Checked` scalar at every read site).
pub(in crate::front::mir) fn compute_binop_result_locals(
    llbc: &Llbc,
    body: &Unstructured,
) -> std::collections::HashSet<usize> {
    let mut set = std::collections::HashSet::new();
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                continue;
            };
            if matches!(rvalue, Rvalue::BinaryOp(..))
                && let PlaceKind::Local(i) = place.kind
            {
                set.insert(i as usize);
            }
        }
    }
    set
}

/// MIR locals assigned more than once anywhere in `body` — statement
/// assigns plus call destinations.  See
/// [`Lowering::multi_assigned_locals`].
pub(in crate::front::mir) fn compute_multi_assigned_locals(
    llbc: &Llbc,
    body: &Unstructured,
) -> std::collections::HashSet<usize> {
    let mut counts: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    let mut bump = |place: &Place| {
        if let PlaceKind::Local(i) = place.kind {
            *counts.entry(i as usize).or_insert(0) += 1;
        }
    };
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            if let Ok(StmtKind::Assign(place, _)) = stmt.stmt_kind_ref() {
                bump(place);
            }
        }
        if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) {
            bump(&call.dest);
        }
    }
    counts
        .into_iter()
        .filter(|(_, c)| *c > 1)
        .map(|(i, _)| i)
        .collect()
}

/// The owning type of a functional-concat string builder: `Wtf8Buf` (the
/// interpreter's repr accumulator) or std `String` (the byte / leaf / module
/// repr helpers).  Both annotate to the immutable `ValueType::Str`, so a
/// `push` / `push_str` append models identically as
/// `buf = ll_strconcat(buf, arg)`;
/// `push_wtf8` is `Wtf8Buf`-only but the leaf gate never matches it on a
/// `String`, so one owner set serves both.
pub(in crate::front::mir) fn is_str_builder_owner(owner: Option<&str>) -> bool {
    matches!(owner, Some("Wtf8Buf") | Some("String"))
}

/// Whether `ty` is a by-value `Wtf8Buf` or `String`.
///
/// A reference or a raw pointer is not an owned accumulator: rebinding the
/// temporary would not rebind the referent.  [`adt_path_of_tyref`] peels
/// `Ref`, so a reference is rejected here before the ADT leaf is read.
/// Both the array form (`{"Ref": [region, ty, kind]}`) and the object form
/// are references.
pub(in crate::front::mir) fn tyref_is_owned_str_builder(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|n| strip_ty_indirections(n, llbc)) else {
        return false;
    };
    let Some(obj) = node.as_object() else {
        return false;
    };
    if obj.contains_key("Ref") || obj.contains_key("RawPtr") {
        return false;
    }
    let Some(path) = adt_path_of_tyref(ty, llbc) else {
        return false;
    };
    is_str_builder_owner(path.rsplit("::").next())
}

/// The owner of a *seed* constructor — one called on a borrowed string to
/// build a fresh buffer, rather than one called on the buffer itself.
///
/// `Wtf8::to_wtf8_buf(&self) -> Wtf8Buf` is `Wtf8Buf { bytes:
/// self.bytes.to_vec() }`: an unconditional fresh allocation holding a copy of
/// the receiver, which is what `Wtf8Buf::from_string` already contributes on
/// the owned side.  Seeding an accumulator from an existing string is the
/// shape `do_stringformat` (`rstr.py`) builds, and a slice feeding one is
/// sound because upstream's `getslice` is a copy, not a view
/// (`ll_stringslice_startonly`, `rstr.py`).
pub(in crate::front::mir) fn is_str_builder_seed_owner(owner: Option<&str>) -> bool {
    matches!(owner, Some("Wtf8"))
}

/// Whether a [`str_builder_ctor_leaf`] leaf CONSTRUCTS a fresh buffer, as
/// opposed to appending to one already built.
///
/// Functional concat does not consult this set.
/// [`owned_builder_dest_locals`] and [`is_fresh_str_builder`] admit any
/// single definition of an owned `Wtf8Buf` / `String`.  Builder mode still
/// requires the unique definition to be one of these leaves
/// ([`single_def_is_str_builder_ctor`]), and the ctor rewrite arm reads the
/// same leaf to choose the `StringBuilder` marker's arguments.  One set,
/// so those two cannot disagree about which leaf constructs a buffer.
pub(in crate::front::mir) fn is_str_builder_ctor(leaf: Option<&str>) -> bool {
    matches!(
        leaf,
        Some("new") | Some("with_capacity") | Some("from_string") | Some("to_wtf8_buf")
    )
}

/// The string-builder ctor leaf a MIR call resolves to, restricted to the
/// fresh-buffer constructors this pass models — `new` / `with_capacity` /
/// `from_string` on a `Wtf8Buf` or `String`, and `to_wtf8_buf` on the borrowed
/// `Wtf8` ([`is_str_builder_seed_owner`]); `None` for any other callee.  Each
/// leaf carries the owner predicate it is admitted under, because a seed ctor
/// is called on the borrowed string and the rest on the buffer.  The cheap
/// leaf check runs before the impl-owner resolution so the type lookup only
/// fires for the candidate names.  Appends are recognised separately by
/// [`str_builder_append_args`].
pub(in crate::front::mir) fn wtf8buf_method_leaf(
    llbc: &Llbc,
    call: &CallPayload,
) -> Option<&'static str> {
    let CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    str_builder_ctor_leaf(llbc, reg)
}

pub(in crate::front::mir) fn str_builder_ctor_leaf(
    llbc: &Llbc,
    reg: &RegularCall,
) -> Option<&'static str> {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    let fd = llbc.fn_by_id(*id)?;
    let np = fd.item_meta.name_path();
    let (leaf, owner_accepted): (&'static str, fn(Option<&str>) -> bool) =
        match np.rsplit("::").next()? {
            "new" => ("new", is_str_builder_owner),
            "with_capacity" => ("with_capacity", is_str_builder_owner),
            "from_string" => ("from_string", is_str_builder_owner),
            "to_wtf8_buf" => ("to_wtf8_buf", is_str_builder_seed_owner),
            "push" => ("push", is_str_builder_owner),
            "push_str" => ("push_str", is_str_builder_owner),
            "push_wtf8" => ("push_wtf8", is_str_builder_owner),
            _ => return None,
        };
    owner_accepted(deref_impl_owner_leaf(llbc, fd).as_deref()).then_some(leaf)
}

/// The `rstring.py` `StringBuilder` method a call resolves to — `new`,
/// `append` or `build` on [`STRING_BUILDER_PATH`] — or `None`.
pub(in crate::front::mir) fn string_builder_method_leaf(
    llbc: &Llbc,
    reg: &RegularCall,
) -> Option<&'static str> {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    let fd = llbc.fn_by_id(*id)?;
    let leaf = match fd.item_meta.name_path().rsplit("::").next()? {
        "new" => "new",
        "append" => "append",
        "build" => "build",
        _ => return None,
    };
    (impl_owner_path(llbc, fd)? == STRING_BUILDER_PATH).then_some(leaf)
}

/// If `reg` is a functional-concat string append, its `(accumulator, piece)`
/// argument positions.  `push` / `push_str` / `push_wtf8` (on `String` /
/// `Wtf8Buf`) take the accumulator as `&mut self` (arg 0) and the appended
/// piece as arg 1.  `String::push(char)` is upstream
/// `AbstractStringBuilderRepr.rtype_method_append`'s `SomeChar` arm;
/// `Wtf8Piece::push_onto(&self, out: &mut Wtf8Buf)` swaps the arguments: the
/// piece is `self` (arg 0) and the accumulator is `out` (arg 1).  Every
/// `Wtf8Piece` impl body is `out.push_str(self)` / `out.push_wtf8(self)`, so
/// `out = out ++ self` models all of them; the accumulator gate
/// ([`append_accumulator_of_arg_temp`]) still restricts the rewrite to a fresh
/// owned local, so a `&mut`-parameter `out` keeps its residual.
pub(in crate::front::mir) fn str_builder_append_args(
    llbc: &Llbc,
    reg: &RegularCall,
) -> Option<(usize, usize)> {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    let fd = llbc.fn_by_id(*id)?;
    let np = fd.item_meta.name_path();
    match np.rsplit("::").next()? {
        "push" | "push_str" | "push_wtf8" => {
            is_str_builder_owner(deref_impl_owner_leaf(llbc, fd).as_deref()).then_some((0, 1))
        }
        // `Wtf8Piece::push_onto`: the `str` / `String` / `Wtf8` impls live
        // under `pyre_interpreter::display`; the `Wtf8Buf` impl resolves to the
        // `Wtf8Buf` owner.  Both append `self` (arg 0) onto the `&mut Wtf8Buf`
        // accumulator (arg 1).
        "push_onto" => (np
            .split("::")
            .take(2)
            .eq([crate::runtime_names::crates::INTERPRETER, "display"])
            || deref_impl_owner_leaf(llbc, fd).as_deref() == Some("Wtf8Buf"))
        .then_some((1, 0)),
        _ => None,
    }
}

/// How an operand references accumulator local `c`: `Some(true)` = a bare
/// `move c` (transfers the whole accumulated string, sound to pass on since
/// the move leaves `c` dead), `Some(false)` = any other reference (`copy c`,
/// or a `move`/`copy` of a projection of `c`) the functional-concat model
/// cannot track, `None` = no reference to `c`.
pub(in crate::front::mir) fn c_operand_kind(op: &Operand, c: usize) -> Option<bool> {
    match op {
        Operand::Move(p) => {
            if matches!(p.kind, PlaceKind::Local(i) if i as usize == c) {
                Some(true)
            } else if place_references_local(p, c) {
                Some(false)
            } else {
                None
            }
        }
        Operand::Copy(p) => place_references_local(p, c).then_some(false),
        Operand::Const(_) => None,
    }
}

/// Whether an operand reads local `l` (a `copy`/`move` whose place bottoms
/// out at `l`).
pub(in crate::front::mir) fn operand_reads_local(op: &Operand, l: usize) -> bool {
    matches!(op, Operand::Copy(p) | Operand::Move(p) if place_references_local(p, l))
}

/// Memo of the per-local string-builder recognizers for one body.
///
/// Each slot is filled at most once. A repeat query returns the stored
/// answer, which is the answer the uncached scan produced for that local.
pub(in crate::front::mir) struct ScanCache {
    pub(in crate::front::mir) fresh: std::cell::RefCell<Vec<Option<bool>>>,
    pub(in crate::front::mir) builder: std::cell::RefCell<Vec<Option<bool>>>,
    pub(in crate::front::mir) clean: std::cell::RefCell<Vec<Option<Option<Vec<usize>>>>>,
    pub(in crate::front::mir) piece: std::cell::RefCell<Vec<Option<Option<usize>>>>,
    /// `sole[local][accumulator as usize]`.
    pub(in crate::front::mir) sole: std::cell::RefCell<Vec<[Option<bool>; 2]>>,
    pub(in crate::front::mir) alias: std::cell::RefCell<Vec<Option<Option<usize>>>>,
    /// `arg_of[local][accumulator as usize]` = [`append_arg_of_borrow`].
    pub(in crate::front::mir) arg_of: std::cell::RefCell<Vec<[Option<Option<usize>>; 2]>>,
    pub(in crate::front::mir) material: std::cell::RefCell<Vec<Option<usize>>>,
    /// [`identity_view_of_borrow`]: a Shared borrow of the accumulator that
    /// is the sole argument of a string-family identity (`as_bytes` /
    /// `deref` / `as_str`), i.e. an `ll_build` read.
    pub(in crate::front::mir) identity: std::cell::RefCell<Vec<Option<Option<usize>>>>,
}

impl ScanCache {
    pub(in crate::front::mir) fn new(n: usize) -> Self {
        Self {
            fresh: std::cell::RefCell::new(vec![None; n]),
            builder: std::cell::RefCell::new(vec![None; n]),
            clean: std::cell::RefCell::new(vec![None; n]),
            piece: std::cell::RefCell::new(vec![None; n]),
            sole: std::cell::RefCell::new(vec![[None, None]; n]),
            alias: std::cell::RefCell::new(vec![None; n]),
            arg_of: std::cell::RefCell::new(vec![[None, None]; n]),
            material: std::cell::RefCell::new(vec![None; n]),
            identity: std::cell::RefCell::new(vec![None; n]),
        }
    }

    pub(in crate::front::mir) fn n(&self) -> usize {
        self.fresh.borrow().len()
    }

    /// One pass of [`is_fresh_str_builder`] over every local.
    pub(in crate::front::mir) fn precompute_fresh(&self, body: &Unstructured, llbc: &Llbc) {
        let n = self.n();
        let def_count = local_def_counts(body, llbc, n);
        let mut fresh = self.fresh.borrow_mut();
        for i in 0..n {
            fresh[i] = Some(owned_single_def_builder(body, llbc, i, def_count[i]));
        }
    }
}

/// Per-body answers of the string-builder recognizers, indexed by MIR local.
///
/// Built once when lowering starts. In-range queries are the values
/// [`is_builder_mode_accumulator`], [`append_accumulator_of_arg_temp`], and
/// [`append_piece_accumulator_of_arg_temp`] return for this body; an index
/// past the local list falls back to those functions.
#[derive(Clone)]
pub(in crate::front::mir) struct AccumulatorFacts {
    pub(in crate::front::mir) builder_mode: Vec<bool>,
    pub(in crate::front::mir) append_accumulator: Vec<Option<usize>>,
    pub(in crate::front::mir) append_piece_accumulator: Vec<Option<usize>>,
    /// Identity-call argument temp → owned builder-mode accumulator.
    /// The temp is the sole argument of `as_bytes` / `deref` / `as_str`
    /// (or a string-family alias of that argument) on a builder-mode
    /// local; the identity arm emits `ll_build` then aliases the dest
    /// (`AbstractStringBuilderRepr.rtype_method_build`).
    pub(in crate::front::mir) identity_accumulator: Vec<Option<usize>>,
    pub(in crate::front::mir) has_builder: bool,
    /// `alloc::fmt::format` dests, their `must_use` / move aliases, and the
    /// shared-ref temps of those: one `Ptr(STR)` (`rstr.py` `do_stringformat`;
    /// the #277 fmt collapsers rewrite the call to `ll_strconcat`, and an
    /// unrecognized call stays unregistered, so its subject never rtypes).
    pub(in crate::front::mir) format_str: Vec<bool>,
}

impl AccumulatorFacts {
    pub(in crate::front::mir) fn build(llbc: &Llbc, body: &Unstructured) -> Self {
        let n = body.locals.locals.len();
        let cache = ScanCache::new(n);
        cache.precompute_fresh(body, llbc);
        let mut builder_mode = vec![false; n];
        for c in 0..n {
            let fresh = cache.fresh.borrow()[c] == Some(true);
            if fresh {
                builder_mode[c] = is_builder_mode_accumulator(&cache, body, llbc, c);
            }
        }
        let owned_dests = owned_builder_dest_locals(body, llbc);
        let mut append_accumulator = vec![None; n];
        for c in owned_dests.iter().copied() {
            let fresh = c < n && cache.fresh.borrow()[c] == Some(true);
            if !fresh {
                continue;
            }
            let Some(receivers) = clean_accumulator_ref_temps(&cache, body, llbc, c) else {
                continue;
            };
            for rt in receivers {
                if let Some(slot) = append_accumulator.get_mut(rt)
                    && slot.is_none()
                {
                    *slot = Some(c);
                }
            }
        }
        let mut append_piece_accumulator = vec![None; n];
        let mut identity_accumulator = vec![None; n];
        let forward_blocks = forward_reachable_mask(llbc, body);
        for c in owned_dests {
            if c >= n || !builder_mode[c] {
                continue;
            }
            for (bb_idx, block) in body.body.iter().enumerate() {
                if !forward_blocks[bb_idx] {
                    continue;
                }
                for stmt in &block.statements {
                    let Ok(StmtKind::Assign(place, Rvalue::Ref { place: source, .. })) =
                        stmt.stmt_kind_ref()
                    else {
                        continue;
                    };
                    if matches!(&source.kind, &PlaceKind::Local(i) if i as usize == c)
                        && let &PlaceKind::Local(direct_borrow) = &place.kind
                    {
                        if let Some(piece) =
                            append_piece_of_borrow(&cache, body, llbc, direct_borrow as usize)
                            && let Some(slot) = append_piece_accumulator.get_mut(piece)
                            && slot.is_none()
                        {
                            *slot = Some(c);
                        }
                        if let Some(id_arg) =
                            identity_view_of_borrow(&cache, body, llbc, direct_borrow as usize)
                            && let Some(slot) = identity_accumulator.get_mut(id_arg)
                            && slot.is_none()
                        {
                            *slot = Some(c);
                        }
                    }
                }
            }
        }
        let has_builder = graph_has_builder_accumulator(llbc, body, &builder_mode);
        let format_str = format_str_locals(body, llbc);
        Self {
            builder_mode,
            append_accumulator,
            append_piece_accumulator,
            identity_accumulator,
            has_builder,
            format_str,
        }
    }
}

pub(in crate::front::mir) enum FormatStrDef {
    /// `alloc::fmt::format` call dest.
    Format,
    /// `must_use(move/copy M)` call dest, `Use(move/copy M)`, or shared `&M`:
    /// marked iff `M` is.
    Alias(usize),
}

/// `alloc::fmt::format` dests, their `must_use` / move aliases, and the
/// shared-ref temps of those. Body locals only (never the return place or a
/// parameter). Least fixpoint: a local is marked when it has at least one
/// def and every whole-local def is format / `must_use` of a marked local /
/// `Use` of a marked local / a shared `&` of a marked local. Any other
/// rvalue, call, or projected write declines the local. A non-Shared borrow
/// of a local (or a projection of it) also declines it: a mutable borrow can
/// change the value without a def of the local (`String::push_str(&mut s)`).
pub(in crate::front::mir) fn format_str_locals(body: &Unstructured, llbc: &Llbc) -> Vec<bool> {
    let n = body.locals.locals.len();
    let arg_count = body.locals.arg_count as usize;
    let mut defs: Vec<Vec<FormatStrDef>> = (0..n).map(|_| Vec::new()).collect();
    let mut declined = vec![false; n];
    let mut saw_format = false;
    let decline_place = |declined: &mut [bool], place: &Place| {
        if let Some(local) = place_root_local(place)
            && local < n
        {
            declined[local] = true;
        }
    };
    for bb in &body.body {
        for st in &bb.statements {
            let Ok(StmtKind::Assign(place, rvalue)) = st.stmt_kind_ref() else {
                continue;
            };
            if let Rvalue::Ref {
                place: src, kind, ..
            } = rvalue
                && kind.as_str() != Some("Shared")
            {
                decline_place(&mut declined, src);
            }
            match &place.kind {
                PlaceKind::Local(dest) => {
                    let dest = *dest as usize;
                    if dest >= n {
                        continue;
                    }
                    match format_str_rvalue_def(rvalue) {
                        Some(FormatStrDef::Format) => {
                            saw_format = true;
                            defs[dest].push(FormatStrDef::Format);
                        }
                        Some(alias) => defs[dest].push(alias),
                        None => declined[dest] = true,
                    }
                }
                PlaceKind::Projection(..) => decline_place(&mut declined, place),
                _ => {}
            }
        }
        if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) {
            match &call.dest.kind {
                PlaceKind::Local(dest) => {
                    let dest = *dest as usize;
                    if dest >= n {
                        continue;
                    }
                    match format_str_call_def(call, llbc) {
                        Some(FormatStrDef::Format) => {
                            saw_format = true;
                            defs[dest].push(FormatStrDef::Format);
                        }
                        Some(alias) => defs[dest].push(alias),
                        None => declined[dest] = true,
                    }
                }
                PlaceKind::Projection(..) => decline_place(&mut declined, &call.dest),
                _ => {}
            }
        }
    }
    if !saw_format {
        return vec![false; n];
    }
    let mut marked = vec![false; n];
    let mut changed = true;
    while changed {
        changed = false;
        for local in arg_count + 1..n {
            if marked[local] || declined[local] || defs[local].is_empty() {
                continue;
            }
            if defs[local].iter().all(|d| match d {
                FormatStrDef::Format => true,
                FormatStrDef::Alias(m) => marked.get(*m).copied() == Some(true),
            }) {
                marked[local] = true;
                changed = true;
            }
        }
    }
    marked
}

pub(in crate::front::mir) fn format_str_rvalue_def(rvalue: &Rvalue) -> Option<FormatStrDef> {
    match rvalue {
        Rvalue::Use(op, _) => operand_local(Some(op)).map(FormatStrDef::Alias),
        Rvalue::Ref { place, kind, .. } => {
            if kind.as_str() != Some("Shared") {
                return None;
            }
            match place.kind {
                PlaceKind::Local(m) => Some(FormatStrDef::Alias(m as usize)),
                _ => None,
            }
        }
        _ => None,
    }
}

pub(in crate::front::mir) fn format_str_call_def(
    call: &CallPayload,
    llbc: &Llbc,
) -> Option<FormatStrDef> {
    let CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    let fd = llbc.fn_by_id(*id)?;
    let np = fd.item_meta.name_path();
    let segs: Vec<String> = np.split("::").map(str::to_string).collect();
    if fmt_path_ends_with(&segs, &["fmt", "format"]) {
        return Some(FormatStrDef::Format);
    }
    if np == "core::hint::must_use" && call.args.len() == 1 {
        return operand_local(call.args.first()).map(FormatStrDef::Alias);
    }
    None
}

/// Whether a builder borrow temp is used exactly once in the selected role of
/// a recognised append (`accumulator == true` selects the receiver, false the
/// appended piece) and nowhere else.  A temp that also escapes declines.
pub(in crate::front::mir) fn ref_temp_is_sole_append_arg(
    cache: &ScanCache,
    body: &Unstructured,
    llbc: &Llbc,
    rt: usize,
    accumulator: bool,
) -> bool {
    let slot = usize::from(accumulator);
    if rt < cache.n()
        && let Some(hit) = cache.sole.borrow()[rt][slot]
    {
        return hit;
    }
    let mut selected = 0usize;
    let mut other = 0usize;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for st in &bb.statements {
            match st.stmt_kind_ref() {
                Ok(StmtKind::Assign(place, rvalue)) => {
                    let (mut derefs, mut others) = (0usize, 0usize);
                    scan_rvalue_dest_ref(rvalue, rt, &mut derefs, &mut others);
                    other += derefs + others;
                    // A write through a projection of `rt` (`*rt = v`) reads
                    // `rt`; a bare `rt := ..` def does not.
                    if matches!(&place.kind, PlaceKind::Projection(..))
                        && place_references_local(place, rt)
                    {
                        other += 1;
                    }
                }
                Ok(StmtKind::Assert(a)) if operand_reads_local(&a.cond, rt) => {
                    other += 1;
                }
                _ => {}
            }
        }
        match bb.term_ref(llbc) {
            Ok(TermKind::Call { call, .. }) => {
                let selected_idx = match &call.func {
                    CallFunc::Regular(reg) => str_builder_append_args(llbc, reg)
                        .map(|(acc, piece)| if accumulator { acc } else { piece }),
                    _ => None,
                };
                if let CallFunc::Dynamic(op, _) = &call.func
                    && operand_reads_local(op, rt)
                {
                    other += 1;
                }
                for (i, arg) in call.args.iter().enumerate() {
                    if operand_reads_local(arg, rt) {
                        if selected_idx == Some(i) {
                            selected += 1;
                        } else {
                            other += 1;
                        }
                    }
                }
            }
            Ok(TermKind::Switch { discr, .. }) => {
                if operand_reads_local(discr, rt) {
                    other += 1;
                }
            }
            Ok(TermKind::Assert { assert, .. }) if operand_reads_local(&assert.cond, rt) => {
                other += 1;
            }
            _ => {}
        }
    }
    let result = selected == 1 && other == 0;
    if rt < cache.n() {
        cache.sole.borrow_mut()[rt][slot] = Some(result);
    }
    result
}

/// Given an owned single-def `Wtf8Buf` / `String` accumulator local
/// `c`, return its append-receiver temps if every use of `c` is append-safe,
/// else `None`.  Append-safe uses: its single definition; a borrow `_r := &(mut) c`
/// that reaches an append accumulator argument ([`append_receiver_of_borrow`],
/// directly or through a single two-phase reborrow); a Shared borrow that is
/// the sole argument of a string-family identity (`as_bytes` / `deref` /
/// `as_str`, [`identity_view_of_borrow`]) — a terminal `ll_build` read
/// (`AbstractStringBuilderRepr.rtype_method_build`); a `mem::take(&mut c)`
/// borrow ([`take_receiver_of_borrow`], including the two-phase reborrow
/// `take` uses) — `old = place; place = empty`; or a bare `move c`
/// (ownership transfer of the complete accumulated string).  Any other use — a
/// `copy`, a projection read/write (`c.field` / `*c`), `&c.field`, `Len`/
/// `Discriminant`, an opaque `escape(&c)`, or `c` in an assert / switch — is an
/// out-of-band access the functional model cannot track, so the whole local
/// declines.  The returned temps are the call-argument locals the append arm
/// resolves against ([`append_accumulator_of_arg_temp`]), which for a
/// reborrowed `&mut` argument is the reborrow, not the direct borrow. Identity
/// and take temps are accepted but not returned: they are not append receivers.
pub(in crate::front::mir) fn clean_accumulator_ref_temps(
    cache: &ScanCache,
    body: &Unstructured,
    llbc: &Llbc,
    c: usize,
) -> Option<Vec<usize>> {
    if c < cache.n()
        && let Some(hit) = cache.clean.borrow()[c].clone()
    {
        return hit;
    }
    let result = clean_accumulator_ref_temps_scan(cache, body, llbc, c);
    if c < cache.n() {
        cache.clean.borrow_mut()[c] = Some(result.clone());
    }
    result
}

pub(in crate::front::mir) fn clean_accumulator_ref_temps_scan(
    cache: &ScanCache,
    body: &Unstructured,
    llbc: &Llbc,
    c: usize,
) -> Option<Vec<usize>> {
    let mut ref_temps: Vec<usize> = Vec::new();
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for st in &bb.statements {
            match st.stmt_kind_ref() {
                Ok(StmtKind::Assign(place, rvalue)) => {
                    // A write into `c`'s storage through a projection
                    // (`*c = ..` / `c.field = ..`) mutates the buffer
                    // out-of-band.
                    if matches!(&place.kind, PlaceKind::Projection(..))
                        && place_references_local(place, c)
                    {
                        return None;
                    }
                    match rvalue {
                        Rvalue::Ref { place: rp, .. } | Rvalue::RawPtr { place: rp, .. } => {
                            if matches!(&rp.kind, &PlaceKind::Local(i) if i as usize == c) {
                                let &PlaceKind::Local(r) = &place.kind else {
                                    return None; // borrow assigned into a projection
                                };
                                ref_temps.push(r as usize);
                            } else if place_references_local(rp, c) {
                                return None; // `&c.field` / `&*c` — escape
                            }
                        }
                        Rvalue::Len(p) | Rvalue::Discriminant(p) => {
                            if place_references_local(p, c) {
                                return None;
                            }
                        }
                        Rvalue::Use(op, _)
                        | Rvalue::UnaryOp(_, op)
                        | Rvalue::Cast(_, op, _)
                        | Rvalue::Repeat(op, _, _, _)
                        | Rvalue::ShallowInitBox(op, _) => {
                            if c_operand_kind(op, c) == Some(false) {
                                return None;
                            }
                        }
                        Rvalue::BinaryOp(_, l, r) => {
                            if c_operand_kind(l, c) == Some(false)
                                || c_operand_kind(r, c) == Some(false)
                            {
                                return None;
                            }
                        }
                        Rvalue::Aggregate(_, ops) => {
                            if ops.iter().any(|op| c_operand_kind(op, c) == Some(false)) {
                                return None;
                            }
                        }
                        Rvalue::NullaryOp(_, _) | Rvalue::Unknown => {}
                    }
                }
                Ok(StmtKind::Assert(a)) => {
                    if c_operand_kind(&a.cond, c).is_some() {
                        return None;
                    }
                }
                Ok(StmtKind::PlaceMention(_))
                | Ok(StmtKind::StorageLive(_))
                | Ok(StmtKind::StorageDead(_))
                | Ok(StmtKind::Borrowck(_))
                | Ok(StmtKind::Unknown) => {}
                Err(_) => return None,
            }
        }
        match bb.term_ref(llbc) {
            Ok(TermKind::Call { call, .. }) => {
                if let CallFunc::Dynamic(op, _) = &call.func
                    && c_operand_kind(op, c).is_some()
                {
                    return None;
                }
                // `move c` into a call transfers the complete string (sound);
                // a `copy` or projected reference declines.
                if call
                    .args
                    .iter()
                    .any(|a| c_operand_kind(a, c) == Some(false))
                {
                    return None;
                }
            }
            Ok(TermKind::Switch { discr, .. }) => {
                if c_operand_kind(discr, c).is_some() {
                    return None;
                }
            }
            Ok(TermKind::Assert { assert, .. }) if c_operand_kind(&assert.cond, c).is_some() => {
                return None;
            }
            _ => {}
        }
    }
    let mut receivers = Vec::with_capacity(ref_temps.len());
    for &rt in &ref_temps {
        if let Some(receiver) = append_receiver_of_borrow(cache, body, llbc, rt) {
            receivers.push(receiver);
        } else if append_piece_of_borrow(cache, body, llbc, rt).is_none()
            && identity_view_of_borrow(cache, body, llbc, rt).is_none()
            && take_receiver_of_borrow(body, llbc, rt).is_none()
        {
            return None;
        }
    }
    Some(receivers)
}

/// Whether `call` is `core::mem::take` / `std::mem::take` of one argument.
pub(in crate::front::mir) fn is_mem_take_call(llbc: &Llbc, call: &CallPayload) -> bool {
    let CallFunc::Regular(reg) = &call.func else {
        return false;
    };
    if call.args.len() != 1 {
        return false;
    }
    let Some(name) = regular_call_name_path(reg, llbc) else {
        return false;
    };
    matches!(name.rsplit("::").next(), Some("take"))
        && (name.starts_with("core::mem::") || name.starts_with("std::mem::"))
}

/// Whether borrow temp `rt` is used exactly once as the sole argument of
/// `mem::take` and nowhere else.
pub(in crate::front::mir) fn ref_temp_is_sole_take_arg(
    body: &Unstructured,
    llbc: &Llbc,
    rt: usize,
) -> bool {
    let mut selected = 0usize;
    let mut other = 0usize;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for st in &bb.statements {
            match st.stmt_kind_ref() {
                Ok(StmtKind::Assign(place, rvalue)) => {
                    let (mut derefs, mut others) = (0usize, 0usize);
                    scan_rvalue_dest_ref(rvalue, rt, &mut derefs, &mut others);
                    other += derefs + others;
                    if matches!(&place.kind, PlaceKind::Projection(..))
                        && place_references_local(place, rt)
                    {
                        other += 1;
                    }
                }
                Ok(StmtKind::Assert(a)) if operand_reads_local(&a.cond, rt) => {
                    other += 1;
                }
                _ => {}
            }
        }
        match bb.term_ref(llbc) {
            Ok(TermKind::Call { call, .. }) => {
                if let CallFunc::Dynamic(op, _) = &call.func
                    && operand_reads_local(op, rt)
                {
                    other += 1;
                }
                let take = is_mem_take_call(llbc, call);
                for (i, arg) in call.args.iter().enumerate() {
                    if operand_reads_local(arg, rt) {
                        if take && i == 0 {
                            selected += 1;
                        } else {
                            other += 1;
                        }
                    }
                }
            }
            Ok(TermKind::Switch { discr, .. }) => {
                if operand_reads_local(discr, rt) {
                    other += 1;
                }
            }
            Ok(TermKind::Assert { assert, .. }) if operand_reads_local(&assert.cond, rt) => {
                other += 1;
            }
            _ => {}
        }
    }
    selected == 1 && other == 0
}

/// Resolve a direct `&(mut) acc` borrow temp `rt` to the temp actually
/// passed as `mem::take`'s argument.  Method-autoref `take(&mut acc)`
/// reaches the call as `rt`; a two-phase reborrow
/// (`rt := &mut acc; recv := &TwoPhaseMut *rt`) is the same walk
/// [`append_receiver_of_borrow`] uses for an explicit `&mut buf`
/// append.  Accepted by [`clean_accumulator_ref_temps`] like an identity
/// view: the borrow does not disqualify the accumulator and is not an
/// append receiver.
pub(in crate::front::mir) fn take_receiver_of_borrow(
    body: &Unstructured,
    llbc: &Llbc,
    rt: usize,
) -> Option<usize> {
    if ref_temp_is_sole_take_arg(body, llbc, rt) {
        return Some(rt);
    }
    let mut recv: Option<usize> = None;
    let mut other = 0usize;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for st in &bb.statements {
            match st.stmt_kind_ref() {
                Ok(StmtKind::Assign(place, rvalue)) => {
                    if let Rvalue::Ref { place: rp, .. } | Rvalue::RawPtr { place: rp, .. } = rvalue
                        && place_is_immediate_deref_of(rp, rt)
                    {
                        let &PlaceKind::Local(r) = &place.kind else {
                            return None;
                        };
                        if recv.replace(r as usize).is_some() {
                            return None;
                        }
                        continue;
                    }
                    let (mut derefs, mut others) = (0usize, 0usize);
                    scan_rvalue_dest_ref(rvalue, rt, &mut derefs, &mut others);
                    other += derefs + others;
                    if matches!(&place.kind, PlaceKind::Projection(..))
                        && place_references_local(place, rt)
                    {
                        other += 1;
                    }
                }
                Ok(StmtKind::Assert(a)) if operand_reads_local(&a.cond, rt) => {
                    other += 1;
                }
                _ => {}
            }
        }
        match bb.term_ref(llbc) {
            Ok(TermKind::Call { call, .. }) => {
                if let CallFunc::Dynamic(op, _) = &call.func
                    && operand_reads_local(op, rt)
                {
                    other += 1;
                }
                if call.args.iter().any(|a| operand_reads_local(a, rt)) {
                    other += 1;
                }
            }
            Ok(TermKind::Switch { discr, .. }) => {
                if operand_reads_local(discr, rt) {
                    other += 1;
                }
            }
            Ok(TermKind::Assert { assert, .. }) if operand_reads_local(&assert.cond, rt) => {
                other += 1;
            }
            _ => {}
        }
    }
    let recv = recv?;
    if other != 0 {
        return None;
    }
    ref_temp_is_sole_take_arg(body, llbc, recv).then_some(recv)
}

/// Whether `call` is a 1-arg string-family identity: `as_bytes` / `bytes`,
/// `as_str` / `as_str_unchecked` / `as_ref` / `borrow`, or string-family
/// `deref` / `deref_mut`.  Same leaves as the identity arms in call
/// lowering; a Shared borrow of a builder that is the sole argument of
/// one of these is an `ll_build` (`AbstractStringBuilderRepr.rtype_method_build`),
/// not an escape.
pub(in crate::front::mir) fn is_string_identity_call(llbc: &Llbc, call: &CallPayload) -> bool {
    let CallFunc::Regular(reg) = &call.func else {
        return false;
    };
    if call.args.len() != 1 {
        return false;
    }
    let Some(first_ty) = operand_tyref(&call.args[0]) else {
        return false;
    };
    if !tyref_is_string_family(first_ty, llbc) {
        return false;
    }
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return false;
    };
    let Some(fd) = llbc.fn_by_id(*id) else {
        return false;
    };
    let np = fd.item_meta.name_path();
    if is_as_view_noop(llbc, reg, Some(first_ty), &call.dest.ty) {
        return true;
    }
    match np.rsplit("::").next() {
        // A Rust-heap `String` receiver keeps the pre-existing `as_bytes`
        // identity for len / eq / iteration consumers; its window conversion
        // is a separate follow-up.
        Some("as_bytes") => true,
        Some("bytes") if np.contains("str") => true,
        Some(leaf @ ("as_ref" | "as_str" | "as_str_unchecked" | "borrow")) => {
            tyref_strips_to_str(&call.dest.ty, llbc)
                || is_as_ref_byte_view(Some(leaf), &call.dest.ty, llbc)
        }
        Some("deref" | "deref_mut") => tyref_is_string_family(&call.dest.ty, llbc),
        _ => false,
    }
}

/// Whether borrow temp `rt` is used exactly once as the sole argument of a
/// string-family identity call ([`is_string_identity_call`]) and nowhere else.
pub(in crate::front::mir) fn ref_temp_is_sole_identity_arg(
    _cache: &ScanCache,
    body: &Unstructured,
    llbc: &Llbc,
    rt: usize,
) -> bool {
    let mut selected = 0usize;
    let mut other = 0usize;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for st in &bb.statements {
            match st.stmt_kind_ref() {
                Ok(StmtKind::Assign(place, rvalue)) => {
                    let (mut derefs, mut others) = (0usize, 0usize);
                    scan_rvalue_dest_ref(rvalue, rt, &mut derefs, &mut others);
                    other += derefs + others;
                    if matches!(&place.kind, PlaceKind::Projection(..))
                        && place_references_local(place, rt)
                    {
                        other += 1;
                    }
                }
                Ok(StmtKind::Assert(a)) if operand_reads_local(&a.cond, rt) => {
                    other += 1;
                }
                _ => {}
            }
        }
        match bb.term_ref(llbc) {
            Ok(TermKind::Call { call, .. }) => {
                if let CallFunc::Dynamic(op, _) = &call.func
                    && operand_reads_local(op, rt)
                {
                    other += 1;
                }
                let identity = is_string_identity_call(llbc, call);
                for (i, arg) in call.args.iter().enumerate() {
                    if operand_reads_local(arg, rt) {
                        if identity && i == 0 {
                            selected += 1;
                        } else {
                            other += 1;
                        }
                    }
                }
            }
            Ok(TermKind::Switch { discr, .. }) => {
                if operand_reads_local(discr, rt) {
                    other += 1;
                }
            }
            Ok(TermKind::Assert { assert, .. }) if operand_reads_local(&assert.cond, rt) => {
                other += 1;
            }
            _ => {}
        }
    }
    selected == 1 && other == 0
}

/// Resolve a Shared/`&` borrow of an owned `Wtf8Buf` / `String` to the temp
/// actually passed as the sole argument of a string-family identity
/// (`as_bytes` / `deref` / `as_str`).  Walks the same reference/coercion
/// shells [`append_piece_of_borrow`] walks (`&Wtf8Buf` → `Deref` → `&Wtf8`
/// → `as_bytes`).  The identity is `ll_build`
/// (`AbstractStringBuilderRepr.rtype_method_build`): the builder is one
/// value, and this read materialises the finished string.
pub(in crate::front::mir) fn identity_view_of_borrow(
    cache: &ScanCache,
    body: &Unstructured,
    llbc: &Llbc,
    rt: usize,
) -> Option<usize> {
    if rt < cache.n()
        && let Some(hit) = cache.identity.borrow()[rt]
    {
        return hit;
    }
    let result = (|| {
        let mut current = rt;
        for _ in 0..ALIAS_WALK_HOPS {
            if ref_temp_is_sole_identity_arg(cache, body, llbc, current) {
                return Some(current);
            }
            current = sole_string_alias_successor(cache, body, llbc, current)?;
        }
        None
    })();
    if rt < cache.n() {
        cache.identity.borrow_mut()[rt] = Some(result);
    }
    result
}

/// Resolve a direct `&(mut) buf` borrow temp `rt` to the temp actually passed
/// as the append's accumulator argument.  A method-autoref receiver
/// (`buf.push(ch)` / `buf.push_str(s)`) reaches the call unchanged, so `rt` is
/// itself the sole
/// append receiver.  An explicit `&mut buf` argument (`Wtf8Piece::push_onto(&s,
/// &mut buf)`) is a two-phase reborrow — `rt := &Mut buf; recv := &TwoPhaseMut
/// *rt` — so the call receives `recv` instead; accept it when `rt`'s only use
/// is that single reborrow and `recv` is itself a sole append receiver.  The
/// returned temp is the call's accumulator-argument local the append arm
/// resolves against ([`append_accumulator_of_arg_temp`]).
pub(in crate::front::mir) fn append_receiver_of_borrow(
    cache: &ScanCache,
    body: &Unstructured,
    llbc: &Llbc,
    rt: usize,
) -> Option<usize> {
    append_arg_of_borrow(cache, body, llbc, rt, true)
}

/// [`append_receiver_of_borrow`]'s twin for the appended PIECE: walk the
/// reference/coercion shells the lowered string model aliases away
/// ([`sole_string_alias_successor`]) until one of them is the sole
/// non-accumulator argument of an append call, and return that temp.
///
/// The walk is bounded because it is a walk over MIR temps and nothing
/// proves the alias chain acyclic; eight hops is far past every chain the
/// real LLBC produces (`&Wtf8Buf` -> `&Wtf8` -> `&str` is three), and a
/// chain longer than that simply declines the builder lift and leaves the
/// residual — the same fail-safe every recognizer in this file takes.
pub(in crate::front::mir) fn append_piece_of_borrow(
    cache: &ScanCache,
    body: &Unstructured,
    llbc: &Llbc,
    rt: usize,
) -> Option<usize> {
    if rt < cache.n()
        && let Some(hit) = cache.piece.borrow()[rt]
    {
        return hit;
    }
    let result = (|| {
        let mut current = rt;
        for _ in 0..ALIAS_WALK_HOPS {
            if ref_temp_is_sole_append_arg(cache, body, llbc, current, false) {
                return Some(current);
            }
            current = sole_string_alias_successor(cache, body, llbc, current)?;
        }
        None
    })();
    if rt < cache.n() {
        cache.piece.borrow_mut()[rt] = Some(result);
    }
    result
}

/// Hop budget for a reference/coercion alias walk over MIR temps.  See
/// [`append_piece_of_borrow`] for why it is bounded and why eight.
pub(in crate::front::mir) const ALIAS_WALK_HOPS: usize = 8;

/// Follow one Rust reference/coercion shell which the lowered string value
/// model aliases away: an immediate reborrow, or `Deref::deref` between two
/// string-family types.  The source temp must have exactly this one use.
pub(in crate::front::mir) fn sole_string_alias_successor(
    cache: &ScanCache,
    body: &Unstructured,
    llbc: &Llbc,
    source: usize,
) -> Option<usize> {
    if source < cache.n()
        && let Some(hit) = cache.alias.borrow()[source]
    {
        return hit;
    }
    let result = sole_string_alias_successor_scan(body, llbc, source);
    if source < cache.n() {
        cache.alias.borrow_mut()[source] = Some(result);
    }
    result
}

pub(in crate::front::mir) fn sole_string_alias_successor_scan(
    body: &Unstructured,
    llbc: &Llbc,
    source: usize,
) -> Option<usize> {
    fn record(slot: &mut Option<usize>, other: &mut usize, candidate: usize) {
        if slot.replace(candidate).is_some() {
            *other += 1;
        }
    }
    let mut successor: Option<usize> = None;
    let mut other = 0usize;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &block.statements {
            match stmt.stmt_kind_ref() {
                Ok(StmtKind::Assign(place, rvalue)) => {
                    if let Rvalue::Ref { place: inner, .. } | Rvalue::RawPtr { place: inner, .. } =
                        rvalue
                        && place_is_immediate_deref_of(inner, source)
                    {
                        if let &PlaceKind::Local(dest) = &place.kind {
                            record(&mut successor, &mut other, dest as usize);
                        } else {
                            other += 1;
                        }
                        continue;
                    }
                    let (mut derefs, mut others) = (0usize, 0usize);
                    scan_rvalue_dest_ref(rvalue, source, &mut derefs, &mut others);
                    other += derefs + others;
                    if matches!(&place.kind, PlaceKind::Projection(..))
                        && place_references_local(place, source)
                    {
                        other += 1;
                    }
                }
                Ok(StmtKind::Assert(assert)) => {
                    other += usize::from(operand_reads_local(&assert.cond, source));
                }
                _ => {}
            }
        }
        match block.term_ref(llbc) {
            Ok(TermKind::Call { call, .. }) => {
                let reads = call
                    .args
                    .iter()
                    .filter(|arg| operand_reads_local(arg, source))
                    .count();
                let string_deref = if reads == 1 && call.args.len() == 1 {
                    matches!(&call.func, CallFunc::Regular(reg) if is_deref_call(reg, llbc))
                        && operand_tyref(&call.args[0])
                            .is_some_and(|ty| tyref_is_string_value(ty, llbc))
                        && tyref_is_string_value(&call.dest.ty, llbc)
                } else {
                    false
                };
                if string_deref {
                    if let &PlaceKind::Local(dest) = &call.dest.kind {
                        record(&mut successor, &mut other, dest as usize);
                    } else {
                        other += 1;
                    }
                } else {
                    other += reads;
                }
                if let CallFunc::Dynamic(op, _) = &call.func {
                    other += usize::from(operand_reads_local(op, source));
                }
            }
            Ok(TermKind::Switch { discr, .. }) => {
                other += usize::from(operand_reads_local(discr, source));
            }
            Ok(TermKind::Assert { assert, .. }) => {
                other += usize::from(operand_reads_local(&assert.cond, source));
            }
            _ => {}
        }
    }
    (other == 0).then_some(successor?)
}

/// The shared body behind [`append_receiver_of_borrow`] (`accumulator =
/// true`) and its piece-side caller: return `rt` when it already is the
/// selected append argument, else the single temp that borrows it and is.
/// More than one such borrower is ambiguous and declines.
pub(in crate::front::mir) fn append_arg_of_borrow(
    cache: &ScanCache,
    body: &Unstructured,
    llbc: &Llbc,
    rt: usize,
    accumulator: bool,
) -> Option<usize> {
    let slot = usize::from(accumulator);
    if rt < cache.n()
        && let Some(hit) = cache.arg_of.borrow()[rt][slot]
    {
        return hit;
    }
    let result = append_arg_of_borrow_scan(cache, body, llbc, rt, accumulator);
    if rt < cache.n() {
        cache.arg_of.borrow_mut()[rt][slot] = Some(result);
    }
    result
}

pub(in crate::front::mir) fn append_arg_of_borrow_scan(
    cache: &ScanCache,
    body: &Unstructured,
    llbc: &Llbc,
    rt: usize,
    accumulator: bool,
) -> Option<usize> {
    let is_selected =
        |candidate| ref_temp_is_sole_append_arg(cache, body, llbc, candidate, accumulator);
    if is_selected(rt) {
        return Some(rt);
    }
    let mut recv: Option<usize> = None;
    let mut other = 0usize;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for st in &bb.statements {
            match st.stmt_kind_ref() {
                Ok(StmtKind::Assign(place, rvalue)) => {
                    if let Rvalue::Ref { place: rp, .. } | Rvalue::RawPtr { place: rp, .. } = rvalue
                        && place_is_immediate_deref_of(rp, rt)
                    {
                        let &PlaceKind::Local(r) = &place.kind else {
                            return None; // reborrow assigned into a projection
                        };
                        if recv.replace(r as usize).is_some() {
                            return None; // more than one reborrow of `rt`
                        }
                        continue;
                    }
                    let (mut derefs, mut others) = (0usize, 0usize);
                    scan_rvalue_dest_ref(rvalue, rt, &mut derefs, &mut others);
                    other += derefs + others;
                    if matches!(&place.kind, PlaceKind::Projection(..))
                        && place_references_local(place, rt)
                    {
                        other += 1;
                    }
                }
                Ok(StmtKind::Assert(a)) if operand_reads_local(&a.cond, rt) => {
                    other += 1;
                }
                _ => {}
            }
        }
        // `rt` is consumed by the reborrow statement; it must not reach any
        // terminator (the append call receives `recv`, not `rt`).
        match bb.term_ref(llbc) {
            Ok(TermKind::Call { call, .. }) => {
                if let CallFunc::Dynamic(op, _) = &call.func
                    && operand_reads_local(op, rt)
                {
                    other += 1;
                }
                if call.args.iter().any(|a| operand_reads_local(a, rt)) {
                    other += 1;
                }
            }
            Ok(TermKind::Switch { discr, .. }) => {
                if operand_reads_local(discr, rt) {
                    other += 1;
                }
            }
            Ok(TermKind::Assert { assert, .. }) if operand_reads_local(&assert.cond, rt) => {
                other += 1;
            }
            _ => {}
        }
    }
    let recv = recv?;
    if other != 0 {
        return None;
    }
    is_selected(recv).then_some(recv)
}

/// Whether MIR local `c` is an owned functional-concat accumulator: not
/// local 0, not a parameter, declared as a by-value `Wtf8Buf` or `String`,
/// and defined exactly once — by a constructor, by any other call, or by a
/// move out of a `?` `Continue` payload.  A second definition, a reference
/// or raw-pointer type, or a parameter keeps its residual.  Rust ownership
/// of that single definition means the local is not aliased, so rebinding
/// it as `ll_strconcat` is sound.
pub(in crate::front::mir) fn is_fresh_str_builder(
    cache: &ScanCache,
    body: &Unstructured,
    llbc: &Llbc,
    c: usize,
) -> bool {
    if c < cache.n()
        && let Some(hit) = cache.fresh.borrow()[c]
    {
        return hit;
    }
    let n = body.locals.locals.len();
    let count = local_def_counts(body, llbc, n).get(c).copied().unwrap_or(0);
    let result = owned_single_def_builder(body, llbc, c, count);
    if c < cache.n() {
        cache.fresh.borrow_mut()[c] = Some(result);
    }
    result
}

/// Resolve the accumulator local a recognised append rebinds, given the MIR
/// local `rt` passed as the append's accumulator argument (`push` / `push_str` /
/// `push_wtf8` receiver, or `Wtf8Piece::push_onto` `&mut buf` argument).  `rt`
/// borrows the accumulator directly (method autoref) or through a single
/// two-phase reborrow; the owning `buf` is the owned single-def `Wtf8Buf` /
/// `String` ([`owned_builder_dest_locals`]) whose every borrow feeds such an
/// append and whose append-receiver set ([`clean_accumulator_ref_temps`])
/// contains `rt`.  Lowering records the answer for every local once per body;
/// the append arm reads that table and rebinds `buf`'s slot to
/// `ll_strconcat(buf, piece)`.
pub(in crate::front::mir) fn append_accumulator_of_arg_temp(
    body: &Unstructured,
    llbc: &Llbc,
    rt: usize,
) -> Option<usize> {
    // Only an owned single-def `Wtf8Buf` / `String` past the parameters can
    // pass [`is_fresh_str_builder`] ([`owned_builder_dest_locals`], which
    // already excludes local 0 and the parameters), and a ref-temp borrows
    // exactly one local so at most one accumulator's receiver set contains
    // `rt`; check those candidates instead of every local — dropping the
    // `n_locals` factor — and first-match order is therefore irrelevant.
    //
    // Call lowering reads the per-body [`AccumulatorFacts`] table. This scan
    // remains for an index outside that table.
    let cache = ScanCache::new(body.locals.locals.len());
    cache.precompute_fresh(body, llbc);
    owned_builder_dest_locals(body, llbc)
        .into_iter()
        .find(|&c| {
            is_fresh_str_builder(&cache, body, llbc, c)
                && clean_accumulator_ref_temps(&cache, body, llbc, c)
                    .is_some_and(|receivers| receivers.contains(&rt))
        })
}

/// Resolve a shared-borrow temp used as an append piece to the completed
/// inner builder it borrows.  Membership is proven from the MIR definition of
/// `rt` and the same single-use append-piece test accepted by
/// [`clean_accumulator_ref_temps`].
///
/// Call lowering reads the per-body [`AccumulatorFacts`] table. This scan
/// remains for an index outside that table.
pub(in crate::front::mir) fn append_piece_accumulator_of_arg_temp(
    body: &Unstructured,
    llbc: &Llbc,
    rt: usize,
) -> Option<usize> {
    let cache = ScanCache::new(body.locals.locals.len());
    cache.precompute_fresh(body, llbc);
    let forward_blocks = forward_reachable_mask(llbc, body);
    for candidate in owned_builder_dest_locals(body, llbc) {
        if !is_builder_mode_accumulator(&cache, body, llbc, candidate) {
            continue;
        }
        for (bb_idx, block) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            for stmt in &block.statements {
                let Ok(StmtKind::Assign(place, Rvalue::Ref { place: source, .. })) =
                    stmt.stmt_kind_ref()
                else {
                    continue;
                };
                if matches!(source.kind, PlaceKind::Local(i) if i as usize == candidate)
                    && let PlaceKind::Local(direct_borrow) = place.kind
                    && append_piece_of_borrow(&cache, body, llbc, direct_borrow as usize)
                        == Some(rt)
                {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

/// Resolve a string-family identity argument temp to the builder-mode
/// accumulator it borrows.  Membership is the MIR definition of `rt` as a
/// whole-local borrow of that accumulator whose [`identity_view_of_borrow`]
/// is `rt` itself (or a successor the identity arm actually consumes).
///
/// Call lowering reads the per-body [`AccumulatorFacts`] table. This scan
/// remains for an index outside that table.
pub(in crate::front::mir) fn identity_accumulator_of_arg_temp(
    body: &Unstructured,
    llbc: &Llbc,
    rt: usize,
) -> Option<usize> {
    let cache = ScanCache::new(body.locals.locals.len());
    cache.precompute_fresh(body, llbc);
    let forward_blocks = forward_reachable_mask(llbc, body);
    for candidate in owned_builder_dest_locals(body, llbc) {
        if !is_builder_mode_accumulator(&cache, body, llbc, candidate) {
            continue;
        }
        for (bb_idx, block) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            for stmt in &block.statements {
                let Ok(StmtKind::Assign(place, Rvalue::Ref { place: source, .. })) =
                    stmt.stmt_kind_ref()
                else {
                    continue;
                };
                if matches!(source.kind, PlaceKind::Local(i) if i as usize == candidate)
                    && let PlaceKind::Local(direct_borrow) = place.kind
                    && identity_view_of_borrow(&cache, body, llbc, direct_borrow as usize)
                        == Some(rt)
                {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

/// Count the terminal consumptions of a fresh accumulator local `c`.
/// [`clean_accumulator_ref_temps`] already proves every other use of `c`
/// is the ctor def, an append borrow, a string-family identity view, or a
/// `mem::take`, so each `move c`, each inner-builder append piece, each
/// identity-view borrow (`as_bytes` / `deref` / `as_str`), and each
/// `mem::take` is a terminal materialisation of the finished string — the
/// point where a real builder must run `ll_build`
/// (`AbstractStringBuilderRepr.rtype_method_build`).  Exclusive-branch
/// identity views are several such points; sequential `build()` is valid
/// on one `StringBuilder`.  See [`is_builder_mode_accumulator`].
#[allow(dead_code)] // wired into the ctor/append/terminal arms in this change
pub(in crate::front::mir) fn accumulator_materialization_site_count(
    cache: &ScanCache,
    body: &Unstructured,
    llbc: &Llbc,
    c: usize,
) -> usize {
    if c < cache.n()
        && let Some(hit) = cache.material.borrow()[c]
    {
        return hit;
    }
    let is_move = |op: &Operand| c_operand_kind(op, c) == Some(true);
    let mut moves = 0usize;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for st in &bb.statements {
            if let Ok(StmtKind::Assign(_, rvalue)) = st.stmt_kind_ref() {
                if let Rvalue::Ref { place: source, .. } = rvalue
                    && matches!(&source.kind, &PlaceKind::Local(i) if i as usize == c)
                    && let Ok(StmtKind::Assign(place, _)) = st.stmt_kind_ref()
                    && let &PlaceKind::Local(rt) = &place.kind
                    && (append_piece_of_borrow(cache, body, llbc, rt as usize).is_some()
                        || identity_view_of_borrow(cache, body, llbc, rt as usize).is_some()
                        || take_receiver_of_borrow(body, llbc, rt as usize).is_some())
                {
                    moves += 1;
                }
                match rvalue {
                    Rvalue::Use(op, _)
                    | Rvalue::UnaryOp(_, op)
                    | Rvalue::Cast(_, op, _)
                    | Rvalue::Repeat(op, _, _, _)
                    | Rvalue::ShallowInitBox(op, _) => {
                        if is_move(op) {
                            moves += 1;
                        }
                    }
                    Rvalue::BinaryOp(_, l, r) => {
                        moves += usize::from(is_move(l)) + usize::from(is_move(r));
                    }
                    Rvalue::Aggregate(_, ops) => {
                        for op in ops {
                            if is_move(op) {
                                moves += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) {
            for a in &call.args {
                if is_move(a) {
                    moves += 1;
                }
            }
        }
    }
    if c < cache.n() {
        cache.material.borrow_mut()[c] = Some(moves);
    }
    moves
}

/// Whether the unique definition of local `c` is an [`is_str_builder_ctor`]
/// call.  [`is_fresh_str_builder`] no longer requires a constructor: a call
/// result or a `?` payload move is one definition of an owned buffer.
/// Builder mode still does, because the ctor rewrite arm replaces that
/// defining call with `__majit_stringbuilder_new`.
pub(in crate::front::mir) fn single_def_is_str_builder_ctor(
    body: &Unstructured,
    llbc: &Llbc,
    c: usize,
) -> bool {
    let mut defs = 0usize;
    let mut ctor = false;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for st in &bb.statements {
            if let Ok(StmtKind::Assign(place, _)) = st.stmt_kind_ref()
                && matches!(place.kind, PlaceKind::Local(i) if i as usize == c)
            {
                return false;
            }
        }
        if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc)
            && matches!(call.dest.kind, PlaceKind::Local(i) if i as usize == c)
        {
            defs += 1;
            if defs > 1 {
                return false;
            }
            ctor = is_str_builder_ctor(wtf8buf_method_leaf(llbc, call));
        }
    }
    defs == 1 && ctor
}

/// Whether fresh accumulator local `c` should lift to a real RPython
/// `StringBuilder` (approach D) rather than the functional-concat fallback:
///
/// * an owned single-def `Wtf8Buf` / `String` ([`is_fresh_str_builder`]),
/// * whose unique definition is a buffer constructor
///   ([`single_def_is_str_builder_ctor`]) — a non-constructor definition is
///   an already-built string, and the ctor rewrite would replace that
///   producer with `__majit_stringbuilder_new`,
/// * every borrow of `c` an append, a string-family identity view, or a
///   `mem::take` ([`clean_accumulator_ref_temps`]), and
/// * at least one terminal consumption
///   ([`accumulator_materialization_site_count`]) — a by-value move, an
///   inner-builder append piece, an identity-view `as_bytes` / `deref` /
///   `as_str`, or a `mem::take` (`AbstractStringBuilderRepr.rtype_method_build`).
///   Exclusive branches may each `ll_build`; sequential `build()` is valid
///   on one `StringBuilder`.
///
/// Zero terminal reads, or a non-constructor definition, keep the
/// `ll_strconcat` fallback so the builder rewrite only fires where the
/// ctor, the appends, and a build point are all unambiguous.
/// Lowering records the answer for every local once per body; emit sites
/// read that table.
#[allow(dead_code)] // wired into the ctor/append/terminal arms in this change
pub(in crate::front::mir) fn is_builder_mode_accumulator(
    cache: &ScanCache,
    body: &Unstructured,
    llbc: &Llbc,
    c: usize,
) -> bool {
    if c < cache.n()
        && let Some(hit) = cache.builder.borrow()[c]
    {
        return hit;
    }
    let result = is_fresh_str_builder(cache, body, llbc, c)
        && single_def_is_str_builder_ctor(body, llbc, c)
        && clean_accumulator_ref_temps(cache, body, llbc, c).is_some()
        && accumulator_materialization_site_count(cache, body, llbc, c) >= 1;
    if c < cache.n() {
        cache.builder.borrow_mut()[c] = Some(result);
    }
    result
}

/// Clear `taken_by_address` on the field read that produced `base` when that
/// field is a declared virtualizable array. See
/// [`Lowering::release_declared_vable_array_address`].
pub(crate) fn release_declared_vable_array_address(
    graph: &mut FunctionGraph,
    base: &Variable,
) -> bool {
    let mut released = false;
    for block in &mut graph.blocks {
        for op in &mut block.operations {
            if op.result.as_ref() != Some(base) {
                continue;
            }
            let OpKind::FieldRead { field, .. } = &mut op.kind else {
                continue;
            };
            if !crate::virtualizable_decl::is_declared_array_field(
                field.owner_root.as_deref(),
                &field.name,
            ) {
                continue;
            }
            field.taken_by_address = false;
            released = true;
        }
    }
    released
}

/// Whether a statically-resolved [`RegularCall`] is a workspace
/// `Index::index` / `IndexMut::index_mut` impl (the `FixedObjectArray`
/// family) — those bottom out at raw-slice construction and are lowered
/// as RPython's getarrayitem rather than a residual call.  Shared by the
/// call-lowering intercept and the liveness pre-pass.
pub(in crate::front::mir) fn is_workspace_index_regular(reg: &RegularCall, llbc: &Llbc) -> bool {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return false;
    };
    let Some(fd) = llbc.fn_by_id(*id) else {
        return false;
    };
    let path = fd.item_meta.name_path();
    let leaf = path.rsplit("::").next().unwrap_or("");
    if leaf != "index" && leaf != "index_mut" {
        return false;
    }
    if !path.starts_with("pyre_") {
        return false;
    }
    // `BytesArray::index` / `UnicodeArray::index` stay in this arm.
    // Their bodies are `&self.as_slice()[i]`, and looking inside hits
    // the pair-return fallback (`ll_slice_len_slot_new`, then a residual
    // `as_slice`). The emit substitutes the `block` field
    // ([`string_array_index_owner`]). `IntArray` / `FloatArray` /
    // `FixedObjectArray` index `*base().add(i)` and read the receiver.
    true
}

/// `BytesArray::index` / `UnicodeArray::index` — the workspace read
/// whose base is the items `block`, not the `{ block, len }` record.
pub(in crate::front::mir) fn string_array_index_owner(
    reg: &RegularCall,
    llbc: &Llbc,
) -> Option<&'static str> {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    let path = llbc.fn_by_id(*id)?.item_meta.name_path();
    let leaf = path.rsplit("::").next().unwrap_or("");
    if leaf != "index" && leaf != "index_mut" {
        return None;
    }
    if path.contains("::bytes_array::") {
        Some("BytesArray")
    } else if path.contains("::unicode_array::") {
        Some("UnicodeArray")
    } else {
        None
    }
}

/// The `Vec<T>` index leaf a statically-resolved [`RegularCall`] resolves
/// to — `"index"` (`<Vec<T> as Index>::index`, read) or `"index_mut"`
/// (`<Vec<T> as IndexMut>::index_mut`, write) — restricted to the
/// integer-index impl.  `Vec`'s `Index<Range<…>>` / `IndexMut<Range<…>>`
/// impls share the same leaf but return a sub-`&[T]` slice, not an element;
/// lowering those to a scalar `ArrayRead` / `ArrayWrite` would mis-index,
/// so the index type must be an integer.  Charon runs `monomorphize:false`,
/// so the signature keeps the generic index param `I` (a `TypeVar` typing
/// as `Ref`); the concrete index type is the callsite substitution
/// `types[1]` of the impl generics `[T, I, A]`, which
/// [`vec_index_type_is_scalar`] classifies: `Index<usize>` types as
/// `Unsigned` and signed integer indices as `Int`, and both occupy RPython's
/// integer register bank, while `Index<Range<…>>` resolves to a `Range*` Adt
/// (`Ref`) and returns `None`.  Owner/leaf are derived the same way
/// [`call_target_segments`]
/// derives the call key (`impl_method_owner_for_fundecl`).  Free so the
/// same gate is shared by the call-lowering intercept and the deferred-write
/// liveness pre-pass.
/// Resolve the scalar Vec index leaf from either source Charon preserves.
///
/// Most calls carry the concrete `Index<I>` substitution in
/// `RegularCall.generics.types[1]`.  A few monomorphized call sites instead
/// retain only the actual index operand type; `bind_builtin_kwargs` is the
/// important one, and leaving its `Vec::index` calls opaque makes the whole
/// builtin gateway miss the RPython rlist `getitem` path.  Both spellings are
/// the same upstream operation, so accept either integer-bank proof while
/// continuing to reject `Range*` (Ref-bank) indices.
pub(in crate::front::mir) fn vec_index_regular_leaf_with_callsite(
    reg: &RegularCall,
    index_ty: Option<&TyRef>,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<&'static str> {
    let leaf = vec_index_regular_unchecked_leaf(reg, llbc)?;
    callsite_or_generic_index_is_scalar(
        reg,
        index_ty,
        llbc,
        gc_struct_ids,
        raw_alloc_flavor_structs,
    )
    .then_some(leaf)
}

/// Whether either source Charon preserves proves an integer-bank index.
///
/// The substitution and the call-site operand are the two spellings of the
/// same upstream fact, so a `Range*` (Ref-bank) index is rejected by both
/// answering `false`.  Shared because `<[T]>::get` classifies its index by
/// exactly this rule, and a fix to one reading has to reach the other.
pub(in crate::front::mir) fn callsite_or_generic_index_is_scalar(
    reg: &RegularCall,
    index_ty: Option<&TyRef>,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> bool {
    let generic_index_is_scalar = reg
        .generics
        .get("types")
        .and_then(serde_json::Value::as_array)
        .and_then(|tys| tys.get(1))
        .and_then(|t| serde_json::from_value::<TyRef>(t.clone()).ok())
        .is_some_and(|t| {
            vec_index_type_is_scalar(&t, llbc, gc_struct_ids, raw_alloc_flavor_structs)
        });
    generic_index_is_scalar
        || index_ty.is_some_and(|ty| {
            vec_index_type_is_scalar(ty, llbc, gc_struct_ids, raw_alloc_flavor_structs)
        })
}

/// The Vec Index/IndexMut leaf before classifying its index argument.  Scalar
/// element access and the `RangeFull` whole-list identity share this exact
/// owner proof but intentionally take different lowerings.
pub(in crate::front::mir) fn vec_index_regular_unchecked_leaf(
    reg: &RegularCall,
    llbc: &Llbc,
) -> Option<&'static str> {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    llbc.fn_by_id(*id).and_then(|fd| {
        impl_method_owner_for_fundecl(llbc, fd).and_then(|(owner, leaf)| {
            if owner != "vec::Vec" {
                return None;
            }
            match leaf.as_str() {
                "index" => Some("index"),
                "index_mut" => Some("index_mut"),
                _ => None,
            }
        })
    })
}

/// `Vec<T>::index(_mut)(RangeFull)` creates only a borrowed whole-list view.
/// The translated RPython list model has no distinct borrow/slice wrapper, so
/// the callsite aliases the receiver instead of residualizing a Rust trait
/// shim.
pub(in crate::front::mir) fn is_vec_rangefull_index_regular(
    reg: &RegularCall,
    llbc: &Llbc,
) -> bool {
    if vec_index_regular_unchecked_leaf(reg, llbc).is_none() {
        return false;
    }
    reg.generics
        .get("types")
        .and_then(serde_json::Value::as_array)
        .and_then(|tys| tys.get(1))
        .and_then(|t| serde_json::from_value::<TyRef>(t.clone()).ok())
        .is_some_and(|t| tyref_to_ast_string(&t, llbc) == "RangeFull")
}

/// Shared vs mutable whole-list view of a `Vec<PyObjectRef>`.
///
/// `&vec` / `&vec[..]` / `as_slice` copy the items into one object array.
/// `&mut vec` / `&mut vec[..]` / `as_mut_slice` keep the header address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::front::mir) enum ObjectVecWholeListView {
    Shared,
    Mut,
}

/// Deref / DerefMut / `as_slice` / `as_mut_slice` / `Index(RangeFull)` of a
/// `Vec`. The caller still checks the receiver is a `Vec<PyObjectRef>`.
pub(in crate::front::mir) fn object_vec_whole_list_view(
    reg: &RegularCall,
    nargs: usize,
    llbc: &Llbc,
) -> Option<ObjectVecWholeListView> {
    let path = regular_call_name_path(reg, llbc)?;
    let leaf = path.rsplit("::").next()?;
    if nargs == 1 {
        // `alloc::vec::<Impl>::deref` after monomorphization. A slice
        // `as_slice` has a slice receiver, not a `Vec`.
        let vec_method = path.starts_with("alloc::vec::");
        return match leaf {
            "deref" | "as_slice" if vec_method => Some(ObjectVecWholeListView::Shared),
            "deref_mut" | "as_mut_slice" if vec_method => Some(ObjectVecWholeListView::Mut),
            _ => None,
        };
    }
    if nargs == 2 && is_vec_rangefull_index_regular(reg, llbc) {
        return match vec_index_regular_unchecked_leaf(reg, llbc)? {
            "index" => Some(ObjectVecWholeListView::Shared),
            "index_mut" => Some(ObjectVecWholeListView::Mut),
            _ => None,
        };
    }
    None
}

/// Whether `ty` is an index type [`vec_index_regular_leaf`] may lower to a
/// scalar `ArrayRead` / `ArrayWrite`.
///
/// Both integer banks count. `usize` — what essentially every real callsite
/// indexes with — serializes as `{"Integer": {"Unsigned": "Usize"}}` and types as
/// [`ValueType::Unsigned`], so an `Int`-only test silently rejects it and
/// leaves the whole fold dead: the `#[ignore]`d real-LLBC anchor
/// `vec_index_mut_fill_user_function_args_real` failed with two residual
/// `index_mut` calls.  A `Range` index types as `Ref` and is still rejected,
/// which is what the gate exists for.
pub(in crate::front::mir) fn vec_index_type_is_scalar(
    ty: &TyRef,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> bool {
    // `tyref_to_value_type` re-harvests every `GcType` fact in the crate.
    // This predicate runs on every call while lowering, so reuse the set
    // the lowering already computed.
    matches!(
        tyref_to_value_type_with(
            ty,
            llbc,
            no_tombstoned_leaves(),
            gc_struct_ids,
            raw_alloc_flavor_structs
        ),
        ValueType::Int | ValueType::Unsigned
    )
}

pub(in crate::front::mir) fn is_vec_index_regular_with_callsite(
    reg: &RegularCall,
    index_ty: Option<&TyRef>,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> bool {
    vec_index_regular_leaf_with_callsite(
        reg,
        index_ty,
        llbc,
        gc_struct_ids,
        raw_alloc_flavor_structs,
    )
    .is_some()
}

pub(in crate::front::mir) fn is_vec_index_mut_call(
    reg: &RegularCall,
    index_ty: Option<&TyRef>,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> bool {
    vec_index_regular_leaf_with_callsite(
        reg,
        index_ty,
        llbc,
        gc_struct_ids,
        raw_alloc_flavor_structs,
    ) == Some("index_mut")
}

/// Blocks whose `from_raw_parts` is an rstr `STR` chars view, paired with
/// the `base` local the slice aliases. See [`str_chars_view_base`].
pub(in crate::front::mir) fn str_chars_view_extra_live(
    body: &Unstructured,
    llbc: &Llbc,
) -> Vec<(usize, usize)> {
    let mut sites = Vec::new();
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            continue;
        };
        let CallFunc::Regular(reg) = &call.func else {
            continue;
        };
        let (Some(ptr_local), Some(len_local)) = (
            operand_local(call.args.first()),
            operand_local(call.args.get(1)),
        ) else {
            continue;
        };
        if let Some(base) = str_chars_view_base(body, llbc, reg, ptr_local, len_local) {
            sites.push((bb_idx, base));
        }
    }
    sites
}

/// `from_raw_parts(ptr, len)` where `ptr` is
/// `(base as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET)` and `len` is
/// `bh_lowlevel_string_len(base)` on that same `base`.
///
/// `rstr.py` `STR` stores `chars` as an inline `Array(Char)`. `ll_strlen`
/// is `len(s.chars)` and `strgetitem` is `s.chars[i]`; the slice is the
/// string, not a second object. Any other shape returns `None` so the
/// caller leaves the raw slice alone. The enclosing function name is not
/// consulted.
pub(in crate::front::mir) fn str_chars_view_base(
    body: &Unstructured,
    llbc: &Llbc,
    reg: &RegularCall,
    ptr_local: usize,
    len_local: usize,
) -> Option<usize> {
    if !regular_call_is_slice_from_raw_parts(reg, llbc) {
        return None;
    }
    let add = defining_call(llbc, body, ptr_local)?;
    let CallFunc::Regular(add_reg) = &add.func else {
        return None;
    };
    if add.args.len() != 2 || !regular_call_is_ptr_add(add_reg, llbc) {
        return None;
    }
    if !operand_is_lowlevel_string_chars_offset(body, llbc, &add.args[1]) {
        return None;
    }
    let ptr_base = peel_copy_and_cast(llbc, body, operand_local(add.args.first())?)?;
    let len_local = peel_copies(llbc, body, len_local)?;
    let len_call = defining_call(llbc, body, len_local)?;
    let CallFunc::Regular(len_reg) = &len_call.func else {
        return None;
    };
    if len_call.args.len() != 1 || !regular_call_is_bh_lowlevel_string_len(len_reg, llbc) {
        return None;
    }
    let len_base = peel_copy_and_cast(llbc, body, operand_local(len_call.args.first())?)?;
    (ptr_base == len_base).then_some(ptr_base)
}

pub(in crate::front::mir) fn regular_call_is_slice_from_raw_parts(
    reg: &RegularCall,
    llbc: &Llbc,
) -> bool {
    regular_call_name_path(reg, llbc).is_some_and(|path| {
        matches!(
            path.as_str(),
            "core::slice::raw::from_raw_parts" | "core::slice::raw::from_raw_parts_mut"
        )
    })
}

pub(in crate::front::mir) fn regular_call_is_bh_lowlevel_string_len(
    reg: &RegularCall,
    llbc: &Llbc,
) -> bool {
    regular_call_name_path(reg, llbc)
        .is_some_and(|path| path == "pyre_object::lowlevel_string::bh_lowlevel_string_len")
}

pub(in crate::front::mir) fn operand_is_lowlevel_string_chars_offset(
    body: &Unstructured,
    llbc: &Llbc,
    op: &Operand,
) -> bool {
    if place_is_chars_offset_global(llbc, operand_place(op)) {
        return true;
    }
    let Some(local) = operand_local(Some(op)) else {
        return false;
    };
    let Some(local) = peel_copies(llbc, body, local) else {
        return false;
    };
    match defining_assign(llbc, body, local) {
        Some(Rvalue::Use(inner, _)) => place_is_chars_offset_global(llbc, operand_place(inner)),
        _ => false,
    }
}

/// `LOWLEVEL_STRING_CHARS_OFFSET` (`lowlevel_string.rs`) — `size_of` the
/// rstr `STR` header (`rstr.py` `STR`).
pub(in crate::front::mir) const LOWLEVEL_STRING_CHARS_OFFSET_PATH: &str =
    "pyre_object::lowlevel_string::LOWLEVEL_STRING_CHARS_OFFSET";

pub(in crate::front::mir) fn place_is_chars_offset_global(
    llbc: &Llbc,
    place: Option<&Place>,
) -> bool {
    let Some(PlaceKind::Global { id, .. }) = place.map(|place| &place.kind) else {
        return false;
    };
    llbc.global_by_id(*id)
        .is_some_and(|global| global.item_meta.name_path() == LOWLEVEL_STRING_CHARS_OFFSET_PATH)
}

/// rstr `STR` (`rstr.py` `STR`) — `W_UnicodeObject._utf8` and the layout
/// type next to `LOWLEVEL_STRING_CHARS_OFFSET`.
pub(in crate::front::mir) fn is_lowlevel_string_header_adt_path(path: &str) -> bool {
    matches!(
        path,
        "pyre_object::unicodeobject::Utf8Str"
            | "pyre_object::lowlevel_string::LowlevelStringHeader"
    )
}

/// `*const Utf8Str` / `*mut Utf8Str` / `&Utf8Str` and the same shapes
/// of `LowlevelStringHeader` — the rstr `STR` header (`rstr.py` `STR`).
/// A reference-derived `(s as *const Utf8Str as *const u8)` peels to
/// `&Utf8Str`; that is still `s.chars` (`ll_stritem_nonneg`).
pub(in crate::front::mir) fn tyref_is_lowlevel_string_header_ptr(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(pointee) =
        tyref_peel_one_raw_ptr_node(ty, llbc).or_else(|| tyref_peel_one_ref_node(ty, llbc))
    else {
        return false;
    };
    adt_node_def_id(pointee)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|td| is_lowlevel_string_header_adt_path(td.item_meta.name_path_str()))
}

pub(in crate::front::mir) fn operand_place(op: &Operand) -> Option<&Place> {
    match op {
        Operand::Copy(place) | Operand::Move(place) => Some(place),
        Operand::Const(_) => None,
    }
}

/// The single call whose destination is `local`, if that local has no
/// other producer.
pub(in crate::front::mir) fn defining_call(
    llbc: &Llbc,
    body: &Unstructured,
    local: usize,
) -> Option<CallPayload> {
    let mut found = None;
    let mut producers = 0usize;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            if let Ok(StmtKind::Assign(place, _)) = stmt.stmt_kind_ref()
                && matches!(place.kind, PlaceKind::Local(i) if i as usize == local)
            {
                producers += 1;
            }
        }
        if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc)
            && matches!(call.dest.kind, PlaceKind::Local(i) if i as usize == local)
        {
            producers += 1;
            found = Some(call.clone());
        }
    }
    (producers == 1).then_some(found).flatten()
}

/// The vtable slot a monomorphized `dyn` call reads: `operand` is a
/// temporary whose single producer is a raw-pointer cast of that slot
/// (`_t = copy (*ptr_metadata(recv)).method_<name> as fn(..)`).
pub(in crate::front::mir) fn vtable_slot_under_fn_ptr_cast(
    llbc: &Llbc,
    body: &Unstructured,
    operand: &Operand,
) -> Option<Operand> {
    let PlaceKind::Local(local) = operand_place(operand)?.kind else {
        return None;
    };
    let Rvalue::UnaryOp(op, slot) = defining_assign(llbc, body, local as usize)? else {
        return None;
    };
    op.get("Cast")?.get("RawPtr")?;
    let PlaceKind::Projection(_, ProjectionElem::Tagged(elem)) = &operand_place(slot)?.kind else {
        return None;
    };
    elem.get("Field")?;
    Some(slot.clone())
}

pub(in crate::front::mir) fn defining_assign<'a>(
    llbc: &Llbc,
    body: &'a Unstructured,
    local: usize,
) -> Option<&'a Rvalue> {
    let mut found = None;
    let mut producers = 0usize;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            if let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref()
                && matches!(place.kind, PlaceKind::Local(i) if i as usize == local)
            {
                producers += 1;
                found = Some(rvalue);
            }
        }
        if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc)
            && matches!(call.dest.kind, PlaceKind::Local(i) if i as usize == local)
        {
            producers += 1;
        }
    }
    (producers == 1).then_some(found).flatten()
}

/// Follow a single chain of `Use` copies. A second producer, or any
/// rvalue that is not a plain local copy, stops the walk.
pub(in crate::front::mir) fn peel_copies(
    llbc: &Llbc,
    body: &Unstructured,
    mut local: usize,
) -> Option<usize> {
    for _ in 0..32 {
        let Some(Rvalue::Use(op, _)) = defining_assign(llbc, body, local) else {
            return Some(local);
        };
        let Some(next) = operand_local(Some(op)) else {
            return Some(local);
        };
        if next == local {
            return Some(local);
        }
        local = next;
    }
    None
}

/// Follow copies and address-preserving pointer casts (`as *const u8`,
/// `as i64`) back to the local they all name.
pub(in crate::front::mir) fn peel_copy_and_cast(
    llbc: &Llbc,
    body: &Unstructured,
    mut local: usize,
) -> Option<usize> {
    for _ in 0..32 {
        match defining_assign(llbc, body, local) {
            Some(Rvalue::Use(op, _)) => {
                let Some(next) = operand_local(Some(op)) else {
                    return Some(local);
                };
                if next == local {
                    return Some(local);
                }
                local = next;
            }
            Some(Rvalue::Cast(_, op, _)) => {
                let Some(next) = operand_local(Some(op)) else {
                    return Some(local);
                };
                if next == local {
                    return Some(local);
                }
                local = next;
            }
            Some(Rvalue::UnaryOp(kind, op)) if unary_op_is_cast(kind) => {
                let Some(next) = operand_local(Some(op)) else {
                    return Some(local);
                };
                if next == local {
                    return Some(local);
                }
                local = next;
            }
            _ => return Some(local),
        }
    }
    None
}

/// Whether a statically-resolved [`RegularCall`] is `<*mut T>::add` /
/// `<*const T>::add` (`core::ptr::{mut_ptr,const_ptr}::<Impl>::add`) —
/// the only `.add` spellings the items-base accessor collapse
/// (brick 1, [`Lowering::is_items_block_base_ptr_add`]) and the list
/// element-access lowering (brick 3,
/// [`Lowering::is_list_items_elem_ptr_add`]) intercept.
pub(in crate::front::mir) fn regular_call_is_ptr_add(reg: &RegularCall, llbc: &Llbc) -> bool {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return false;
    };
    llbc.fn_by_id(*id).is_some_and(|fd| {
        matches!(
            fd.item_meta.name_path().as_str(),
            "core::ptr::mut_ptr::<Impl>::add" | "core::ptr::const_ptr::<Impl>::add"
        )
    })
}

/// The wrapping twin used only by `IntArray::base` / `FloatArray::base`.
/// Their null storage form deliberately computes an aligned, non-null address
/// for Rust's zero-length slice contract, while every live element access has
/// first installed a real `TypedItemsBlock`.  In the translated graph that
/// live path is RPython's `l.items` GcArray header, just like the direct
/// `typed_items_block_items_base` accessor.
pub(in crate::front::mir) fn regular_call_is_ptr_wrapping_add(
    reg: &RegularCall,
    llbc: &Llbc,
) -> bool {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return false;
    };
    llbc.fn_by_id(*id).is_some_and(|fd| {
        matches!(
            fd.item_meta.name_path().as_str(),
            "core::ptr::mut_ptr::<Impl>::wrapping_add"
                | "core::ptr::const_ptr::<Impl>::wrapping_add"
        )
    })
}

/// Whether a statically-resolved [`RegularCall`] is one of the two
/// `ItemsBlock` items-base accessors brick 1 rewrites to return the
/// block *header* pointer (`items_block_items_base` /
/// `items_block_items_ptr`).  brick 3's `*base.add(idx)` lowering only
/// fires when the base traces to one of these (their header return is
/// what the `base_size = ITEMS_BLOCK_ITEMS_OFFSET` array descr re-adds);
/// an items pointer from any other producer already points at
/// `items[0]` and must keep its residual `add`.
///
/// Recognised through [`is_object_items_block_base_accessor`], the same
/// module-qualified suffix brick 1 keys on. Matching whole `pyre_object::`
/// paths here would make brick 3 strictly narrower than the rewrite it is
/// paired with; see that function for why the two must not disagree.
pub(in crate::front::mir) fn regular_call_is_items_block_accessor(
    reg: &RegularCall,
    llbc: &Llbc,
) -> bool {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return false;
    };
    llbc.fn_by_id(*id)
        .is_some_and(|fd| is_object_items_block_base_accessor(fd.item_meta.name_path().as_str()))
}

pub(in crate::front::mir) fn regular_call_name_path(
    reg: &RegularCall,
    llbc: &Llbc,
) -> Option<String> {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    llbc.fn_by_id(*id).map(|fd| fd.item_meta.name_path())
}

/// `core::ptr::null_mut` / `null` and the `std::ptr` re-exports.
pub(in crate::front::mir) fn is_core_ptr_null_fun_path(path: &str) -> bool {
    matches!(
        path,
        "core::ptr::null_mut" | "core::ptr::null" | "std::ptr::null_mut" | "std::ptr::null"
    )
}

/// A Rust abort, lowered as `RaiseImplicit(AssertionError)` the way
/// `TermKind::Abort` is. Declared `Never` / `!` output is that path
/// (`ll_assert`; `rtyper/debug.py`). A panicking-module callee without
/// that spelling still matches.
pub(in crate::front::mir) fn regular_call_is_panicking_abort(
    reg: &RegularCall,
    llbc: &Llbc,
) -> bool {
    if let CallKind::Fun(FunId::Regular { id }) = &reg.kind
        && let Some(fd) = llbc.fn_by_id(*id)
        && tyref_is_never(&fd.signature.output, llbc)
    {
        return true;
    }
    let Some(path) = regular_call_name_path(reg, llbc) else {
        return false;
    };
    let mut segs = path.split("::");
    matches!(segs.next(), Some("core") | Some("std")) && segs.next() == Some("panicking")
}

/// Charon `PathElem::Builtin(PromotedConst, N)` on a FunDecl.
pub(in crate::front::mir) fn fundecl_is_promoted_const(fd: &FunDecl) -> bool {
    fd.item_meta.name.last().is_some_and(|seg| match seg {
        NameSeg::Other(v) => {
            v.as_object()
                .and_then(|m| m.get("Builtin"))
                .and_then(serde_json::Value::as_array)
                .and_then(|arr| arr.first())
                .and_then(serde_json::Value::as_str)
                == Some("PromotedConst")
        }
        _ => false,
    })
}

pub(in crate::front::mir) fn inner_global_id(place: &Place) -> u64 {
    match place.kind {
        PlaceKind::Global { id, .. } => id,
        _ => u64::MAX,
    }
}

/// Walk `body` from the entry block following `Goto` only, stopping at
/// `Return`. Any other terminator, a cleanup block on the path, or a
/// cycle is `None`.
pub(in crate::front::mir) fn promoted_const_linear_blocks(
    body: &Unstructured,
) -> Option<Vec<&BasicBlock>> {
    let n = body.body.len();
    if n == 0 {
        return None;
    }
    let mut blocks = Vec::new();
    let mut seen = vec![false; n];
    let mut bb = 0usize;
    loop {
        if bb >= n || seen[bb] {
            return None;
        }
        seen[bb] = true;
        let block = &body.body[bb];
        if block.is_cleanup {
            return None;
        }
        blocks.push(block);
        let kind = block.terminator.kind_value();
        // Charon writes `"Return"` or `{"Return": ...}`; both end the walk.
        if kind.as_str() == Some("Return")
            || kind
                .as_object()
                .is_some_and(|map| map.contains_key("Return"))
        {
            return Some(blocks);
        }
        let target = kind.get("Goto").and_then(|g| {
            g.as_u64()
                .or_else(|| g.get("target").and_then(serde_json::Value::as_u64))
        })?;
        bb = target as usize;
    }
}

pub(in crate::front::mir) fn place_is_bare_local(place: &Place, id: u64) -> bool {
    matches!(place.kind, PlaceKind::Local(got) if got == id)
}

pub(in crate::front::mir) fn place_is_bare_global(place: &Place) -> bool {
    matches!(place.kind, PlaceKind::Global { .. })
}

/// The `Place::Global` a promoted-const initializer copies before returning.
///
/// `flowspace/flowcontext.py` `find_global` / `LOAD_GLOBAL` fold a
/// module-global read to `const(value)` — the value folded is exactly
/// the read value. rustc emits either `_1 = copy Global(id); _0 = &_1;
/// return` or `_0 = copy Global(id); return`. A field projection on the
/// returned local, a local-to-local copy chain, or a global copied only
/// in an unreachable block is not that fold. `resolve_place` does not
/// extract a field from a projected global (the fallback resolves the
/// inner Global), so the copied place must be a bare `PlaceKind::Global`.
pub(in crate::front::mir) fn promoted_const_copied_global(body: &Unstructured) -> Option<Place> {
    let blocks = promoted_const_linear_blocks(body)?;
    let mut copied: Option<(u64, Place)> = None;
    let mut returned_ref: Option<u64> = None;
    let mut copied_to_ret: Option<Place> = None;
    for block in blocks {
        for stmt in &block.statements {
            match stmt.stmt_kind_ref().ok()? {
                StmtKind::StorageLive(_)
                | StmtKind::StorageDead(_)
                | StmtKind::PlaceMention(_)
                | StmtKind::Borrowck(_) => {}
                StmtKind::Assign(place, rvalue) => match rvalue {
                    Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) => {
                        if !place_is_bare_global(src) {
                            return None;
                        }
                        let PlaceKind::Local(dst) = place.kind else {
                            return None;
                        };
                        if dst == 0 {
                            if copied.is_some() || copied_to_ret.is_some() || returned_ref.is_some()
                            {
                                return None;
                            }
                            copied_to_ret = Some(clone_place(src));
                        } else {
                            if copied.is_some() || copied_to_ret.is_some() {
                                return None;
                            }
                            copied = Some((dst, clone_place(src)));
                        }
                    }
                    Rvalue::Ref { place: src, .. } | Rvalue::RawPtr { place: src, .. } => {
                        if !place_is_bare_local(place, 0) {
                            return None;
                        }
                        let PlaceKind::Local(k) = src.kind else {
                            return None;
                        };
                        if k == 0 || returned_ref.is_some() || copied_to_ret.is_some() {
                            return None;
                        }
                        returned_ref = Some(k);
                    }
                    _ => return None,
                },
                StmtKind::Assert(_) => {}
                StmtKind::Unknown => return None,
            }
        }
    }
    if let Some(global) = copied_to_ret {
        if copied.is_some() || returned_ref.is_some() {
            return None;
        }
        return Some(global);
    }
    let (k, global) = copied?;
    if returned_ref? != k {
        return None;
    }
    Some(global)
}

/// The `Operand::Const` a promoted-const initializer copies before
/// returning a shared borrow of that local.
///
/// rustc emits `_1 = const C; _0 = &_1; return` for a promoted `&str`
/// (or other by-value literal). Anything richer is not this wrapper.
/// The block walk is the same linear `Goto`/`Return` walk as
/// [`promoted_const_copied_global`].
pub(in crate::front::mir) fn promoted_const_copied_const(
    body: &Unstructured,
) -> Option<serde_json::Value> {
    let blocks = promoted_const_linear_blocks(body)?;
    let mut copied: Option<(u64, serde_json::Value)> = None;
    let mut returned_ref: Option<u64> = None;
    for block in blocks {
        for stmt in &block.statements {
            match stmt.stmt_kind_ref().ok()? {
                StmtKind::StorageLive(_)
                | StmtKind::StorageDead(_)
                | StmtKind::PlaceMention(_)
                | StmtKind::Borrowck(_) => {}
                StmtKind::Assign(place, rvalue) => match rvalue {
                    Rvalue::Use(Operand::Const(value), _) => {
                        let PlaceKind::Local(dst) = place.kind else {
                            return None;
                        };
                        if copied.is_some() {
                            return None;
                        }
                        copied = Some((dst, value.clone()));
                    }
                    Rvalue::Ref { place: src, .. } => {
                        let PlaceKind::Local(0) = place.kind else {
                            return None;
                        };
                        let PlaceKind::Local(src) = src.kind else {
                            return None;
                        };
                        if returned_ref.is_some() {
                            return None;
                        }
                        returned_ref = Some(src);
                    }
                    Rvalue::RawPtr { .. } => {}
                    _ => return None,
                },
                StmtKind::Assert(_) => {}
                StmtKind::Unknown => return None,
            }
        }
    }
    let (dst, value) = copied?;
    if returned_ref? != dst {
        return None;
    }
    Some(value)
}

/// `Result::branch` behind `?`. Charon records the impl method
/// `core::result::<Impl>::branch`. A trait `Try::branch` is the same
/// operator. `Result::expect` is a different leaf and stays out.
pub(in crate::front::mir) fn call_is_result_try_branch(func: &CallFunc, llbc: &Llbc) -> bool {
    let CallFunc::Regular(reg) = func else {
        return false;
    };
    if let Some(path) = regular_call_name_path(reg, llbc) {
        return path_leaf_is(&path, "branch")
            && path
                .split("::")
                .any(|seg| seg == "result" || seg == "try_trait" || seg == "Try");
    }
    let CallKind::Trait(payload) = &reg.kind else {
        return false;
    };
    trait_payload_path_and_method(payload, llbc).is_some_and(|(trait_path, method)| {
        method == "branch" && trait_path == "core::ops::try_trait::Try"
    })
}

/// `Result::expect` on a whole local. `unwrap` is a different leaf: it
/// still observes the `Result`.
pub(in crate::front::mir) fn call_is_result_expect(func: &CallFunc, llbc: &Llbc) -> bool {
    let CallFunc::Regular(reg) = func else {
        return false;
    };
    let Some(path) = regular_call_name_path(reg, llbc) else {
        return false;
    };
    path_leaf_is(&path, "expect") && path.split("::").any(|seg| seg == "result")
}

pub(in crate::front::mir) fn rvalue_copied_local(rvalue: &Rvalue) -> Option<usize> {
    let Rvalue::Use(op, _) = rvalue else {
        return None;
    };
    operand_local(Some(op))
}

/// The `divmod` `Result` is still a `Result` for its consumer.
///
/// Local 0 is the function return place (`return this.int_divmod(...)`).
/// `core::result::<Impl>::branch` is `?` (`RBigInt::floordiv`). A whole-local
/// `Rvalue::Use` copy extends the alias set and is not itself a consumer.
/// `Result::expect` on that alias is the always-`Ok` pair residual. An
/// aggregate, another helper, a discriminant read, or a projected place
/// still observes the `Result`, so tuple lowering declines.
pub(in crate::front::mir) fn divmod_result_observed_as_result(
    body: &Unstructured,
    llbc: &Llbc,
    dest_local: usize,
) -> bool {
    let mut alias = vec![dest_local];
    let mut grew = true;
    let forward_blocks = forward_reachable_mask(llbc, body);
    while grew {
        grew = false;
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            for stmt in &bb.statements {
                let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                    continue;
                };
                let PlaceKind::Local(dst) = place.kind else {
                    continue;
                };
                let Some(src) = rvalue_copied_local(rvalue) else {
                    continue;
                };
                let dst = dst as usize;
                if alias.contains(&src) && !alias.contains(&dst) {
                    alias.push(dst);
                    grew = true;
                }
            }
        }
    }
    if alias.contains(&0) {
        return true;
    }
    body.body.iter().enumerate().any(|(bb_idx, bb)| {
        forward_blocks[bb_idx]
            && (bb
                .statements
                .iter()
                .any(|stmt| stmt_observes_divmod_result(stmt, &alias))
                || bb
                    .term_ref(llbc)
                    .is_ok_and(|term| term_observes_divmod_result(term, llbc, &alias)))
    })
}

pub(in crate::front::mir) fn stmt_observes_divmod_result(
    stmt: &majit_charon_reader::ullbc::Statement,
    alias: &[usize],
) -> bool {
    let Ok(kind) = stmt.stmt_kind_ref() else {
        return false;
    };
    match kind {
        StmtKind::Assign(place, rvalue) => {
            if rvalue_is_alias_copy(rvalue, alias) && matches!(place.kind, PlaceKind::Local(_)) {
                return false;
            }
            rvalue_mentions_alias(rvalue, alias)
                || (place_root_in_alias(place, alias) && !matches!(place.kind, PlaceKind::Local(_)))
        }
        StmtKind::Assert(assert) => operand_mentions_alias(&assert.cond, alias),
        StmtKind::StorageLive(_)
        | StmtKind::StorageDead(_)
        | StmtKind::Borrowck(_)
        | StmtKind::PlaceMention(_)
        | StmtKind::Unknown => false,
    }
}

pub(in crate::front::mir) fn term_observes_divmod_result(
    term: &TermKind,
    llbc: &Llbc,
    alias: &[usize],
) -> bool {
    match term {
        TermKind::Switch { discr, .. } => operand_mentions_alias(discr, alias),
        TermKind::Assert { assert, .. } => operand_mentions_alias(&assert.cond, alias),
        TermKind::Call { call, .. } => call_observes_divmod_result(call, llbc, alias),
        TermKind::Return
        | TermKind::UnwindResume
        | TermKind::UnwindTerminate
        | TermKind::Abort(_)
        | TermKind::Panic { .. }
        | TermKind::UndefinedBehavior
        | TermKind::Goto { .. }
        | TermKind::Drop { .. }
        | TermKind::Unknown => false,
    }
}

/// A call observes the `Result` when an alias is an operand and the callee
/// is not `Result::expect` on that whole local. `Result::branch` (`?`) is
/// an observation. The call that defines the destination does not mention it.
pub(in crate::front::mir) fn call_observes_divmod_result(
    call: &CallPayload,
    llbc: &Llbc,
    alias: &[usize],
) -> bool {
    let mut receiver_only = false;
    for (index, arg) in call.args.iter().enumerate() {
        if !operand_mentions_alias(arg, alias) {
            continue;
        }
        let whole_receiver =
            index == 0 && operand_local(Some(arg)).is_some_and(|local| alias.contains(&local));
        if !whole_receiver {
            return true;
        }
        receiver_only = true;
    }
    if let CallFunc::Dynamic(op, _) = &call.func
        && operand_mentions_alias(op, alias)
    {
        return true;
    }
    receiver_only
        && (call_is_result_try_branch(&call.func, llbc) || !call_is_result_expect(&call.func, llbc))
}

pub(in crate::front::mir) fn place_root_local(place: &Place) -> Option<usize> {
    match &place.kind {
        PlaceKind::Local(id) => Some(*id as usize),
        PlaceKind::Projection(inner, _) => place_root_local(inner),
        PlaceKind::Global { .. } | PlaceKind::Unknown => None,
    }
}

pub(in crate::front::mir) fn place_root_in_alias(place: &Place, alias: &[usize]) -> bool {
    place_root_local(place).is_some_and(|id| alias.contains(&id))
}

pub(in crate::front::mir) fn operand_mentions_alias(op: &Operand, alias: &[usize]) -> bool {
    match op {
        Operand::Copy(place) | Operand::Move(place) => place_root_in_alias(place, alias),
        Operand::Const(_) => false,
    }
}

pub(in crate::front::mir) fn rvalue_is_alias_copy(rvalue: &Rvalue, alias: &[usize]) -> bool {
    rvalue_copied_local(rvalue).is_some_and(|src| alias.contains(&src))
}

pub(in crate::front::mir) fn rvalue_mentions_alias(rvalue: &Rvalue, alias: &[usize]) -> bool {
    match rvalue {
        Rvalue::Use(op, _)
        | Rvalue::UnaryOp(_, op)
        | Rvalue::Cast(_, op, _)
        | Rvalue::ShallowInitBox(op, _) => operand_mentions_alias(op, alias),
        Rvalue::BinaryOp(_, lhs, rhs) => {
            operand_mentions_alias(lhs, alias) || operand_mentions_alias(rhs, alias)
        }
        Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
            place_root_in_alias(place, alias)
        }
        Rvalue::Aggregate(_, operands) => {
            operands.iter().any(|op| operand_mentions_alias(op, alias))
        }
        Rvalue::Discriminant(place) | Rvalue::Len(place) => place_root_in_alias(place, alias),
        Rvalue::Repeat(op, _, _, _) => operand_mentions_alias(op, alias),
        Rvalue::NullaryOp(_, _) | Rvalue::Unknown => false,
    }
}

pub(in crate::front::mir) fn regular_call_fun_decl_id(kind: &CallKind) -> Option<u64> {
    match kind {
        CallKind::Fun(FunId::Regular { id }) => Some(*id),
        // The payload is `[trait_ref, method_idx]` and does not carry a
        // fun-decl id. Callers that need the method use
        // `trait_payload_path_and_method`.
        CallKind::Trait(_) => None,
        _ => None,
    }
}

/// `RDict::insert` / `RDict::get` receiver spelling for
/// `__cast_instance_intrinsic`.
///
/// `ll_dict_setitem` / `ll_dict_lookup` (`rordereddict.py`) see an ordered
/// dict. The cast root is the instantiated `RDict<K,V,S>` (references and
/// the raw-pointer prefix stripped) so the receiver annotates as
/// `SomeDict`. A bare `RDict` is not that dict. The monomorphized leaf's
/// hash is not part of the match.
pub(in crate::front::mir) fn rdict_insert_get_cast_root(
    segments: &[String],
    recv_ty: Option<&TyRef>,
    llbc: &Llbc,
) -> Option<String> {
    if segments.len() < 3 || segments[segments.len() - 2] != "RDict" {
        return None;
    }
    let leaf = crate::front::clause_spec::unspecialized_leaf(segments.last()?);
    if leaf != "insert" && leaf != "get" {
        return None;
    }
    let spelled = tyref_to_ast_string(recv_ty?, llbc);
    let spelled = if let Some(rest) = spelled.strip_prefix("*const ") {
        rest.to_string()
    } else if let Some(rest) = spelled.strip_prefix("*mut ") {
        rest.to_string()
    } else {
        spelled
    };
    if spelled.starts_with("RDict<") {
        Some(spelled)
    } else {
        None
    }
}

/// Call-path segments for a specialized leaf of this declaration.
pub(in crate::front::mir) fn spec_segments(llbc: &Llbc, fd: &FunDecl, leaf: &str) -> Vec<String> {
    if let Some((owner, _)) = impl_method_owner_for_fundecl(llbc, fd) {
        // `method-call-expr`: inherent methods are looked up before trait
        // methods for the same receiver. `CallControl.graphs_from` /
        // `Bookkeeper.getdesc` keys on the function object; Charon
        // records that callee as a FunDeclId. An ADT-owned trait-impl
        // method shares the owner's leaf with an inherent of the same
        // name, so it occupies the identity path `lib.rs` already
        // registers; the inherent keeps `for_impl_method`.
        if let Some(impl_id) = trait_impl_id_for_fundecl(fd)
            && let Some(identity) = crate::front::clause_spec::render_trait_impl(llbc, impl_id)
        {
            return trait_impl_identity_path(&owner, &identity, leaf);
        }
        return crate::parse::CallPath::for_impl_method(&owner, leaf).segments;
    }
    let stripped = strip_crate_prefix(&fd.item_meta.name_path());
    let module = stripped
        .rsplit_once("::")
        .map(|(module, _)| module)
        .unwrap_or("");
    // Primitive `Self` (`impl PartialOrd<char> for char`): no ADT owner,
    // so `name_path` is `cmp::impls::<Impl>::le` for every width.
    // `CallControl.graphs_from` keeps one graph per function object;
    // the impl's trait ref and Self type is that identity.
    if let Some(impl_id) = trait_impl_id_for_fundecl(fd)
        && let Some(identity) = crate::front::clause_spec::render_trait_impl(llbc, impl_id)
        && !module.is_empty()
    {
        return trait_impl_identity_path(module, &identity, leaf);
    }
    // Same key `free_function_alias_paths` registers: the module portion
    // of `name_path` is one segment when it contains `<Impl>`, not a
    // split of that token.
    let mut segments: Vec<String> = if module.is_empty() {
        Vec::new()
    } else {
        module.split("::").map(str::to_string).collect()
    };
    segments.push(leaf.to_string());
    segments
}

/// The `CallPath` `lib.rs` registers for this FunDecl: the same
/// segments [`spec_segments`] emits, so a Fun call and a Trait call
/// that resolve to this declaration share one registry key.
pub(in crate::front::mir) fn registered_path_for_fun_decl(
    llbc: &Llbc,
    fd: &FunDecl,
) -> crate::parse::CallPath {
    let leaf = match instance_leaf(llbc, fd) {
        Some(leaf) => leaf,
        None => fd
            .item_meta
            .name_path()
            .rsplit("::")
            .next()
            .unwrap_or("fn")
            .to_string(),
    };
    crate::parse::CallPath::from_segments(spec_segments(llbc, fd, &leaf))
}

/// The `()` type expected by drop glue.
pub(in crate::front::mir) fn unit_tyref() -> TyRef {
    TyRef::Other(serde_json::json!({
        "Adt": {
            "id": 0, "builtin": "Tuple",
            "generics": {
                "regions": [],
                "types": [],
                "const_generics": [],
                "trait_refs": []
            }
        }
    }))
}
/// The scalar `TypedItemsBlock` accessor, kept separate from the object-array
/// pair because its consumers must emit `GcArray(Signed|Float)` operations,
/// never reference-array operations.
pub(in crate::front::mir) fn regular_call_is_typed_items_block_accessor(
    reg: &RegularCall,
    llbc: &Llbc,
) -> bool {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return false;
    };
    llbc.fn_by_id(*id).is_some_and(|fd| {
        let name = fd.item_meta.name_path();
        is_typed_items_block_base_accessor(&name) || is_typed_array_base_adapter(&name)
    })
}

/// The two unwrapped-string adapters whose Rust return type is the physical
/// `*mut PyObjectRef`, while their RPython list item is externally `str` and
/// internally GCREF.  Keep these out of the generic object-array predicate:
/// sharing the machine word does not make `[str]` and `[W_Root]` the same
/// ListDef / ARRAY identity upstream.
pub(in crate::front::mir) fn regular_call_is_string_items_block_accessor(
    reg: &RegularCall,
    llbc: &Llbc,
) -> bool {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return false;
    };
    llbc.fn_by_id(*id)
        .is_some_and(|fd| is_string_array_base_adapter(&fd.item_meta.name_path()))
}

/// The [`TyRef`] of a plain-place [`Operand`], or `None` for a
/// constant.  The borrow tracks the operand, so callers reading a
/// receiver type from a raw `CallPayload` (the deferred-write liveness
/// pre-pass) need not clone it like [`Lowering::lower_call`] does.
pub(in crate::front::mir) fn operand_tyref(op: &Operand) -> Option<&TyRef> {
    match op {
        Operand::Copy(p) | Operand::Move(p) => Some(&p.ty),
        Operand::Const(_) => None,
    }
}

/// Declared type of an `Operand::Const`.
///
/// Charon stores it beside the literal (`{"ty": {"Deduplicated": N}}` or an
/// inline `Value`). [`operand_tyref`] stays place-only; the rbigint
/// constructor gate is the caller that has to see a constant word.
pub(in crate::front::mir) fn const_operand_tyref(op: &Operand, llbc: &Llbc) -> Option<TyRef> {
    let Operand::Const(value) = op else {
        return None;
    };
    serde_json::from_value(llbc.const_expr_ty(value)?.clone()).ok()
}

/// `{"Scalar": "Bool"}`, after dedup.
pub(in crate::front::mir) fn tyref_is_literal_bool(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc) else {
        return false;
    };
    node.as_object()
        .and_then(|obj| obj.get("Scalar"))
        .and_then(serde_json::Value::as_str)
        == Some("Bool")
}

/// The decomposed brick-3 element-`add` gate
/// ([`Lowering::is_list_items_elem_ptr_add`]), taking pieces both the
/// call-lowering intercept (which has already moved `call.func` /
/// `call.args` out of the payload) and the deferred-write liveness
/// pre-pass (which still holds the raw [`CallPayload`]) can supply, so
/// both agree on exactly which `.add` calls become array ops.
pub(in crate::front::mir) fn is_list_items_elem_ptr_add_parts(
    reg: &RegularCall,
    args_len: usize,
    base_local: Option<usize>,
    base_ty: Option<&TyRef>,
    dest_local: usize,
    body: &Unstructured,
    llbc: &Llbc,
) -> bool {
    args_len == 2
        && regular_call_is_ptr_add(reg, llbc)
        && base_ty.is_some_and(|ty| is_object_ref_items_ptr(ty, llbc))
        && base_local.is_some_and(|base| base_traces_to_items_block_accessor(body, base, llbc))
        && add_dest_used_only_as_single_deref(llbc, body, dest_local)
}

/// Scalar half of brick 3. The physical `TypedItemsBlock` stores only the two
/// RPython array families declared by `rlist.rs`: `GcArray(Signed)` (`i64`)
/// and `GcArray(Float)` (`f64`). Other raw-pointer pointees remain residual.
pub(in crate::front::mir) fn is_typed_items_elem_ptr_add_parts(
    reg: &RegularCall,
    args_len: usize,
    base_local: Option<usize>,
    base_ty: Option<&TyRef>,
    dest_local: usize,
    body: &Unstructured,
    llbc: &Llbc,
) -> bool {
    args_len == 2
        && regular_call_is_ptr_add(reg, llbc)
        && base_ty.is_some_and(|ty| raw_ptr_typed_items_element(ty, llbc).is_some())
        && base_local
            .is_some_and(|base| base_traces_to_typed_items_block_accessor(body, base, llbc))
        && add_dest_used_only_as_single_deref(llbc, body, dest_local)
}

pub(in crate::front::mir) fn is_u8_slice_elem_ptr_add_parts(
    reg: &RegularCall,
    args_len: usize,
    base_local: Option<usize>,
    base_ty: Option<&TyRef>,
    index_local: Option<usize>,
    dest_local: usize,
    body: &Unstructured,
    llbc: &Llbc,
) -> bool {
    args_len == 2
        && regular_call_is_ptr_add(reg, llbc)
        && base_ty.is_some_and(|ty| raw_ptr_u8_slice_element(ty, llbc).is_some())
        && index_local.is_some()
        && base_local.is_some_and(|base| {
            base_traces_to_items_block_accessor_matching(
                body,
                base,
                llbc,
                regular_call_is_slice_as_ptr,
            )
        })
        && add_dest_used_only_as_single_deref(llbc, body, dest_local)
}

pub(in crate::front::mir) fn regular_call_is_slice_as_ptr(reg: &RegularCall, llbc: &Llbc) -> bool {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return false;
    };
    llbc.fn_by_id(*id).is_some_and(|fd| {
        matches!(
            fd.item_meta.name_path().as_str(),
            "core::slice::<Impl>::as_ptr" | "core::slice::<Impl>::as_mut_ptr"
        )
    })
}

/// String-list half of brick 3.  The pointee is physically PyObjectRef/GCREF,
/// but only the two named adapters prove that it is the internal storage of a
/// `SomeList(SomeString)` rather than a list of W_Root instances.
pub(in crate::front::mir) fn is_string_items_elem_ptr_add_parts(
    reg: &RegularCall,
    args_len: usize,
    base_local: Option<usize>,
    base_ty: Option<&TyRef>,
    dest_local: usize,
    body: &Unstructured,
    llbc: &Llbc,
) -> bool {
    args_len == 2
        && regular_call_is_ptr_add(reg, llbc)
        && base_ty.is_some_and(|ty| is_object_ref_items_ptr(ty, llbc))
        && base_local.is_some_and(|base| {
            base_traces_to_items_block_accessor_matching(
                body,
                base,
                llbc,
                regular_call_is_string_items_block_accessor,
            )
        })
        && add_dest_used_only_as_single_deref(llbc, body, dest_local)
}

/// Whether MIR local `base` — the receiver of a `*base.add(idx)` over
/// `*mut PyObjectRef` — traces, through plain `Copy` / `Move` aliases,
/// to the result of an `items_block_items_base` / `items_block_items_ptr`
/// call.  Those accessors are the only `*mut PyObjectRef` producers
/// brick 1 rewrites to return the `ItemsBlock` *header*, so the matching
/// `ArrayRead` / `ArrayWrite` `base_size = ITEMS_BLOCK_ITEMS_OFFSET`
/// re-adds the header offset to reach `items[0]`.  A pointer from any
/// other source (`FixedObjectArray::items_mut_ptr`, `null_mut`, a
/// chained `.add`) already points at the items, so re-adding the offset
/// would overshoot by one element — such a base keeps its residual
/// `add` and falls to the legacy walker.  A multiply-defined or
/// non-`Copy`-produced local is ambiguous and rejected.
pub(in crate::front::mir) fn base_traces_to_items_block_accessor(
    body: &Unstructured,
    base: usize,
    llbc: &Llbc,
) -> bool {
    base_traces_to_items_block_accessor_matching(
        body,
        base,
        llbc,
        regular_call_is_items_block_accessor,
    )
}

pub(in crate::front::mir) fn base_traces_to_typed_items_block_accessor(
    body: &Unstructured,
    base: usize,
    llbc: &Llbc,
) -> bool {
    base_traces_to_items_block_accessor_matching(
        body,
        base,
        llbc,
        regular_call_is_typed_items_block_accessor,
    )
}

pub(in crate::front::mir) fn base_traces_to_items_block_accessor_matching(
    body: &Unstructured,
    base: usize,
    llbc: &Llbc,
    matches_accessor: fn(&RegularCall, &Llbc) -> bool,
) -> bool {
    let mut cur = base;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for _ in 0..32 {
        let mut producers = 0usize;
        let mut copy_src: Option<usize> = None;
        let mut is_accessor = false;
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            for stmt in &bb.statements {
                if let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref()
                    && matches!(&place.kind, PlaceKind::Local(i) if *i as usize == cur)
                {
                    producers += 1;
                    match &rvalue {
                        Rvalue::Use(op, _) => copy_src = operand_local(Some(op)),
                        // `typed_items_block_items_base(..) as *const i64`
                        // changes only the raw-pointer pointee spelling. The
                        // address is identical, so follow it exactly as
                        // RPython follows `cast_pointer` while retaining the
                        // final pointee type for the array descr.
                        Rvalue::Cast(_, op, target_ty)
                            if tyref_node(target_ty, llbc)
                                .and_then(|node| type_node_raw_ptr_pointee(node, llbc))
                                .is_some()
                                && operand_tyref(op).is_some_and(|source_ty| {
                                    tyref_node(source_ty, llbc)
                                        .and_then(|node| type_node_raw_ptr_pointee(node, llbc))
                                        .is_some()
                                }) =>
                        {
                            copy_src = operand_local(Some(op));
                        }
                        // Charon also serializes raw pointer casts through its
                        // older `UnaryOp({Cast: RawPtr(..)}, operand)` form.
                        // It is the same address-preserving cast as the
                        // dedicated `Rvalue::Cast` arm above.
                        Rvalue::UnaryOp(kind, op)
                            if unary_op_is_cast(kind)
                                && tyref_node(&place.ty, llbc)
                                    .and_then(|node| type_node_raw_ptr_pointee(node, llbc))
                                    .is_some()
                                && operand_tyref(op).is_some_and(|source_ty| {
                                    tyref_node(source_ty, llbc)
                                        .and_then(|node| type_node_raw_ptr_pointee(node, llbc))
                                        .is_some()
                                }) =>
                        {
                            copy_src = operand_local(Some(op));
                        }
                        _ => {}
                    }
                }
            }
            if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc)
                && matches!(call.dest.kind, PlaceKind::Local(i) if i as usize == cur)
            {
                producers += 1;
                if let CallFunc::Regular(reg) = &call.func
                    && matches_accessor(reg, llbc)
                {
                    is_accessor = true;
                }
            }
        }
        if producers != 1 {
            return false;
        }
        if is_accessor {
            return true;
        }
        match copy_src {
            Some(src) => cur = src,
            None => return false,
        }
    }
    false
}

/// How a place / operand references the brick-3 `add`-result local
/// `dest`, as seen by [`add_dest_used_only_as_single_deref`].
pub(in crate::front::mir) enum DestRef {
    /// Does not reference `dest`.
    None,
    /// `*dest` exactly — the element load / store the array op models.
    Deref,
    /// References `dest` some other way (passed by value, re-borrowed,
    /// a deeper projection) — the raw pointer escapes the load / store.
    Other,
}

/// Whether a place's projection chain bottoms out at `Local(dest)`.
pub(in crate::front::mir) fn place_references_local(place: &Place, dest: usize) -> bool {
    match &place.kind {
        PlaceKind::Local(i) => *i as usize == dest,
        PlaceKind::Projection(inner, _) => place_references_local(inner, dest),
        PlaceKind::Global { .. } | PlaceKind::Unknown => false,
    }
}

/// Whether a place is exactly `*Local(dest)` — `Projection(Local(dest),
/// Deref)` with nothing layered on top.
pub(in crate::front::mir) fn place_is_immediate_deref_of(place: &Place, dest: usize) -> bool {
    let PlaceKind::Projection(inner, elem) = &place.kind else {
        return false;
    };
    matches!(elem, ProjectionElem::Atom(s) if s == "Deref")
        && matches!(inner.kind, PlaceKind::Local(i) if i as usize == dest)
}

/// Classify how a read-position operand references `dest`.
pub(in crate::front::mir) fn operand_dest_ref(op: &Operand, dest: usize) -> DestRef {
    let (Operand::Copy(p) | Operand::Move(p)) = op else {
        return DestRef::None;
    };
    if place_is_immediate_deref_of(p, dest) {
        DestRef::Deref
    } else if place_references_local(p, dest) {
        DestRef::Other
    } else {
        DestRef::None
    }
}

pub(in crate::front::mir) fn bump_dest_ref(r: DestRef, derefs: &mut usize, other: &mut usize) {
    match r {
        DestRef::Deref => *derefs += 1,
        DestRef::Other => *other += 1,
        DestRef::None => {}
    }
}

/// Classify a write-target place's reference to `dest`: a bare
/// `Local(dest)` is the `add` definition, `*dest` a deref store, and any
/// other reference an escape.
pub(in crate::front::mir) fn classify_write_place(
    place: &Place,
    dest: usize,
    defs: &mut usize,
    derefs: &mut usize,
    other: &mut usize,
) {
    if matches!(place.kind, PlaceKind::Local(i) if i as usize == dest) {
        *defs += 1;
    } else if place_is_immediate_deref_of(place, dest) {
        *derefs += 1;
    } else if place_references_local(place, dest) {
        *other += 1;
    }
}

pub(in crate::front::mir) fn scan_rvalue_dest_ref(
    rvalue: &Rvalue,
    dest: usize,
    derefs: &mut usize,
    other: &mut usize,
) {
    match rvalue {
        Rvalue::Use(op, _)
        | Rvalue::UnaryOp(_, op)
        | Rvalue::Cast(_, op, _)
        | Rvalue::Repeat(op, _, _, _)
        | Rvalue::ShallowInitBox(op, _) => bump_dest_ref(operand_dest_ref(op, dest), derefs, other),
        Rvalue::BinaryOp(_, lhs, rhs) => {
            bump_dest_ref(operand_dest_ref(lhs, dest), derefs, other);
            bump_dest_ref(operand_dest_ref(rhs, dest), derefs, other);
        }
        // A re-borrow / `Len` / `Discriminant` of `dest` is not a plain
        // element load, so any reference to `dest` here escapes.
        Rvalue::Ref { place, .. }
        | Rvalue::RawPtr { place, .. }
        | Rvalue::Len(place)
        | Rvalue::Discriminant(place) => {
            if place_references_local(place, dest) {
                *other += 1;
            }
        }
        Rvalue::Aggregate(_, operands) => {
            for op in operands {
                bump_dest_ref(operand_dest_ref(op, dest), derefs, other);
            }
        }
        Rvalue::NullaryOp(_, _) | Rvalue::Unknown => {}
    }
}

/// The brick-3 escape guard: whether the `.add`-result local `dest` is
/// used in the body *only* as a single dereference — exactly one
/// definition (the `add` itself), exactly one `*dest` read **or** `*dest
/// = v` write, and no other appearance.  A raw pointer that is passed by
/// value (`ptr::copy(p, ..)`), used as the base of a further `.add`,
/// re-borrowed, or read twice leaves the element load / store model the
/// `ArrayRead` / `ArrayWrite` captures, so the `add` must stay residual.
/// `StorageLive` / `StorageDead` / `PlaceMention` are borrow-ck markers,
/// not loads, so they are ignored.
pub(in crate::front::mir) struct DestDerefCensus {
    pub(in crate::front::mir) defs: usize,
    /// `*dest` appearing in a read position.
    pub(in crate::front::mir) read_derefs: usize,
    /// `*dest = v`.
    pub(in crate::front::mir) write_derefs: usize,
    pub(in crate::front::mir) other: usize,
}

pub(in crate::front::mir) fn dest_deref_census(
    llbc: &Llbc,
    body: &Unstructured,
    dest: usize,
) -> DestDerefCensus {
    let mut census = DestDerefCensus {
        defs: 0,
        read_derefs: 0,
        write_derefs: 0,
        other: 0,
    };
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            match stmt.stmt_kind_ref() {
                Ok(StmtKind::Assign(place, rvalue)) => {
                    scan_rvalue_dest_ref(rvalue, dest, &mut census.read_derefs, &mut census.other);
                    classify_write_place(
                        place,
                        dest,
                        &mut census.defs,
                        &mut census.write_derefs,
                        &mut census.other,
                    );
                }
                Ok(StmtKind::Assert(assert)) => bump_dest_ref(
                    operand_dest_ref(&assert.cond, dest),
                    &mut census.read_derefs,
                    &mut census.other,
                ),
                _ => {}
            }
        }
        match bb.term_ref(llbc) {
            Ok(TermKind::Switch { discr, .. }) => bump_dest_ref(
                operand_dest_ref(discr, dest),
                &mut census.read_derefs,
                &mut census.other,
            ),
            Ok(TermKind::Call { call, .. }) => {
                if let CallFunc::Dynamic(op, _) = &call.func {
                    bump_dest_ref(
                        operand_dest_ref(op, dest),
                        &mut census.read_derefs,
                        &mut census.other,
                    );
                }
                for arg in &call.args {
                    bump_dest_ref(
                        operand_dest_ref(arg, dest),
                        &mut census.read_derefs,
                        &mut census.other,
                    );
                }
                classify_write_place(
                    &call.dest,
                    dest,
                    &mut census.defs,
                    &mut census.write_derefs,
                    &mut census.other,
                );
            }
            Ok(TermKind::Assert { assert, .. }) => bump_dest_ref(
                operand_dest_ref(&assert.cond, dest),
                &mut census.read_derefs,
                &mut census.other,
            ),
            _ => {}
        }
    }
    census
}

pub(in crate::front::mir) fn add_dest_used_only_as_single_deref(
    llbc: &Llbc,
    body: &Unstructured,
    dest: usize,
) -> bool {
    let census = dest_deref_census(llbc, body, dest);
    census.defs == 1 && census.read_derefs + census.write_derefs == 1 && census.other == 0
}

/// `*dest = v` with no load of `*dest` and no escaping use.
///
/// `ll_setitem_fast` does not read the old item. The `index_mut` arm
/// otherwise emits that read (`list.getitem`) before the store.
pub(in crate::front::mir) fn add_dest_used_only_as_single_write_deref(
    llbc: &Llbc,
    body: &Unstructured,
    dest: usize,
) -> bool {
    let census = dest_deref_census(llbc, body, dest);
    census.defs == 1 && census.write_derefs == 1 && census.read_derefs == 0 && census.other == 0
}

/// `*p` read of an `.add` result, with no store and no escaping use.
pub(in crate::front::mir) fn add_dest_used_only_as_single_read_deref(
    llbc: &Llbc,
    body: &Unstructured,
    dest: usize,
) -> bool {
    let census = dest_deref_census(llbc, body, dest);
    census.defs == 1 && census.read_derefs == 1 && census.write_derefs == 0 && census.other == 0
}

/// The MIR local `d` of `(*d).field`.
///
/// `elem` is the outer `Field` projection and `inner` is the `*d` place.
/// A bare `*d`, a deeper projection, or an index is `None`.
pub(in crate::front::mir) fn struct_field_deref_local(
    inner: &Place,
    elem: &ProjectionElem,
) -> Option<usize> {
    let ProjectionElem::Tagged(payload) = elem else {
        return None;
    };
    if !payload
        .as_object()
        .is_some_and(|map| map.contains_key("Field"))
    {
        return None;
    }
    let PlaceKind::Projection(pre, ProjectionElem::Atom(name)) = &inner.kind else {
        return None;
    };
    if name != "Deref" {
        return None;
    }
    match pre.kind {
        PlaceKind::Local(local) => Some(local as usize),
        _ => None,
    }
}

/// Whether `place` is exactly `(*Local(dest)).field`.
pub(in crate::front::mir) fn place_is_immediate_struct_field_of(
    place: &Place,
    dest: usize,
) -> bool {
    let PlaceKind::Projection(inner, elem) = &place.kind else {
        return false;
    };
    struct_field_deref_local(inner, elem) == Some(dest)
}

/// How a place / operand references the struct-field `add` destination.
pub(in crate::front::mir) enum StructFieldUse {
    None,
    /// `(*dest).field` as a value, including a cast or aggregate operand.
    Field,
    /// The pointer escapes: a copy, a bare deref, a reborrow, or a
    /// deeper projection. `&(*dest).field` is this case too.
    Other,
}

pub(in crate::front::mir) fn operand_struct_field_use(op: &Operand, dest: usize) -> StructFieldUse {
    let (Operand::Copy(place) | Operand::Move(place)) = op else {
        return StructFieldUse::None;
    };
    if place_is_immediate_struct_field_of(place, dest) {
        StructFieldUse::Field
    } else if place_references_local(place, dest) {
        StructFieldUse::Other
    } else {
        StructFieldUse::None
    }
}

pub(in crate::front::mir) fn bump_struct_field_use(
    use_kind: StructFieldUse,
    fields: &mut usize,
    other: &mut usize,
) {
    match use_kind {
        StructFieldUse::Field => *fields += 1,
        StructFieldUse::Other => *other += 1,
        StructFieldUse::None => {}
    }
}

pub(in crate::front::mir) fn classify_struct_field_write(
    place: &Place,
    dest: usize,
    defs: &mut usize,
    fields: &mut usize,
    other: &mut usize,
) {
    if matches!(place.kind, PlaceKind::Local(local) if local as usize == dest) {
        *defs += 1;
    } else if place_is_immediate_struct_field_of(place, dest) {
        *fields += 1;
    } else if place_references_local(place, dest) {
        *other += 1;
    }
}

pub(in crate::front::mir) fn scan_rvalue_struct_field(
    rvalue: &Rvalue,
    dest: usize,
    fields: &mut usize,
    other: &mut usize,
) {
    match rvalue {
        Rvalue::Use(op, _)
        | Rvalue::UnaryOp(_, op)
        | Rvalue::Cast(_, op, _)
        | Rvalue::Repeat(op, _, _, _)
        | Rvalue::ShallowInitBox(op, _) => {
            bump_struct_field_use(operand_struct_field_use(op, dest), fields, other);
        }
        Rvalue::BinaryOp(_, lhs, rhs) => {
            bump_struct_field_use(operand_struct_field_use(lhs, dest), fields, other);
            bump_struct_field_use(operand_struct_field_use(rhs, dest), fields, other);
        }
        // Address-of, length, and discriminant are not value field uses,
        // even when the place is exactly `(*dest).field`.
        Rvalue::Ref { place, .. }
        | Rvalue::RawPtr { place, .. }
        | Rvalue::Len(place)
        | Rvalue::Discriminant(place) => {
            if place_references_local(place, dest) {
                *other += 1;
            }
        }
        Rvalue::Aggregate(_, operands) => {
            for op in operands {
                bump_struct_field_use(operand_struct_field_use(op, dest), fields, other);
            }
        }
        Rvalue::NullaryOp(_, _) | Rvalue::Unknown => {}
    }
}

pub(in crate::front::mir) struct StructFieldCensus {
    pub(in crate::front::mir) defs: usize,
    pub(in crate::front::mir) fields: usize,
    pub(in crate::front::mir) other: usize,
}

pub(in crate::front::mir) fn struct_field_census(
    llbc: &Llbc,
    body: &Unstructured,
    dest: usize,
) -> StructFieldCensus {
    let mut census = StructFieldCensus {
        defs: 0,
        fields: 0,
        other: 0,
    };
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            match stmt.stmt_kind_ref() {
                Ok(StmtKind::Assign(place, rvalue)) => {
                    scan_rvalue_struct_field(rvalue, dest, &mut census.fields, &mut census.other);
                    classify_struct_field_write(
                        place,
                        dest,
                        &mut census.defs,
                        &mut census.fields,
                        &mut census.other,
                    );
                }
                Ok(StmtKind::Assert(assert)) => bump_struct_field_use(
                    operand_struct_field_use(&assert.cond, dest),
                    &mut census.fields,
                    &mut census.other,
                ),
                _ => {}
            }
        }
        match bb.term_ref(llbc) {
            Ok(TermKind::Switch { discr, .. }) => bump_struct_field_use(
                operand_struct_field_use(discr, dest),
                &mut census.fields,
                &mut census.other,
            ),
            Ok(TermKind::Call { call, .. }) => {
                if let CallFunc::Dynamic(op, _) = &call.func {
                    bump_struct_field_use(
                        operand_struct_field_use(op, dest),
                        &mut census.fields,
                        &mut census.other,
                    );
                }
                for arg in &call.args {
                    bump_struct_field_use(
                        operand_struct_field_use(arg, dest),
                        &mut census.fields,
                        &mut census.other,
                    );
                }
                classify_struct_field_write(
                    &call.dest,
                    dest,
                    &mut census.defs,
                    &mut census.fields,
                    &mut census.other,
                );
            }
            Ok(TermKind::Assert { assert, .. }) => bump_struct_field_use(
                operand_struct_field_use(&assert.cond, dest),
                &mut census.fields,
                &mut census.other,
            ),
            _ => {}
        }
    }
    census
}

/// `ptr::add`'s destination is used only as one or more `(*dest).field`
/// value reads or writes. A bare deref, a copy of the pointer, a reborrow,
/// or a second definition keeps the add residual. `StorageLive` /
/// `StorageDead` / `PlaceMention` are ignored, same as
/// [`add_dest_used_only_as_single_deref`].
pub(in crate::front::mir) fn add_dest_used_only_as_struct_field(
    llbc: &Llbc,
    body: &Unstructured,
    dest: usize,
) -> bool {
    let census = struct_field_census(llbc, body, dest);
    census.defs == 1 && census.fields >= 1 && census.other == 0
}

/// Header recovered from the add's base before any flowspace Variable exists.
pub(in crate::front::mir) enum TracedPtrAddHeader {
    /// `entries_item_ptr(header)`. `local` is that argument, the GcEntries
    /// object, not the item pointer the call returns.
    EntriesItem { local: usize },
    /// `entry_ptr(&dict)` on a receiver whose struct has an `entries` field.
    EntryPtr { recv_local: usize, recv_ty: TyRef },
}

pub(in crate::front::mir) struct StructFieldPtrAdd {
    pub(in crate::front::mir) header: TracedPtrAddHeader,
    pub(in crate::front::mir) index_local: usize,
    pub(in crate::front::mir) array_type_id: String,
}

pub(in crate::front::mir) fn path_leaf_is(path: &str, leaf: &str) -> bool {
    path.rsplit("::").next() == Some(leaf)
}

pub(in crate::front::mir) fn path_is_rordereddict_entry_ptr(path: &str) -> bool {
    path.split("::").any(|seg| seg == "rordereddict") && path_leaf_is(path, "entry_ptr")
}

pub(in crate::front::mir) fn tyref_is_raw_ptr_value(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(|node| type_node_raw_ptr_pointee(node, llbc))
        .is_some()
}

/// Index of a struct field named `name`. Enums are excluded: `entries` is
/// a struct field of `RDict`, not a variant payload.
pub(in crate::front::mir) fn struct_field_index_named(
    ty: &TyRef,
    llbc: &Llbc,
    name: &str,
) -> Option<usize> {
    let node = strip_ty_wrappers(tyref_node(ty, llbc)?, llbc)?;
    let decl = llbc.type_by_id(adt_node_def_id(node)?)?;
    if decl.published_struct_copy {
        return None;
    }
    let fields = type_decl_struct_fields(decl, llbc)?;
    fields
        .iter()
        .position(|field| field.name.as_deref() == Some(name))
}

/// `GcArray<Point>` / `GcArray<Entry>` for a `*mut`/`*const` named,
/// non-transparent struct pointee.
///
/// A scalar, a pointer, or a `repr(transparent)` pointee is `None`, so list
/// `ArrayRead` intercepts and `direct_ptradd` keep those `.add` calls.
///
/// `get_array_descr` keys one ARRAY lltype, and `nolength` is a property of
/// that lltype. `interiorfielddescrof_keyed` mints the length-prefixed
/// shape (`!nolength`, items at `array_items_base`). A `[T]`
/// spelling is an item run (`nolength_from_array_type_id`), so it cannot
/// share an atid with that mint. `GcArray<T>` stays length-prefixed and
/// `extract_element_type_from_str` still peels it to `T`.
pub(in crate::front::mir) fn struct_pointee_array_type_id(
    base_ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
) -> Option<String> {
    let pointee = tyref_peel_one_raw_ptr_node(base_ty, llbc)?;
    let decl = llbc.type_by_id(adt_node_def_id(pointee)?)?;
    if decl.published_struct_copy || !matches!(decl.kind, TypeDeclKind::Struct(_)) {
        return None;
    }
    if tyref_transparent_nonzst_field(&TyRef::Other(pointee.clone()), llbc).is_some() {
        return None;
    }
    let root = adt_node_class_root_with(pointee, llbc, tombstoned)?;
    let leaf = majit_ir::descr::strip_instantiation_suffix(&root);
    Some(format!("GcArray<{leaf}>"))
}

pub(in crate::front::mir) enum HeaderProd {
    Copy(Option<usize>),
    EntriesItem(Option<usize>),
    EntryPtr {
        recv: Option<usize>,
        recv_ty: Option<TyRef>,
    },
    Stop,
}

pub(in crate::front::mir) enum HeaderStep {
    Follow(usize),
    Done(TracedPtrAddHeader),
}

pub(in crate::front::mir) fn copy_prod_if_raw_ptr(
    op: &Operand,
    source_ty: Option<&TyRef>,
    target_ty: &TyRef,
    llbc: &Llbc,
) -> HeaderProd {
    let source_is_ptr = source_ty.is_some_and(|ty| tyref_is_raw_ptr_value(ty, llbc));
    if source_is_ptr && tyref_is_raw_ptr_value(target_ty, llbc) {
        HeaderProd::Copy(operand_local(Some(op)))
    } else {
        HeaderProd::Stop
    }
}

pub(in crate::front::mir) fn operand_root_local(op: &Operand) -> Option<usize> {
    match op {
        Operand::Copy(place) | Operand::Move(place) => place_root_local(place),
        Operand::Const(_) => None,
    }
}
/// The `ptr::add` that writes `dest`, and its normal successor.
///
/// [`add_dest_used_only_as_struct_field`] already requires one definition.
/// A missing or repeated call declines the interior alias.
pub(in crate::front::mir) fn ptr_add_def_site(
    llbc: &Llbc,
    body: &Unstructured,
    dest: usize,
) -> Option<(usize, usize)> {
    let mut found = None;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        let Ok(TermKind::Call { call, target, .. }) = bb.term_ref(llbc) else {
            continue;
        };
        if matches!(call.dest.kind, PlaceKind::Local(local) if local as usize == dest) {
            if found.is_some() {
                return None;
            }
            found = Some((bb_idx, *target as usize));
        }
    }
    found
}

pub(in crate::front::mir) fn stmt_assigns_local(
    stmt: &majit_charon_reader::ullbc::Statement,
    local: usize,
) -> bool {
    matches!(
        stmt.stmt_kind_ref(),
        Ok(StmtKind::Assign(place, _))
            if matches!(place.kind, PlaceKind::Local(id) if id as usize == local)
    )
}

pub(in crate::front::mir) fn stmt_has_struct_field_value_use(
    stmt: &majit_charon_reader::ullbc::Statement,
    dest: usize,
) -> bool {
    let mut fields = 0usize;
    let mut other = 0usize;
    match stmt.stmt_kind_ref() {
        Ok(StmtKind::Assign(place, rvalue)) => {
            scan_rvalue_struct_field(rvalue, dest, &mut fields, &mut other);
            if place_is_immediate_struct_field_of(place, dest) {
                fields += 1;
            }
        }
        Ok(StmtKind::Assert(assert)) => {
            bump_struct_field_use(
                operand_struct_field_use(&assert.cond, dest),
                &mut fields,
                &mut other,
            );
        }
        _ => {}
    }
    fields > 0
}

pub(in crate::front::mir) fn term_assigns_local(term: &TermKind, local: usize) -> bool {
    match term {
        TermKind::Call { call, .. } => {
            matches!(call.dest.kind, PlaceKind::Local(id) if id as usize == local)
        }
        _ => false,
    }
}

pub(in crate::front::mir) fn term_has_struct_field_value_use(term: &TermKind, dest: usize) -> bool {
    let mut fields = 0usize;
    let mut other = 0usize;
    match term {
        TermKind::Switch { discr, .. } => {
            bump_struct_field_use(
                operand_struct_field_use(discr, dest),
                &mut fields,
                &mut other,
            );
        }
        TermKind::Call { call, .. } => {
            if let CallFunc::Dynamic(op, _) = &call.func {
                bump_struct_field_use(operand_struct_field_use(op, dest), &mut fields, &mut other);
            }
            for arg in &call.args {
                bump_struct_field_use(operand_struct_field_use(arg, dest), &mut fields, &mut other);
            }
        }
        TermKind::Assert { assert, .. } => {
            bump_struct_field_use(
                operand_struct_field_use(&assert.cond, dest),
                &mut fields,
                &mut other,
            );
        }
        _ => {}
    }
    fields > 0
}

/// Every `(*dest).field` use sees `local` as `ptr::add` captured it.
///
/// The walk starts at the add's normal successor, where the local still
/// holds that captured value. An assignment marks the local stale.
/// Reaching the add again recaptures on its normal successor only: the
/// call's operands are the values in that block, and the unwind edge
/// keeps the incoming stale bit. A join that has seen a stale predecessor
/// stays stale.
pub(in crate::front::mir) fn captured_local_reaches_field_uses(
    llbc: &Llbc,
    body: &Unstructured,
    dest: usize,
    local: usize,
) -> bool {
    let Some((add_bb, add_target)) = ptr_add_def_site(llbc, body, dest) else {
        return false;
    };
    let n = body.body.len();
    if add_target >= n {
        return false;
    }
    let mut state = vec![None; n];
    let mut queue = vec![add_target];
    state[add_target] = Some(false);
    while let Some(bb) = queue.pop() {
        let mut stale = state[bb].unwrap_or(false);
        let block = &body.body[bb];
        for stmt in &block.statements {
            if stale && stmt_has_struct_field_value_use(stmt, dest) {
                return false;
            }
            if stmt_assigns_local(stmt, local) {
                stale = true;
            }
        }
        let Ok(term) = block.term_ref(llbc) else {
            continue;
        };
        if stale && term_has_struct_field_value_use(term, dest) {
            return false;
        }
        let term_assigns = term_assigns_local(term, local);
        for succ in block_successors(llbc, body, bb) {
            let succ_stale = if bb == add_bb && succ == add_target {
                false
            } else if term_assigns {
                true
            } else {
                stale
            };
            match state[succ] {
                None => {
                    state[succ] = Some(succ_stale);
                    queue.push(succ);
                }
                Some(false) if succ_stale => {
                    state[succ] = Some(true);
                    queue.push(succ);
                }
                Some(_) => {}
            }
        }
    }
    true
}

#[allow(dead_code)]
pub(in crate::front::mir) fn result_value_is_branched_or_returned(
    body: &Unstructured,
    llbc: &Llbc,
    dest: usize,
) -> bool {
    for bb in &body.body {
        match bb.term_ref(llbc) {
            Ok(TermKind::Switch { discr, .. }) => {
                if operand_root_local(discr) == Some(dest) {
                    return true;
                }
            }
            Ok(TermKind::Return) => {}
            Ok(TermKind::Call { call, .. }) => {
                if matches!(call.dest.kind, PlaceKind::Local(i) if i as usize == dest) {
                    continue;
                }
            }
            _ => {}
        }
        for stmt in &bb.statements {
            let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                continue;
            };
            // The return slot. A later `expect` call is not an assignment.
            if !matches!(place.kind, PlaceKind::Local(0)) {
                continue;
            }
            if let Rvalue::Use(op, _) = rvalue
                && operand_root_local(op) == Some(dest)
            {
                return true;
            }
        }
    }
    false
}

pub(in crate::front::mir) fn mir_successor_ids(llbc: &Llbc, bb: &BasicBlock) -> Vec<usize> {
    let Ok(term) = bb.term_ref(llbc) else {
        return Vec::new();
    };
    let raw: Vec<u64> = match term {
        TermKind::Goto { target } => vec![*target],
        TermKind::Call {
            target, on_unwind, ..
        }
        | TermKind::Assert {
            target, on_unwind, ..
        }
        | TermKind::Drop {
            target, on_unwind, ..
        } => vec![*target, *on_unwind],
        TermKind::Switch { targets, .. } => match targets {
            SwitchTargets::If(a, b) => vec![*a, *b],
            SwitchTargets::SwitchInt(_, arms, default) => {
                let mut v: Vec<u64> = arms.iter().map(|(_, id)| *id).collect();
                v.push(*default);
                v
            }
        },
        TermKind::Return
        | TermKind::UnwindResume
        | TermKind::UnwindTerminate
        | TermKind::Abort(_)
        | TermKind::Panic { .. }
        | TermKind::UndefinedBehavior
        | TermKind::Unknown => Vec::new(),
    };
    raw.into_iter().map(|id| id as usize).collect()
}

pub(in crate::front::mir) fn block_assigns_local(
    llbc: &Llbc,
    bb: &BasicBlock,
    local: usize,
) -> bool {
    for stmt in &bb.statements {
        if let Ok(StmtKind::Assign(place, _)) = stmt.stmt_kind_ref()
            && matches!(place.kind, PlaceKind::Local(i) if i as usize == local)
        {
            return true;
        }
    }
    matches!(
        bb.term_ref(llbc),
        Ok(TermKind::Call { call, .. })
            if matches!(call.dest.kind, PlaceKind::Local(i) if i as usize == local)
    )
}

/// True when `locals` are stored in any block reachable from `start`.
pub(in crate::front::mir) fn locals_assigned_reachable(
    body: &Unstructured,
    llbc: &Llbc,
    start: usize,
    locals: &[usize],
) -> bool {
    let mut seen = vec![false; body.body.len()];
    let mut stack = vec![start];
    while let Some(bb_id) = stack.pop() {
        if bb_id >= body.body.len() || seen[bb_id] {
            continue;
        }
        seen[bb_id] = true;
        let bb = &body.body[bb_id];
        if locals
            .iter()
            .any(|local| block_assigns_local(llbc, bb, *local))
        {
            return true;
        }
        stack.extend(mir_successor_ids(llbc, bb));
    }
    false
}

/// One producer of `cur`.
///
/// A copy or raw-pointer cast is followed. `entries_item_ptr` and
/// `entry_ptr` stop the walk: their argument is the header. A local
/// with no producer is the pointer itself, already at element 0, and
/// `getinteriorfield` would add `array_items_base` again. Any other
/// rvalue, including a field read of `self.ptr`, aborts so an item
/// pointer is not given a second header offset.
pub(in crate::front::mir) fn header_producer(
    body: &Unstructured,
    llbc: &Llbc,
    cur: usize,
) -> Option<HeaderStep> {
    let mut producers = 0usize;
    let mut kind = HeaderProd::Stop;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                continue;
            };
            if !matches!(place.kind, PlaceKind::Local(local) if local as usize == cur) {
                continue;
            }
            producers += 1;
            kind = match &rvalue {
                Rvalue::Use(op, _) => HeaderProd::Copy(operand_local(Some(op))),
                Rvalue::Cast(_, op, target_ty) => {
                    copy_prod_if_raw_ptr(op, operand_tyref(op), target_ty, llbc)
                }
                Rvalue::UnaryOp(op_kind, op) if unary_op_is_cast(op_kind) => {
                    copy_prod_if_raw_ptr(op, operand_tyref(op), &place.ty, llbc)
                }
                _ => HeaderProd::Stop,
            };
        }
        if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc)
            && matches!(call.dest.kind, PlaceKind::Local(local) if local as usize == cur)
        {
            producers += 1;
            kind = match &call.func {
                CallFunc::Regular(reg) => {
                    let path = regular_call_name_path(reg, llbc).unwrap_or_default();
                    let arg = call.args.first();
                    if path_leaf_is(&path, "entries_item_ptr") {
                        HeaderProd::EntriesItem(operand_local(arg))
                    } else if path_is_rordereddict_entry_ptr(&path) {
                        HeaderProd::EntryPtr {
                            recv: operand_local(arg),
                            recv_ty: arg.and_then(operand_tyref).map(clone_tyref),
                        }
                    } else {
                        HeaderProd::Stop
                    }
                }
                _ => HeaderProd::Stop,
            };
        }
    }
    // No producer means the local is already the pointer that was
    // passed in. A struct-field `ptr::add` of that pointer is the header.
    if producers == 0 {
        return Some(HeaderStep::Done(TracedPtrAddHeader::EntriesItem {
            local: cur,
        }));
    }
    if producers != 1 {
        return None;
    }
    match kind {
        HeaderProd::Copy(Some(src)) if src != cur => Some(HeaderStep::Follow(src)),
        HeaderProd::EntriesItem(Some(local)) => {
            Some(HeaderStep::Done(TracedPtrAddHeader::EntriesItem { local }))
        }
        HeaderProd::EntryPtr {
            recv: Some(recv_local),
            recv_ty: Some(recv_ty),
        } if struct_field_index_named(&recv_ty, llbc, "entries").is_some() => {
            Some(HeaderStep::Done(TracedPtrAddHeader::EntryPtr {
                recv_local,
                recv_ty,
            }))
        }
        _ => None,
    }
}

pub(in crate::front::mir) fn trace_struct_ptr_add_header(
    body: &Unstructured,
    llbc: &Llbc,
    base: usize,
) -> Option<TracedPtrAddHeader> {
    let mut cur = base;
    for _ in 0..32 {
        match header_producer(body, llbc, cur)? {
            HeaderStep::Done(done) => return Some(done),
            HeaderStep::Follow(next) => cur = next,
        }
    }
    None
}

/// Shared gate for the extra-live pass and the `ptr::add` intercept.
///
/// Requires `ptr::add` (not `sub`), a MIR-local index, a named
/// non-transparent struct pointee, and a destination used only as
/// `(*dest).field`. The array identity is taken from the add's pointer
/// type, which is `*mut Entry` even when the header is a `GcEntries`.
///
/// A reassignment of the index or the header that reaches a `(*dest).field`
/// use is a miss. [`Lowering::realias_operand`] reads the local's binding
/// at that use, which is the new value, while `ptr::add` captured the old
/// one.
#[allow(clippy::too_many_arguments)]
pub(in crate::front::mir) fn struct_field_ptr_add_trace(
    reg: &RegularCall,
    args_len: usize,
    base_local: Option<usize>,
    base_ty: Option<&TyRef>,
    index_local: Option<usize>,
    dest_local: usize,
    body: &Unstructured,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
) -> Option<StructFieldPtrAdd> {
    if args_len != 2 || !regular_call_is_ptr_add(reg, llbc) {
        return None;
    }
    let base_local = base_local?;
    let index_local = index_local?;
    let array_type_id = struct_pointee_array_type_id(base_ty?, llbc, tombstoned)?;
    if !add_dest_used_only_as_struct_field(llbc, body, dest_local) {
        return None;
    }
    let header = trace_struct_ptr_add_header(body, llbc, base_local)?;
    let header_local = match &header {
        TracedPtrAddHeader::EntriesItem { local } => *local,
        TracedPtrAddHeader::EntryPtr { recv_local, .. } => *recv_local,
    };
    if !captured_local_reaches_field_uses(llbc, body, dest_local, index_local) {
        return None;
    }
    if header_local != index_local
        && !captured_local_reaches_field_uses(llbc, body, dest_local, header_local)
    {
        return None;
    }
    Some(StructFieldPtrAdd {
        header,
        index_local,
        array_type_id,
    })
}

pub(in crate::front::mir) fn block_has_struct_field_value_use(
    llbc: &Llbc,
    bb: &BasicBlock,
    dest: usize,
) -> bool {
    let mut fields = 0usize;
    let mut other = 0usize;
    for stmt in &bb.statements {
        match stmt.stmt_kind_ref() {
            Ok(StmtKind::Assign(place, rvalue)) => {
                scan_rvalue_struct_field(rvalue, dest, &mut fields, &mut other);
                if place_is_immediate_struct_field_of(place, dest) {
                    fields += 1;
                }
            }
            Ok(StmtKind::Assert(assert)) => {
                bump_struct_field_use(
                    operand_struct_field_use(&assert.cond, dest),
                    &mut fields,
                    &mut other,
                );
            }
            _ => {}
        }
    }
    match bb.term_ref(llbc) {
        Ok(TermKind::Switch { discr, .. }) => bump_struct_field_use(
            operand_struct_field_use(discr, dest),
            &mut fields,
            &mut other,
        ),
        Ok(TermKind::Call { call, .. }) => {
            if let CallFunc::Dynamic(op, _) = &call.func {
                bump_struct_field_use(operand_struct_field_use(op, dest), &mut fields, &mut other);
            }
            for arg in &call.args {
                bump_struct_field_use(operand_struct_field_use(arg, dest), &mut fields, &mut other);
            }
        }
        Ok(TermKind::Assert { assert, .. }) => bump_struct_field_use(
            operand_struct_field_use(&assert.cond, dest),
            &mut fields,
            &mut other,
        ),
        _ => {}
    }
    fields > 0
}

/// `index_mut` result used only as the element: one deref load, one deref
/// store, or both (`*p += k`). Two loads, or any use of the pointer itself,
/// stay residual.
pub(in crate::front::mir) fn index_mut_result_is_element(
    llbc: &Llbc,
    body: &Unstructured,
    dest: usize,
) -> bool {
    let census = dest_deref_census(llbc, body, dest);
    census.defs == 1
        && census.other == 0
        && census.read_derefs <= 1
        && census.write_derefs <= 1
        && census.read_derefs + census.write_derefs >= 1
}

/// The MIR local behind a plain-local [`Operand`], or `None` for a
/// constant or a projected place.
pub(in crate::front::mir) fn operand_local(op: Option<&Operand>) -> Option<usize> {
    match op? {
        Operand::Copy(p) | Operand::Move(p) => match p.kind {
            PlaceKind::Local(i) => Some(i as usize),
            _ => None,
        },
        Operand::Const(_) => None,
    }
}

/// The base MIR local of a `*p = v` deref write (`Projection(Local(p),
/// Deref)`), or `None` for any other place shape — the same destination
/// shape [`Lowering::emit_projection_write`] re-expresses as an
/// `ArrayWrite` when `p` was bound by a workspace index call.
pub(in crate::front::mir) fn deref_write_base_local(place: &Place) -> Option<usize> {
    let PlaceKind::Projection(inner, elem) = &place.kind else {
        return None;
    };
    let ProjectionElem::Atom(s) = elem else {
        return None;
    };
    if s != "Deref" {
        return None;
    }
    match inner.kind {
        PlaceKind::Local(i) => Some(i as usize),
        _ => None,
    }
}

/// Clone a [`TyRef`] (no `Clone` impl on the schema enum).  Used by
/// [`Lowering::resolve_adt_field`] when handing the resolved field's
/// type to [`tyref_to_value_type`].
/// The borrowed place behind `_i = &place` / `&raw place`.
///
/// Consumers are [`Lowering::is_atomic_store`] and
/// [`Lowering::try_lower_mem_exchange`].  The referent is not filtered:
/// `mem::replace` writes whatever place the borrow names.
/// A place `mem::replace` can read and write. `Index` is an `index_mut`
/// result: the element value is already the local, and the write is the
/// `Deref` arm of [`Lowering::emit_projection_write`].
pub(in crate::front::mir) enum MemSlot {
    Place(Place),
    Index(usize),
}

/// One field of a multi-word `mem::replace` / `swap` / `take`.
pub(in crate::front::mir) struct MovePlan {
    pub(in crate::front::mir) ctor_id: u64,
    pub(in crate::front::mir) spans: Vec<MoveSpan>,
    /// Empty for a struct. An enum move switches on `spans`' discriminant
    /// and copies `arms` (one live variant).
    pub(in crate::front::mir) arms: Vec<EnumMoveArm>,
}

pub(in crate::front::mir) struct EnumMoveArm {
    pub(in crate::front::mir) discr: i64,
    pub(in crate::front::mir) spans: Vec<MoveSpan>,
}

#[derive(Clone)]
pub(in crate::front::mir) struct MoveSpan {
    pub(in crate::front::mir) offset: u64,
    /// Physical width. Overlap uses this, not the JIT field's word size.
    pub(in crate::front::mir) bytes: usize,
    pub(in crate::front::mir) kind: SpanKind,
}

#[derive(Clone)]
pub(in crate::front::mir) enum SpanKind {
    /// Named field. `getfield_gc` / `setfield_gc`.
    Field {
        name: String,
        owner: String,
        owner_id: majit_ir::descr::StructId,
        ty: ValueType,
    },
}

#[cfg(test)]
mod tests;
