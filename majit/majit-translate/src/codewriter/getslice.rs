//! `l[start:]` and `l[:-1]` on a GC array in a graph the rtyper never lifted.
//!
//! The front spells a Rust `&slice[start..]` as the deferred marker call
//! `__getslice_rangefrom(slice, start)`, `&slice[..end]` as
//! `__getslice_rangeto(slice, end)`, `&slice[start..end]` as
//! `__getslice_range(slice, start, end)`, and `&slice[..slice.len() - 1]` as
//! `__getslice_minusone(slice)` (`front/slice_index.rs`).  A lifted
//! graph meets the rtyper, whose `rtype_getslice` (`rlist.py`) replaces the
//! `getslice` with `gendirectcall(ll_listslice_startonly, …)` or
//! `ll_listslice_minusone` and mints that helper graph
//! (`translator/rtyper/rlist.rs`).  A graph that stays on the rich-`OpKind`
//! spine never meets the rtyper, so the marker survived to the codewriter as
//! a call with no graph — a symbolic residual no host symbol backs, and the
//! wall between a builtin gateway body and its trace whenever the body
//! passes `&args[1..]` or `&args[..n-1]` on.
//!
//! This module gives that spine the same answer: [`listslice_startonly_path`]
//! mints an `ll_listslice_startonly` graph in the rich model for the array's
//! item kind and registers it as an ordinary callee, and `jtransform`
//! redirects the marker call to it.  The graph is `rlist.py`'s helper
//! line for line:
//!
//! ```python
//! def ll_listslice_startonly(RESLIST, l1, start):
//!     len1 = l1.ll_length()
//!     newlength = len1 - start
//!     l = RESLIST.ll_newlist(newlength)
//!     ll_arraycopy(l1, l, start, 0, newlength)
//!     return l
//! ```
//!
//! The helper calls `ll_arraycopy` the same way `rlist.py` does; jtransform
//! turns that call into `OS_ARRAYCOPY` (`do_fixed_list_ll_arraycopy`).
//! The helper is therefore loop-free, so `look_inside_graph` admits it and
//! it is a candidate — the residual is the copy, not the slice helper.
//!
//! The marker may carry the array's item kind as a second path segment.
//! [`listslice_marker_suffix`] spells that segment the same way
//! [`helper_name`] spells a minted helper (`int___i64_`,
//! `ref__majit__object_ref_gcarray`). [`getslice_marker_identity`] reads it
//! first. A marker with no suffix, or a suffix that does not round-trip to
//! an item kind and array identity, falls back to [`array_identity_of_base`].

use crate::codewriter::call::CallControl;
use crate::flowspace::model::Variable;
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};
use crate::parse::CallPath;

/// The leaf every minted helper shares, suffixed per array identity.
pub const LL_LISTSLICE_STARTONLY: &str = "ll_listslice_startonly";
/// `rlist.py ll_listslice_minusone` — `l[:-1]`.
pub const LL_LISTSLICE_MINUSONE: &str = "ll_listslice_minusone";
/// `rlist.py ll_listslice_startstop` with start 0 — `l[:end]`.
pub const LL_LISTSLICE_RANGETO: &str = "ll_listslice_rangeto";
/// `rlist.py ll_listslice_startstop` — `l[start:end]`.
pub const LL_LISTSLICE_STARTSTOP: &str = "ll_listslice_startstop";

/// The item kind and ARRAY identity of the GC array `base` names, read off an
/// `ArrayRead` / `ArrayWrite` / `ArrayLen` / `NewArrayClear` on the same
/// value anywhere in the graph.  `None` when the graph never touches it as
/// an array.
///
/// "The same value" is judged across block links, not by `Variable`
/// identity: the front threads a live value through every block as a fresh
/// phi `inputarg`, so the array a slice reads in one block and the array the
/// marker slices in a later block are distinct `Variable`s bound together
/// only by the links between them.
pub fn array_identity_of_base(
    graph: &FunctionGraph,
    base: &Variable,
) -> Option<(ValueType, Option<String>)> {
    let classes = LinkClasses::of(graph);
    let same = |v: &Variable| classes.same(v, base);
    let mut len_only: Option<Option<String>> = None;
    let mut object_slice_input = false;
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        match &op.kind {
            OpKind::ArrayRead {
                base: b,
                item_ty,
                array_type_id,
                ..
            }
            | OpKind::ArrayWrite {
                base: b,
                item_ty,
                array_type_id,
                ..
            } if same(b) => return Some((item_ty.clone(), array_type_id.clone())),
            OpKind::NewArray {
                item_ty,
                array_type_id,
                ..
            }
            | OpKind::NewArrayClear {
                item_ty,
                array_type_id,
                ..
            } if op.result.as_ref().is_some_and(same) => {
                return Some((item_ty.clone(), array_type_id.clone()));
            }
            OpKind::ArrayLen {
                base: b,
                array_type_id,
                ..
            } if len_only.is_none() && same(b) => len_only = Some(array_type_id.clone()),
            OpKind::Input {
                class_root: Some(root),
                ..
            } if root == OBJECT_REF_SLICE_CLASS_ROOT && op.result.as_ref().is_some_and(same) => {
                object_slice_input = true;
            }
            _ => {}
        }
    }
    // A length read alone names the ARRAY but not its item kind; the object
    // array is the one identity whose items are known from the name.
    match len_only {
        Some(Some(id)) if id == crate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID => {
            Some((ValueType::Ref(None), Some(id)))
        }
        // A `&[PyObjectRef]` parameter names its own identity: the front
        // stamps the slice's class root even where no sibling array op
        // carries an id.  A thin-pointer element keeps the unnamed id —
        // the identity-less descr mint already sizes it right, and it is
        // the id the graph's own length reads use.
        _ if object_slice_input => Some((ValueType::Ref(None), None)),
        _ => None,
    }
}

/// The class root the front stamps on a `&[PyObjectRef]` input
/// (`front/mir.rs` field/param layout spelling).
const OBJECT_REF_SLICE_CLASS_ROOT: &str = "[*mut PyObject]";

/// Union-find over the graph's links: `link.args[i]` and
/// `target.inputargs[i]` carry one value.
pub(crate) struct LinkClasses {
    parent: std::collections::HashMap<Variable, Variable>,
}

impl LinkClasses {
    pub(crate) fn of(graph: &FunctionGraph) -> Self {
        let mut classes = Self {
            parent: std::collections::HashMap::new(),
        };
        for block in &graph.blocks {
            for link in &block.exits {
                let target = graph.block(link.target);
                for (arg, input) in link.args.iter().zip(&target.inputargs) {
                    if let LinkArg::Value(value) = arg {
                        classes.union(value, input);
                    }
                }
            }
        }
        classes
    }

    pub(crate) fn find(&self, var: &Variable) -> Variable {
        let mut cur = var.clone();
        while let Some(next) = self.parent.get(&cur) {
            if *next == cur {
                break;
            }
            cur = next.clone();
        }
        cur
    }

    fn union(&mut self, a: &Variable, b: &Variable) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent.insert(ra, rb);
        }
    }

    pub(crate) fn same(&self, a: &Variable, b: &Variable) -> bool {
        a == b || self.find(a) == self.find(b)
    }
}

/// The helper path for `(item_ty, array_type_id)`, minting and registering
/// the graph on first use.
pub fn listslice_startonly_path(
    cc: &mut CallControl,
    item_ty: &ValueType,
    array_type_id: Option<&str>,
) -> CallPath {
    let name = helper_name(LL_LISTSLICE_STARTONLY, item_ty, array_type_id);
    let path = CallPath::from_segments([name.as_str()]);
    if !cc.has_function_graph(&path) {
        let graph = build_ll_listslice_startonly_graph(&name, item_ty, array_type_id);
        register_listslice_helper(cc, path.clone(), graph);
    }
    path
}

/// The helper path for `__getslice_minusone`, minting
/// `rlist.py ll_listslice_minusone` on first use.
pub fn listslice_minusone_path(
    cc: &mut CallControl,
    item_ty: &ValueType,
    array_type_id: Option<&str>,
) -> CallPath {
    let name = helper_name(LL_LISTSLICE_MINUSONE, item_ty, array_type_id);
    let path = CallPath::from_segments([name.as_str()]);
    if !cc.has_function_graph(&path) {
        let graph = build_ll_listslice_minusone_graph(&name, item_ty, array_type_id);
        register_listslice_helper(cc, path.clone(), graph);
    }
    path
}

/// The helper path for `__getslice_rangeto`, minting
/// `rlist.py ll_listslice_startstop` with start 0 on first use.
pub fn listslice_rangeto_path(
    cc: &mut CallControl,
    item_ty: &ValueType,
    array_type_id: Option<&str>,
) -> CallPath {
    let name = helper_name(LL_LISTSLICE_RANGETO, item_ty, array_type_id);
    let path = CallPath::from_segments([name.as_str()]);
    if !cc.has_function_graph(&path) {
        let graph = build_ll_listslice_rangeto_graph(&name, item_ty, array_type_id);
        register_listslice_helper(cc, path.clone(), graph);
    }
    path
}

/// The helper path for `__getslice_range`, minting
/// `rlist.py ll_listslice_startstop` on first use.
pub fn listslice_range_path(
    cc: &mut CallControl,
    item_ty: &ValueType,
    array_type_id: Option<&str>,
) -> CallPath {
    let name = helper_name(LL_LISTSLICE_STARTSTOP, item_ty, array_type_id);
    let path = CallPath::from_segments([name.as_str()]);
    if !cc.has_function_graph(&path) {
        let graph = build_ll_listslice_startstop_graph(&name, item_ty, array_type_id);
        register_listslice_helper(cc, path.clone(), graph);
    }
    path
}

/// Register a minted `ll_listslice_*` graph and admit it as a candidate.
/// The helper is loop-free (`ll_arraycopy` is the residual), so
/// `look_inside_graph` accepts it — the same seed `call.py` uses for a
/// helper the BFS could not have reached.
fn register_listslice_helper(cc: &mut CallControl, path: CallPath, graph: FunctionGraph) {
    cc.register_function_graph(path.clone(), graph);
    cc.add_candidate_graph(path);
}

/// `ll_listslice_*__<item>[__<array identity>]`, the identity
/// reduced to identifier characters so the leaf stays a plain path segment.
fn helper_name(leaf: &str, item_ty: &ValueType, array_type_id: Option<&str>) -> String {
    let item = match item_ty {
        ValueType::Ref(_) => "ref",
        ValueType::Float => "float",
        _ => "int",
    };
    let mut name = format!("{leaf}__{item}");
    if let Some(id) = array_type_id {
        name.push_str("__");
        name.push_str(&sanitize_array_id(id));
    }
    name
}

fn sanitize_array_id(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// The second path segment of a `__getslice_*` marker: the same
/// `{item}[__{array identity}]` tail [`helper_name`] appends.
///
/// `None` when `item_ty` is not one of the three helper banks, or when
/// sanitizing `array_type_id` cannot be reversed to the same string. A
/// missing suffix means the codewriter falls back to
/// [`array_identity_of_base`] rather than minting a helper for a guessed
/// element width.
pub fn listslice_marker_suffix(item_ty: &ValueType, array_type_id: Option<&str>) -> Option<String> {
    let item = match item_ty {
        ValueType::Ref(None) => "ref",
        ValueType::Float => "float",
        ValueType::Int | ValueType::Unsigned => "int",
        _ => return None,
    };
    let mut suffix = item.to_string();
    if let Some(id) = array_type_id {
        let sanitized = sanitize_array_id(id);
        if restore_array_id(&sanitized).as_deref() != Some(id) {
            return None;
        }
        suffix.push_str("__");
        suffix.push_str(&sanitized);
    }
    Some(suffix)
}

fn restore_array_id(sanitized: &str) -> Option<String> {
    if let Some(inner) = sanitized
        .strip_prefix('_')
        .and_then(|s| s.strip_suffix('_'))
        && matches!(
            inner,
            "u8" | "u16"
                | "u32"
                | "u64"
                | "usize"
                | "i8"
                | "i16"
                | "i32"
                | "i64"
                | "isize"
                | "f32"
                | "f64"
                | "bool"
                | "str"
        )
    {
        let id = format!("[{inner}]");
        if sanitize_array_id(&id) == sanitized {
            return Some(id);
        }
    }
    let restored = sanitized.replace("__", "::");
    (sanitize_array_id(&restored) == sanitized).then_some(restored)
}

/// Inverse of [`listslice_marker_suffix`]. `None` when `suffix` is not a
/// suffix this module would emit.
pub fn decode_listslice_marker_suffix(suffix: &str) -> Option<(ValueType, Option<String>)> {
    let (item, raw_id) = match suffix.split_once("__") {
        Some((item, rest)) => (item, Some(rest)),
        None => (suffix, None),
    };
    let item_ty = match item {
        "ref" => ValueType::Ref(None),
        "float" => ValueType::Float,
        "int" => ValueType::Int,
        _ => return None,
    };
    let array_type_id = match raw_id {
        None => None,
        Some(sanitized) => Some(restore_array_id(sanitized)?),
    };
    let again = listslice_marker_suffix(&item_ty, array_type_id.as_deref())?;
    (again == suffix).then_some((item_ty, array_type_id))
}

/// Item kind for a `__getslice_*` marker. The carried suffix wins. Without
/// one, or when it does not decode, the sibling-array scan applies.
pub fn getslice_marker_identity(
    graph: &FunctionGraph,
    op: &SpaceOperation,
) -> Option<(ValueType, Option<String>)> {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        args,
        ..
    } = &op.kind
    else {
        return None;
    };
    if let Some(suffix) = segments.get(1)
        && let Some(decoded) = decode_listslice_marker_suffix(suffix)
    {
        return Some(decoded);
    }
    let base = args.first()?;
    array_identity_of_base(graph, base)
}

/// `rlist.py ll_listslice_startonly` in the rich model, with `ll_arraycopy`
/// as the residual call (`list.ll_arraycopy` / `OS_ARRAYCOPY`).
pub fn build_ll_listslice_startonly_graph(
    name: &str,
    item_ty: &ValueType,
    array_type_id: Option<&str>,
) -> FunctionGraph {
    let array_type_id = array_type_id.map(str::to_string);
    let mut graph = FunctionGraph::new(name);
    let start_block = graph.startblock;

    // Parameters: `l1`, `start`.
    let l1 = graph.alloc_value_var();
    let start = graph.alloc_value_var();
    for (var, param, ty) in [
        (&l1, "l1", ValueType::Ref(None)),
        (&start, "start", ValueType::Int),
    ] {
        graph.push_inputarg_var(start_block, var.clone());
        graph.push_op_with_result_var(
            start_block,
            OpKind::Input {
                name: param.to_string(),
                ty,
                class_root: None,
            },
            var.clone(),
        );
    }

    // len1 = l1.ll_length(); newlength = len1 - start;
    // l = RESLIST.ll_newlist(newlength)
    let len1 = push(
        &mut graph,
        start_block,
        OpKind::ArrayLen {
            base: l1.clone(),
            array_type_id: array_type_id.clone(),
            nolength: false,
        },
    );
    let newlength = push(
        &mut graph,
        start_block,
        OpKind::BinOp {
            op: "sub".into(),
            lhs: len1,
            rhs: start.clone(),
            result_ty: ValueType::Int,
        },
    );
    let new_list = push(
        &mut graph,
        start_block,
        OpKind::NewArrayClear {
            length: newlength.clone(),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
    );
    emit_ll_arraycopy_call(&mut graph, start_block, l1, start, new_list, newlength);
    graph
}

/// `rlist.py ll_listslice_minusone` in the rich model.
///
/// ```python
/// def ll_listslice_minusone(RESLIST, l1):
///     newlength = l1.ll_length() - 1
///     l = RESLIST.ll_newlist(newlength)
///     ll_arraycopy(l1, l, 0, 0, newlength)
///     return l
/// ```
pub fn build_ll_listslice_minusone_graph(
    name: &str,
    item_ty: &ValueType,
    array_type_id: Option<&str>,
) -> FunctionGraph {
    let array_type_id = array_type_id.map(str::to_string);
    let mut graph = FunctionGraph::new(name);
    let start_block = graph.startblock;

    let l1 = graph.alloc_value_var();
    graph.push_inputarg_var(start_block, l1.clone());
    graph.push_op_with_result_var(
        start_block,
        OpKind::Input {
            name: "l1".to_string(),
            ty: ValueType::Ref(None),
            class_root: None,
        },
        l1.clone(),
    );

    let len1 = push(
        &mut graph,
        start_block,
        OpKind::ArrayLen {
            base: l1.clone(),
            array_type_id: array_type_id.clone(),
            nolength: false,
        },
    );
    let one = push(&mut graph, start_block, OpKind::ConstInt(1));
    let raw = push(
        &mut graph,
        start_block,
        OpKind::BinOp {
            op: "sub".into(),
            lhs: len1,
            rhs: one,
            result_ty: ValueType::Int,
        },
    );
    let zero = push(&mut graph, start_block, OpKind::ConstInt(0));
    // `[][:-1]` is `[]`.  Upstream `ll_assert(newlength >= 0)` translates
    // out; a negative `NewArrayClear` is rejected by the wasm allocator.
    let nonneg = push(
        &mut graph,
        start_block,
        OpKind::BinOp {
            op: "ge".into(),
            lhs: raw.clone(),
            rhs: zero.clone(),
            result_ty: ValueType::Bool,
        },
    );
    let (use_raw, use_raw_args) = graph.create_block_with_arg_vars(2);
    let (use_zero, use_zero_args) = graph.create_block_with_arg_vars(2);
    let (copy, copy_args) = graph.create_block_with_arg_vars(2);
    graph.set_branch(
        start_block,
        nonneg,
        use_raw,
        vec![l1.clone(), raw],
        use_zero,
        vec![l1, zero],
    );
    graph.set_goto(use_raw, copy, use_raw_args);
    graph.set_goto(use_zero, copy, use_zero_args);
    let [c_l1, newlength] = copy_args.as_slice() else {
        unreachable!("copy block was created with two inputargs")
    };
    let new_list = push(
        &mut graph,
        copy,
        OpKind::NewArrayClear {
            length: newlength.clone(),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
    );
    let start = push(&mut graph, copy, OpKind::ConstInt(0));
    emit_ll_arraycopy_call(
        &mut graph,
        copy,
        c_l1.clone(),
        start,
        new_list,
        newlength.clone(),
    );
    graph
}

/// `rlist.py ll_listslice_startstop(RESLIST, l1, 0, stop)` in the rich model.
///
/// ```python
/// length = l1.ll_length()
/// if stop > length:
///     stop = length
/// newlength = stop
/// l = RESLIST.ll_newlist(newlength)
/// ll_arraycopy(l1, l, 0, 0, newlength)
/// ```
pub fn build_ll_listslice_rangeto_graph(
    name: &str,
    item_ty: &ValueType,
    array_type_id: Option<&str>,
) -> FunctionGraph {
    let array_type_id = array_type_id.map(str::to_string);
    let mut graph = FunctionGraph::new(name);
    let start_block = graph.startblock;

    let l1 = graph.alloc_value_var();
    let stop = graph.alloc_value_var();
    for (var, param, ty) in [
        (&l1, "l1", ValueType::Ref(None)),
        (&stop, "stop", ValueType::Int),
    ] {
        graph.push_inputarg_var(start_block, var.clone());
        graph.push_op_with_result_var(
            start_block,
            OpKind::Input {
                name: param.to_string(),
                ty,
                class_root: None,
            },
            var.clone(),
        );
    }

    let len1 = push(
        &mut graph,
        start_block,
        OpKind::ArrayLen {
            base: l1.clone(),
            array_type_id: array_type_id.clone(),
            nolength: false,
        },
    );
    let in_range = push(
        &mut graph,
        start_block,
        OpKind::BinOp {
            op: "le".into(),
            lhs: stop.clone(),
            rhs: len1.clone(),
            result_ty: ValueType::Bool,
        },
    );
    let (use_stop, use_stop_args) = graph.create_block_with_arg_vars(2);
    let (use_len, use_len_args) = graph.create_block_with_arg_vars(2);
    let (copy, copy_args) = graph.create_block_with_arg_vars(2);
    graph.set_branch(
        start_block,
        in_range,
        use_stop,
        vec![l1.clone(), stop],
        use_len,
        vec![l1, len1],
    );
    graph.set_goto(use_stop, copy, use_stop_args);
    graph.set_goto(use_len, copy, use_len_args);

    let [c_l1, c_len] = copy_args.as_slice() else {
        unreachable!("copy block was created with two inputargs")
    };
    let new_list = push(
        &mut graph,
        copy,
        OpKind::NewArrayClear {
            length: c_len.clone(),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
    );
    let start = push(&mut graph, copy, OpKind::ConstInt(0));
    emit_ll_arraycopy_call(
        &mut graph,
        copy,
        c_l1.clone(),
        start,
        new_list,
        c_len.clone(),
    );
    graph
}

/// `rlist.py ll_listslice_startstop(RESLIST, l1, start, stop)` in the rich model.
///
/// ```python
/// length = l1.ll_length()
/// if stop > length:
///     stop = length
/// newlength = stop - start
/// l = RESLIST.ll_newlist(newlength)
/// ll_arraycopy(l1, l, start, 0, newlength)
/// ```
pub fn build_ll_listslice_startstop_graph(
    name: &str,
    item_ty: &ValueType,
    array_type_id: Option<&str>,
) -> FunctionGraph {
    let array_type_id = array_type_id.map(str::to_string);
    let mut graph = FunctionGraph::new(name);
    let start_block = graph.startblock;

    let l1 = graph.alloc_value_var();
    let start = graph.alloc_value_var();
    let stop = graph.alloc_value_var();
    for (var, param, ty) in [
        (&l1, "l1", ValueType::Ref(None)),
        (&start, "start", ValueType::Int),
        (&stop, "stop", ValueType::Int),
    ] {
        graph.push_inputarg_var(start_block, var.clone());
        graph.push_op_with_result_var(
            start_block,
            OpKind::Input {
                name: param.to_string(),
                ty,
                class_root: None,
            },
            var.clone(),
        );
    }

    let len1 = push(
        &mut graph,
        start_block,
        OpKind::ArrayLen {
            base: l1.clone(),
            array_type_id: array_type_id.clone(),
            nolength: false,
        },
    );
    let in_range = push(
        &mut graph,
        start_block,
        OpKind::BinOp {
            op: "le".into(),
            lhs: stop.clone(),
            rhs: len1.clone(),
            result_ty: ValueType::Bool,
        },
    );
    let (use_stop, use_stop_args) = graph.create_block_with_arg_vars(3);
    let (use_len, use_len_args) = graph.create_block_with_arg_vars(3);
    let (copy, copy_args) = graph.create_block_with_arg_vars(3);
    graph.set_branch(
        start_block,
        in_range,
        use_stop,
        vec![l1.clone(), start.clone(), stop],
        use_len,
        vec![l1, start, len1],
    );
    graph.set_goto(use_stop, copy, use_stop_args);
    graph.set_goto(use_len, copy, use_len_args);

    let [c_l1, c_start, c_stop] = copy_args.as_slice() else {
        unreachable!("copy block was created with three inputargs")
    };
    let newlength = push(
        &mut graph,
        copy,
        OpKind::BinOp {
            op: "sub".into(),
            lhs: c_stop.clone(),
            rhs: c_start.clone(),
            result_ty: ValueType::Int,
        },
    );
    let new_list = push(
        &mut graph,
        copy,
        OpKind::NewArrayClear {
            length: newlength.clone(),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
    );
    emit_ll_arraycopy_call(
        &mut graph,
        copy,
        c_l1.clone(),
        c_start.clone(),
        new_list,
        newlength,
    );
    graph
}

/// `ll_arraycopy(l1, l, start, 0, newlength)` as the residual call
/// `rgc.py` / `rlist.py` emit.  jtransform's `do_fixed_list_ll_arraycopy`
/// turns this into `OS_ARRAYCOPY`.
fn emit_ll_arraycopy_call(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    l1: Variable,
    start: Variable,
    new_list: Variable,
    newlength: Variable,
) {
    let dest_start = push(graph, block, OpKind::ConstInt(0));
    graph.push_op_var(
        block,
        OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["ll_arraycopy".into()],
                fun_decl_id: None,
                generic_rust_args: Vec::new(),
            },
            args: crate::model::call_args(vec![l1, new_list.clone(), start, dest_start, newlength]),
            result_ty: ValueType::Void,
        },
        false,
    );
    graph.set_return(block, Some(new_list));
}

fn push(graph: &mut FunctionGraph, block: crate::model::BlockId, kind: OpKind) -> Variable {
    graph
        .push_op_var(block, kind, true)
        .expect("a value-producing op allocates its result")
}

/// The op is the front's `__getslice_rangefrom(slice, start)` marker.
pub fn is_getslice_rangefrom(op: &SpaceOperation) -> bool {
    matches!(
        &op.kind,
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if is_getslice_marker(segments, "__getslice_rangefrom") && args.len() == 2
    )
}

/// The op is the front's `__getslice_minusone(slice)` marker (`l[:-1]`).
pub fn is_getslice_minusone(op: &SpaceOperation) -> bool {
    matches!(
        &op.kind,
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if is_getslice_marker(segments, "__getslice_minusone") && args.len() == 1
    )
}

/// The op is the front's `__getslice_rangeto(slice, end)` marker (`l[:end]`).
pub fn is_getslice_rangeto(op: &SpaceOperation) -> bool {
    matches!(
        &op.kind,
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if is_getslice_marker(segments, "__getslice_rangeto") && args.len() == 2
    )
}

/// The op is the front's `__getslice_range(slice, start, end)` marker (`l[start:end]`).
pub fn is_getslice_range(op: &SpaceOperation) -> bool {
    matches!(
        &op.kind,
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } if is_getslice_marker(segments, "__getslice_range") && args.len() == 3
    )
}

fn is_getslice_marker(segments: &[String], name: &str) -> bool {
    segments.first().map(String::as_str) == Some(name) && segments.len() <= 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    #[test]
    fn helper_graph_has_the_rlist_shape() {
        let graph = build_ll_listslice_startonly_graph(
            "ll_listslice_startonly__ref",
            &ValueType::Ref(None),
            Some(crate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID),
        );
        // start / return / except — the copy is a residual call, not a loop.
        assert_eq!(graph.blocks.len(), 3);
        let start = graph.block(graph.startblock);
        assert_eq!(start.inputargs.len(), 2);
        assert!(
            start
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ArrayLen { .. }))
        );
        assert!(
            start
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::NewArrayClear { .. }))
        );
        assert!(
            start.operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments == &["ll_arraycopy"]
            )),
            "the helper calls ll_arraycopy"
        );
        assert!(
            !start.operations.iter().any(|op| matches!(
                op.kind,
                OpKind::ArrayRead { .. } | OpKind::ArrayWrite { .. }
            )),
            "the copy is not an item loop"
        );
        assert_eq!(start.exits[0].target, graph.returnblock);
    }

    #[test]
    fn minusone_helper_has_one_list_arg_and_arraycopy() {
        let graph = build_ll_listslice_minusone_graph(
            "ll_listslice_minusone__ref",
            &ValueType::Ref(None),
            Some(crate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID),
        );
        let start = graph.block(graph.startblock);
        assert_eq!(start.inputargs.len(), 1);
        assert!(
            start
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ArrayLen { .. }))
        );
        assert_eq!(start.exits.len(), 2);
        assert!(graph.blocks.iter().any(|block| {
            block
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::NewArrayClear { .. }))
        }));
        assert!(graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments == &["ll_arraycopy"]
                )
            })
        }));
        assert!(!graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                matches!(
                    op.kind,
                    OpKind::ArrayRead { .. } | OpKind::ArrayWrite { .. }
                )
            })
        }));
    }

    #[test]
    fn rangeto_helper_clamps_stop_then_copies() {
        let graph = build_ll_listslice_rangeto_graph(
            "ll_listslice_rangeto__ref",
            &ValueType::Ref(None),
            Some(crate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID),
        );
        let start = graph.block(graph.startblock);
        assert_eq!(start.inputargs.len(), 2);
        assert!(
            start
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ArrayLen { .. }))
        );
        assert_eq!(start.exits.len(), 2);
        assert!(graph.blocks.iter().any(|block| {
            block
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::NewArrayClear { .. }))
        }));
        assert!(graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments == &["ll_arraycopy"]
                )
            })
        }));
        assert!(is_getslice_rangeto(&SpaceOperation {
            result: Some(Variable::new()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__getslice_rangeto".into()],
                    fun_decl_id: None,
                    generic_rust_args: Vec::new(),
                },
                args: crate::model::call_args(vec![Variable::new(), Variable::new()]),
                result_ty: ValueType::Ref(None),
            },
        }));
    }

    #[test]
    fn startstop_helper_clamps_stop_then_copies_from_start() {
        let graph = build_ll_listslice_startstop_graph(
            "ll_listslice_startstop__ref",
            &ValueType::Ref(None),
            Some(crate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID),
        );
        let start = graph.block(graph.startblock);
        assert_eq!(start.inputargs.len(), 3);
        assert!(
            start
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ArrayLen { .. }))
        );
        assert_eq!(start.exits.len(), 2);
        assert!(graph.blocks.iter().any(|block| {
            block
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::NewArrayClear { .. }))
        }));
        assert!(graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments == &["ll_arraycopy"]
                )
            })
        }));
        assert!(graph.blocks.iter().any(|block| {
            block
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "sub"))
        }));
        assert!(is_getslice_range(&SpaceOperation {
            result: Some(Variable::new()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__getslice_range".into()],
                    fun_decl_id: None,
                    generic_rust_args: Vec::new(),
                },
                args: crate::model::call_args(vec![
                    Variable::new(),
                    Variable::new(),
                    Variable::new()
                ]),
                result_ty: ValueType::Ref(None),
            },
        }));
    }

    #[test]
    fn array_identity_comes_from_a_sibling_read() {
        let mut graph = FunctionGraph::new("f");
        let args = graph.alloc_value_var();
        let index = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayRead {
                base: args.clone(),
                index,
                item_ty: ValueType::Ref(None),
                array_type_id: Some("objref".into()),
                nolength: false,
                pure: false,
            },
            true,
        );
        assert_eq!(
            array_identity_of_base(&graph, &args),
            Some((ValueType::Ref(None), Some("objref".into())))
        );
        let other = graph.alloc_value_var();
        assert_eq!(array_identity_of_base(&graph, &other), None);
    }

    #[test]
    fn array_identity_follows_the_value_across_links() {
        let mut graph = FunctionGraph::new("f");
        let args = graph.alloc_value_var();
        let index = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayRead {
                base: args.clone(),
                index,
                item_ty: ValueType::Ref(None),
                array_type_id: Some("objref".into()),
                nolength: false,
                pure: false,
            },
            true,
        );
        let (next, next_args) = graph.create_block_with_arg_vars(1);
        graph.set_goto(graph.startblock, next, vec![args]);
        assert_eq!(
            array_identity_of_base(&graph, &next_args[0]),
            Some((ValueType::Ref(None), Some("objref".into())))
        );
    }

    #[test]
    fn an_object_slice_input_names_its_own_identity() {
        let mut graph = FunctionGraph::new("f");
        let args = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, args.clone());
        graph.push_op_with_result_var(
            graph.startblock,
            OpKind::Input {
                name: "args_w".into(),
                ty: ValueType::Ref(None),
                class_root: Some(OBJECT_REF_SLICE_CLASS_ROOT.into()),
            },
            args.clone(),
        );
        assert_eq!(
            array_identity_of_base(&graph, &args),
            Some((ValueType::Ref(None), None))
        );
        let mut graph = FunctionGraph::new("g");
        let other = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, other.clone());
        graph.push_op_with_result_var(
            graph.startblock,
            OpKind::Input {
                name: "w_obj".into(),
                ty: ValueType::Ref(None),
                class_root: Some("PyObject".into()),
            },
            other.clone(),
        );
        assert_eq!(array_identity_of_base(&graph, &other), None);
    }

    #[test]
    fn a_bare_length_read_names_only_the_object_array() {
        let mut graph = FunctionGraph::new("f");
        let args = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayLen {
                base: args.clone(),
                array_type_id: Some(crate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID.into()),
                nolength: false,
            },
            true,
        );
        assert_eq!(
            array_identity_of_base(&graph, &args),
            Some((
                ValueType::Ref(None),
                Some(crate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID.into())
            ))
        );
        let mut graph = FunctionGraph::new("g");
        let bytes = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayLen {
                base: bytes.clone(),
                array_type_id: Some("bytes".into()),
                nolength: false,
            },
            true,
        );
        assert_eq!(array_identity_of_base(&graph, &bytes), None);
    }

    #[test]
    fn the_helper_is_minted_once_and_is_a_candidate() {
        use crate::codewriter::call::CallKind;
        let mut cc = CallControl::new();
        let first = listslice_startonly_path(&mut cc, &ValueType::Ref(None), Some("objref"));
        let again = listslice_startonly_path(&mut cc, &ValueType::Ref(None), Some("objref"));
        assert_eq!(first, again);
        let ints = listslice_startonly_path(&mut cc, &ValueType::Int, Some("ints"));
        assert_ne!(first, ints);
        assert!(cc.has_function_graph(&first));
        let call = SpaceOperation {
            result: Some(Variable::new()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![first.last_segment().unwrap().to_string()],
                    fun_decl_id: None,
                    generic_rust_args: Vec::new(),
                },
                args: crate::model::call_args(vec![Variable::new(), Variable::new()]),
                result_ty: ValueType::Ref(None),
            },
        };
        // Loop-free helper: `look_inside_graph` admits it, so Regular.
        assert_eq!(cc.guess_call_kind(&call), CallKind::Regular);
    }

    #[test]
    fn minusone_helper_is_minted_once_and_is_a_candidate() {
        use crate::codewriter::call::CallKind;
        let mut cc = CallControl::new();
        let first = listslice_minusone_path(&mut cc, &ValueType::Ref(None), Some("objref"));
        let again = listslice_minusone_path(&mut cc, &ValueType::Ref(None), Some("objref"));
        assert_eq!(first, again);
        let startonly = listslice_startonly_path(&mut cc, &ValueType::Ref(None), Some("objref"));
        assert_ne!(first, startonly);
        assert!(cc.has_function_graph(&first));
        let call = SpaceOperation {
            result: Some(Variable::new()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![first.last_segment().unwrap().to_string()],
                    fun_decl_id: None,
                    generic_rust_args: Vec::new(),
                },
                args: crate::model::call_args(vec![Variable::new()]),
                result_ty: ValueType::Ref(None),
            },
        };
        assert_eq!(cc.guess_call_kind(&call), CallKind::Regular);
        assert!(is_getslice_minusone(&SpaceOperation {
            result: Some(Variable::new()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__getslice_minusone".into()],
                    fun_decl_id: None,
                    generic_rust_args: Vec::new(),
                },
                args: crate::model::call_args(vec![Variable::new()]),
                result_ty: ValueType::Ref(None),
            },
        }));
    }

    #[test]
    fn carried_integer_suffix_resolves_without_a_sibling_read() {
        let mut graph = FunctionGraph::new("f");
        let items = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayLen {
                base: items.clone(),
                array_type_id: Some("[i64]".into()),
                nolength: false,
            },
            true,
        );
        assert_eq!(array_identity_of_base(&graph, &items), None);
        let suffix = listslice_marker_suffix(&ValueType::Int, Some("[i64]")).unwrap();
        assert_eq!(suffix, "int___i64_");
        let op = SpaceOperation {
            result: Some(Variable::new()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__getslice_range".into(), suffix],
                    fun_decl_id: None,
                    generic_rust_args: Vec::new(),
                },
                args: crate::model::call_args(vec![items, Variable::new(), Variable::new()]),
                result_ty: ValueType::Ref(None),
            },
        };
        assert!(is_getslice_range(&op));
        assert_eq!(
            getslice_marker_identity(&graph, &op),
            Some((ValueType::Int, Some("[i64]".into())))
        );
    }

    #[test]
    fn unsuffixed_marker_resolves_through_the_array_scan() {
        let mut graph = FunctionGraph::new("f");
        let items = graph.alloc_value_var();
        let index = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayRead {
                base: items.clone(),
                index,
                item_ty: ValueType::Float,
                array_type_id: Some("[f64]".into()),
                nolength: false,
                pure: false,
            },
            true,
        );
        let op = SpaceOperation {
            result: Some(Variable::new()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__getslice_rangefrom".into()],
                    fun_decl_id: None,
                    generic_rust_args: Vec::new(),
                },
                args: crate::model::call_args(vec![items, Variable::new()]),
                result_ty: ValueType::Ref(None),
            },
        };
        assert!(is_getslice_rangefrom(&op));
        assert_eq!(
            getslice_marker_identity(&graph, &op),
            Some((ValueType::Float, Some("[f64]".into())))
        );
    }
}
