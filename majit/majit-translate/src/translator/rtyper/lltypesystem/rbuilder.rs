//! RPython `rpython/rtyper/lltypesystem/rbuilder.py` parity module.
//!
//! This module carries the low-level builder container shapes, reprs, and
//! append/grow/build helper graphs. The public lltype names match upstream:
//! `STRINGPIECE`, `STRINGBUILDER`, `UNICODEPIECE`, `UNICODEBUILDER`,
//! `BaseStringBuilderRepr`, `StringBuilderRepr`, and
//! `UnicodeBuilderRepr`.

#![allow(non_snake_case, non_upper_case_globals)]

use std::sync::{Arc, LazyLock, OnceLock};

use crate::flowspace::model::{
    Block, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation, Variable, c_last_exception,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::backendopt::constfold::WE_ARE_JITTED_TAG_ID;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    Array, ForwardReference, LowLevelType, Ptr, PtrTarget, Struct,
};
use crate::translator::rtyper::lltypesystem::rstr::{STRPTR, UNICODEPTR};
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{
    ConvertedTo, HighLevelOp, constant_with_lltype, exception_args, functionptr_const,
    helper_pygraph_from_graph, variable_with_lltype, void_field_const,
};

fn ptr_to_lowlevel(target: LowLevelType) -> LowLevelType {
    match target {
        LowLevelType::ForwardReference(fwd) => LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::ForwardReference(*fwd),
        })),
        LowLevelType::Struct(t) => LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(*t),
        })),
        other => panic!("expected container lowleveltype for Ptr(...), got {other:?}"),
    }
}

fn ptr_to_forward_reference(target: &LowLevelType) -> LowLevelType {
    let LowLevelType::ForwardReference(fwd) = target.clone() else {
        panic!("builder piece type must be a ForwardReference");
    };
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::ForwardReference(*fwd),
    }))
}

/// RPython `STRINGPIECE = lltype.GcStruct('stringpiece', ...)`.
pub static STRINGPIECE: LazyLock<LowLevelType> = LazyLock::new(|| {
    let fwd = ForwardReference::gc();
    let body = Struct::gc(
        "stringpiece",
        vec![
            ("buf".into(), STRPTR.clone()),
            (
                "prev_piece".into(),
                LowLevelType::Ptr(Box::new(Ptr {
                    TO: PtrTarget::ForwardReference(fwd.clone()),
                })),
            ),
        ],
    );
    fwd.r#become(LowLevelType::Struct(Box::new(body)))
        .expect("STRINGPIECE.prev_piece.TO.become(STRINGPIECE)");
    LowLevelType::ForwardReference(Box::new(fwd))
});

/// RPython `Ptr(STRINGPIECE)`.
pub static STRINGPIECEPTR: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_forward_reference(&STRINGPIECE));

/// RPython `STRINGBUILDER = lltype.GcStruct('stringbuilder', ...)`.
pub static STRINGBUILDER: LazyLock<LowLevelType> = LazyLock::new(|| {
    LowLevelType::Struct(Box::new(Struct::gc(
        "stringbuilder",
        vec![
            ("current_buf".into(), STRPTR.clone()),
            ("current_pos".into(), LowLevelType::Signed),
            ("current_end".into(), LowLevelType::Signed),
            ("total_size".into(), LowLevelType::Signed),
            ("extra_pieces".into(), STRINGPIECEPTR.clone()),
        ],
    )))
});

/// RPython `Ptr(STRINGBUILDER)`.
pub static STRINGBUILDERPTR: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_lowlevel(STRINGBUILDER.clone()));

/// RPython `UNICODEPIECE = lltype.GcStruct('unicodepiece', ...)`.
pub static UNICODEPIECE: LazyLock<LowLevelType> = LazyLock::new(|| {
    let fwd = ForwardReference::gc();
    let body = Struct::gc(
        "unicodepiece",
        vec![
            ("buf".into(), UNICODEPTR.clone()),
            (
                "prev_piece".into(),
                LowLevelType::Ptr(Box::new(Ptr {
                    TO: PtrTarget::ForwardReference(fwd.clone()),
                })),
            ),
        ],
    );
    fwd.r#become(LowLevelType::Struct(Box::new(body)))
        .expect("UNICODEPIECE.prev_piece.TO.become(UNICODEPIECE)");
    LowLevelType::ForwardReference(Box::new(fwd))
});

/// RPython `Ptr(UNICODEPIECE)`.
pub static UNICODEPIECEPTR: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_forward_reference(&UNICODEPIECE));

/// RPython `UNICODEBUILDER = lltype.GcStruct('unicodebuilder', ...)`.
pub static UNICODEBUILDER: LazyLock<LowLevelType> = LazyLock::new(|| {
    LowLevelType::Struct(Box::new(Struct::gc(
        "unicodebuilder",
        vec![
            ("current_buf".into(), UNICODEPTR.clone()),
            ("current_pos".into(), LowLevelType::Signed),
            ("current_end".into(), LowLevelType::Signed),
            ("total_size".into(), LowLevelType::Signed),
            ("extra_pieces".into(), UNICODEPIECEPTR.clone()),
        ],
    )))
});

/// RPython `Ptr(UNICODEBUILDER)`.
pub static UNICODEBUILDERPTR: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_lowlevel(UNICODEBUILDER.clone()));

fn builder_runtime_deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!(
        "lltypesystem.rbuilder.{name} - low-level StringBuilder runtime helper deferred"
    ))
}

pub fn _ll_append() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("_ll_append"))
}

pub fn ll_grow_by() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_grow_by"))
}

pub fn ll_grow_and_append() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_grow_and_append"))
}

pub fn ll_append() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_append"))
}

pub fn ll_jit_append() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_jit_append"))
}

pub fn ll_append_res0() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_append_res0"))
}

pub fn ll_append_char() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_append_char"))
}

pub fn ll_append_slice() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_append_slice"))
}

pub fn ll_jit_append_slice() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_jit_append_slice"))
}

pub fn ll_append_res_slice() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_append_res_slice"))
}

pub const MAX_N: usize = 10;

pub fn make_func_for_size(N: usize) -> (String, String, usize) {
    (
        format!("ll_append_0_{N}"),
        format!("ll_append_start_{N}"),
        N,
    )
}

pub static unroll_func_for_size: LazyLock<Vec<(String, String, usize)>> =
    LazyLock::new(|| (2..=MAX_N).map(make_func_for_size).collect());

pub fn ll_jit_try_append_slice() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_jit_try_append_slice"))
}

pub fn ll_append_multiple_char() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_append_multiple_char"))
}

pub fn _ll_append_multiple_char() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("_ll_append_multiple_char"))
}

pub fn ll_jit_try_append_multiple_char() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_jit_try_append_multiple_char"))
}

pub fn ll_append_charpsize() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_append_charpsize"))
}

pub fn ll_getlength() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_getlength"))
}

/// Synthesise `ll_getlength(ll_builder)` (`rbuilder.py`):
/// `ll_builder.total_size - (ll_builder.current_end - ll_builder.current_pos)`.
pub fn build_ll_getlength_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(ll_builder.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let current_end = variable_with_lltype("current_end", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_end"),
        ],
        Hlvalue::Variable(current_end.clone()),
    ));
    let current_pos = variable_with_lltype("current_pos", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(current_pos.clone()),
    ));
    let num_chars_missing_from_last_piece =
        variable_with_lltype("num_chars_missing_from_last_piece", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![
            Hlvalue::Variable(current_end),
            Hlvalue::Variable(current_pos),
        ],
        Hlvalue::Variable(num_chars_missing_from_last_piece.clone()),
    ));
    let total_size = variable_with_lltype("total_size", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder),
            void_field_const("total_size"),
        ],
        Hlvalue::Variable(total_size.clone()),
    ));
    let result = variable_with_lltype("result", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![
            Hlvalue::Variable(total_size),
            Hlvalue::Variable(num_chars_missing_from_last_piece),
        ],
        Hlvalue::Variable(result.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ll_builder".to_string()],
        func,
    ))
}

/// SPIKE (L4-first, task #49): a minimal `ll_build(ll_builder)` helper
/// that returns `ll_builder.current_buf` unchanged. Used only to prove
/// the rtyper method-dispatch pipeline; **not runtime-correct** (it does
/// not fold `extra_pieces` or shrink to `current_pos`). The full
/// `build_ll_build_helper_graph` (fold_pieces + shrink_final closure)
/// replaces it in L2-full.
pub fn build_ll_build_spike_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(ll_builder.clone())]);
    let return_var = variable_with_lltype("result", buf_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let current_buf = variable_with_lltype("current_buf", buf_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder),
            void_field_const("current_buf"),
        ],
        Hlvalue::Variable(current_buf.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(current_buf)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ll_builder".to_string()],
        func,
    ))
}

pub fn ll_build() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_build"))
}

pub fn ll_shrink_final() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_shrink_final"))
}

pub fn ll_fold_pieces() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_fold_pieces"))
}

pub fn ll_bool() -> Result<(), TyperError> {
    Err(builder_runtime_deferred("ll_bool"))
}

/// Synthesise `ll_bool(ll_builder)` (`rbuilder.py`):
/// `ll_builder != nullptr(lltype.typeOf(ll_builder).TO)`.
pub fn build_ll_bool_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(ll_builder.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // result = ptr_ne(ll_builder, nullptr(TO))
    let null_builder = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        builder_ptr_lltype,
    ));
    let result = variable_with_lltype("result", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_ne",
        vec![Hlvalue::Variable(ll_builder), null_builder],
        Hlvalue::Variable(result.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ll_builder".to_string()],
        func,
    ))
}

/// Synthesise `ll_new(init_size)` (`rbuilder.py` / `469-478`):
///
/// ```python
/// init_size = intmask(min(r_uint(init_size), r_uint(1280)))
/// ll_builder = lltype.malloc(STRINGBUILDER)
/// ll_builder.current_buf = ll_builder.mallocfn(init_size)
/// ll_builder.current_pos = 0
/// ll_builder.current_end = init_size
/// ll_builder.total_size = init_size
/// return ll_builder
/// ```
///
/// `min` is `rbuiltin.ll_min` (`rbuiltin.py`) and `mallocfn` is the
/// specialization's `staticAdtMethod(rstr.mallocstr / mallocunicode)`
/// (`rbuilder.py:54`/`72`) — both baked in as `direct_call` callee consts,
/// mirroring [`build_ll_call_lookup_function_helper_graph`]. `buf_lltype`
/// is `STRPTR`/`UNICODEPTR` (the `current_buf` field and `mallocfn` result).
/// Synthesise `copy_string_contents(src, dst, srcstart, dststart, length)`
/// (`rstr.py` `copy_string_contents` / `copy_unicode_contents`): a single
/// `copystrcontent` / `copyunicodecontent` op over whole string objects.
/// The builder append/build trees thread this as a `direct_call` callee
/// const (`copy_fn`), so the shared copy primitive is materialised once and
/// baked in via [`functionptr_const`], mirroring the `ll_min` / `mallocstr`
/// consts of [`build_ll_new_helper_graph`]. `buf_lltype` is `STRPTR` /
/// `UNICODEPTR`; `copy_opname` selects `copystrcontent` (STR) vs
/// `copyunicodecontent` (UNICODE). Returns `Void`.
pub fn build_ll_copy_string_contents_helper_graph(
    name: &str,
    buf_lltype: LowLevelType,
    copy_opname: &'static str,
) -> Result<PyGraph, TyperError> {
    let src = variable_with_lltype("src", buf_lltype.clone());
    let dst = variable_with_lltype("dst", buf_lltype);
    let srcstart = variable_with_lltype("srcstart", LowLevelType::Signed);
    let dststart = variable_with_lltype("dststart", LowLevelType::Signed);
    let length = variable_with_lltype("length", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(src.clone()),
        Hlvalue::Variable(dst.clone()),
        Hlvalue::Variable(srcstart.clone()),
        Hlvalue::Variable(dststart.clone()),
        Hlvalue::Variable(length.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let void_result = variable_with_lltype("v", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        copy_opname,
        vec![
            Hlvalue::Variable(src),
            Hlvalue::Variable(dst),
            Hlvalue::Variable(srcstart),
            Hlvalue::Variable(dststart),
            Hlvalue::Variable(length),
        ],
        Hlvalue::Variable(void_result),
    ));
    let none_const = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ));
    startblock.closeblock(vec![
        Link::new(vec![none_const], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "src".to_string(),
            "dst".to_string(),
            "srcstart".to_string(),
            "dststart".to_string(),
            "length".to_string(),
        ],
        func,
    ))
}

pub fn build_ll_new_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    builder_struct: LowLevelType,
    buf_lltype: LowLevelType,
    min_fn: Constant,
    mallocfn: Constant,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let init_size = variable_with_lltype("init_size", LowLevelType::Signed);
    let startblock = Block::shared(vec![Hlvalue::Variable(init_size.clone())]);
    let return_var = variable_with_lltype("result", builder_ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let void_result = || variable_with_lltype("v", LowLevelType::Void);

    // init_size = intmask(min(r_uint(init_size), r_uint(1280)))
    let uint_size = variable_with_lltype("uint_size", LowLevelType::Unsigned);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_int_to_uint",
        vec![Hlvalue::Variable(init_size)],
        Hlvalue::Variable(uint_size.clone()),
    ));
    let uint_min = variable_with_lltype("uint_min", LowLevelType::Unsigned);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(min_fn),
            Hlvalue::Variable(uint_size),
            constant_with_lltype(ConstValue::UInt(1280), LowLevelType::Unsigned),
        ],
        Hlvalue::Variable(uint_min.clone()),
    ));
    let size = variable_with_lltype("size", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_uint_to_int",
        vec![Hlvalue::Variable(uint_min)],
        Hlvalue::Variable(size.clone()),
    ));

    // ll_builder = lltype.malloc(STRINGBUILDER)
    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc",
        vec![lowlevel_type_const(builder_struct), gc_flavor_const()?],
        Hlvalue::Variable(ll_builder.clone()),
    ));

    // ll_builder.current_buf = ll_builder.mallocfn(init_size)
    let current_buf = variable_with_lltype("current_buf", buf_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![Hlvalue::Constant(mallocfn), Hlvalue::Variable(size.clone())],
        Hlvalue::Variable(current_buf.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_buf"),
            Hlvalue::Variable(current_buf),
        ],
        Hlvalue::Variable(void_result()),
    ));
    // ll_builder.current_pos = 0
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_pos"),
            constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
        ],
        Hlvalue::Variable(void_result()),
    ));
    // ll_builder.current_end = init_size
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_end"),
            Hlvalue::Variable(size.clone()),
        ],
        Hlvalue::Variable(void_result()),
    ));
    // ll_builder.total_size = init_size
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("total_size"),
            Hlvalue::Variable(size),
        ],
        Hlvalue::Variable(void_result()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(ll_builder)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["init_size".to_string()],
        func,
    ))
}

/// Stub `ll_shrink_array(buf, smallerlength) -> buf` — an identity the
/// codewriter never executes.  `rgc.ll_shrink_array` is an `@jit.oopspec`
/// (`rgc.py`), so the `direct_call` naming this helper lowers to a
/// `ShrinkArray` residual (`jtransform._handle_rgc_call`), which does the real
/// shrink at runtime (`eval.rs ll_shrink_array` / `opt_call_shrink_array`).
/// The body exists only so [`functionptr_const`] has a graph to name; it
/// carries NO `shrink_array` op (which has no rtyper lowering arm), so it
/// stays drain-safe even if the codewriter enumerates it as a helper.  The
/// binding is `mark_oopspec(["ll_shrink_array"], "rgc.ll_shrink_array")` in
/// `lib.rs`.
pub fn build_ll_shrink_array_stub_helper_graph(
    name: &str,
    buf_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let buf = variable_with_lltype("buf", buf_lltype.clone());
    let smallerlength = variable_with_lltype("smallerlength", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(buf.clone()),
        Hlvalue::Variable(smallerlength),
    ]);
    let return_var = variable_with_lltype("result", buf_lltype);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(buf)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["buf".to_string(), "smallerlength".to_string()],
        func,
    ))
}

/// Synthesise `ll_shrink_final(ll_builder)` (`rbuilder.py`):
///
/// ```python
/// final_size = ll_builder.current_pos
/// ll_assert(final_size <= ll_builder.total_size, "...")   # debug-only, omitted
/// buf = rgc.ll_shrink_array(ll_builder.current_buf, final_size)
/// ll_builder.current_buf = buf
/// ll_builder.current_end = final_size
/// ll_builder.total_size = final_size
/// ```
///
/// `rgc.ll_shrink_array` (`rgc.py`) is baked in as a `direct_call`
/// callee const (`shrink_array_fn`); `buf_lltype` is `STRPTR`/`UNICODEPTR`.
/// Returns `Void`.
pub fn build_ll_shrink_final_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    shrink_array_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(ll_builder.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };

    // final_size = ll_builder.current_pos
    let final_size = variable_with_lltype("final_size", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(final_size.clone()),
    ));
    // buf = rgc.ll_shrink_array(ll_builder.current_buf, final_size)
    let old_buf = variable_with_lltype("old_buf", buf_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_buf"),
        ],
        Hlvalue::Variable(old_buf.clone()),
    ));
    let buf = variable_with_lltype("buf", buf_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(shrink_array_fn),
            Hlvalue::Variable(old_buf),
            Hlvalue::Variable(final_size.clone()),
        ],
        Hlvalue::Variable(buf.clone()),
    ));
    // ll_builder.current_buf = buf
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_buf"),
            Hlvalue::Variable(buf),
        ],
        Hlvalue::Variable(void_result()),
    ));
    // ll_builder.current_end = final_size
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_end"),
            Hlvalue::Variable(final_size.clone()),
        ],
        Hlvalue::Variable(void_result()),
    ));
    // ll_builder.total_size = final_size
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(ll_builder),
            void_field_const("total_size"),
            Hlvalue::Variable(final_size),
        ],
        Hlvalue::Variable(void_result()),
    ));
    startblock.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ll_builder".to_string()],
        func,
    ))
}

/// Synthesise `ll_grow_and_append(ll_builder, ll_str, start, size)`
/// (`rbuilder.py`):
///
/// Fast path when `size > 1280 and current_pos == 0 and start == 0 and
/// size == len(ll_str.chars)`: append `ll_str` directly as a new piece
/// (`total_size = ovfcheck(total_size + size)`, malloc PIECE, link it).
/// Else the slow path copies the head into the current buffer, `ll_grow_by`s
/// a fresh buffer, and copies the tail. The short-circuit `and` becomes four
/// chained tests each falling to the slow path. The fast body's
/// `total_size = ovfcheck(total_size + size)` is split into `block_fast_check`
/// (the trailing `int_add_ovf`) and `block_fast_body`; the `except
/// OverflowError: pass` edge routes the overflow to the slow path.
/// `mallocfn` / `copy_string_contents` / `ll_grow_by` are `direct_call` callee
/// consts. Debug-only `ll_assert`s omitted. Returns `Void`.
#[allow(clippy::too_many_arguments)]
pub fn build_ll_grow_and_append_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    piece_ptr_lltype: LowLevelType,
    piece_struct: LowLevelType,
    buf_lltype: LowLevelType,
    chars_array_ptr_lltype: LowLevelType,
    copy_fn: Constant,
    grow_by_fn: Constant,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let bool_case = |b: bool| {
        Some(constant_with_lltype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let push = |block: &crate::flowspace::model::BlockRef,
                opname: &str,
                args: Vec<Hlvalue>,
                out: Hlvalue| {
        block
            .borrow_mut()
            .operations
            .push(SpaceOperation::new(opname, args, out));
    };
    // Fresh (ll_builder, ll_str, start, size) input tuple for a block.
    let arg_tuple = |builder: &LowLevelType, buf: &LowLevelType| {
        (
            variable_with_lltype("ll_builder", builder.clone()),
            variable_with_lltype("ll_str", buf.clone()),
            variable_with_lltype("start", LowLevelType::Signed),
            variable_with_lltype("size", LowLevelType::Signed),
        )
    };
    let tuple_vals = |t: &(Variable, Variable, Variable, Variable)| {
        vec![
            Hlvalue::Variable(t.0.clone()),
            Hlvalue::Variable(t.1.clone()),
            Hlvalue::Variable(t.2.clone()),
            Hlvalue::Variable(t.3.clone()),
        ]
    };

    let (llb, ll_str, start, size) = arg_tuple(&builder_ptr_lltype, &buf_lltype);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(llb.clone()),
        Hlvalue::Variable(ll_str.clone()),
        Hlvalue::Variable(start.clone()),
        Hlvalue::Variable(size.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let overflow_error = exception_args("OverflowError")?;

    // block_fast_body: append ll_str as a new big piece (no-overflow)
    // Receives (ll_builder, ll_str, total_size) where total_size is the
    // ovfcheck result computed in block_fast_check.
    let fb_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let fb_str = variable_with_lltype("ll_str", buf_lltype.clone());
    let fb_total = variable_with_lltype("total_size", LowLevelType::Signed);
    let block_fast_body = Block::shared(vec![
        Hlvalue::Variable(fb_llb.clone()),
        Hlvalue::Variable(fb_str.clone()),
        Hlvalue::Variable(fb_total.clone()),
    ]);
    let fb_piece = variable_with_lltype("old_piece", piece_ptr_lltype.clone());
    push(
        &block_fast_body,
        "malloc",
        vec![
            lowlevel_type_const(piece_struct.clone()),
            gc_flavor_const()?,
        ],
        Hlvalue::Variable(fb_piece.clone()),
    );
    push(
        &block_fast_body,
        "setfield",
        vec![
            Hlvalue::Variable(fb_piece.clone()),
            void_field_const("buf"),
            Hlvalue::Variable(fb_str),
        ],
        Hlvalue::Variable(void_result()),
    );
    let fb_extra = variable_with_lltype("extra_pieces", piece_ptr_lltype.clone());
    push(
        &block_fast_body,
        "getfield",
        vec![
            Hlvalue::Variable(fb_llb.clone()),
            void_field_const("extra_pieces"),
        ],
        Hlvalue::Variable(fb_extra.clone()),
    );
    push(
        &block_fast_body,
        "setfield",
        vec![
            Hlvalue::Variable(fb_piece.clone()),
            void_field_const("prev_piece"),
            Hlvalue::Variable(fb_extra),
        ],
        Hlvalue::Variable(void_result()),
    );
    push(
        &block_fast_body,
        "setfield",
        vec![
            Hlvalue::Variable(fb_llb.clone()),
            void_field_const("total_size"),
            Hlvalue::Variable(fb_total),
        ],
        Hlvalue::Variable(void_result()),
    );
    push(
        &block_fast_body,
        "setfield",
        vec![
            Hlvalue::Variable(fb_llb),
            void_field_const("extra_pieces"),
            Hlvalue::Variable(fb_piece),
        ],
        Hlvalue::Variable(void_result()),
    );
    block_fast_body.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // block_fast_check: total_size = ovfcheck(total_size + size)
    // `except OverflowError: pass` -> the overflow edge falls to the slow path
    // (block_slow), so the trailing int_add_ovf is the block's last op and its
    // exception edge carries the four slow-path inputs, not [class, instance].
    let fc = arg_tuple(&builder_ptr_lltype, &buf_lltype);
    let block_fast_check = Block::shared(tuple_vals(&fc));
    let fc_orig_total = variable_with_lltype("total_size", LowLevelType::Signed);
    push(
        &block_fast_check,
        "getfield",
        vec![
            Hlvalue::Variable(fc.0.clone()),
            void_field_const("total_size"),
        ],
        Hlvalue::Variable(fc_orig_total.clone()),
    );
    let fc_total_new = variable_with_lltype("total_size", LowLevelType::Signed);
    push(
        &block_fast_check,
        "int_add_ovf",
        vec![
            Hlvalue::Variable(fc_orig_total),
            Hlvalue::Variable(fc.3.clone()),
        ],
        Hlvalue::Variable(fc_total_new.clone()),
    );
    // Closed after block_slow is built (see close_ovfcheck_block call below).

    // block_slow: copy head, grow, copy tail
    let s = arg_tuple(&builder_ptr_lltype, &buf_lltype);
    let block_slow = Block::shared(tuple_vals(&s));
    let s_pos = variable_with_lltype("current_pos", LowLevelType::Signed);
    push(
        &block_slow,
        "getfield",
        vec![
            Hlvalue::Variable(s.0.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(s_pos.clone()),
    );
    let s_end = variable_with_lltype("current_end", LowLevelType::Signed);
    push(
        &block_slow,
        "getfield",
        vec![
            Hlvalue::Variable(s.0.clone()),
            void_field_const("current_end"),
        ],
        Hlvalue::Variable(s_end.clone()),
    );
    let s_part1 = variable_with_lltype("part1", LowLevelType::Signed);
    push(
        &block_slow,
        "int_sub",
        vec![Hlvalue::Variable(s_end), Hlvalue::Variable(s_pos.clone())],
        Hlvalue::Variable(s_part1.clone()),
    );
    let s_buf = variable_with_lltype("current_buf", buf_lltype.clone());
    push(
        &block_slow,
        "getfield",
        vec![
            Hlvalue::Variable(s.0.clone()),
            void_field_const("current_buf"),
        ],
        Hlvalue::Variable(s_buf.clone()),
    );
    // copy_string_contents(ll_str, current_buf, start, current_pos, part1)
    push(
        &block_slow,
        "direct_call",
        vec![
            Hlvalue::Constant(copy_fn.clone()),
            Hlvalue::Variable(s.1.clone()),
            Hlvalue::Variable(s_buf),
            Hlvalue::Variable(s.2.clone()),
            Hlvalue::Variable(s_pos),
            Hlvalue::Variable(s_part1.clone()),
        ],
        Hlvalue::Variable(void_result()),
    );
    let s_start2 = variable_with_lltype("start", LowLevelType::Signed);
    push(
        &block_slow,
        "int_add",
        vec![
            Hlvalue::Variable(s.2.clone()),
            Hlvalue::Variable(s_part1.clone()),
        ],
        Hlvalue::Variable(s_start2.clone()),
    );
    let s_size2 = variable_with_lltype("size", LowLevelType::Signed);
    push(
        &block_slow,
        "int_sub",
        vec![Hlvalue::Variable(s.3.clone()), Hlvalue::Variable(s_part1)],
        Hlvalue::Variable(s_size2.clone()),
    );
    // ll_grow_by(ll_builder, size)
    push(
        &block_slow,
        "direct_call",
        vec![
            Hlvalue::Constant(grow_by_fn),
            Hlvalue::Variable(s.0.clone()),
            Hlvalue::Variable(s_size2.clone()),
        ],
        Hlvalue::Variable(void_result()),
    );
    // ll_builder.current_pos = size
    push(
        &block_slow,
        "setfield",
        vec![
            Hlvalue::Variable(s.0.clone()),
            void_field_const("current_pos"),
            Hlvalue::Variable(s_size2.clone()),
        ],
        Hlvalue::Variable(void_result()),
    );
    let s_buf2 = variable_with_lltype("current_buf", buf_lltype.clone());
    push(
        &block_slow,
        "getfield",
        vec![
            Hlvalue::Variable(s.0.clone()),
            void_field_const("current_buf"),
        ],
        Hlvalue::Variable(s_buf2.clone()),
    );
    // copy_string_contents(ll_str, current_buf, start, 0, size)
    push(
        &block_slow,
        "direct_call",
        vec![
            Hlvalue::Constant(copy_fn),
            Hlvalue::Variable(s.1.clone()),
            Hlvalue::Variable(s_buf2),
            Hlvalue::Variable(s_start2),
            signed(0),
            Hlvalue::Variable(s_size2),
        ],
        Hlvalue::Variable(void_result()),
    );
    block_slow.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // Wire the fast-path ovfcheck now that block_slow exists: no-overflow
    // threads [ll_builder, ll_str, total_new] to block_fast_body; the
    // OverflowError edge carries the four slow-path inputs to block_slow
    // (`except OverflowError: pass`). exc_args (four data vars) differ from the
    // [class, instance] extravars, so `make_exception_link` routes to block_slow
    // instead of re-raising.
    close_ovfcheck_block(
        &block_fast_check,
        &overflow_error,
        &block_fast_body,
        vec![
            Hlvalue::Variable(fc.0.clone()),
            Hlvalue::Variable(fc.1.clone()),
            Hlvalue::Variable(fc_total_new),
        ],
        &block_slow,
        tuple_vals(&fc),
    );

    // Condition chain: size > 1280 -> pos == 0 -> start == 0 ->
    // size == len(ll_str.chars); any failure jumps to block_slow.
    // block_b3: size == len(ll_str.chars)
    let b3 = arg_tuple(&builder_ptr_lltype, &buf_lltype);
    let block_b3 = Block::shared(tuple_vals(&b3));
    let b3_chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    push(
        &block_b3,
        "getsubstruct",
        vec![
            Hlvalue::Variable(b3.1.clone()),
            constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void),
        ],
        Hlvalue::Variable(b3_chars.clone()),
    );
    let b3_len = variable_with_lltype("length", LowLevelType::Signed);
    push(
        &block_b3,
        "getarraysize",
        vec![Hlvalue::Variable(b3_chars)],
        Hlvalue::Variable(b3_len.clone()),
    );
    let b3_eq = variable_with_lltype("size_eq_len", LowLevelType::Bool);
    push(
        &block_b3,
        "int_eq",
        vec![Hlvalue::Variable(b3.3.clone()), Hlvalue::Variable(b3_len)],
        Hlvalue::Variable(b3_eq.clone()),
    );
    block_b3.borrow_mut().exitswitch = Some(Hlvalue::Variable(b3_eq));
    block_b3.closeblock(vec![
        Link::new(tuple_vals(&b3), Some(block_fast_check), bool_case(true)).into_ref(),
        Link::new(tuple_vals(&b3), Some(block_slow.clone()), bool_case(false)).into_ref(),
    ]);

    // block_b2: start == 0
    let b2 = arg_tuple(&builder_ptr_lltype, &buf_lltype);
    let block_b2 = Block::shared(tuple_vals(&b2));
    let b2_eq = variable_with_lltype("start_eq0", LowLevelType::Bool);
    push(
        &block_b2,
        "int_eq",
        vec![Hlvalue::Variable(b2.2.clone()), signed(0)],
        Hlvalue::Variable(b2_eq.clone()),
    );
    block_b2.borrow_mut().exitswitch = Some(Hlvalue::Variable(b2_eq));
    block_b2.closeblock(vec![
        Link::new(tuple_vals(&b2), Some(block_b3), bool_case(true)).into_ref(),
        Link::new(tuple_vals(&b2), Some(block_slow.clone()), bool_case(false)).into_ref(),
    ]);

    // block_b1: current_pos == 0
    let b1 = arg_tuple(&builder_ptr_lltype, &buf_lltype);
    let block_b1 = Block::shared(tuple_vals(&b1));
    let b1_pos = variable_with_lltype("current_pos", LowLevelType::Signed);
    push(
        &block_b1,
        "getfield",
        vec![
            Hlvalue::Variable(b1.0.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(b1_pos.clone()),
    );
    let b1_eq = variable_with_lltype("pos_eq0", LowLevelType::Bool);
    push(
        &block_b1,
        "int_eq",
        vec![Hlvalue::Variable(b1_pos), signed(0)],
        Hlvalue::Variable(b1_eq.clone()),
    );
    block_b1.borrow_mut().exitswitch = Some(Hlvalue::Variable(b1_eq));
    block_b1.closeblock(vec![
        Link::new(tuple_vals(&b1), Some(block_b2), bool_case(true)).into_ref(),
        Link::new(tuple_vals(&b1), Some(block_slow.clone()), bool_case(false)).into_ref(),
    ]);

    // startblock (block_b0): size > 1280
    let b0_gt = variable_with_lltype("big", LowLevelType::Bool);
    push(
        &startblock,
        "int_gt",
        vec![Hlvalue::Variable(size.clone()), signed(1280)],
        Hlvalue::Variable(b0_gt.clone()),
    );
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(b0_gt));
    let b0 = (llb, ll_str, start, size);
    startblock.closeblock(vec![
        Link::new(tuple_vals(&b0), Some(block_b1), bool_case(true)).into_ref(),
        Link::new(tuple_vals(&b0), Some(block_slow), bool_case(false)).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "ll_builder".to_string(),
            "ll_str".to_string(),
            "start".to_string(),
            "size".to_string(),
        ],
        func,
    ))
}

/// Synthesise `ll_append_multiple_char(ll_builder, char, times)`
/// (`rbuilder.py:275-280`):
///
/// ```python
/// if jit.we_are_jitted():
///     if ll_jit_try_append_multiple_char(ll_builder, char, times):
///         return
/// _ll_append_multiple_char(ll_builder, char, times)
/// ```
///
/// Outer branch is the `we_are_jitted()` symbolic exitswitch (see
/// [`build_ll_append_helper_graph`]); its true arm runs the inner
/// `ll_jit_try_append_multiple_char` guard. The `we_are_jitted()`-false
/// and guard-failed edges both fall through to a shared block that
/// `direct_call`s `_ll_append_multiple_char`. All callees are baked in
/// as `direct_call` consts. Returns `Void`.
pub fn build_ll_append_multiple_char_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    char_lltype: LowLevelType,
    jit_try_fn: Constant,
    append_multiple_char_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let bool_case = |b: bool| {
        Some(constant_with_lltype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };

    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let char = variable_with_lltype("char", char_lltype.clone());
    let times = variable_with_lltype("times", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(ll_builder.clone()),
        Hlvalue::Variable(char.clone()),
        Hlvalue::Variable(times.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Shared fallback: _ll_append_multiple_char(ll_builder, char, times).
    let fb_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let fb_char = variable_with_lltype("char", char_lltype.clone());
    let fb_times = variable_with_lltype("times", LowLevelType::Signed);
    let block_fb = Block::shared(vec![
        Hlvalue::Variable(fb_llb.clone()),
        Hlvalue::Variable(fb_char.clone()),
        Hlvalue::Variable(fb_times.clone()),
    ]);
    block_fb.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(append_multiple_char_fn),
            Hlvalue::Variable(fb_llb),
            Hlvalue::Variable(fb_char),
            Hlvalue::Variable(fb_times),
        ],
        Hlvalue::Variable(void_result()),
    ));
    block_fb.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // jit arm: handled = ll_jit_try_append_multiple_char(...); branch.
    let jit_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let jit_char = variable_with_lltype("char", char_lltype);
    let jit_times = variable_with_lltype("times", LowLevelType::Signed);
    let block_jit = Block::shared(vec![
        Hlvalue::Variable(jit_llb.clone()),
        Hlvalue::Variable(jit_char.clone()),
        Hlvalue::Variable(jit_times.clone()),
    ]);
    let handled = variable_with_lltype("handled", LowLevelType::Bool);
    block_jit.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(jit_try_fn),
            Hlvalue::Variable(jit_llb.clone()),
            Hlvalue::Variable(jit_char.clone()),
            Hlvalue::Variable(jit_times.clone()),
        ],
        Hlvalue::Variable(handled.clone()),
    ));
    block_jit.borrow_mut().exitswitch = Some(Hlvalue::Variable(handled));
    block_jit.closeblock(vec![
        // handled -> return.
        Link::new(
            vec![none_const()],
            Some(graph.returnblock.clone()),
            bool_case(true),
        )
        .into_ref(),
        // not handled -> shared fallback.
        Link::new(
            vec![
                Hlvalue::Variable(jit_llb),
                Hlvalue::Variable(jit_char),
                Hlvalue::Variable(jit_times),
            ],
            Some(block_fb.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    // startblock: branch on `we_are_jitted()` symbolic exitswitch.
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::SpecTag(WE_ARE_JITTED_TAG_ID),
        LowLevelType::Bool,
    )));
    let arm_args = |llb: &Variable, c: &Variable, t: &Variable| {
        vec![
            Hlvalue::Variable(llb.clone()),
            Hlvalue::Variable(c.clone()),
            Hlvalue::Variable(t.clone()),
        ]
    };
    startblock.closeblock(vec![
        Link::new(
            arm_args(&ll_builder, &char, &times),
            Some(block_jit),
            bool_case(true),
        )
        .into_ref(),
        // we_are_jitted() false -> shared fallback.
        Link::new(
            arm_args(&ll_builder, &char, &times),
            Some(block_fb),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "ll_builder".to_string(),
            "char".to_string(),
            "times".to_string(),
        ],
        func,
    ))
}

/// Synthesise `_ll_append_multiple_char(ll_builder, char, times)`
/// (`rbuilder.py:283-297`):
///
/// ```python
/// part1 = ll_builder.current_end - ll_builder.current_pos
/// if times > part1:
///     times -= part1
///     buf = ll_builder.current_buf
///     for i in xrange(ll_builder.current_pos, ll_builder.current_end):
///         buf.chars[i] = char
///     ll_grow_by(ll_builder, times)
/// buf = ll_builder.current_buf
/// pos = ll_builder.current_pos
/// end = pos + times
/// ll_builder.current_pos = end
/// for i in xrange(pos, end):
///     buf.chars[i] = char
/// ```
///
/// The two `xrange` loops become `int_lt` header / `setarrayitem`+`int_add`
/// body pairs (each `buf.chars[i]` is `getsubstruct('chars')` +
/// `setarrayitem`). `ll_grow_by` is a `direct_call` callee const. Returns
/// `Void`.
pub fn build_ll__ll_append_multiple_char_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    char_lltype: LowLevelType,
    chars_array_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    grow_by_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let bool_case = |b: bool| {
        Some(constant_with_lltype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let push = |block: &crate::flowspace::model::BlockRef,
                opname: &str,
                args: Vec<Hlvalue>,
                out: Hlvalue| {
        block
            .borrow_mut()
            .operations
            .push(SpaceOperation::new(opname, args, out));
    };
    // Emit `chars = getsubstruct(buf, 'chars'); setarrayitem(chars, i, char)`.
    let write_char =
        |block: &crate::flowspace::model::BlockRef, buf: &Variable, i: &Variable, ch: &Variable| {
            let chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
            push(
                block,
                "getsubstruct",
                vec![
                    Hlvalue::Variable(buf.clone()),
                    constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void),
                ],
                Hlvalue::Variable(chars.clone()),
            );
            push(
                block,
                "setarrayitem",
                vec![
                    Hlvalue::Variable(chars),
                    Hlvalue::Variable(i.clone()),
                    Hlvalue::Variable(ch.clone()),
                ],
                Hlvalue::Variable(void_result()),
            );
        };

    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let char_v = variable_with_lltype("char", char_lltype.clone());
    let times = variable_with_lltype("times", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(ll_builder.clone()),
        Hlvalue::Variable(char_v.clone()),
        Hlvalue::Variable(times.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Second (unconditional) loop, built first as it is the tail.
    // block_after(ll_builder, char, times)
    let a_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let a_char = variable_with_lltype("char", char_lltype.clone());
    let a_times = variable_with_lltype("times", LowLevelType::Signed);
    let block_after = Block::shared(vec![
        Hlvalue::Variable(a_llb.clone()),
        Hlvalue::Variable(a_char.clone()),
        Hlvalue::Variable(a_times.clone()),
    ]);
    // block_l2_header(char, buf, i, end) / block_l2_body(char, buf, i, end)
    let h2_char = variable_with_lltype("char", char_lltype.clone());
    let h2_buf = variable_with_lltype("buf", buf_lltype.clone());
    let h2_i = variable_with_lltype("i", LowLevelType::Signed);
    let h2_end = variable_with_lltype("end", LowLevelType::Signed);
    let block_l2_header = Block::shared(vec![
        Hlvalue::Variable(h2_char.clone()),
        Hlvalue::Variable(h2_buf.clone()),
        Hlvalue::Variable(h2_i.clone()),
        Hlvalue::Variable(h2_end.clone()),
    ]);
    let b2_char = variable_with_lltype("char", char_lltype.clone());
    let b2_buf = variable_with_lltype("buf", buf_lltype.clone());
    let b2_i = variable_with_lltype("i", LowLevelType::Signed);
    let b2_end = variable_with_lltype("end", LowLevelType::Signed);
    let block_l2_body = Block::shared(vec![
        Hlvalue::Variable(b2_char.clone()),
        Hlvalue::Variable(b2_buf.clone()),
        Hlvalue::Variable(b2_i.clone()),
        Hlvalue::Variable(b2_end.clone()),
    ]);
    // body: buf.chars[i] = char; i += 1; loop back.
    write_char(&block_l2_body, &b2_buf, &b2_i, &b2_char);
    let b2_i2 = variable_with_lltype("i", LowLevelType::Signed);
    push(
        &block_l2_body,
        "int_add",
        vec![Hlvalue::Variable(b2_i), signed(1)],
        Hlvalue::Variable(b2_i2.clone()),
    );
    block_l2_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(b2_char),
                Hlvalue::Variable(b2_buf),
                Hlvalue::Variable(b2_i2),
                Hlvalue::Variable(b2_end),
            ],
            Some(block_l2_header.clone()),
            None,
        )
        .into_ref(),
    ]);
    // header: if i < end -> body else return.
    let h2_cont = variable_with_lltype("cont", LowLevelType::Bool);
    push(
        &block_l2_header,
        "int_lt",
        vec![
            Hlvalue::Variable(h2_i.clone()),
            Hlvalue::Variable(h2_end.clone()),
        ],
        Hlvalue::Variable(h2_cont.clone()),
    );
    block_l2_header.borrow_mut().exitswitch = Some(Hlvalue::Variable(h2_cont));
    block_l2_header.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(h2_char),
                Hlvalue::Variable(h2_buf),
                Hlvalue::Variable(h2_i),
                Hlvalue::Variable(h2_end),
            ],
            Some(block_l2_body),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![none_const()],
            Some(graph.returnblock.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);
    // block_after: buf/pos re-read, end = pos + times, current_pos = end, enter loop2.
    let a_buf = variable_with_lltype("buf", buf_lltype.clone());
    push(
        &block_after,
        "getfield",
        vec![
            Hlvalue::Variable(a_llb.clone()),
            void_field_const("current_buf"),
        ],
        Hlvalue::Variable(a_buf.clone()),
    );
    let a_pos = variable_with_lltype("pos", LowLevelType::Signed);
    push(
        &block_after,
        "getfield",
        vec![
            Hlvalue::Variable(a_llb.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(a_pos.clone()),
    );
    let a_end = variable_with_lltype("end", LowLevelType::Signed);
    push(
        &block_after,
        "int_add",
        vec![Hlvalue::Variable(a_pos.clone()), Hlvalue::Variable(a_times)],
        Hlvalue::Variable(a_end.clone()),
    );
    push(
        &block_after,
        "setfield",
        vec![
            Hlvalue::Variable(a_llb),
            void_field_const("current_pos"),
            Hlvalue::Variable(a_end.clone()),
        ],
        Hlvalue::Variable(void_result()),
    );
    block_after.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(a_char),
                Hlvalue::Variable(a_buf),
                Hlvalue::Variable(a_pos),
                Hlvalue::Variable(a_end),
            ],
            Some(block_l2_header),
            None,
        )
        .into_ref(),
    ]);

    // First (conditional) loop + grow.
    // block_grow_tail(ll_builder, char, times): ll_grow_by then jump to after.
    let gt_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let gt_char = variable_with_lltype("char", char_lltype.clone());
    let gt_times = variable_with_lltype("times", LowLevelType::Signed);
    let block_grow_tail = Block::shared(vec![
        Hlvalue::Variable(gt_llb.clone()),
        Hlvalue::Variable(gt_char.clone()),
        Hlvalue::Variable(gt_times.clone()),
    ]);
    push(
        &block_grow_tail,
        "direct_call",
        vec![
            Hlvalue::Constant(grow_by_fn),
            Hlvalue::Variable(gt_llb.clone()),
            Hlvalue::Variable(gt_times.clone()),
        ],
        Hlvalue::Variable(void_result()),
    );
    block_grow_tail.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(gt_llb),
                Hlvalue::Variable(gt_char),
                Hlvalue::Variable(gt_times),
            ],
            Some(block_after.clone()),
            None,
        )
        .into_ref(),
    ]);
    // block_l1_header(ll_builder, char, times, buf, i, end0)
    let h1_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let h1_char = variable_with_lltype("char", char_lltype.clone());
    let h1_times = variable_with_lltype("times", LowLevelType::Signed);
    let h1_buf = variable_with_lltype("buf", buf_lltype.clone());
    let h1_i = variable_with_lltype("i", LowLevelType::Signed);
    let h1_end0 = variable_with_lltype("end0", LowLevelType::Signed);
    let block_l1_header = Block::shared(vec![
        Hlvalue::Variable(h1_llb.clone()),
        Hlvalue::Variable(h1_char.clone()),
        Hlvalue::Variable(h1_times.clone()),
        Hlvalue::Variable(h1_buf.clone()),
        Hlvalue::Variable(h1_i.clone()),
        Hlvalue::Variable(h1_end0.clone()),
    ]);
    let b1_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let b1_char = variable_with_lltype("char", char_lltype.clone());
    let b1_times = variable_with_lltype("times", LowLevelType::Signed);
    let b1_buf = variable_with_lltype("buf", buf_lltype.clone());
    let b1_i = variable_with_lltype("i", LowLevelType::Signed);
    let b1_end0 = variable_with_lltype("end0", LowLevelType::Signed);
    let block_l1_body = Block::shared(vec![
        Hlvalue::Variable(b1_llb.clone()),
        Hlvalue::Variable(b1_char.clone()),
        Hlvalue::Variable(b1_times.clone()),
        Hlvalue::Variable(b1_buf.clone()),
        Hlvalue::Variable(b1_i.clone()),
        Hlvalue::Variable(b1_end0.clone()),
    ]);
    write_char(&block_l1_body, &b1_buf, &b1_i, &b1_char);
    let b1_i2 = variable_with_lltype("i", LowLevelType::Signed);
    push(
        &block_l1_body,
        "int_add",
        vec![Hlvalue::Variable(b1_i), signed(1)],
        Hlvalue::Variable(b1_i2.clone()),
    );
    block_l1_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(b1_llb),
                Hlvalue::Variable(b1_char),
                Hlvalue::Variable(b1_times),
                Hlvalue::Variable(b1_buf),
                Hlvalue::Variable(b1_i2),
                Hlvalue::Variable(b1_end0),
            ],
            Some(block_l1_header.clone()),
            None,
        )
        .into_ref(),
    ]);
    let h1_cont = variable_with_lltype("cont", LowLevelType::Bool);
    push(
        &block_l1_header,
        "int_lt",
        vec![
            Hlvalue::Variable(h1_i.clone()),
            Hlvalue::Variable(h1_end0.clone()),
        ],
        Hlvalue::Variable(h1_cont.clone()),
    );
    block_l1_header.borrow_mut().exitswitch = Some(Hlvalue::Variable(h1_cont));
    block_l1_header.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(h1_llb.clone()),
                Hlvalue::Variable(h1_char.clone()),
                Hlvalue::Variable(h1_times.clone()),
                Hlvalue::Variable(h1_buf),
                Hlvalue::Variable(h1_i),
                Hlvalue::Variable(h1_end0),
            ],
            Some(block_l1_body),
            bool_case(true),
        )
        .into_ref(),
        // loop done -> grow_tail(ll_builder, char, times).
        Link::new(
            vec![
                Hlvalue::Variable(h1_llb),
                Hlvalue::Variable(h1_char),
                Hlvalue::Variable(h1_times),
            ],
            Some(block_grow_tail),
            bool_case(false),
        )
        .into_ref(),
    ]);
    // block_setup(ll_builder, char, times, part1, pos0, end0): times-=part1;
    // buf=current_buf; enter loop1 at i=pos0.
    let su_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let su_char = variable_with_lltype("char", char_lltype.clone());
    let su_times = variable_with_lltype("times", LowLevelType::Signed);
    let su_part1 = variable_with_lltype("part1", LowLevelType::Signed);
    let su_pos0 = variable_with_lltype("pos0", LowLevelType::Signed);
    let su_end0 = variable_with_lltype("end0", LowLevelType::Signed);
    let block_setup = Block::shared(vec![
        Hlvalue::Variable(su_llb.clone()),
        Hlvalue::Variable(su_char.clone()),
        Hlvalue::Variable(su_times.clone()),
        Hlvalue::Variable(su_part1.clone()),
        Hlvalue::Variable(su_pos0.clone()),
        Hlvalue::Variable(su_end0.clone()),
    ]);
    let su_times2 = variable_with_lltype("times", LowLevelType::Signed);
    push(
        &block_setup,
        "int_sub",
        vec![Hlvalue::Variable(su_times), Hlvalue::Variable(su_part1)],
        Hlvalue::Variable(su_times2.clone()),
    );
    let su_buf = variable_with_lltype("buf", buf_lltype.clone());
    push(
        &block_setup,
        "getfield",
        vec![
            Hlvalue::Variable(su_llb.clone()),
            void_field_const("current_buf"),
        ],
        Hlvalue::Variable(su_buf.clone()),
    );
    block_setup.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(su_llb),
                Hlvalue::Variable(su_char),
                Hlvalue::Variable(su_times2),
                Hlvalue::Variable(su_buf),
                Hlvalue::Variable(su_pos0),
                Hlvalue::Variable(su_end0),
            ],
            Some(block_l1_header),
            None,
        )
        .into_ref(),
    ]);

    // startblock: part1 = current_end - current_pos; if times > part1.
    let pos0 = variable_with_lltype("pos0", LowLevelType::Signed);
    push(
        &startblock,
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(pos0.clone()),
    );
    let end0 = variable_with_lltype("end0", LowLevelType::Signed);
    push(
        &startblock,
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_end"),
        ],
        Hlvalue::Variable(end0.clone()),
    );
    let part1 = variable_with_lltype("part1", LowLevelType::Signed);
    push(
        &startblock,
        "int_sub",
        vec![
            Hlvalue::Variable(end0.clone()),
            Hlvalue::Variable(pos0.clone()),
        ],
        Hlvalue::Variable(part1.clone()),
    );
    let big = variable_with_lltype("big", LowLevelType::Bool);
    push(
        &startblock,
        "int_gt",
        vec![
            Hlvalue::Variable(times.clone()),
            Hlvalue::Variable(part1.clone()),
        ],
        Hlvalue::Variable(big.clone()),
    );
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(big));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(ll_builder.clone()),
                Hlvalue::Variable(char_v.clone()),
                Hlvalue::Variable(times.clone()),
                Hlvalue::Variable(part1),
                Hlvalue::Variable(pos0),
                Hlvalue::Variable(end0),
            ],
            Some(block_setup),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(ll_builder),
                Hlvalue::Variable(char_v),
                Hlvalue::Variable(times),
            ],
            Some(block_after),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "ll_builder".to_string(),
            "char".to_string(),
            "times".to_string(),
        ],
        func,
    ))
}

/// Synthesise `ll_jit_try_append_multiple_char(ll_builder, char, size)`
/// (`rbuilder.py:299-323`). Returns `Bool`.
#[allow(clippy::too_many_arguments)]
pub fn build_ll_jit_try_append_multiple_char_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    char_lltype: LowLevelType,
    chars_array_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    isconstant_fn: Constant,
    ll_append_char_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let bool_case = |b: bool| {
        Some(constant_with_lltype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let push = |block: &crate::flowspace::model::BlockRef,
                opname: &str,
                args: Vec<Hlvalue>,
                out: Hlvalue| {
        block
            .borrow_mut()
            .operations
            .push(SpaceOperation::new(opname, args, out));
    };
    let base_args = |llb: &Variable, ch: &Variable, size: &Variable| {
        vec![
            Hlvalue::Variable(llb.clone()),
            Hlvalue::Variable(ch.clone()),
            Hlvalue::Variable(size.clone()),
        ]
    };
    let mk_base_block = |builder_ty: &LowLevelType, char_ty: &LowLevelType| {
        let llb = variable_with_lltype("ll_builder", builder_ty.clone());
        let ch = variable_with_lltype("char", char_ty.clone());
        let size = variable_with_lltype("size", LowLevelType::Signed);
        let block = Block::shared(vec![
            Hlvalue::Variable(llb.clone()),
            Hlvalue::Variable(ch.clone()),
            Hlvalue::Variable(size.clone()),
        ]);
        (llb, ch, size, block)
    };
    let write_char = |block: &crate::flowspace::model::BlockRef,
                      buf: &Variable,
                      pos: &Variable,
                      ch: &Variable| {
        let chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
        push(
            block,
            "getsubstruct",
            vec![
                Hlvalue::Variable(buf.clone()),
                constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void),
            ],
            Hlvalue::Variable(chars.clone()),
        );
        push(
            block,
            "setarrayitem",
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(pos.clone()),
                Hlvalue::Variable(ch.clone()),
            ],
            Hlvalue::Variable(void_result()),
        );
    };

    let (ll_builder, char_v, size, startblock) = mk_base_block(&builder_ptr_lltype, &char_lltype);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let a_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let a_char = variable_with_lltype("char", char_lltype.clone());
    let block_append_one = Block::shared(vec![
        Hlvalue::Variable(a_llb.clone()),
        Hlvalue::Variable(a_char.clone()),
    ]);
    push(
        &block_append_one,
        "direct_call",
        vec![
            Hlvalue::Constant(ll_append_char_fn),
            Hlvalue::Variable(a_llb),
            Hlvalue::Variable(a_char),
        ],
        Hlvalue::Variable(void_result()),
    );
    block_append_one.closeblock(vec![
        Link::new(
            vec![bool_const(true)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let (s1_llb, s1_char, s1_size, block_size_one) =
        mk_base_block(&builder_ptr_lltype, &char_lltype);
    let is_one = variable_with_lltype("is_one", LowLevelType::Bool);
    push(
        &block_size_one,
        "int_eq",
        vec![Hlvalue::Variable(s1_size), signed(1)],
        Hlvalue::Variable(is_one.clone()),
    );
    block_size_one.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_one));
    block_size_one.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(s1_llb.clone()),
                Hlvalue::Variable(s1_char.clone()),
            ],
            Some(block_append_one),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![bool_const(false)],
            Some(graph.returnblock.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let h_char = variable_with_lltype("char", char_lltype.clone());
    let h_buf = variable_with_lltype("buf", buf_lltype.clone());
    let h_pos = variable_with_lltype("pos", LowLevelType::Signed);
    let h_stop = variable_with_lltype("stop", LowLevelType::Signed);
    let block_loop_header = Block::shared(vec![
        Hlvalue::Variable(h_char.clone()),
        Hlvalue::Variable(h_buf.clone()),
        Hlvalue::Variable(h_pos.clone()),
        Hlvalue::Variable(h_stop.clone()),
    ]);
    let b_char = variable_with_lltype("char", char_lltype.clone());
    let b_buf = variable_with_lltype("buf", buf_lltype.clone());
    let b_pos = variable_with_lltype("pos", LowLevelType::Signed);
    let b_stop = variable_with_lltype("stop", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(b_char.clone()),
        Hlvalue::Variable(b_buf.clone()),
        Hlvalue::Variable(b_pos.clone()),
        Hlvalue::Variable(b_stop.clone()),
    ]);
    write_char(&block_loop_body, &b_buf, &b_pos, &b_char);
    let b_pos_next = variable_with_lltype("pos", LowLevelType::Signed);
    push(
        &block_loop_body,
        "int_add",
        vec![Hlvalue::Variable(b_pos), signed(1)],
        Hlvalue::Variable(b_pos_next.clone()),
    );
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(b_char),
                Hlvalue::Variable(b_buf),
                Hlvalue::Variable(b_pos_next),
                Hlvalue::Variable(b_stop),
            ],
            Some(block_loop_header.clone()),
            None,
        )
        .into_ref(),
    ]);
    let h_cont = variable_with_lltype("cont", LowLevelType::Bool);
    push(
        &block_loop_header,
        "int_lt",
        vec![
            Hlvalue::Variable(h_pos.clone()),
            Hlvalue::Variable(h_stop.clone()),
        ],
        Hlvalue::Variable(h_cont.clone()),
    );
    block_loop_header.borrow_mut().exitswitch = Some(Hlvalue::Variable(h_cont));
    block_loop_header.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(h_char),
                Hlvalue::Variable(h_buf),
                Hlvalue::Variable(h_pos),
                Hlvalue::Variable(h_stop),
            ],
            Some(block_loop_body),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![bool_const(true)],
            Some(graph.returnblock.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let f_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let f_char = variable_with_lltype("char", char_lltype.clone());
    let f_size = variable_with_lltype("size", LowLevelType::Signed);
    let f_pos = variable_with_lltype("current_pos", LowLevelType::Signed);
    let block_fit = Block::shared(vec![
        Hlvalue::Variable(f_llb.clone()),
        Hlvalue::Variable(f_char.clone()),
        Hlvalue::Variable(f_size.clone()),
        Hlvalue::Variable(f_pos.clone()),
    ]);
    let f_buf = variable_with_lltype("buf", buf_lltype);
    push(
        &block_fit,
        "getfield",
        vec![
            Hlvalue::Variable(f_llb.clone()),
            void_field_const("current_buf"),
        ],
        Hlvalue::Variable(f_buf.clone()),
    );
    let f_stop = variable_with_lltype("stop", LowLevelType::Signed);
    push(
        &block_fit,
        "int_add",
        vec![Hlvalue::Variable(f_pos.clone()), Hlvalue::Variable(f_size)],
        Hlvalue::Variable(f_stop.clone()),
    );
    push(
        &block_fit,
        "setfield",
        vec![
            Hlvalue::Variable(f_llb),
            void_field_const("current_pos"),
            Hlvalue::Variable(f_stop.clone()),
        ],
        Hlvalue::Variable(void_result()),
    );
    block_fit.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(f_char),
                Hlvalue::Variable(f_buf),
                Hlvalue::Variable(f_pos),
                Hlvalue::Variable(f_stop),
            ],
            Some(block_loop_header),
            None,
        )
        .into_ref(),
    ]);

    let l_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let l_char = variable_with_lltype("char", char_lltype.clone());
    let l_size = variable_with_lltype("size", LowLevelType::Signed);
    let l_pos = variable_with_lltype("current_pos", LowLevelType::Signed);
    let block_le_16 = Block::shared(vec![
        Hlvalue::Variable(l_llb.clone()),
        Hlvalue::Variable(l_char.clone()),
        Hlvalue::Variable(l_size.clone()),
        Hlvalue::Variable(l_pos.clone()),
    ]);
    let le_16 = variable_with_lltype("le_16", LowLevelType::Bool);
    push(
        &block_le_16,
        "int_le",
        vec![Hlvalue::Variable(l_size.clone()), signed(16)],
        Hlvalue::Variable(le_16.clone()),
    );
    block_le_16.borrow_mut().exitswitch = Some(Hlvalue::Variable(le_16));
    block_le_16.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_llb.clone()),
                Hlvalue::Variable(l_char.clone()),
                Hlvalue::Variable(l_size.clone()),
                Hlvalue::Variable(l_pos),
            ],
            Some(block_fit),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            base_args(&l_llb, &l_char, &l_size),
            Some(block_size_one.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let av_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let av_char = variable_with_lltype("char", char_lltype.clone());
    let av_size = variable_with_lltype("size", LowLevelType::Signed);
    let av_pos = variable_with_lltype("current_pos", LowLevelType::Signed);
    let av_end = variable_with_lltype("current_end", LowLevelType::Signed);
    let block_avail = Block::shared(vec![
        Hlvalue::Variable(av_llb.clone()),
        Hlvalue::Variable(av_char.clone()),
        Hlvalue::Variable(av_size.clone()),
        Hlvalue::Variable(av_pos.clone()),
        Hlvalue::Variable(av_end.clone()),
    ]);
    let avail = variable_with_lltype("avail", LowLevelType::Signed);
    push(
        &block_avail,
        "int_sub",
        vec![Hlvalue::Variable(av_end), Hlvalue::Variable(av_pos.clone())],
        Hlvalue::Variable(avail.clone()),
    );
    let fits = variable_with_lltype("fits", LowLevelType::Bool);
    push(
        &block_avail,
        "int_le",
        vec![Hlvalue::Variable(av_size.clone()), Hlvalue::Variable(avail)],
        Hlvalue::Variable(fits.clone()),
    );
    block_avail.borrow_mut().exitswitch = Some(Hlvalue::Variable(fits));
    block_avail.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(av_llb.clone()),
                Hlvalue::Variable(av_char.clone()),
                Hlvalue::Variable(av_size.clone()),
                Hlvalue::Variable(av_pos),
            ],
            Some(block_le_16),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            base_args(&av_llb, &av_char, &av_size),
            Some(block_size_one.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let ce_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let ce_char = variable_with_lltype("char", char_lltype.clone());
    let ce_size = variable_with_lltype("size", LowLevelType::Signed);
    let ce_pos = variable_with_lltype("current_pos", LowLevelType::Signed);
    let block_end_const = Block::shared(vec![
        Hlvalue::Variable(ce_llb.clone()),
        Hlvalue::Variable(ce_char.clone()),
        Hlvalue::Variable(ce_size.clone()),
        Hlvalue::Variable(ce_pos.clone()),
    ]);
    let ce_end = variable_with_lltype("current_end", LowLevelType::Signed);
    push(
        &block_end_const,
        "getfield",
        vec![
            Hlvalue::Variable(ce_llb.clone()),
            void_field_const("current_end"),
        ],
        Hlvalue::Variable(ce_end.clone()),
    );
    let end_is_const = variable_with_lltype("end_is_const", LowLevelType::Bool);
    push(
        &block_end_const,
        "direct_call",
        vec![
            Hlvalue::Constant(isconstant_fn.clone()),
            Hlvalue::Variable(ce_end.clone()),
        ],
        Hlvalue::Variable(end_is_const.clone()),
    );
    block_end_const.borrow_mut().exitswitch = Some(Hlvalue::Variable(end_is_const));
    block_end_const.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(ce_llb.clone()),
                Hlvalue::Variable(ce_char.clone()),
                Hlvalue::Variable(ce_size.clone()),
                Hlvalue::Variable(ce_pos),
                Hlvalue::Variable(ce_end),
            ],
            Some(block_avail),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            base_args(&ce_llb, &ce_char, &ce_size),
            Some(block_size_one.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let (cp_llb, cp_char, cp_size, block_pos_const) =
        mk_base_block(&builder_ptr_lltype, &char_lltype);
    let cp_pos = variable_with_lltype("current_pos", LowLevelType::Signed);
    push(
        &block_pos_const,
        "getfield",
        vec![
            Hlvalue::Variable(cp_llb.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(cp_pos.clone()),
    );
    let pos_is_const = variable_with_lltype("pos_is_const", LowLevelType::Bool);
    push(
        &block_pos_const,
        "direct_call",
        vec![
            Hlvalue::Constant(isconstant_fn.clone()),
            Hlvalue::Variable(cp_pos.clone()),
        ],
        Hlvalue::Variable(pos_is_const.clone()),
    );
    block_pos_const.borrow_mut().exitswitch = Some(Hlvalue::Variable(pos_is_const));
    block_pos_const.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(cp_llb.clone()),
                Hlvalue::Variable(cp_char.clone()),
                Hlvalue::Variable(cp_size.clone()),
                Hlvalue::Variable(cp_pos),
            ],
            Some(block_end_const),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            base_args(&cp_llb, &cp_char, &cp_size),
            Some(block_size_one.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let (z_llb, z_char, z_size, block_size_zero) = mk_base_block(&builder_ptr_lltype, &char_lltype);
    let is_zero = variable_with_lltype("is_zero", LowLevelType::Bool);
    push(
        &block_size_zero,
        "int_eq",
        vec![Hlvalue::Variable(z_size.clone()), signed(0)],
        Hlvalue::Variable(is_zero.clone()),
    );
    block_size_zero.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_zero));
    block_size_zero.closeblock(vec![
        Link::new(
            vec![bool_const(true)],
            Some(graph.returnblock.clone()),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            base_args(&z_llb, &z_char, &z_size),
            Some(block_pos_const),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let size_is_const = variable_with_lltype("size_is_const", LowLevelType::Bool);
    push(
        &startblock,
        "direct_call",
        vec![
            Hlvalue::Constant(isconstant_fn),
            Hlvalue::Variable(size.clone()),
        ],
        Hlvalue::Variable(size_is_const.clone()),
    );
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(size_is_const));
    startblock.closeblock(vec![
        Link::new(
            base_args(&ll_builder, &char_v, &size),
            Some(block_size_zero),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![bool_const(false)],
            Some(graph.returnblock.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "ll_builder".to_string(),
            "char".to_string(),
            "size".to_string(),
        ],
        func,
    ))
}

/// Synthesise `ll_append_charpsize(ll_builder, charp, size)`
/// (`rbuilder.py:329-341`). Returns `Void`.
pub fn build_ll_append_charpsize_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    charp_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    copy_raw_to_string_fn: Constant,
    grow_by_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let bool_case = |b: bool| {
        Some(constant_with_lltype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let push = |block: &crate::flowspace::model::BlockRef,
                opname: &str,
                args: Vec<Hlvalue>,
                out: Hlvalue| {
        block
            .borrow_mut()
            .operations
            .push(SpaceOperation::new(opname, args, out));
    };

    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let charp = variable_with_lltype("charp", charp_lltype.clone());
    let size = variable_with_lltype("size", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(ll_builder.clone()),
        Hlvalue::Variable(charp.clone()),
        Hlvalue::Variable(size.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Shared tail: current_pos = current_pos + size; copy_raw_to_string(...).
    let t_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let t_charp = variable_with_lltype("charp", charp_lltype.clone());
    let t_size = variable_with_lltype("size", LowLevelType::Signed);
    let block_tail = Block::shared(vec![
        Hlvalue::Variable(t_llb.clone()),
        Hlvalue::Variable(t_charp.clone()),
        Hlvalue::Variable(t_size.clone()),
    ]);
    let t_pos = variable_with_lltype("pos", LowLevelType::Signed);
    push(
        &block_tail,
        "getfield",
        vec![
            Hlvalue::Variable(t_llb.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(t_pos.clone()),
    );
    let t_newpos = variable_with_lltype("newpos", LowLevelType::Signed);
    push(
        &block_tail,
        "int_add",
        vec![
            Hlvalue::Variable(t_pos.clone()),
            Hlvalue::Variable(t_size.clone()),
        ],
        Hlvalue::Variable(t_newpos.clone()),
    );
    push(
        &block_tail,
        "setfield",
        vec![
            Hlvalue::Variable(t_llb.clone()),
            void_field_const("current_pos"),
            Hlvalue::Variable(t_newpos),
        ],
        Hlvalue::Variable(void_result()),
    );
    let t_buf = variable_with_lltype("current_buf", buf_lltype.clone());
    push(
        &block_tail,
        "getfield",
        vec![Hlvalue::Variable(t_llb), void_field_const("current_buf")],
        Hlvalue::Variable(t_buf.clone()),
    );
    push(
        &block_tail,
        "direct_call",
        vec![
            Hlvalue::Constant(copy_raw_to_string_fn.clone()),
            Hlvalue::Variable(t_charp),
            Hlvalue::Variable(t_buf),
            Hlvalue::Variable(t_pos),
            Hlvalue::Variable(t_size),
        ],
        Hlvalue::Variable(void_result()),
    );
    block_tail.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // True arm: copy first part, advance charp, shrink size, grow.
    let b_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let b_charp = variable_with_lltype("charp", charp_lltype.clone());
    let b_size = variable_with_lltype("size", LowLevelType::Signed);
    let b_part1 = variable_with_lltype("part1", LowLevelType::Signed);
    let b_pos = variable_with_lltype("current_pos", LowLevelType::Signed);
    let block_body = Block::shared(vec![
        Hlvalue::Variable(b_llb.clone()),
        Hlvalue::Variable(b_charp.clone()),
        Hlvalue::Variable(b_size.clone()),
        Hlvalue::Variable(b_part1.clone()),
        Hlvalue::Variable(b_pos.clone()),
    ]);
    let b_buf = variable_with_lltype("current_buf", buf_lltype);
    push(
        &block_body,
        "getfield",
        vec![
            Hlvalue::Variable(b_llb.clone()),
            void_field_const("current_buf"),
        ],
        Hlvalue::Variable(b_buf.clone()),
    );
    push(
        &block_body,
        "direct_call",
        vec![
            Hlvalue::Constant(copy_raw_to_string_fn),
            Hlvalue::Variable(b_charp.clone()),
            Hlvalue::Variable(b_buf),
            Hlvalue::Variable(b_pos),
            Hlvalue::Variable(b_part1.clone()),
        ],
        Hlvalue::Variable(void_result()),
    );
    let b_charp2 = variable_with_lltype("charp", charp_lltype);
    push(
        &block_body,
        "direct_ptradd",
        vec![
            Hlvalue::Variable(b_charp),
            Hlvalue::Variable(b_part1.clone()),
        ],
        Hlvalue::Variable(b_charp2.clone()),
    );
    let b_size2 = variable_with_lltype("size", LowLevelType::Signed);
    push(
        &block_body,
        "int_sub",
        vec![Hlvalue::Variable(b_size), Hlvalue::Variable(b_part1)],
        Hlvalue::Variable(b_size2.clone()),
    );
    push(
        &block_body,
        "direct_call",
        vec![
            Hlvalue::Constant(grow_by_fn),
            Hlvalue::Variable(b_llb.clone()),
            Hlvalue::Variable(b_size2.clone()),
        ],
        Hlvalue::Variable(void_result()),
    );
    block_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(b_llb),
                Hlvalue::Variable(b_charp2),
                Hlvalue::Variable(b_size2),
            ],
            Some(block_tail.clone()),
            None,
        )
        .into_ref(),
    ]);

    // part1 = current_end - current_pos; if size > part1.
    let pos0 = variable_with_lltype("current_pos", LowLevelType::Signed);
    push(
        &startblock,
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(pos0.clone()),
    );
    let end0 = variable_with_lltype("current_end", LowLevelType::Signed);
    push(
        &startblock,
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_end"),
        ],
        Hlvalue::Variable(end0.clone()),
    );
    let part1 = variable_with_lltype("part1", LowLevelType::Signed);
    push(
        &startblock,
        "int_sub",
        vec![Hlvalue::Variable(end0), Hlvalue::Variable(pos0.clone())],
        Hlvalue::Variable(part1.clone()),
    );
    let too_big = variable_with_lltype("too_big", LowLevelType::Bool);
    push(
        &startblock,
        "int_gt",
        vec![
            Hlvalue::Variable(size.clone()),
            Hlvalue::Variable(part1.clone()),
        ],
        Hlvalue::Variable(too_big.clone()),
    );
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(too_big));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(ll_builder.clone()),
                Hlvalue::Variable(charp.clone()),
                Hlvalue::Variable(size.clone()),
                Hlvalue::Variable(part1),
                Hlvalue::Variable(pos0),
            ],
            Some(block_body),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(ll_builder),
                Hlvalue::Variable(charp),
                Hlvalue::Variable(size),
            ],
            Some(block_tail),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "ll_builder".to_string(),
            "charp".to_string(),
            "size".to_string(),
        ],
        func,
    ))
}

/// Synthesise `ll_append(ll_builder, ll_str)` (`rbuilder.py`):
///
/// ```python
/// if jit.we_are_jitted():
///     ll_jit_append(ll_builder, ll_str)
/// else:
///     # no-jit case: inline the logic of _ll_append() in the caller
///     _ll_append(ll_builder, ll_str, 0, len(ll_str.chars))
/// ```
///
/// `jit.we_are_jitted()` rtypes to the identity-bearing symbolic
/// `Constant(ConstValue::SpecTag(WE_ARE_JITTED_TAG_ID), Bool)`
/// (`jit.py`), emitted here as the branch exitswitch.
/// `replace_we_are_jitted` folds it to `false` on the interpreter
/// path and `jtransform` folds it to `true` on the JIT path.
/// `ll_jit_append` / `_ll_append` are `direct_call` callee consts.
/// Returns `Void`.
pub fn build_ll_append_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    chars_array_ptr_lltype: LowLevelType,
    jit_append_fn: Constant,
    ll_append_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let bool_case = |b: bool| {
        Some(constant_with_lltype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };

    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let ll_str = variable_with_lltype("ll_str", buf_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(ll_builder.clone()),
        Hlvalue::Variable(ll_str.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // jit arm: ll_jit_append(ll_builder, ll_str).
    let jit_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let jit_str = variable_with_lltype("ll_str", buf_lltype.clone());
    let block_jit = Block::shared(vec![
        Hlvalue::Variable(jit_llb.clone()),
        Hlvalue::Variable(jit_str.clone()),
    ]);
    block_jit.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(jit_append_fn),
            Hlvalue::Variable(jit_llb),
            Hlvalue::Variable(jit_str),
        ],
        Hlvalue::Variable(void_result()),
    ));
    block_jit.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // no-jit arm: _ll_append(ll_builder, ll_str, 0, len(ll_str.chars)).
    let nj_llb = variable_with_lltype("ll_builder", builder_ptr_lltype);
    let nj_str = variable_with_lltype("ll_str", buf_lltype);
    let block_nojit = Block::shared(vec![
        Hlvalue::Variable(nj_llb.clone()),
        Hlvalue::Variable(nj_str.clone()),
    ]);
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    block_nojit
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![
                Hlvalue::Variable(nj_str.clone()),
                constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void),
            ],
            Hlvalue::Variable(chars.clone()),
        ));
    let len = variable_with_lltype("length", LowLevelType::Signed);
    block_nojit
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(chars)],
            Hlvalue::Variable(len.clone()),
        ));
    block_nojit
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(ll_append_fn),
                Hlvalue::Variable(nj_llb),
                Hlvalue::Variable(nj_str),
                constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
                Hlvalue::Variable(len),
            ],
            Hlvalue::Variable(void_result()),
        ));
    block_nojit.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // startblock: branch on `we_are_jitted()` symbolic. The exitswitch
    // is the identity-bearing `SpecTag` constant so `replace_symbolic`
    // (interpreter path) / `jtransform` (JIT path) can fold the branch.
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::SpecTag(WE_ARE_JITTED_TAG_ID),
        LowLevelType::Bool,
    )));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(ll_builder.clone()),
                Hlvalue::Variable(ll_str.clone()),
            ],
            Some(block_jit),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(ll_builder), Hlvalue::Variable(ll_str)],
            Some(block_nojit),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ll_builder".to_string(), "ll_str".to_string()],
        func,
    ))
}

/// Synthesise `ll_jit_append(ll_builder, ll_str)` (`rbuilder.py`):
///
/// ```python
/// if ll_jit_try_append_slice(ll_builder, ll_str, 0, len(ll_str.chars)):
///     return
/// ll_append_res0(ll_builder, ll_str)
/// ```
///
/// `ll_jit_try_append_slice` / `ll_append_res0` are `direct_call` callee
/// consts. Returns `Void`.
pub fn build_ll_jit_append_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    chars_array_ptr_lltype: LowLevelType,
    try_append_slice_fn: Constant,
    res0_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let bool_case = |b: bool| {
        Some(constant_with_lltype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };

    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let ll_str = variable_with_lltype("ll_str", buf_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(ll_builder.clone()),
        Hlvalue::Variable(ll_str.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // fallback: ll_append_res0(ll_builder, ll_str).
    let fb_llb = variable_with_lltype("ll_builder", builder_ptr_lltype);
    let fb_str = variable_with_lltype("ll_str", buf_lltype);
    let block_fb = Block::shared(vec![
        Hlvalue::Variable(fb_llb.clone()),
        Hlvalue::Variable(fb_str.clone()),
    ]);
    block_fb.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(res0_fn),
            Hlvalue::Variable(fb_llb),
            Hlvalue::Variable(fb_str),
        ],
        Hlvalue::Variable(void_result()),
    ));
    block_fb.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // startblock: len(ll_str.chars); handled = try_append_slice(...); branch.
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![
            Hlvalue::Variable(ll_str.clone()),
            constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void),
        ],
        Hlvalue::Variable(chars.clone()),
    ));
    let len = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars)],
        Hlvalue::Variable(len.clone()),
    ));
    let handled = variable_with_lltype("handled", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(try_append_slice_fn),
            Hlvalue::Variable(ll_builder.clone()),
            Hlvalue::Variable(ll_str.clone()),
            constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
            Hlvalue::Variable(len),
        ],
        Hlvalue::Variable(handled.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(handled));
    startblock.closeblock(vec![
        // handled -> return.
        Link::new(
            vec![none_const()],
            Some(graph.returnblock.clone()),
            bool_case(true),
        )
        .into_ref(),
        // not handled -> fallback.
        Link::new(
            vec![Hlvalue::Variable(ll_builder), Hlvalue::Variable(ll_str)],
            Some(block_fb),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ll_builder".to_string(), "ll_str".to_string()],
        func,
    ))
}

/// Synthesise `ll_append_char(ll_builder, char)` (`rbuilder.py`):
///
/// ```python
/// jit.conditional_call(ll_builder.current_pos == ll_builder.current_end,
///                      ll_grow_by, ll_builder, 1)
/// pos = ll_builder.current_pos
/// ll_builder.current_pos = pos + 1
/// ll_builder.current_buf.chars[pos] = char
/// ```
///
/// `jit.conditional_call` is `_always_inline_ = 'try'` (`rlib/jit.py
/// conditional_call`), so the call is the inlined `we_are_jitted()`
/// switch: true → `jit_conditional_call`; false → `if cond:
/// direct_call(ll_grow_by, ll_builder, 1)`. Both arms continue to the
/// write-char tail. `ll_grow_by` is the `direct_call`-style funcptr
/// const. `current_pos` is re-read after the conditional call since
/// `ll_grow_by` may reset it. Returns `Void`.
pub fn build_ll_append_char_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    char_lltype: LowLevelType,
    chars_array_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    grow_by_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };

    let bool_case = |b: bool| {
        Some(constant_with_lltype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };
    let one = || constant_with_lltype(ConstValue::Int(1), LowLevelType::Signed);

    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let char = variable_with_lltype("char", char_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(ll_builder.clone()),
        Hlvalue::Variable(char.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let llb_jit = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let char_jit = variable_with_lltype("char", char_lltype.clone());
    let cond_jit = variable_with_lltype("cond", LowLevelType::Bool);
    let block_jit = Block::shared(vec![
        Hlvalue::Variable(llb_jit.clone()),
        Hlvalue::Variable(char_jit.clone()),
        Hlvalue::Variable(cond_jit.clone()),
    ]);

    let llb_interp_if = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let char_interp_if = variable_with_lltype("char", char_lltype.clone());
    let cond_interp_if = variable_with_lltype("cond", LowLevelType::Bool);
    let block_interp_if = Block::shared(vec![
        Hlvalue::Variable(llb_interp_if.clone()),
        Hlvalue::Variable(char_interp_if.clone()),
        Hlvalue::Variable(cond_interp_if.clone()),
    ]);

    let llb_interp_direct = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let char_interp_direct = variable_with_lltype("char", char_lltype.clone());
    let block_interp_direct = Block::shared(vec![
        Hlvalue::Variable(llb_interp_direct.clone()),
        Hlvalue::Variable(char_interp_direct.clone()),
    ]);

    let llb_tail = variable_with_lltype("ll_builder", builder_ptr_lltype);
    let char_tail = variable_with_lltype("char", char_lltype);
    let block_tail = Block::shared(vec![
        Hlvalue::Variable(llb_tail.clone()),
        Hlvalue::Variable(char_tail.clone()),
    ]);

    // cond = (ll_builder.current_pos == ll_builder.current_end).
    let pos0 = variable_with_lltype("pos", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(pos0.clone()),
    ));
    let end0 = variable_with_lltype("end", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_end"),
        ],
        Hlvalue::Variable(end0.clone()),
    ));
    let cond = variable_with_lltype("cond", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![Hlvalue::Variable(pos0), Hlvalue::Variable(end0)],
        Hlvalue::Variable(cond.clone()),
    ));

    // `rlib/jit.py conditional_call` is always inlined, so this is the
    // identity-bearing `we_are_jitted()` SpecTag exitswitch (true link
    // first, matching `build_ll_list_resize_ge_graph`): true →
    // jit_conditional_call; false → if cond: direct_call.
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::SpecTag(WE_ARE_JITTED_TAG_ID),
        LowLevelType::Bool,
    )));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(ll_builder.clone()),
                Hlvalue::Variable(char.clone()),
                Hlvalue::Variable(cond.clone()),
            ],
            Some(block_jit.clone()),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(ll_builder),
                Hlvalue::Variable(char),
                Hlvalue::Variable(cond),
            ],
            Some(block_interp_if.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    block_jit.borrow_mut().operations.push(SpaceOperation::new(
        "jit_conditional_call",
        vec![
            Hlvalue::Variable(cond_jit),
            Hlvalue::Constant(grow_by_fn.clone()),
            Hlvalue::Variable(llb_jit.clone()),
            one(),
        ],
        Hlvalue::Variable(void_result()),
    ));
    block_jit.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(llb_jit), Hlvalue::Variable(char_jit)],
            Some(block_tail.clone()),
            None,
        )
        .into_ref(),
    ]);

    block_interp_if.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond_interp_if));
    block_interp_if.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(llb_interp_if.clone()),
                Hlvalue::Variable(char_interp_if.clone()),
            ],
            Some(block_interp_direct.clone()),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(llb_interp_if),
                Hlvalue::Variable(char_interp_if),
            ],
            Some(block_tail.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    block_interp_direct
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(grow_by_fn),
                Hlvalue::Variable(llb_interp_direct.clone()),
                one(),
            ],
            Hlvalue::Variable(void_result()),
        ));
    block_interp_direct.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(llb_interp_direct),
                Hlvalue::Variable(char_interp_direct),
            ],
            Some(block_tail.clone()),
            None,
        )
        .into_ref(),
    ]);

    // pos = ll_builder.current_pos (re-read after possible grow).
    let pos = variable_with_lltype("pos", LowLevelType::Signed);
    block_tail.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(llb_tail.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(pos.clone()),
    ));
    // ll_builder.current_pos = pos + 1.
    let newpos = variable_with_lltype("newpos", LowLevelType::Signed);
    block_tail.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![Hlvalue::Variable(pos.clone()), one()],
        Hlvalue::Variable(newpos.clone()),
    ));
    block_tail.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(llb_tail.clone()),
            void_field_const("current_pos"),
            Hlvalue::Variable(newpos),
        ],
        Hlvalue::Variable(void_result()),
    ));

    // ll_builder.current_buf.chars[pos] = char.
    let buf = variable_with_lltype("buf", buf_lltype);
    block_tail.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(llb_tail), void_field_const("current_buf")],
        Hlvalue::Variable(buf.clone()),
    ));
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    block_tail.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![
            Hlvalue::Variable(buf),
            constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void),
        ],
        Hlvalue::Variable(chars.clone()),
    ));
    block_tail.borrow_mut().operations.push(SpaceOperation::new(
        "setarrayitem",
        vec![
            Hlvalue::Variable(chars),
            Hlvalue::Variable(pos),
            Hlvalue::Variable(char_tail),
        ],
        Hlvalue::Variable(void_result()),
    ));

    block_tail.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ll_builder".to_string(), "char".to_string()],
        func,
    ))
}

/// Synthesise `ll_append_slice(ll_builder, ll_str, start, end)`
/// (`rbuilder.py:189-195`):
///
/// ```python
/// if jit.we_are_jitted():
///     ll_jit_append_slice(ll_builder, ll_str, start, end)
/// else:
///     # no-jit case: inline the logic of _ll_append() in the caller
///     _ll_append(ll_builder, ll_str, start, end - start)
/// ```
///
/// Branches on the `we_are_jitted()` symbolic exitswitch (see
/// [`build_ll_append_helper_graph`]). `ll_jit_append_slice` /
/// `_ll_append` are `direct_call` callee consts. Returns `Void`.
pub fn build_ll_append_slice_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    jit_append_slice_fn: Constant,
    ll_append_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let bool_case = |b: bool| {
        Some(constant_with_lltype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };

    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let ll_str = variable_with_lltype("ll_str", buf_lltype.clone());
    let start = variable_with_lltype("start", LowLevelType::Signed);
    let end = variable_with_lltype("end", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(ll_builder.clone()),
        Hlvalue::Variable(ll_str.clone()),
        Hlvalue::Variable(start.clone()),
        Hlvalue::Variable(end.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // jit arm: ll_jit_append_slice(ll_builder, ll_str, start, end).
    let jit_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let jit_str = variable_with_lltype("ll_str", buf_lltype.clone());
    let jit_start = variable_with_lltype("start", LowLevelType::Signed);
    let jit_end = variable_with_lltype("end", LowLevelType::Signed);
    let block_jit = Block::shared(vec![
        Hlvalue::Variable(jit_llb.clone()),
        Hlvalue::Variable(jit_str.clone()),
        Hlvalue::Variable(jit_start.clone()),
        Hlvalue::Variable(jit_end.clone()),
    ]);
    block_jit.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(jit_append_slice_fn),
            Hlvalue::Variable(jit_llb),
            Hlvalue::Variable(jit_str),
            Hlvalue::Variable(jit_start),
            Hlvalue::Variable(jit_end),
        ],
        Hlvalue::Variable(void_result()),
    ));
    block_jit.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // no-jit arm: _ll_append(ll_builder, ll_str, start, end - start).
    let nj_llb = variable_with_lltype("ll_builder", builder_ptr_lltype);
    let nj_str = variable_with_lltype("ll_str", buf_lltype);
    let nj_start = variable_with_lltype("start", LowLevelType::Signed);
    let nj_end = variable_with_lltype("end", LowLevelType::Signed);
    let block_nojit = Block::shared(vec![
        Hlvalue::Variable(nj_llb.clone()),
        Hlvalue::Variable(nj_str.clone()),
        Hlvalue::Variable(nj_start.clone()),
        Hlvalue::Variable(nj_end.clone()),
    ]);
    let size = variable_with_lltype("size", LowLevelType::Signed);
    block_nojit
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_sub",
            vec![
                Hlvalue::Variable(nj_end),
                Hlvalue::Variable(nj_start.clone()),
            ],
            Hlvalue::Variable(size.clone()),
        ));
    block_nojit
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(ll_append_fn),
                Hlvalue::Variable(nj_llb),
                Hlvalue::Variable(nj_str),
                Hlvalue::Variable(nj_start),
                Hlvalue::Variable(size),
            ],
            Hlvalue::Variable(void_result()),
        ));
    block_nojit.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // startblock: branch on `we_are_jitted()` symbolic exitswitch.
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::SpecTag(WE_ARE_JITTED_TAG_ID),
        LowLevelType::Bool,
    )));
    let arm_args = |llb: &Variable, s: &Variable, st: &Variable, en: &Variable| {
        vec![
            Hlvalue::Variable(llb.clone()),
            Hlvalue::Variable(s.clone()),
            Hlvalue::Variable(st.clone()),
            Hlvalue::Variable(en.clone()),
        ]
    };
    startblock.closeblock(vec![
        Link::new(
            arm_args(&ll_builder, &ll_str, &start, &end),
            Some(block_jit),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            arm_args(&ll_builder, &ll_str, &start, &end),
            Some(block_nojit),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "ll_builder".to_string(),
            "ll_str".to_string(),
            "start".to_string(),
            "end".to_string(),
        ],
        func,
    ))
}

/// Synthesise `ll_jit_append_slice(ll_builder, ll_str, start, end)`
/// (`rbuilder.py:198-203`):
///
/// ```python
/// if ll_jit_try_append_slice(ll_builder, ll_str, start, end - start):
///     return
/// ll_append_res_slice(ll_builder, ll_str, start, end)
/// ```
///
/// `ll_jit_try_append_slice` / `ll_append_res_slice` are `direct_call`
/// callee consts. Returns `Void`.
pub fn build_ll_jit_append_slice_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    try_append_slice_fn: Constant,
    res_slice_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let bool_case = |b: bool| {
        Some(constant_with_lltype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };

    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let ll_str = variable_with_lltype("ll_str", buf_lltype.clone());
    let start = variable_with_lltype("start", LowLevelType::Signed);
    let end = variable_with_lltype("end", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(ll_builder.clone()),
        Hlvalue::Variable(ll_str.clone()),
        Hlvalue::Variable(start.clone()),
        Hlvalue::Variable(end.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // fallback: ll_append_res_slice(ll_builder, ll_str, start, end).
    let fb_llb = variable_with_lltype("ll_builder", builder_ptr_lltype);
    let fb_str = variable_with_lltype("ll_str", buf_lltype);
    let fb_start = variable_with_lltype("start", LowLevelType::Signed);
    let fb_end = variable_with_lltype("end", LowLevelType::Signed);
    let block_fb = Block::shared(vec![
        Hlvalue::Variable(fb_llb.clone()),
        Hlvalue::Variable(fb_str.clone()),
        Hlvalue::Variable(fb_start.clone()),
        Hlvalue::Variable(fb_end.clone()),
    ]);
    block_fb.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(res_slice_fn),
            Hlvalue::Variable(fb_llb),
            Hlvalue::Variable(fb_str),
            Hlvalue::Variable(fb_start),
            Hlvalue::Variable(fb_end),
        ],
        Hlvalue::Variable(void_result()),
    ));
    block_fb.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // startblock: size = end - start; handled = try_append_slice(...); branch.
    let size = variable_with_lltype("size", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![
            Hlvalue::Variable(end.clone()),
            Hlvalue::Variable(start.clone()),
        ],
        Hlvalue::Variable(size.clone()),
    ));
    let handled = variable_with_lltype("handled", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(try_append_slice_fn),
            Hlvalue::Variable(ll_builder.clone()),
            Hlvalue::Variable(ll_str.clone()),
            Hlvalue::Variable(start.clone()),
            Hlvalue::Variable(size),
        ],
        Hlvalue::Variable(handled.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(handled));
    startblock.closeblock(vec![
        Link::new(
            vec![none_const()],
            Some(graph.returnblock.clone()),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(ll_builder),
                Hlvalue::Variable(ll_str),
                Hlvalue::Variable(start),
                Hlvalue::Variable(end),
            ],
            Some(block_fb),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "ll_builder".to_string(),
            "ll_str".to_string(),
            "start".to_string(),
            "end".to_string(),
        ],
        func,
    ))
}

/// Synthesise `ll_jit_try_append_slice(ll_builder, ll_str, start, size)`
/// (`rbuilder.py:233-270`). Returns `Bool`.
#[allow(clippy::too_many_arguments)]
pub fn build_ll_jit_try_append_slice_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    chars_array_ptr_lltype: LowLevelType,
    char_lltype: LowLevelType,
    isconstant_fn: Constant,
    ll_append_char_fn: Constant,
    size_specialized: &[(i64, Constant, Constant)],
) -> Result<PyGraph, TyperError> {
    let bool_case = |b: bool| {
        Some(constant_with_lltype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };
    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let push = |block: &crate::flowspace::model::BlockRef,
                opname: &str,
                args: Vec<Hlvalue>,
                out: Hlvalue| {
        block
            .borrow_mut()
            .operations
            .push(SpaceOperation::new(opname, args, out));
    };
    let slice_args = |llb: &Variable, s: &Variable, start: &Variable, size: &Variable| {
        vec![
            Hlvalue::Variable(llb.clone()),
            Hlvalue::Variable(s.clone()),
            Hlvalue::Variable(start.clone()),
            Hlvalue::Variable(size.clone()),
        ]
    };
    let start_args = |llb: &Variable, s: &Variable, start: &Variable| {
        vec![
            Hlvalue::Variable(llb.clone()),
            Hlvalue::Variable(s.clone()),
            Hlvalue::Variable(start.clone()),
        ]
    };
    let mk_slice_block = |builder_ty: &LowLevelType, buf_ty: &LowLevelType| {
        let llb = variable_with_lltype("ll_builder", builder_ty.clone());
        let s = variable_with_lltype("ll_str", buf_ty.clone());
        let start = variable_with_lltype("start", LowLevelType::Signed);
        let size = variable_with_lltype("size", LowLevelType::Signed);
        let block = Block::shared(vec![
            Hlvalue::Variable(llb.clone()),
            Hlvalue::Variable(s.clone()),
            Hlvalue::Variable(start.clone()),
            Hlvalue::Variable(size.clone()),
        ]);
        (llb, s, start, size, block)
    };
    let mk_start_block = |builder_ty: &LowLevelType, buf_ty: &LowLevelType| {
        let llb = variable_with_lltype("ll_builder", builder_ty.clone());
        let s = variable_with_lltype("ll_str", buf_ty.clone());
        let start = variable_with_lltype("start", LowLevelType::Signed);
        let block = Block::shared(vec![
            Hlvalue::Variable(llb.clone()),
            Hlvalue::Variable(s.clone()),
            Hlvalue::Variable(start.clone()),
        ]);
        (llb, s, start, block)
    };
    let read_char =
        |block: &crate::flowspace::model::BlockRef, ll_str: &Variable, start: &Variable| {
            let chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
            push(
                block,
                "getsubstruct",
                vec![
                    Hlvalue::Variable(ll_str.clone()),
                    constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void),
                ],
                Hlvalue::Variable(chars.clone()),
            );
            let ch = variable_with_lltype("char", char_lltype.clone());
            push(
                block,
                "getarrayitem",
                vec![Hlvalue::Variable(chars), Hlvalue::Variable(start.clone())],
                Hlvalue::Variable(ch.clone()),
            );
            ch
        };
    let write_char = |block: &crate::flowspace::model::BlockRef,
                      buf: &Variable,
                      pos: &Variable,
                      ch: &Variable| {
        let chars = variable_with_lltype("chars", chars_array_ptr_lltype.clone());
        push(
            block,
            "getsubstruct",
            vec![
                Hlvalue::Variable(buf.clone()),
                constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void),
            ],
            Hlvalue::Variable(chars.clone()),
        );
        push(
            block,
            "setarrayitem",
            vec![
                Hlvalue::Variable(chars),
                Hlvalue::Variable(pos.clone()),
                Hlvalue::Variable(ch.clone()),
            ],
            Hlvalue::Variable(void_result()),
        );
    };

    let (ll_builder, ll_str, start, size, startblock) =
        mk_slice_block(&builder_ptr_lltype, &buf_lltype);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Unrolled size-specialized arms for sizes 2..10.
    let mut next_check: Option<crate::flowspace::model::BlockRef> = None;
    for (for_size, func0_fn, funcstart_fn) in size_specialized.iter().rev() {
        let (cs_llb, cs_str, cs_start, block_call_start) =
            mk_start_block(&builder_ptr_lltype, &buf_lltype);
        push(
            &block_call_start,
            "direct_call",
            vec![
                Hlvalue::Constant(funcstart_fn.clone()),
                Hlvalue::Variable(cs_llb),
                Hlvalue::Variable(cs_str),
                Hlvalue::Variable(cs_start),
            ],
            Hlvalue::Variable(void_result()),
        );
        block_call_start.closeblock(vec![
            Link::new(
                vec![bool_const(true)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let c0_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
        let c0_str = variable_with_lltype("ll_str", buf_lltype.clone());
        let block_call0 = Block::shared(vec![
            Hlvalue::Variable(c0_llb.clone()),
            Hlvalue::Variable(c0_str.clone()),
        ]);
        push(
            &block_call0,
            "direct_call",
            vec![
                Hlvalue::Constant(func0_fn.clone()),
                Hlvalue::Variable(c0_llb),
                Hlvalue::Variable(c0_str),
            ],
            Hlvalue::Variable(void_result()),
        );
        block_call0.closeblock(vec![
            Link::new(
                vec![bool_const(true)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let (z_llb, z_str, z_start, block_start_zero) =
            mk_start_block(&builder_ptr_lltype, &buf_lltype);
        let start_zero = variable_with_lltype("start_zero", LowLevelType::Bool);
        push(
            &block_start_zero,
            "int_eq",
            vec![Hlvalue::Variable(z_start.clone()), signed(0)],
            Hlvalue::Variable(start_zero.clone()),
        );
        block_start_zero.borrow_mut().exitswitch = Some(Hlvalue::Variable(start_zero));
        block_start_zero.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(z_llb.clone()),
                    Hlvalue::Variable(z_str.clone()),
                ],
                Some(block_call0),
                bool_case(true),
            )
            .into_ref(),
            Link::new(
                start_args(&z_llb, &z_str, &z_start),
                Some(block_call_start.clone()),
                bool_case(false),
            )
            .into_ref(),
        ]);

        let (ic_llb, ic_str, ic_start, block_start_const) =
            mk_start_block(&builder_ptr_lltype, &buf_lltype);
        let start_is_const = variable_with_lltype("start_is_const", LowLevelType::Bool);
        push(
            &block_start_const,
            "direct_call",
            vec![
                Hlvalue::Constant(isconstant_fn.clone()),
                Hlvalue::Variable(ic_start.clone()),
            ],
            Hlvalue::Variable(start_is_const.clone()),
        );
        block_start_const.borrow_mut().exitswitch = Some(Hlvalue::Variable(start_is_const));
        block_start_const.closeblock(vec![
            Link::new(
                start_args(&ic_llb, &ic_str, &ic_start),
                Some(block_start_zero),
                bool_case(true),
            )
            .into_ref(),
            Link::new(
                start_args(&ic_llb, &ic_str, &ic_start),
                Some(block_call_start),
                bool_case(false),
            )
            .into_ref(),
        ]);

        let (chk_llb, chk_str, chk_start, chk_size, block_check) =
            mk_slice_block(&builder_ptr_lltype, &buf_lltype);
        let size_matches = variable_with_lltype("size_matches", LowLevelType::Bool);
        push(
            &block_check,
            "int_eq",
            vec![Hlvalue::Variable(chk_size.clone()), signed(*for_size)],
            Hlvalue::Variable(size_matches.clone()),
        );
        block_check.borrow_mut().exitswitch = Some(Hlvalue::Variable(size_matches));
        let false_link = if let Some(next) = next_check.clone() {
            Link::new(
                slice_args(&chk_llb, &chk_str, &chk_start, &chk_size),
                Some(next),
                bool_case(false),
            )
            .into_ref()
        } else {
            Link::new(
                vec![bool_const(false)],
                Some(graph.returnblock.clone()),
                bool_case(false),
            )
            .into_ref()
        };
        block_check.closeblock(vec![
            Link::new(
                start_args(&chk_llb, &chk_str, &chk_start),
                Some(block_start_const),
                bool_case(true),
            )
            .into_ref(),
            false_link,
        ]);
        next_check = Some(block_check);
    }
    let first_size_specialized = next_check;

    // size == 1: ll_append_char(ll_builder, ll_str.chars[start]); return True.
    let (a_llb, a_str, a_start, block_append_one) =
        mk_start_block(&builder_ptr_lltype, &buf_lltype);
    let a_char = read_char(&block_append_one, &a_str, &a_start);
    push(
        &block_append_one,
        "direct_call",
        vec![
            Hlvalue::Constant(ll_append_char_fn),
            Hlvalue::Variable(a_llb),
            Hlvalue::Variable(a_char),
        ],
        Hlvalue::Variable(void_result()),
    );
    block_append_one.closeblock(vec![
        Link::new(
            vec![bool_const(true)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let (s1_llb, s1_str, s1_start, s1_size, block_size_one) =
        mk_slice_block(&builder_ptr_lltype, &buf_lltype);
    let is_one = variable_with_lltype("is_one", LowLevelType::Bool);
    push(
        &block_size_one,
        "int_eq",
        vec![Hlvalue::Variable(s1_size.clone()), signed(1)],
        Hlvalue::Variable(is_one.clone()),
    );
    block_size_one.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_one));
    let false_link = if let Some(first_size_specialized) = first_size_specialized {
        Link::new(
            slice_args(&s1_llb, &s1_str, &s1_start, &s1_size),
            Some(first_size_specialized),
            bool_case(false),
        )
        .into_ref()
    } else {
        Link::new(
            vec![bool_const(false)],
            Some(graph.returnblock.clone()),
            bool_case(false),
        )
        .into_ref()
    };
    block_size_one.closeblock(vec![
        Link::new(
            start_args(&s1_llb, &s1_str, &s1_start),
            Some(block_append_one),
            bool_case(true),
        )
        .into_ref(),
        false_link,
    ]);

    // while pos < stop: buf.chars[pos] = ll_str.chars[start]; pos += 1; start += 1
    let h_str = variable_with_lltype("ll_str", buf_lltype.clone());
    let h_start = variable_with_lltype("start", LowLevelType::Signed);
    let h_buf = variable_with_lltype("buf", buf_lltype.clone());
    let h_pos = variable_with_lltype("pos", LowLevelType::Signed);
    let h_stop = variable_with_lltype("stop", LowLevelType::Signed);
    let block_loop_header = Block::shared(vec![
        Hlvalue::Variable(h_str.clone()),
        Hlvalue::Variable(h_start.clone()),
        Hlvalue::Variable(h_buf.clone()),
        Hlvalue::Variable(h_pos.clone()),
        Hlvalue::Variable(h_stop.clone()),
    ]);
    let b_str = variable_with_lltype("ll_str", buf_lltype.clone());
    let b_start = variable_with_lltype("start", LowLevelType::Signed);
    let b_buf = variable_with_lltype("buf", buf_lltype.clone());
    let b_pos = variable_with_lltype("pos", LowLevelType::Signed);
    let b_stop = variable_with_lltype("stop", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(b_str.clone()),
        Hlvalue::Variable(b_start.clone()),
        Hlvalue::Variable(b_buf.clone()),
        Hlvalue::Variable(b_pos.clone()),
        Hlvalue::Variable(b_stop.clone()),
    ]);
    let b_char = read_char(&block_loop_body, &b_str, &b_start);
    write_char(&block_loop_body, &b_buf, &b_pos, &b_char);
    let b_pos_next = variable_with_lltype("pos", LowLevelType::Signed);
    push(
        &block_loop_body,
        "int_add",
        vec![Hlvalue::Variable(b_pos), signed(1)],
        Hlvalue::Variable(b_pos_next.clone()),
    );
    let b_start_next = variable_with_lltype("start", LowLevelType::Signed);
    push(
        &block_loop_body,
        "int_add",
        vec![Hlvalue::Variable(b_start), signed(1)],
        Hlvalue::Variable(b_start_next.clone()),
    );
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(b_str),
                Hlvalue::Variable(b_start_next),
                Hlvalue::Variable(b_buf),
                Hlvalue::Variable(b_pos_next),
                Hlvalue::Variable(b_stop),
            ],
            Some(block_loop_header.clone()),
            None,
        )
        .into_ref(),
    ]);
    let h_cont = variable_with_lltype("cont", LowLevelType::Bool);
    push(
        &block_loop_header,
        "int_lt",
        vec![
            Hlvalue::Variable(h_pos.clone()),
            Hlvalue::Variable(h_stop.clone()),
        ],
        Hlvalue::Variable(h_cont.clone()),
    );
    block_loop_header.borrow_mut().exitswitch = Some(Hlvalue::Variable(h_cont));
    block_loop_header.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(h_str),
                Hlvalue::Variable(h_start),
                Hlvalue::Variable(h_buf),
                Hlvalue::Variable(h_pos),
                Hlvalue::Variable(h_stop),
            ],
            Some(block_loop_body),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![bool_const(true)],
            Some(graph.returnblock.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let f_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let f_str = variable_with_lltype("ll_str", buf_lltype.clone());
    let f_start = variable_with_lltype("start", LowLevelType::Signed);
    let f_size = variable_with_lltype("size", LowLevelType::Signed);
    let f_pos = variable_with_lltype("current_pos", LowLevelType::Signed);
    let block_fit = Block::shared(vec![
        Hlvalue::Variable(f_llb.clone()),
        Hlvalue::Variable(f_str.clone()),
        Hlvalue::Variable(f_start.clone()),
        Hlvalue::Variable(f_size.clone()),
        Hlvalue::Variable(f_pos.clone()),
    ]);
    let f_buf = variable_with_lltype("buf", buf_lltype.clone());
    push(
        &block_fit,
        "getfield",
        vec![
            Hlvalue::Variable(f_llb.clone()),
            void_field_const("current_buf"),
        ],
        Hlvalue::Variable(f_buf.clone()),
    );
    let f_stop = variable_with_lltype("stop", LowLevelType::Signed);
    push(
        &block_fit,
        "int_add",
        vec![Hlvalue::Variable(f_pos.clone()), Hlvalue::Variable(f_size)],
        Hlvalue::Variable(f_stop.clone()),
    );
    push(
        &block_fit,
        "setfield",
        vec![
            Hlvalue::Variable(f_llb),
            void_field_const("current_pos"),
            Hlvalue::Variable(f_stop.clone()),
        ],
        Hlvalue::Variable(void_result()),
    );
    block_fit.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(f_str),
                Hlvalue::Variable(f_start),
                Hlvalue::Variable(f_buf),
                Hlvalue::Variable(f_pos),
                Hlvalue::Variable(f_stop),
            ],
            Some(block_loop_header),
            None,
        )
        .into_ref(),
    ]);

    let l_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let l_str = variable_with_lltype("ll_str", buf_lltype.clone());
    let l_start = variable_with_lltype("start", LowLevelType::Signed);
    let l_size = variable_with_lltype("size", LowLevelType::Signed);
    let l_pos = variable_with_lltype("current_pos", LowLevelType::Signed);
    let block_le_16 = Block::shared(vec![
        Hlvalue::Variable(l_llb.clone()),
        Hlvalue::Variable(l_str.clone()),
        Hlvalue::Variable(l_start.clone()),
        Hlvalue::Variable(l_size.clone()),
        Hlvalue::Variable(l_pos.clone()),
    ]);
    let le_16 = variable_with_lltype("le_16", LowLevelType::Bool);
    push(
        &block_le_16,
        "int_le",
        vec![Hlvalue::Variable(l_size.clone()), signed(16)],
        Hlvalue::Variable(le_16.clone()),
    );
    block_le_16.borrow_mut().exitswitch = Some(Hlvalue::Variable(le_16));
    block_le_16.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_llb.clone()),
                Hlvalue::Variable(l_str.clone()),
                Hlvalue::Variable(l_start.clone()),
                Hlvalue::Variable(l_size.clone()),
                Hlvalue::Variable(l_pos),
            ],
            Some(block_fit),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            slice_args(&l_llb, &l_str, &l_start, &l_size),
            Some(block_size_one.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let av_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let av_str = variable_with_lltype("ll_str", buf_lltype.clone());
    let av_start = variable_with_lltype("start", LowLevelType::Signed);
    let av_size = variable_with_lltype("size", LowLevelType::Signed);
    let av_pos = variable_with_lltype("current_pos", LowLevelType::Signed);
    let av_end = variable_with_lltype("current_end", LowLevelType::Signed);
    let block_avail = Block::shared(vec![
        Hlvalue::Variable(av_llb.clone()),
        Hlvalue::Variable(av_str.clone()),
        Hlvalue::Variable(av_start.clone()),
        Hlvalue::Variable(av_size.clone()),
        Hlvalue::Variable(av_pos.clone()),
        Hlvalue::Variable(av_end.clone()),
    ]);
    let avail = variable_with_lltype("avail", LowLevelType::Signed);
    push(
        &block_avail,
        "int_sub",
        vec![Hlvalue::Variable(av_end), Hlvalue::Variable(av_pos.clone())],
        Hlvalue::Variable(avail.clone()),
    );
    let fits = variable_with_lltype("fits", LowLevelType::Bool);
    push(
        &block_avail,
        "int_le",
        vec![Hlvalue::Variable(av_size.clone()), Hlvalue::Variable(avail)],
        Hlvalue::Variable(fits.clone()),
    );
    block_avail.borrow_mut().exitswitch = Some(Hlvalue::Variable(fits));
    block_avail.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(av_llb.clone()),
                Hlvalue::Variable(av_str.clone()),
                Hlvalue::Variable(av_start.clone()),
                Hlvalue::Variable(av_size.clone()),
                Hlvalue::Variable(av_pos),
            ],
            Some(block_le_16),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            slice_args(&av_llb, &av_str, &av_start, &av_size),
            Some(block_size_one.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let ce_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let ce_str = variable_with_lltype("ll_str", buf_lltype.clone());
    let ce_start = variable_with_lltype("start", LowLevelType::Signed);
    let ce_size = variable_with_lltype("size", LowLevelType::Signed);
    let ce_pos = variable_with_lltype("current_pos", LowLevelType::Signed);
    let block_end_const = Block::shared(vec![
        Hlvalue::Variable(ce_llb.clone()),
        Hlvalue::Variable(ce_str.clone()),
        Hlvalue::Variable(ce_start.clone()),
        Hlvalue::Variable(ce_size.clone()),
        Hlvalue::Variable(ce_pos.clone()),
    ]);
    let ce_end = variable_with_lltype("current_end", LowLevelType::Signed);
    push(
        &block_end_const,
        "getfield",
        vec![
            Hlvalue::Variable(ce_llb.clone()),
            void_field_const("current_end"),
        ],
        Hlvalue::Variable(ce_end.clone()),
    );
    let end_is_const = variable_with_lltype("end_is_const", LowLevelType::Bool);
    push(
        &block_end_const,
        "direct_call",
        vec![
            Hlvalue::Constant(isconstant_fn.clone()),
            Hlvalue::Variable(ce_end.clone()),
        ],
        Hlvalue::Variable(end_is_const.clone()),
    );
    block_end_const.borrow_mut().exitswitch = Some(Hlvalue::Variable(end_is_const));
    block_end_const.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(ce_llb.clone()),
                Hlvalue::Variable(ce_str.clone()),
                Hlvalue::Variable(ce_start.clone()),
                Hlvalue::Variable(ce_size.clone()),
                Hlvalue::Variable(ce_pos),
                Hlvalue::Variable(ce_end),
            ],
            Some(block_avail),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            slice_args(&ce_llb, &ce_str, &ce_start, &ce_size),
            Some(block_size_one.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let (cp_llb, cp_str, cp_start, cp_size, block_pos_const) =
        mk_slice_block(&builder_ptr_lltype, &buf_lltype);
    let cp_pos = variable_with_lltype("current_pos", LowLevelType::Signed);
    push(
        &block_pos_const,
        "getfield",
        vec![
            Hlvalue::Variable(cp_llb.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(cp_pos.clone()),
    );
    let pos_is_const = variable_with_lltype("pos_is_const", LowLevelType::Bool);
    push(
        &block_pos_const,
        "direct_call",
        vec![
            Hlvalue::Constant(isconstant_fn.clone()),
            Hlvalue::Variable(cp_pos.clone()),
        ],
        Hlvalue::Variable(pos_is_const.clone()),
    );
    block_pos_const.borrow_mut().exitswitch = Some(Hlvalue::Variable(pos_is_const));
    block_pos_const.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(cp_llb.clone()),
                Hlvalue::Variable(cp_str.clone()),
                Hlvalue::Variable(cp_start.clone()),
                Hlvalue::Variable(cp_size.clone()),
                Hlvalue::Variable(cp_pos),
            ],
            Some(block_end_const),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            slice_args(&cp_llb, &cp_str, &cp_start, &cp_size),
            Some(block_size_one.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let (z_llb, z_str, z_start, z_size, block_size_zero) =
        mk_slice_block(&builder_ptr_lltype, &buf_lltype);
    let is_zero = variable_with_lltype("is_zero", LowLevelType::Bool);
    push(
        &block_size_zero,
        "int_eq",
        vec![Hlvalue::Variable(z_size.clone()), signed(0)],
        Hlvalue::Variable(is_zero.clone()),
    );
    block_size_zero.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_zero));
    block_size_zero.closeblock(vec![
        Link::new(
            vec![bool_const(true)],
            Some(graph.returnblock.clone()),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            slice_args(&z_llb, &z_str, &z_start, &z_size),
            Some(block_pos_const),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let size_is_const = variable_with_lltype("size_is_const", LowLevelType::Bool);
    push(
        &startblock,
        "direct_call",
        vec![
            Hlvalue::Constant(isconstant_fn),
            Hlvalue::Variable(size.clone()),
        ],
        Hlvalue::Variable(size_is_const.clone()),
    );
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(size_is_const));
    startblock.closeblock(vec![
        Link::new(
            slice_args(&ll_builder, &ll_str, &start, &size),
            Some(block_size_zero),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![bool_const(false)],
            Some(graph.returnblock.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "ll_builder".to_string(),
            "ll_str".to_string(),
            "start".to_string(),
            "size".to_string(),
        ],
        func,
    ))
}

/// Close a block whose LAST operation is an `int_add_ovf` (`ovfcheck`) with the
/// OverflowError-catching exit shape `flatten.rs insert_exits` requires for any
/// block containing an `_ovf` op: a `c_last_exception` exitswitch with exit 0
/// the no-overflow fall-through (`normal_args` → `normal_target`) and exit 1 the
/// OverflowError catch (`exc_args` → `exc_target`). `overflow_error` is
/// `exception_args("OverflowError")` (`[class, instance]`); its elements are the
/// exit's `exitcase` and its `last_exception` / `last_exc_value` extravars.
/// `exc_args` must differ from `[class, instance]` so `make_exception_link`
/// routes to `exc_target` instead of taking its bare-reraise fast path.
pub(crate) fn close_ovfcheck_block(
    block: &BlockRef,
    overflow_error: &[Hlvalue],
    normal_target: &BlockRef,
    normal_args: Vec<Hlvalue>,
    exc_target: &BlockRef,
    exc_args: Vec<Hlvalue>,
) {
    block.borrow_mut().exitswitch = Some(Hlvalue::Constant(c_last_exception()));
    let normal_link = Link::new(normal_args, Some(normal_target.clone()), None).into_ref();
    let mut exc_link = Link::new(
        exc_args,
        Some(exc_target.clone()),
        Some(overflow_error[0].clone()),
    );
    exc_link.extravars(
        Some(overflow_error[0].clone()),
        Some(overflow_error[1].clone()),
    );
    block.closeblock(vec![normal_link, exc_link.into_ref()]);
}

/// Synthesise `ll_grow_by(ll_builder, needed)` (`rbuilder.py`):
///
/// ```python
/// try:
///     needed = ovfcheck(needed + ll_builder.total_size)
///     needed = ovfcheck(needed + 63) & ~63
///     total_size = ovfcheck(ll_builder.total_size + needed)
/// except OverflowError:
///     raise MemoryError
/// new_string = ll_builder.mallocfn(needed)
/// old_piece = lltype.malloc(PIECE)
/// old_piece.buf = ll_builder.current_buf
/// old_piece.prev_piece = ll_builder.extra_pieces
/// ll_builder.current_buf = new_string
/// ll_builder.current_pos = 0
/// ll_builder.current_end = needed
/// ll_builder.total_size = total_size
/// ll_builder.extra_pieces = old_piece
/// ```
///
/// Each `ovfcheck` is the trailing `int_add_ovf` of its own block, switched on
/// `c_last_exception` ([`close_ovfcheck_block`]): the no-overflow edge threads
/// forward, the OverflowError edge lands in the shared handler that raises
/// `MemoryError` (a link to the graph's `exceptblock` carrying the prebuilt
/// MemoryError class/instance). `flatten.rs` rejects an `_ovf` op in a block
/// that does not catch OverflowError, so the edges are load-bearing, not
/// decorative. `~63` is the `int_and` mask `-64`. `mallocfn` is a `direct_call`
/// callee const; `piece_struct` is `PIECE` (`STRINGPIECE`/`UNICODEPIECE`).
/// Debug-only `ll_assert` omitted. Returns `Void`.
pub fn build_ll_grow_by_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    piece_ptr_lltype: LowLevelType,
    piece_struct: LowLevelType,
    buf_lltype: LowLevelType,
    mallocfn: Constant,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, out: Hlvalue| {
        block
            .borrow_mut()
            .operations
            .push(SpaceOperation::new(opname, args, out));
    };
    let overflow_error = exception_args("OverflowError")?;

    // ── block_a (startblock): orig_total = total_size;
    //    needed1 = ovfcheck(needed + orig_total) ──
    let a_builder = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let a_needed = variable_with_lltype("needed", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(a_builder.clone()),
        Hlvalue::Variable(a_needed.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    // orig_total is read once; the 3rd ovfcheck uses the pre-update field value.
    let a_total = variable_with_lltype("total_size", LowLevelType::Signed);
    push(
        &startblock,
        "getfield",
        vec![
            Hlvalue::Variable(a_builder.clone()),
            void_field_const("total_size"),
        ],
        Hlvalue::Variable(a_total.clone()),
    );
    let a_needed1 = variable_with_lltype("needed", LowLevelType::Signed);
    push(
        &startblock,
        "int_add_ovf",
        vec![
            Hlvalue::Variable(a_needed),
            Hlvalue::Variable(a_total.clone()),
        ],
        Hlvalue::Variable(a_needed1.clone()),
    );

    // ── block_b: needed2 = ovfcheck(needed1 + 63) ──
    let b_builder = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let b_total = variable_with_lltype("total_size", LowLevelType::Signed);
    let b_needed1 = variable_with_lltype("needed", LowLevelType::Signed);
    let block_b = Block::shared(vec![
        Hlvalue::Variable(b_builder.clone()),
        Hlvalue::Variable(b_total.clone()),
        Hlvalue::Variable(b_needed1.clone()),
    ]);
    let b_needed2 = variable_with_lltype("needed", LowLevelType::Signed);
    push(
        &block_b,
        "int_add_ovf",
        vec![Hlvalue::Variable(b_needed1), signed(63)],
        Hlvalue::Variable(b_needed2.clone()),
    );

    // ── block_c: needed3 = needed2 & ~63; total_new = ovfcheck(orig_total + needed3) ──
    let c_builder = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let c_total = variable_with_lltype("total_size", LowLevelType::Signed);
    let c_needed2 = variable_with_lltype("needed", LowLevelType::Signed);
    let block_c = Block::shared(vec![
        Hlvalue::Variable(c_builder.clone()),
        Hlvalue::Variable(c_total.clone()),
        Hlvalue::Variable(c_needed2.clone()),
    ]);
    let c_needed3 = variable_with_lltype("needed", LowLevelType::Signed);
    push(
        &block_c,
        "int_and",
        vec![Hlvalue::Variable(c_needed2), signed(-64)],
        Hlvalue::Variable(c_needed3.clone()),
    );
    let c_total_new = variable_with_lltype("total_size", LowLevelType::Signed);
    push(
        &block_c,
        "int_add_ovf",
        vec![
            Hlvalue::Variable(c_total),
            Hlvalue::Variable(c_needed3.clone()),
        ],
        Hlvalue::Variable(c_total_new.clone()),
    );

    // ── block_d: mallocfn(needed), chain the old buffer into a fresh piece,
    //    install the new empty buffer ──
    let d_builder = variable_with_lltype("ll_builder", builder_ptr_lltype);
    let d_needed = variable_with_lltype("needed", LowLevelType::Signed);
    let d_total_new = variable_with_lltype("total_size", LowLevelType::Signed);
    let block_d = Block::shared(vec![
        Hlvalue::Variable(d_builder.clone()),
        Hlvalue::Variable(d_needed.clone()),
        Hlvalue::Variable(d_total_new.clone()),
    ]);
    // new_string = ll_builder.mallocfn(needed)
    let new_string = variable_with_lltype("new_string", buf_lltype.clone());
    push(
        &block_d,
        "direct_call",
        vec![
            Hlvalue::Constant(mallocfn),
            Hlvalue::Variable(d_needed.clone()),
        ],
        Hlvalue::Variable(new_string.clone()),
    );
    // old_piece = lltype.malloc(PIECE)
    let old_piece = variable_with_lltype("old_piece", piece_ptr_lltype.clone());
    push(
        &block_d,
        "malloc",
        vec![lowlevel_type_const(piece_struct), gc_flavor_const()?],
        Hlvalue::Variable(old_piece.clone()),
    );
    // old_piece.buf = ll_builder.current_buf
    let cur_buf = variable_with_lltype("current_buf", buf_lltype);
    push(
        &block_d,
        "getfield",
        vec![
            Hlvalue::Variable(d_builder.clone()),
            void_field_const("current_buf"),
        ],
        Hlvalue::Variable(cur_buf.clone()),
    );
    push(
        &block_d,
        "setfield",
        vec![
            Hlvalue::Variable(old_piece.clone()),
            void_field_const("buf"),
            Hlvalue::Variable(cur_buf),
        ],
        Hlvalue::Variable(void_result()),
    );
    // old_piece.prev_piece = ll_builder.extra_pieces
    let cur_extra = variable_with_lltype("extra_pieces", piece_ptr_lltype);
    push(
        &block_d,
        "getfield",
        vec![
            Hlvalue::Variable(d_builder.clone()),
            void_field_const("extra_pieces"),
        ],
        Hlvalue::Variable(cur_extra.clone()),
    );
    push(
        &block_d,
        "setfield",
        vec![
            Hlvalue::Variable(old_piece.clone()),
            void_field_const("prev_piece"),
            Hlvalue::Variable(cur_extra),
        ],
        Hlvalue::Variable(void_result()),
    );
    // ll_builder.current_buf = new_string; current_pos = 0; current_end = needed;
    // total_size = total_new; extra_pieces = old_piece.
    for (field, value) in [
        ("current_buf", Hlvalue::Variable(new_string)),
        ("current_pos", signed(0)),
        ("current_end", Hlvalue::Variable(d_needed)),
        ("total_size", Hlvalue::Variable(d_total_new)),
        ("extra_pieces", Hlvalue::Variable(old_piece)),
    ] {
        push(
            &block_d,
            "setfield",
            vec![
                Hlvalue::Variable(d_builder.clone()),
                void_field_const(field),
                value,
            ],
            Hlvalue::Variable(void_result()),
        );
    }
    block_d.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // ── handler: except OverflowError -> raise MemoryError ──
    // No operations; a single link to the graph's exceptblock carrying the
    // prebuilt MemoryError class/instance is the low-level `raise MemoryError`.
    let handler = Block::shared(vec![]);
    handler.closeblock(vec![
        Link::new(
            exception_args("MemoryError")?,
            Some(graph.exceptblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    // Wire the ovfcheck exits: no-overflow threads a->b->c->d; each OverflowError
    // edge lands in the shared MemoryError handler. `exc_args` is empty (≠ the
    // [class, instance] extravars) so `make_exception_link` enters the handler.
    close_ovfcheck_block(
        &startblock,
        &overflow_error,
        &block_b,
        vec![
            Hlvalue::Variable(a_builder),
            Hlvalue::Variable(a_total),
            Hlvalue::Variable(a_needed1),
        ],
        &handler,
        vec![],
    );
    close_ovfcheck_block(
        &block_b,
        &overflow_error,
        &block_c,
        vec![
            Hlvalue::Variable(b_builder),
            Hlvalue::Variable(b_total),
            Hlvalue::Variable(b_needed2),
        ],
        &handler,
        vec![],
    );
    close_ovfcheck_block(
        &block_c,
        &overflow_error,
        &block_d,
        vec![
            Hlvalue::Variable(c_builder),
            Hlvalue::Variable(c_needed3),
            Hlvalue::Variable(c_total_new),
        ],
        &handler,
        vec![],
    );

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ll_builder".to_string(), "needed".to_string()],
        func,
    ))
}

/// Synthesise `ll_append_res0(ll_builder, ll_str)` (`rbuilder.py`):
/// `_ll_append(ll_builder, ll_str, 0, len(ll_str.chars))`. `_ll_append` is
/// baked in as a `direct_call` callee const. Returns `Void`.
pub fn build_ll_append_res0_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    chars_array_ptr_lltype: LowLevelType,
    ll_append_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };

    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype);
    let ll_str = variable_with_lltype("ll_str", buf_lltype);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(ll_builder.clone()),
        Hlvalue::Variable(ll_str.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // len(ll_str.chars)
    let chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![
            Hlvalue::Variable(ll_str.clone()),
            constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void),
        ],
        Hlvalue::Variable(chars.clone()),
    ));
    let len = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(chars)],
        Hlvalue::Variable(len.clone()),
    ));
    // _ll_append(ll_builder, ll_str, 0, len)
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(ll_append_fn),
            Hlvalue::Variable(ll_builder),
            Hlvalue::Variable(ll_str),
            constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
            Hlvalue::Variable(len),
        ],
        Hlvalue::Variable(void_result()),
    ));
    startblock.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ll_builder".to_string(), "ll_str".to_string()],
        func,
    ))
}

/// Synthesise `ll_append_res_slice(ll_builder, ll_str, start, end)`
/// (`rbuilder.py`): `_ll_append(ll_builder, ll_str, start,
/// end - start)`. `_ll_append` baked in as a `direct_call` callee const.
/// Returns `Void`.
pub fn build_ll_append_res_slice_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    ll_append_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };

    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype);
    let ll_str = variable_with_lltype("ll_str", buf_lltype);
    let start = variable_with_lltype("start", LowLevelType::Signed);
    let end = variable_with_lltype("end", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(ll_builder.clone()),
        Hlvalue::Variable(ll_str.clone()),
        Hlvalue::Variable(start.clone()),
        Hlvalue::Variable(end.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // size = end - start
    let size = variable_with_lltype("size", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![Hlvalue::Variable(end), Hlvalue::Variable(start.clone())],
        Hlvalue::Variable(size.clone()),
    ));
    // _ll_append(ll_builder, ll_str, start, size)
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(ll_append_fn),
            Hlvalue::Variable(ll_builder),
            Hlvalue::Variable(ll_str),
            Hlvalue::Variable(start),
            Hlvalue::Variable(size),
        ],
        Hlvalue::Variable(void_result()),
    ));
    startblock.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "ll_builder".to_string(),
            "ll_str".to_string(),
            "start".to_string(),
            "end".to_string(),
        ],
        func,
    ))
}

/// Synthesise the `make_func_for_size(N)` residual helpers
/// (`rbuilder.py:217-228`):
///
/// ```python
/// ll_append_0_N(ll_builder, ll_str):
///     _ll_append(ll_builder, ll_str, 0, N)
/// ll_append_start_N(ll_builder, ll_str, start):
///     _ll_append(ll_builder, ll_str, start, N)
/// ```
///
/// `start_is_zero` selects the two-argument `ll_append_0_N` shape vs the
/// three-argument `ll_append_start_N` shape. `_ll_append` is baked in as a
/// `direct_call` callee const. Returns `Void`.
pub fn build_ll_append_sized_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    start_is_zero: bool,
    n: i64,
    ll_append_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let signed = |x: i64| constant_with_lltype(ConstValue::Int(x), LowLevelType::Signed);

    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype);
    let ll_str = variable_with_lltype("ll_str", buf_lltype);
    let start = if start_is_zero {
        None
    } else {
        Some(variable_with_lltype("start", LowLevelType::Signed))
    };
    let mut inputargs = vec![
        Hlvalue::Variable(ll_builder.clone()),
        Hlvalue::Variable(ll_str.clone()),
    ];
    if let Some(start) = &start {
        inputargs.push(Hlvalue::Variable(start.clone()));
    }
    let startblock = Block::shared(inputargs);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let start_arg = match start {
        Some(start) => Hlvalue::Variable(start),
        None => signed(0),
    };
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(ll_append_fn),
            Hlvalue::Variable(ll_builder),
            Hlvalue::Variable(ll_str),
            start_arg,
            signed(n),
        ],
        Hlvalue::Variable(void_result()),
    ));
    startblock.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    let argnames = if start_is_zero {
        vec!["ll_builder".to_string(), "ll_str".to_string()]
    } else {
        vec![
            "ll_builder".to_string(),
            "ll_str".to_string(),
            "start".to_string(),
        ]
    };
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

/// Synthesise `_ll_append(ll_builder, ll_str, start, size)`
/// (`rbuilder.py:80-89`):
///
/// ```python
/// pos = ll_builder.current_pos
/// end = ll_builder.current_end
/// if (end - pos) < size:
///     ll_grow_and_append(ll_builder, ll_str, start, size)
/// else:
///     ll_builder.current_pos = pos + size
///     ll_builder.copy_string_contents(ll_str, ll_builder.current_buf,
///                                     start, pos, size)
/// ```
///
/// `ll_grow_and_append` / `copy_string_contents` are baked in as
/// `direct_call` callee consts. `buf_lltype` = `STRPTR`/`UNICODEPTR`
/// (both `ll_str` and `current_buf`). Returns `Void`.
pub fn build_ll__ll_append_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    grow_and_append_fn: Constant,
    copy_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let bool_case = |b: bool| {
        Some(constant_with_lltype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };

    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let ll_str = variable_with_lltype("ll_str", buf_lltype.clone());
    let start = variable_with_lltype("start", LowLevelType::Signed);
    let size = variable_with_lltype("size", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(ll_builder.clone()),
        Hlvalue::Variable(ll_str.clone()),
        Hlvalue::Variable(start.clone()),
        Hlvalue::Variable(size.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // grow arm: ll_grow_and_append(ll_builder, ll_str, start, size).
    let g_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let g_str = variable_with_lltype("ll_str", buf_lltype.clone());
    let g_start = variable_with_lltype("start", LowLevelType::Signed);
    let g_size = variable_with_lltype("size", LowLevelType::Signed);
    let block_grow = Block::shared(vec![
        Hlvalue::Variable(g_llb.clone()),
        Hlvalue::Variable(g_str.clone()),
        Hlvalue::Variable(g_start.clone()),
        Hlvalue::Variable(g_size.clone()),
    ]);
    block_grow.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(grow_and_append_fn),
            Hlvalue::Variable(g_llb),
            Hlvalue::Variable(g_str),
            Hlvalue::Variable(g_start),
            Hlvalue::Variable(g_size),
        ],
        Hlvalue::Variable(void_result()),
    ));
    block_grow.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // copy arm: current_pos = pos + size; copy_string_contents(...).
    let c_llb = variable_with_lltype("ll_builder", builder_ptr_lltype);
    let c_str = variable_with_lltype("ll_str", buf_lltype.clone());
    let c_start = variable_with_lltype("start", LowLevelType::Signed);
    let c_size = variable_with_lltype("size", LowLevelType::Signed);
    let c_pos = variable_with_lltype("pos", LowLevelType::Signed);
    let block_copy = Block::shared(vec![
        Hlvalue::Variable(c_llb.clone()),
        Hlvalue::Variable(c_str.clone()),
        Hlvalue::Variable(c_start.clone()),
        Hlvalue::Variable(c_size.clone()),
        Hlvalue::Variable(c_pos.clone()),
    ]);
    let c_newpos = variable_with_lltype("newpos", LowLevelType::Signed);
    block_copy.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![
            Hlvalue::Variable(c_pos.clone()),
            Hlvalue::Variable(c_size.clone()),
        ],
        Hlvalue::Variable(c_newpos.clone()),
    ));
    block_copy.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(c_llb.clone()),
            void_field_const("current_pos"),
            Hlvalue::Variable(c_newpos),
        ],
        Hlvalue::Variable(void_result()),
    ));
    let c_buf = variable_with_lltype("current_buf", buf_lltype);
    block_copy.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(c_llb), void_field_const("current_buf")],
        Hlvalue::Variable(c_buf.clone()),
    ));
    block_copy.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(copy_fn),
            Hlvalue::Variable(c_str),
            Hlvalue::Variable(c_buf),
            Hlvalue::Variable(c_start),
            Hlvalue::Variable(c_pos),
            Hlvalue::Variable(c_size),
        ],
        Hlvalue::Variable(void_result()),
    ));
    block_copy.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // startblock: pos/end read; branch on (end - pos) < size.
    let pos = variable_with_lltype("pos", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(pos.clone()),
    ));
    let end = variable_with_lltype("end", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_end"),
        ],
        Hlvalue::Variable(end.clone()),
    ));
    let avail = variable_with_lltype("avail", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![Hlvalue::Variable(end), Hlvalue::Variable(pos.clone())],
        Hlvalue::Variable(avail.clone()),
    ));
    let too_small = variable_with_lltype("too_small", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(avail), Hlvalue::Variable(size.clone())],
        Hlvalue::Variable(too_small.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(too_small));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(ll_builder.clone()),
                Hlvalue::Variable(ll_str.clone()),
                Hlvalue::Variable(start.clone()),
                Hlvalue::Variable(size.clone()),
            ],
            Some(block_grow),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(ll_builder),
                Hlvalue::Variable(ll_str),
                Hlvalue::Variable(start),
                Hlvalue::Variable(size),
                Hlvalue::Variable(pos),
            ],
            Some(block_copy),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "ll_builder".to_string(),
            "ll_str".to_string(),
            "start".to_string(),
            "size".to_string(),
        ],
        func,
    ))
}

/// Synthesise `ll_fold_pieces(ll_builder)` (`rbuilder.py`):
///
/// ```python
/// final_size = BaseStringBuilderRepr.ll_getlength(ll_builder)
/// extra = ll_builder.extra_pieces
/// ll_builder.extra_pieces = lltype.nullptr(...)
/// if ll_builder.current_pos == 0 and not extra.prev_piece:   # fast path
///     piece = extra.buf
///     ll_builder.total_size = final_size
///     ll_builder.current_buf = piece
///     ll_builder.current_pos = final_size
///     ll_builder.current_end = final_size
///     return
/// result = ll_builder.mallocfn(final_size)
/// piece = ll_builder.current_buf
/// piece_lgt = ll_builder.current_pos
/// ll_builder.total_size = final_size
/// ll_builder.current_buf = result
/// ll_builder.current_pos = final_size
/// ll_builder.current_end = final_size
/// dst = final_size
/// while True:
///     dst -= piece_lgt
///     ll_builder.copy_string_contents(piece, result, 0, dst, piece_lgt)
///     if not extra:
///         break
///     piece = extra.buf
///     piece_lgt = len(piece.chars)
///     extra = extra.prev_piece
/// ```
///
/// `ll_getlength` / `mallocfn` / `copy_string_contents` are baked in as
/// `direct_call` callee consts. The short-circuit `and` splits the header
/// into a `current_pos == 0` test then a `not extra.prev_piece` test; both
/// `not <ptr>` conditions use `ptr_nonzero` with the null case on the False
/// arm (the ported string-helper convention). Debug-only `ll_assert`s
/// omitted. `piece_ptr_lltype` = `STRINGPIECEPTR`/`UNICODEPIECEPTR`,
/// `buf_lltype` = `STRPTR`/`UNICODEPTR`, `chars_array_ptr_lltype` the
/// `getsubstruct('chars')` interior array pointer. Returns `Void`.
#[allow(clippy::too_many_arguments)]
pub fn build_ll_fold_pieces_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    piece_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    chars_array_ptr_lltype: LowLevelType,
    getlength_fn: Constant,
    mallocfn: Constant,
    copy_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let bool_case = |b: bool| {
        Some(constant_with_lltype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };
    let void_result = || variable_with_lltype("v", LowLevelType::Void);
    let none_const = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    };
    let signed_zero = || constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed);

    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(ll_builder.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Loop blocks (created first so the back-edge can be wired)
    // block_loop(dst, piece, piece_lgt, extra, result)
    let l_dst = variable_with_lltype("dst", LowLevelType::Signed);
    let l_piece = variable_with_lltype("piece", buf_lltype.clone());
    let l_piece_lgt = variable_with_lltype("piece_lgt", LowLevelType::Signed);
    let l_extra = variable_with_lltype("extra", piece_ptr_lltype.clone());
    let l_result = variable_with_lltype("result", buf_lltype.clone());
    let block_loop = Block::shared(vec![
        Hlvalue::Variable(l_dst.clone()),
        Hlvalue::Variable(l_piece.clone()),
        Hlvalue::Variable(l_piece_lgt.clone()),
        Hlvalue::Variable(l_extra.clone()),
        Hlvalue::Variable(l_result.clone()),
    ]);
    // block_loop_next(dst, extra, result)
    let n_dst = variable_with_lltype("dst", LowLevelType::Signed);
    let n_extra = variable_with_lltype("extra", piece_ptr_lltype.clone());
    let n_result = variable_with_lltype("result", buf_lltype.clone());
    let block_loop_next = Block::shared(vec![
        Hlvalue::Variable(n_dst.clone()),
        Hlvalue::Variable(n_extra.clone()),
        Hlvalue::Variable(n_result.clone()),
    ]);

    // block_loop_next: piece = extra.buf; piece_lgt = len(piece.chars);
    //                  extra = extra.prev_piece; jump back to block_loop.
    let n_piece = variable_with_lltype("piece", buf_lltype.clone());
    block_loop_next
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(n_extra.clone()), void_field_const("buf")],
            Hlvalue::Variable(n_piece.clone()),
        ));
    let n_chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    block_loop_next
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getsubstruct",
            vec![
                Hlvalue::Variable(n_piece.clone()),
                constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void),
            ],
            Hlvalue::Variable(n_chars.clone()),
        ));
    let n_piece_lgt = variable_with_lltype("piece_lgt", LowLevelType::Signed);
    block_loop_next
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(n_chars)],
            Hlvalue::Variable(n_piece_lgt.clone()),
        ));
    let n_extra2 = variable_with_lltype("extra", piece_ptr_lltype.clone());
    block_loop_next
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(n_extra), void_field_const("prev_piece")],
            Hlvalue::Variable(n_extra2.clone()),
        ));
    block_loop_next.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(n_dst),
                Hlvalue::Variable(n_piece),
                Hlvalue::Variable(n_piece_lgt),
                Hlvalue::Variable(n_extra2),
                Hlvalue::Variable(n_result),
            ],
            Some(block_loop.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_loop: dst -= piece_lgt; copy_string_contents(piece, result, 0,
    //             dst, piece_lgt); if not extra: break.
    let l_dst2 = variable_with_lltype("dst", LowLevelType::Signed);
    block_loop.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![
            Hlvalue::Variable(l_dst),
            Hlvalue::Variable(l_piece_lgt.clone()),
        ],
        Hlvalue::Variable(l_dst2.clone()),
    ));
    block_loop.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(copy_fn),
            Hlvalue::Variable(l_piece),
            Hlvalue::Variable(l_result.clone()),
            signed_zero(),
            Hlvalue::Variable(l_dst2.clone()),
            Hlvalue::Variable(l_piece_lgt),
        ],
        Hlvalue::Variable(void_result()),
    ));
    let l_has_extra = variable_with_lltype("has_extra", LowLevelType::Bool);
    block_loop.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_nonzero",
        vec![Hlvalue::Variable(l_extra.clone())],
        Hlvalue::Variable(l_has_extra.clone()),
    ));
    block_loop.borrow_mut().exitswitch = Some(Hlvalue::Variable(l_has_extra));
    block_loop.closeblock(vec![
        // extra non-null -> continue.
        Link::new(
            vec![
                Hlvalue::Variable(l_dst2.clone()),
                Hlvalue::Variable(l_extra),
                Hlvalue::Variable(l_result),
            ],
            Some(block_loop_next),
            bool_case(true),
        )
        .into_ref(),
        // not extra -> break -> return.
        Link::new(
            vec![none_const()],
            Some(graph.returnblock.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    // block_fast(ll_builder, extra, final_size)
    let f_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let f_extra = variable_with_lltype("extra", piece_ptr_lltype.clone());
    let f_fs = variable_with_lltype("final_size", LowLevelType::Signed);
    let block_fast = Block::shared(vec![
        Hlvalue::Variable(f_llb.clone()),
        Hlvalue::Variable(f_extra.clone()),
        Hlvalue::Variable(f_fs.clone()),
    ]);
    let f_piece = variable_with_lltype("piece", buf_lltype.clone());
    block_fast.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(f_extra), void_field_const("buf")],
        Hlvalue::Variable(f_piece.clone()),
    ));
    for (field, value) in [
        ("total_size", Hlvalue::Variable(f_fs.clone())),
        ("current_buf", Hlvalue::Variable(f_piece)),
        ("current_pos", Hlvalue::Variable(f_fs.clone())),
        ("current_end", Hlvalue::Variable(f_fs)),
    ] {
        block_fast.borrow_mut().operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(f_llb.clone()),
                void_field_const(field),
                value,
            ],
            Hlvalue::Variable(void_result()),
        ));
    }
    block_fast.closeblock(vec![
        Link::new(vec![none_const()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // block_slow(ll_builder, extra, final_size)
    let s_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let s_extra = variable_with_lltype("extra", piece_ptr_lltype.clone());
    let s_fs = variable_with_lltype("final_size", LowLevelType::Signed);
    let block_slow = Block::shared(vec![
        Hlvalue::Variable(s_llb.clone()),
        Hlvalue::Variable(s_extra.clone()),
        Hlvalue::Variable(s_fs.clone()),
    ]);
    let s_result = variable_with_lltype("result", buf_lltype.clone());
    block_slow.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![Hlvalue::Constant(mallocfn), Hlvalue::Variable(s_fs.clone())],
        Hlvalue::Variable(s_result.clone()),
    ));
    let s_piece = variable_with_lltype("piece", buf_lltype);
    block_slow.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(s_llb.clone()),
            void_field_const("current_buf"),
        ],
        Hlvalue::Variable(s_piece.clone()),
    ));
    let s_piece_lgt = variable_with_lltype("piece_lgt", LowLevelType::Signed);
    block_slow.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(s_llb.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(s_piece_lgt.clone()),
    ));
    for (field, value) in [
        ("total_size", Hlvalue::Variable(s_fs.clone())),
        ("current_buf", Hlvalue::Variable(s_result.clone())),
        ("current_pos", Hlvalue::Variable(s_fs.clone())),
        ("current_end", Hlvalue::Variable(s_fs.clone())),
    ] {
        block_slow.borrow_mut().operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(s_llb.clone()),
                void_field_const(field),
                value,
            ],
            Hlvalue::Variable(void_result()),
        ));
    }
    block_slow.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(s_fs), // dst = final_size
                Hlvalue::Variable(s_piece),
                Hlvalue::Variable(s_piece_lgt),
                Hlvalue::Variable(s_extra),
                Hlvalue::Variable(s_result),
            ],
            Some(block_loop),
            None,
        )
        .into_ref(),
    ]);

    // block_cond2(ll_builder, extra, final_size): not extra.prev_piece
    let c_llb = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let c_extra = variable_with_lltype("extra", piece_ptr_lltype.clone());
    let c_fs = variable_with_lltype("final_size", LowLevelType::Signed);
    let block_cond2 = Block::shared(vec![
        Hlvalue::Variable(c_llb.clone()),
        Hlvalue::Variable(c_extra.clone()),
        Hlvalue::Variable(c_fs.clone()),
    ]);
    let c_prev = variable_with_lltype("prev_piece", piece_ptr_lltype.clone());
    block_cond2
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(c_extra.clone()),
                void_field_const("prev_piece"),
            ],
            Hlvalue::Variable(c_prev.clone()),
        ));
    let c_has_prev = variable_with_lltype("has_prev", LowLevelType::Bool);
    block_cond2
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "ptr_nonzero",
            vec![Hlvalue::Variable(c_prev)],
            Hlvalue::Variable(c_has_prev.clone()),
        ));
    block_cond2.borrow_mut().exitswitch = Some(Hlvalue::Variable(c_has_prev));
    block_cond2.closeblock(vec![
        // prev_piece non-null -> slow path.
        Link::new(
            vec![
                Hlvalue::Variable(c_llb.clone()),
                Hlvalue::Variable(c_extra.clone()),
                Hlvalue::Variable(c_fs.clone()),
            ],
            Some(block_slow.clone()),
            bool_case(true),
        )
        .into_ref(),
        // not prev_piece -> fast path.
        Link::new(
            vec![
                Hlvalue::Variable(c_llb),
                Hlvalue::Variable(c_extra),
                Hlvalue::Variable(c_fs),
            ],
            Some(block_fast),
            bool_case(false),
        )
        .into_ref(),
    ]);

    // startblock: header
    let fs = variable_with_lltype("final_size", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(getlength_fn),
            Hlvalue::Variable(ll_builder.clone()),
        ],
        Hlvalue::Variable(fs.clone()),
    ));
    let extra = variable_with_lltype("extra", piece_ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("extra_pieces"),
        ],
        Hlvalue::Variable(extra.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("extra_pieces"),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::None,
                piece_ptr_lltype,
            )),
        ],
        Hlvalue::Variable(void_result()),
    ));
    let pos0 = variable_with_lltype("current_pos", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(pos0.clone()),
    ));
    let is_empty = variable_with_lltype("is_empty", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![Hlvalue::Variable(pos0), signed_zero()],
        Hlvalue::Variable(is_empty.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_empty));
    startblock.closeblock(vec![
        // current_pos == 0 -> evaluate `not extra.prev_piece`.
        Link::new(
            vec![
                Hlvalue::Variable(ll_builder.clone()),
                Hlvalue::Variable(extra.clone()),
                Hlvalue::Variable(fs.clone()),
            ],
            Some(block_cond2),
            bool_case(true),
        )
        .into_ref(),
        // current_pos != 0 -> slow path.
        Link::new(
            vec![
                Hlvalue::Variable(ll_builder),
                Hlvalue::Variable(extra),
                Hlvalue::Variable(fs),
            ],
            Some(block_slow),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ll_builder".to_string()],
        func,
    ))
}

/// Synthesise `ll_build(ll_builder)` (`rbuilder.py`):
///
/// ```python
/// if ll_builder.extra_pieces:
///     ll_fold_pieces(ll_builder)
/// elif ll_builder.current_pos != ll_builder.total_size:
///     ll_shrink_final(ll_builder)
/// return ll_builder.current_buf
/// ```
///
/// `ll_fold_pieces` / `ll_shrink_final` are baked in as `direct_call`
/// callee consts. The three arms merge into a returnblock-feeding tail
/// that reads `current_buf` (`buf_lltype` = `STRPTR`/`UNICODEPTR`).
pub fn build_ll_build_helper_graph(
    name: &str,
    builder_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    fold_pieces_fn: Constant,
    shrink_final_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let bool_case = |b: bool| {
        Some(constant_with_lltype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };
    let void_call_result = || variable_with_lltype("v", LowLevelType::Void);

    let ll_builder = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(ll_builder.clone())]);
    let return_var = variable_with_lltype("result", buf_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Tail merge block: return ll_builder.current_buf.
    let llb_ret = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let block_ret = Block::shared(vec![Hlvalue::Variable(llb_ret.clone())]);
    let buf = variable_with_lltype("buf", buf_lltype);
    block_ret.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(llb_ret), void_field_const("current_buf")],
        Hlvalue::Variable(buf.clone()),
    ));
    block_ret.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(buf)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    // fold arm: ll_fold_pieces(ll_builder).
    let llb_fold = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let block_fold = Block::shared(vec![Hlvalue::Variable(llb_fold.clone())]);
    block_fold.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(fold_pieces_fn),
            Hlvalue::Variable(llb_fold.clone()),
        ],
        Hlvalue::Variable(void_call_result()),
    ));
    block_fold.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(llb_fold)],
            Some(block_ret.clone()),
            None,
        )
        .into_ref(),
    ]);

    // shrink arm: ll_shrink_final(ll_builder).
    let llb_shrink = variable_with_lltype("ll_builder", builder_ptr_lltype.clone());
    let block_shrink = Block::shared(vec![Hlvalue::Variable(llb_shrink.clone())]);
    block_shrink
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(shrink_final_fn),
                Hlvalue::Variable(llb_shrink.clone()),
            ],
            Hlvalue::Variable(void_call_result()),
        ));
    block_shrink.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(llb_shrink)],
            Some(block_ret.clone()),
            None,
        )
        .into_ref(),
    ]);

    // elif block: current_pos != total_size.
    let llb_elif = variable_with_lltype("ll_builder", builder_ptr_lltype);
    let block_elif = Block::shared(vec![Hlvalue::Variable(llb_elif.clone())]);
    let pos = variable_with_lltype("current_pos", LowLevelType::Signed);
    block_elif.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(llb_elif.clone()),
            void_field_const("current_pos"),
        ],
        Hlvalue::Variable(pos.clone()),
    ));
    let tot = variable_with_lltype("total_size", LowLevelType::Signed);
    block_elif.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(llb_elif.clone()),
            void_field_const("total_size"),
        ],
        Hlvalue::Variable(tot.clone()),
    ));
    let needs_shrink = variable_with_lltype("needs_shrink", LowLevelType::Bool);
    block_elif.borrow_mut().operations.push(SpaceOperation::new(
        "int_ne",
        vec![Hlvalue::Variable(pos), Hlvalue::Variable(tot)],
        Hlvalue::Variable(needs_shrink.clone()),
    ));
    block_elif.borrow_mut().exitswitch = Some(Hlvalue::Variable(needs_shrink));
    block_elif.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(llb_elif.clone())],
            Some(block_shrink),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(llb_elif)],
            Some(block_ret),
            bool_case(false),
        )
        .into_ref(),
    ]);

    // start block: if ll_builder.extra_pieces.
    let extra = variable_with_lltype("extra", STRINGPIECEPTR.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(ll_builder.clone()),
            void_field_const("extra_pieces"),
        ],
        Hlvalue::Variable(extra.clone()),
    ));
    let has_extra = variable_with_lltype("has_extra", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_nonzero",
        vec![Hlvalue::Variable(extra)],
        Hlvalue::Variable(has_extra.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(has_extra));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(ll_builder.clone())],
            Some(block_fold),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(ll_builder)],
            Some(block_elif),
            bool_case(false),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ll_builder".to_string()],
        func,
    ))
}

/// `Ptr(GcArray(ITEM))` for an item lltype — the array pointer shape both
/// `ll_arrayclear` and `ll_arrayfill` operate on (`p` is the array itself:
/// `len(p)`, `p[i]`, `typeOf(p).TO`).
fn array_ptr_lltype(item_lltype: &LowLevelType) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(Array::gc(item_lltype.clone())),
    }))
}

/// Synthesise `rgc.ll_arrayclear(p)` (`rpython/rlib/rgc.py`):
///
/// ```python
/// def ll_arrayclear(p):
///     length = len(p)
///     ARRAY = typeOf(p).TO
///     if must_split_gc_address_space():
///         i = 0
///         if isinstance(ARRAY.OF, Ptr):
///             while i < length:
///                 p[i] = nullptr(ARRAY.OF.TO)
///                 i += 1
///         else:
///             ZERO = rffi.cast(ARRAY.OF, 0)
///             while i < length:
///                 p[i] = ZERO
///                 i += 1
///     else:
///         offset = itemoffsetof(ARRAY, 0)
///         dest_addr = cast_ptr_to_adr(p) + offset
///         raw_memclear(dest_addr, sizeof(ARRAY.OF) * length)
///     keepalive_until_here(p)
/// ```
///
/// Three-block loop over the bare `Ptr(GcArray(ITEM))` operand: writes the
/// null/zero element into every slot. Writing NULL never creates an
/// old->young reference, so no write barrier is emitted. `@specialize.ll()`
/// picks the two `isinstance(ARRAY.OF, Ptr)` arms per concrete item type;
/// the Rust-side match on `item_lltype` folds that specialisation to one
/// `ZERO` element constant (`nullptr` for a `Ptr` item, `0` otherwise). The
/// `raw_memclear` fast path (the `must_split_gc_address_space()` `else`
/// branch) is deferred as a translation-time concern, mirroring
/// [`build_ll_arraycopy_helper_graph`](../../rlist.rs) which lowers the
/// sibling `rgc.ll_arraycopy` to the same bare element loop and defers its
/// `raw_memcopy` fast path (`rgc.py`); the `itemoffsetof` / `sizeof`
/// / `cast_ptr_to_adr` / `raw_memclear` address machinery has no
/// construction-side surface here.
pub fn build_ll_arrayclear_helper_graph(
    name: &str,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let items_ptr = array_ptr_lltype(&item_lltype);
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_case = |b: bool| {
        Some(constant_with_lltype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };
    let none_void = || constant_with_lltype(ConstValue::None, LowLevelType::Void);

    // `ZERO`: the null pointer for a `Ptr` item, else the zero primitive.
    let zero_value = match &item_lltype {
        LowLevelType::Ptr(_) => constant_with_lltype(ConstValue::None, item_lltype.clone()),
        LowLevelType::Float | LowLevelType::SingleFloat | LowLevelType::LongFloat => {
            constant_with_lltype(ConstValue::float(0.0), item_lltype.clone())
        }
        LowLevelType::Unsigned | LowLevelType::UnsignedLongLong => {
            constant_with_lltype(ConstValue::UInt(0), item_lltype.clone())
        }
        LowLevelType::UnsignedLongLongLong => {
            constant_with_lltype(ConstValue::UInt128(0), item_lltype.clone())
        }
        LowLevelType::SignedLongLongLong => {
            constant_with_lltype(ConstValue::Int128(0), item_lltype.clone())
        }
        _ => constant_with_lltype(ConstValue::Int(0), item_lltype.clone()),
    };

    let p = variable_with_lltype("p", items_ptr.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(p.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Loop blocks carry (p, length, i) as fresh inputargs.
    let p_c = variable_with_lltype("p", items_ptr.clone());
    let len_c = variable_with_lltype("length", LowLevelType::Signed);
    let i_c = variable_with_lltype("i", LowLevelType::Signed);
    let block_cond = Block::shared(vec![
        Hlvalue::Variable(p_c.clone()),
        Hlvalue::Variable(len_c.clone()),
        Hlvalue::Variable(i_c.clone()),
    ]);

    let p_b = variable_with_lltype("p", items_ptr.clone());
    let len_b = variable_with_lltype("length", LowLevelType::Signed);
    let i_b = variable_with_lltype("i", LowLevelType::Signed);
    let block_body = Block::shared(vec![
        Hlvalue::Variable(p_b.clone()),
        Hlvalue::Variable(len_b.clone()),
        Hlvalue::Variable(i_b.clone()),
    ]);

    // startblock: length = len(p); i = 0.
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(p.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(p), Hlvalue::Variable(length), signed(0)],
            Some(block_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_cond: int_lt(i, length). True -> body; False -> return None.
    let cond = variable_with_lltype("cond", LowLevelType::Bool);
    block_cond.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![
            Hlvalue::Variable(i_c.clone()),
            Hlvalue::Variable(len_c.clone()),
        ],
        Hlvalue::Variable(cond.clone()),
    ));
    block_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
    block_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(p_c),
                Hlvalue::Variable(len_c),
                Hlvalue::Variable(i_c),
            ],
            Some(block_body.clone()),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![none_void()],
            Some(graph.returnblock.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    // block_body: p[i] = ZERO; i += 1.
    let store_void = variable_with_lltype("v", LowLevelType::Void);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "setarrayitem",
        vec![
            Hlvalue::Variable(p_b.clone()),
            Hlvalue::Variable(i_b.clone()),
            zero_value,
        ],
        Hlvalue::Variable(store_void),
    ));
    let i_next = variable_with_lltype("i", LowLevelType::Signed);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![Hlvalue::Variable(i_b), signed(1)],
        Hlvalue::Variable(i_next.clone()),
    ));
    block_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(p_b),
                Hlvalue::Variable(len_b),
                Hlvalue::Variable(i_next),
            ],
            Some(block_cond),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["p".to_string()],
        func,
    ))
}

/// Synthesise `rgc.ll_arrayfill(p, item)` (`rpython/rlib/rgc.py`,
/// `@jit.dont_look_inside @specialize.ll()`):
///
/// ```python
/// def ll_arrayfill(p, item):
///     i = 0
///     length = len(p)
///     if we_are_translated():
///         llop.gc_writebarrier(Void, p)
///         while i < length:
///             llop.bare_setarrayitem(Void, p, i, item)
///             i += 1
///     else:
///         while i < length:
///             p[i] = item
///             i += 1
///     keepalive_until_here(p)
/// ```
///
/// A lifted low-level helper is part of the translated program, so
/// `we_are_translated()` folds to the translated arm (as
/// [`build_ll_mallocstr_helper_graph`](../rstr.rs) resolves the same
/// predicate): one `gc_writebarrier(p)` before the loop covers every slot
/// (if `p` is old it is added to `old_objects_pointing_to_young`; if young
/// the barrier is a no-op), then the loop stores `item` into each slot via
/// `bare_setarrayitem`. Three-block loop over the bare `Ptr(GcArray(ITEM))`
/// operand.
pub fn build_ll_arrayfill_helper_graph(
    name: &str,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let items_ptr = array_ptr_lltype(&item_lltype);
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let bool_case = |b: bool| {
        Some(constant_with_lltype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };
    let none_void = || constant_with_lltype(ConstValue::None, LowLevelType::Void);

    let p = variable_with_lltype("p", items_ptr.clone());
    let item = variable_with_lltype("item", item_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(p.clone()),
        Hlvalue::Variable(item.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Loop blocks carry (p, item, length, i) as fresh inputargs.
    let p_c = variable_with_lltype("p", items_ptr.clone());
    let item_c = variable_with_lltype("item", item_lltype.clone());
    let len_c = variable_with_lltype("length", LowLevelType::Signed);
    let i_c = variable_with_lltype("i", LowLevelType::Signed);
    let block_cond = Block::shared(vec![
        Hlvalue::Variable(p_c.clone()),
        Hlvalue::Variable(item_c.clone()),
        Hlvalue::Variable(len_c.clone()),
        Hlvalue::Variable(i_c.clone()),
    ]);

    let p_b = variable_with_lltype("p", items_ptr.clone());
    let item_b = variable_with_lltype("item", item_lltype.clone());
    let len_b = variable_with_lltype("length", LowLevelType::Signed);
    let i_b = variable_with_lltype("i", LowLevelType::Signed);
    let block_body = Block::shared(vec![
        Hlvalue::Variable(p_b.clone()),
        Hlvalue::Variable(item_b.clone()),
        Hlvalue::Variable(len_b.clone()),
        Hlvalue::Variable(i_b.clone()),
    ]);

    // startblock: length = len(p); gc_writebarrier(p); i = 0.
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(p.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let wb_void = variable_with_lltype("v", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "gc_writebarrier",
        vec![Hlvalue::Variable(p.clone())],
        Hlvalue::Variable(wb_void),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(p),
                Hlvalue::Variable(item),
                Hlvalue::Variable(length),
                signed(0),
            ],
            Some(block_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_cond: int_lt(i, length). True -> body; False -> return None.
    let cond = variable_with_lltype("cond", LowLevelType::Bool);
    block_cond.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![
            Hlvalue::Variable(i_c.clone()),
            Hlvalue::Variable(len_c.clone()),
        ],
        Hlvalue::Variable(cond.clone()),
    ));
    block_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
    block_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(p_c),
                Hlvalue::Variable(item_c),
                Hlvalue::Variable(len_c),
                Hlvalue::Variable(i_c),
            ],
            Some(block_body.clone()),
            bool_case(true),
        )
        .into_ref(),
        Link::new(
            vec![none_void()],
            Some(graph.returnblock.clone()),
            bool_case(false),
        )
        .into_ref(),
    ]);

    // block_body: bare_setarrayitem(p, i, item); i += 1.
    let store_void = variable_with_lltype("v", LowLevelType::Void);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "bare_setarrayitem",
        vec![
            Hlvalue::Variable(p_b.clone()),
            Hlvalue::Variable(i_b.clone()),
            Hlvalue::Variable(item_b.clone()),
        ],
        Hlvalue::Variable(store_void),
    ));
    let i_next = variable_with_lltype("i", LowLevelType::Signed);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![Hlvalue::Variable(i_b), signed(1)],
        Hlvalue::Variable(i_next.clone()),
    ));
    block_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(p_b),
                Hlvalue::Variable(item_b),
                Hlvalue::Variable(len_b),
                Hlvalue::Variable(i_next),
            ],
            Some(block_cond),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["p".to_string(), "item".to_string()],
        func,
    ))
}

/// RPython `class BaseStringBuilderRepr(AbstractStringBuilderRepr)`.
#[derive(Debug, Default)]
pub struct BaseStringBuilderRepr;

/// RPython `class StringBuilderRepr(BaseStringBuilderRepr)`.
#[derive(Debug)]
pub struct StringBuilderRepr {
    state: ReprState,
}

impl StringBuilderRepr {
    pub fn new() -> Self {
        StringBuilderRepr {
            state: ReprState::new(),
        }
    }

    /// RPython `StringBuilderRepr.basetp = STR`.
    pub fn basetp(&self) -> &'static LowLevelType {
        &crate::translator::rtyper::lltypesystem::rstr::STR
    }

    /// RPython `AbstractStringBuilderRepr.rtyper_new(self, hop)`
    /// (`rtyper/rbuilder.py`), inherited by `StringBuilderRepr`.
    pub fn rtyper_new(&self, hop: &HighLevelOp) -> RTypeResult {
        rtype_builder_new(
            hop,
            STRINGBUILDERPTR.clone(),
            STRINGBUILDER.clone(),
            STRPTR.clone(),
            "mallocstr",
        )
    }
}

/// RPython `BaseStringBuilderRepr.rtype_method_getlength(self, hop)`
/// (`rbuilder.py:42-45`):
///
/// ```python
/// def rtype_method_getlength(self, hop):
///     vlist = hop.inputargs(self)
///     hop.exception_cannot_occur()
///     return hop.gendirectcall(self.ll_getlength, *vlist)
/// ```
///
/// Shared body for the StringBuilderRepr / UnicodeBuilderRepr impls —
/// both bind the same `ll_getlength` container-length lowering, only the
/// builder pointer lltype (`Ptr(STRINGBUILDER)` vs `Ptr(UNICODEBUILDER)`)
/// differs, which the helper-graph cache key keeps distinct.
fn rtype_builder_getlength(
    self_repr: &dyn Repr,
    hop: &HighLevelOp,
    builder_ptr_lltype: LowLevelType,
) -> RTypeResult {
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(self_repr)])?;
    hop.exception_cannot_occur()?;
    let ptr_for_builder = builder_ptr_lltype.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        "ll_getlength".to_string(),
        vec![builder_ptr_lltype],
        LowLevelType::Signed,
        move |_rtyper, _args, _result| {
            build_ll_getlength_helper_graph("ll_getlength", ptr_for_builder)
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

/// RPython `AbstractStringBuilderRepr.rtype_method_append(self, hop)`
/// (`rbuilder.py`):
///
/// ```python
/// def rtype_method_append(self, hop):
///     vlist = hop.inputargs(self, rstr.string_repr)
///     hop.exception_cannot_occur()
///     return hop.gendirectcall(self.ll_append, *vlist)
/// ```
///
/// `self.ll_append` resolves to the O(n) `build_ll_append_helper_graph`
/// tree: a `we_are_jitted()` branch whose jit arm delegates to
/// `ll_append_res0` and whose no-jit arm inlines the same
/// `_ll_append(builder, str, 0, len)`. Both funnel into `_ll_append`,
/// which either copies into the current buffer or, when it would
/// overflow, calls `ll_grow_and_append` (which chains the filled buffer
/// into a fresh `STRINGPIECE` via `ll_grow_by` + `mallocfn`). The whole
/// callee closure — `copy_string_contents`, `mallocfn`, `ll_grow_by`,
/// `ll_grow_and_append`, `_ll_append`, `ll_append_res0` — is materialised
/// here and baked in as `direct_call` callee consts by [`functionptr_const`],
/// mirroring the `ll_min` / `mallocstr` consts of [`rtype_builder_new`].
/// `piece_ptr_lltype` / `piece_struct` are `STRINGPIECEPTR` / `STRINGPIECE`
/// (or the UNICODE pair); `malloc_name` and `copy_opname` select the
/// STR (`mallocstr` / `copystrcontent`) vs UNICODE variants.
#[allow(clippy::too_many_arguments)]
fn rtype_builder_append(
    self_repr: &dyn Repr,
    string_repr: &dyn Repr,
    hop: &HighLevelOp,
    builder_ptr_lltype: LowLevelType,
    piece_ptr_lltype: LowLevelType,
    piece_struct: LowLevelType,
    buf_lltype: LowLevelType,
    malloc_name: &'static str,
    copy_opname: &'static str,
) -> RTypeResult {
    use crate::translator::rtyper::lltypesystem::rstr::{
        build_ll_mallocstr_helper_graph, chars_array_ptr_lltype_from_strptr,
    };
    // The appended piece is a string; coerce it through `string_repr`
    // (lowleveltype == STRPTR == buf_lltype) so `ll_append`'s second
    // param receives an STRPTR, mirroring `rtype_method_append`'s
    // `hop.inputargs(self, rstr.string_repr)`.
    let vlist = hop.inputargs(vec![
        ConvertedTo::Repr(self_repr),
        ConvertedTo::Repr(string_repr),
    ])?;
    hop.exception_cannot_occur()?;

    let outer_builder = builder_ptr_lltype.clone();
    let outer_buf = buf_lltype.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        "ll_append".to_string(),
        vec![builder_ptr_lltype, buf_lltype],
        LowLevelType::Void,
        move |rtyper, _args, _result| {
            let chars_ptr = chars_array_ptr_lltype_from_strptr(&outer_buf)?;

            // copy_fn = copy_string_contents(src, dst, srcstart, dststart, length)
            let copy_buf = outer_buf.clone();
            let copy = rtyper.lowlevel_helper_function_with_builder(
                "ll_copy_string_contents".to_string(),
                vec![
                    outer_buf.clone(),
                    outer_buf.clone(),
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                ],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    build_ll_copy_string_contents_helper_graph(
                        "ll_copy_string_contents",
                        copy_buf,
                        copy_opname,
                    )
                },
            )?;
            let copy_fn = functionptr_const(rtyper, &copy)?;

            // mallocfn = mallocstr / mallocunicode
            let malloc_buf = outer_buf.clone();
            let malloc = rtyper.lowlevel_helper_function_with_builder(
                malloc_name.to_string(),
                vec![LowLevelType::Signed],
                outer_buf.clone(),
                move |_r, _a, _res| build_ll_mallocstr_helper_graph(malloc_name, malloc_buf),
            )?;
            let mallocfn = functionptr_const(rtyper, &malloc)?;

            // grow_by_fn = ll_grow_by(ll_builder, needed)
            let gb_builder = outer_builder.clone();
            let gb_piece_ptr = piece_ptr_lltype.clone();
            let gb_piece = piece_struct.clone();
            let gb_buf = outer_buf.clone();
            let grow_by = rtyper.lowlevel_helper_function_with_builder(
                "ll_grow_by".to_string(),
                vec![outer_builder.clone(), LowLevelType::Signed],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    build_ll_grow_by_helper_graph(
                        "ll_grow_by",
                        gb_builder,
                        gb_piece_ptr,
                        gb_piece,
                        gb_buf,
                        mallocfn,
                    )
                },
            )?;
            let grow_by_fn = functionptr_const(rtyper, &grow_by)?;

            // grow_and_append_fn = ll_grow_and_append(ll_builder, ll_str, start, size)
            let ga_builder = outer_builder.clone();
            let ga_piece_ptr = piece_ptr_lltype.clone();
            let ga_piece = piece_struct.clone();
            let ga_buf = outer_buf.clone();
            let ga_chars = chars_ptr.clone();
            let ga_copy_fn = copy_fn.clone();
            let grow_and_append = rtyper.lowlevel_helper_function_with_builder(
                "ll_grow_and_append".to_string(),
                vec![
                    outer_builder.clone(),
                    outer_buf.clone(),
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                ],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    build_ll_grow_and_append_helper_graph(
                        "ll_grow_and_append",
                        ga_builder,
                        ga_piece_ptr,
                        ga_piece,
                        ga_buf,
                        ga_chars,
                        ga_copy_fn,
                        grow_by_fn,
                    )
                },
            )?;
            let grow_and_append_fn = functionptr_const(rtyper, &grow_and_append)?;

            // _ll_append(ll_builder, ll_str, start, size): inline-copy or grow.
            let lla_builder = outer_builder.clone();
            let lla_buf = outer_buf.clone();
            let lla_copy_fn = copy_fn.clone();
            let ll_append_inner = rtyper.lowlevel_helper_function_with_builder(
                "_ll_append".to_string(),
                vec![
                    outer_builder.clone(),
                    outer_buf.clone(),
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                ],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    build_ll__ll_append_helper_graph(
                        "_ll_append",
                        lla_builder,
                        lla_buf,
                        grow_and_append_fn,
                        lla_copy_fn,
                    )
                },
            )?;
            let ll_append_fn = functionptr_const(rtyper, &ll_append_inner)?;

            // jit arm: ll_append_res0(ll_builder, ll_str) = _ll_append(builder,
            // str, 0, len). Correctness-first — identical to the no-jit arm;
            // the per-size JIT specialisation (try_append_slice) is a later task.
            let res0_builder = outer_builder.clone();
            let res0_buf = outer_buf.clone();
            let res0_chars = chars_ptr.clone();
            let res0_ll_append_fn = ll_append_fn.clone();
            let res0 = rtyper.lowlevel_helper_function_with_builder(
                "ll_append_res0".to_string(),
                vec![outer_builder.clone(), outer_buf.clone()],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    build_ll_append_res0_helper_graph(
                        "ll_append_res0",
                        res0_builder,
                        res0_buf,
                        res0_chars,
                        res0_ll_append_fn,
                    )
                },
            )?;
            let jit_append_fn = functionptr_const(rtyper, &res0)?;

            build_ll_append_helper_graph(
                "ll_append",
                outer_builder.clone(),
                outer_buf.clone(),
                chars_ptr,
                jit_append_fn,
                ll_append_fn,
            )
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

/// RPython `AbstractStringBuilderRepr.rtype_method_append`'s character arm:
/// select `self.char_repr` and call `ll_append_char` rather than coercing the
/// character through the string repr.
#[allow(clippy::too_many_arguments)]
fn rtype_builder_append_char(
    self_repr: &dyn Repr,
    char_repr: &dyn Repr,
    hop: &HighLevelOp,
    builder_ptr_lltype: LowLevelType,
    piece_ptr_lltype: LowLevelType,
    piece_struct: LowLevelType,
    buf_lltype: LowLevelType,
    char_lltype: LowLevelType,
    malloc_name: &'static str,
) -> RTypeResult {
    use crate::translator::rtyper::lltypesystem::rstr::{
        build_ll_mallocstr_helper_graph, chars_array_ptr_lltype_from_strptr,
    };
    let vlist = hop.inputargs(vec![
        ConvertedTo::Repr(self_repr),
        ConvertedTo::Repr(char_repr),
    ])?;
    hop.exception_cannot_occur()?;

    let outer_builder = builder_ptr_lltype.clone();
    let outer_buf = buf_lltype.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        "ll_append_char".to_string(),
        vec![builder_ptr_lltype, char_lltype.clone()],
        LowLevelType::Void,
        move |rtyper, _args, _result| {
            let chars_ptr = chars_array_ptr_lltype_from_strptr(&outer_buf)?;
            let malloc_buf = outer_buf.clone();
            let malloc = rtyper.lowlevel_helper_function_with_builder(
                malloc_name.to_string(),
                vec![LowLevelType::Signed],
                outer_buf.clone(),
                move |_r, _a, _res| build_ll_mallocstr_helper_graph(malloc_name, malloc_buf),
            )?;
            let mallocfn = functionptr_const(rtyper, &malloc)?;
            let gb_builder = outer_builder.clone();
            let gb_piece_ptr = piece_ptr_lltype.clone();
            let gb_piece = piece_struct.clone();
            let gb_buf = outer_buf.clone();
            let grow_by = rtyper.lowlevel_helper_function_with_builder(
                "ll_grow_by".to_string(),
                vec![outer_builder.clone(), LowLevelType::Signed],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    build_ll_grow_by_helper_graph(
                        "ll_grow_by",
                        gb_builder,
                        gb_piece_ptr,
                        gb_piece,
                        gb_buf,
                        mallocfn,
                    )
                },
            )?;
            let grow_by_fn = functionptr_const(rtyper, &grow_by)?;
            build_ll_append_char_helper_graph(
                "ll_append_char",
                outer_builder.clone(),
                char_lltype,
                chars_ptr,
                outer_buf.clone(),
                grow_by_fn,
            )
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

/// RPython `AbstractStringBuilderRepr.rtype_method_append_slice(self, hop)`
/// (`rbuilder.py`):
///
/// ```python
/// def rtype_method_append_slice(self, hop):
///     vlist = hop.inputargs(self, self.string_repr, lltype.Signed, lltype.Signed)
///     hop.exception_cannot_occur()
///     return hop.gendirectcall(self.ll_append_slice, *vlist)
/// ```
///
/// `ll_append_slice` branches on `we_are_jitted`; both arms funnel into
/// `_ll_append(builder, str, start, size)`.  The jit arm is wired to the general
/// `ll_append_res_slice` residual rather than the per-size-specialised
/// `ll_jit_append_slice`; that size specialisation is a later task, matching
/// `rtype_builder_append`'s own correctness-first jit arm.  The nested funcptr
/// consts are built bottom-up exactly as `rtype_builder_append`; the identical
/// `(name, args, result)` keys share its cached sub-graphs.
#[allow(clippy::too_many_arguments)]
fn rtype_builder_append_slice(
    self_repr: &dyn Repr,
    string_repr: &dyn Repr,
    hop: &HighLevelOp,
    builder_ptr_lltype: LowLevelType,
    piece_ptr_lltype: LowLevelType,
    piece_struct: LowLevelType,
    buf_lltype: LowLevelType,
    malloc_name: &'static str,
    copy_opname: &'static str,
) -> RTypeResult {
    use crate::translator::rtyper::lltypesystem::rstr::{
        build_ll_mallocstr_helper_graph, chars_array_ptr_lltype_from_strptr,
    };
    let vlist = hop.inputargs(vec![
        ConvertedTo::Repr(self_repr),
        ConvertedTo::Repr(string_repr),
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
    ])?;
    hop.exception_cannot_occur()?;

    let outer_builder = builder_ptr_lltype.clone();
    let outer_buf = buf_lltype.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        "ll_append_slice".to_string(),
        vec![
            builder_ptr_lltype,
            buf_lltype,
            LowLevelType::Signed,
            LowLevelType::Signed,
        ],
        LowLevelType::Void,
        move |rtyper, _args, _result| {
            let chars_ptr = chars_array_ptr_lltype_from_strptr(&outer_buf)?;

            // copy_fn = copy_string_contents(src, dst, srcstart, dststart, length)
            let copy_buf = outer_buf.clone();
            let copy = rtyper.lowlevel_helper_function_with_builder(
                "ll_copy_string_contents".to_string(),
                vec![
                    outer_buf.clone(),
                    outer_buf.clone(),
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                ],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    build_ll_copy_string_contents_helper_graph(
                        "ll_copy_string_contents",
                        copy_buf,
                        copy_opname,
                    )
                },
            )?;
            let copy_fn = functionptr_const(rtyper, &copy)?;

            // mallocfn = mallocstr / mallocunicode
            let malloc_buf = outer_buf.clone();
            let malloc = rtyper.lowlevel_helper_function_with_builder(
                malloc_name.to_string(),
                vec![LowLevelType::Signed],
                outer_buf.clone(),
                move |_r, _a, _res| build_ll_mallocstr_helper_graph(malloc_name, malloc_buf),
            )?;
            let mallocfn = functionptr_const(rtyper, &malloc)?;

            // grow_by_fn = ll_grow_by(ll_builder, needed)
            let gb_builder = outer_builder.clone();
            let gb_piece_ptr = piece_ptr_lltype.clone();
            let gb_piece = piece_struct.clone();
            let gb_buf = outer_buf.clone();
            let grow_by = rtyper.lowlevel_helper_function_with_builder(
                "ll_grow_by".to_string(),
                vec![outer_builder.clone(), LowLevelType::Signed],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    build_ll_grow_by_helper_graph(
                        "ll_grow_by",
                        gb_builder,
                        gb_piece_ptr,
                        gb_piece,
                        gb_buf,
                        mallocfn,
                    )
                },
            )?;
            let grow_by_fn = functionptr_const(rtyper, &grow_by)?;

            // grow_and_append_fn = ll_grow_and_append(ll_builder, ll_str, start, size)
            let ga_builder = outer_builder.clone();
            let ga_piece_ptr = piece_ptr_lltype.clone();
            let ga_piece = piece_struct.clone();
            let ga_buf = outer_buf.clone();
            let ga_chars = chars_ptr.clone();
            let ga_copy_fn = copy_fn.clone();
            let grow_and_append = rtyper.lowlevel_helper_function_with_builder(
                "ll_grow_and_append".to_string(),
                vec![
                    outer_builder.clone(),
                    outer_buf.clone(),
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                ],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    build_ll_grow_and_append_helper_graph(
                        "ll_grow_and_append",
                        ga_builder,
                        ga_piece_ptr,
                        ga_piece,
                        ga_buf,
                        ga_chars,
                        ga_copy_fn,
                        grow_by_fn,
                    )
                },
            )?;
            let grow_and_append_fn = functionptr_const(rtyper, &grow_and_append)?;

            // _ll_append(ll_builder, ll_str, start, size): inline-copy or grow.
            let lla_builder = outer_builder.clone();
            let lla_buf = outer_buf.clone();
            let lla_copy_fn = copy_fn.clone();
            let ll_append_inner = rtyper.lowlevel_helper_function_with_builder(
                "_ll_append".to_string(),
                vec![
                    outer_builder.clone(),
                    outer_buf.clone(),
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                ],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    build_ll__ll_append_helper_graph(
                        "_ll_append",
                        lla_builder,
                        lla_buf,
                        grow_and_append_fn,
                        lla_copy_fn,
                    )
                },
            )?;
            let ll_append_fn = functionptr_const(rtyper, &ll_append_inner)?;

            // jit arm = ll_append_res_slice(ll_builder, ll_str, start, end):
            // `_ll_append(builder, str, start, end - start)`. Correctness-first —
            // both arms reach the same `_ll_append`; the per-size specialisation
            // (`ll_jit_append_slice`) is a later task.
            let rs_builder = outer_builder.clone();
            let rs_buf = outer_buf.clone();
            let rs_ll_append_fn = ll_append_fn.clone();
            let res_slice = rtyper.lowlevel_helper_function_with_builder(
                "ll_append_res_slice".to_string(),
                vec![
                    outer_builder.clone(),
                    outer_buf.clone(),
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                ],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    build_ll_append_res_slice_helper_graph(
                        "ll_append_res_slice",
                        rs_builder,
                        rs_buf,
                        rs_ll_append_fn,
                    )
                },
            )?;
            let res_slice_fn = functionptr_const(rtyper, &res_slice)?;

            build_ll_append_slice_helper_graph(
                "ll_append_slice",
                outer_builder.clone(),
                outer_buf.clone(),
                res_slice_fn,
                ll_append_fn,
            )
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

/// RPython `AbstractStringBuilderRepr.rtype_method_build(self, hop)`
/// (`rbuilder.py`):
///
/// ```python
/// def rtype_method_build(self, hop):
///     vlist = hop.inputargs(self)
///     hop.exception_cannot_occur()
///     return hop.gendirectcall(self.ll_build, *vlist)
/// ```
///
/// `self.ll_build` is [`build_ll_build_helper_graph`] (fold_pieces +
/// shrink_final).  `ll_fold_pieces` folds the `extra_pieces` chain into one
/// buffer (`ll_getlength` / `mallocfn` / `copy`); `ll_shrink_final` trims the
/// single-buffer case with `rgc.ll_shrink_array` (an oopspec residual).  The
/// nested funcptr consts are built bottom-up exactly as [`rtype_builder_append`].
fn rtype_builder_build(
    self_repr: &dyn Repr,
    hop: &HighLevelOp,
    builder_ptr_lltype: LowLevelType,
    piece_ptr_lltype: LowLevelType,
    buf_lltype: LowLevelType,
    malloc_name: &'static str,
    copy_opname: &'static str,
) -> RTypeResult {
    use crate::translator::rtyper::lltypesystem::rstr::{
        build_ll_mallocstr_helper_graph, chars_array_ptr_lltype_from_strptr,
    };
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(self_repr)])?;
    hop.exception_cannot_occur()?;

    let outer_builder = builder_ptr_lltype.clone();
    let outer_piece_ptr = piece_ptr_lltype.clone();
    let outer_buf = buf_lltype.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        "ll_build".to_string(),
        vec![builder_ptr_lltype],
        buf_lltype,
        move |rtyper, _args, _result| {
            let chars_ptr = chars_array_ptr_lltype_from_strptr(&outer_buf)?;

            // getlength_fn = ll_getlength(ll_builder)
            let gl_builder = outer_builder.clone();
            let getlength = rtyper.lowlevel_helper_function_with_builder(
                "ll_getlength".to_string(),
                vec![outer_builder.clone()],
                LowLevelType::Signed,
                move |_r, _a, _res| build_ll_getlength_helper_graph("ll_getlength", gl_builder),
            )?;
            let getlength_fn = functionptr_const(rtyper, &getlength)?;

            // copy_fn = copy_string_contents(src, dst, srcstart, dststart, length)
            let copy_buf = outer_buf.clone();
            let copy = rtyper.lowlevel_helper_function_with_builder(
                "ll_copy_string_contents".to_string(),
                vec![
                    outer_buf.clone(),
                    outer_buf.clone(),
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                    LowLevelType::Signed,
                ],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    build_ll_copy_string_contents_helper_graph(
                        "ll_copy_string_contents",
                        copy_buf,
                        copy_opname,
                    )
                },
            )?;
            let copy_fn = functionptr_const(rtyper, &copy)?;

            // mallocfn = mallocstr / mallocunicode
            let malloc_buf = outer_buf.clone();
            let malloc = rtyper.lowlevel_helper_function_with_builder(
                malloc_name.to_string(),
                vec![LowLevelType::Signed],
                outer_buf.clone(),
                move |_r, _a, _res| build_ll_mallocstr_helper_graph(malloc_name, malloc_buf),
            )?;
            let mallocfn = functionptr_const(rtyper, &malloc)?;

            // fold_pieces_fn = ll_fold_pieces(ll_builder)
            let fp_builder = outer_builder.clone();
            let fp_piece_ptr = outer_piece_ptr.clone();
            let fp_buf = outer_buf.clone();
            let fp_chars = chars_ptr.clone();
            let fold_pieces = rtyper.lowlevel_helper_function_with_builder(
                "ll_fold_pieces".to_string(),
                vec![outer_builder.clone()],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    build_ll_fold_pieces_helper_graph(
                        "ll_fold_pieces",
                        fp_builder,
                        fp_piece_ptr,
                        fp_buf,
                        fp_chars,
                        getlength_fn,
                        mallocfn,
                        copy_fn,
                    )
                },
            )?;
            let fold_pieces_fn = functionptr_const(rtyper, &fold_pieces)?;

            // shrink_array_fn = rgc.ll_shrink_array (oopspec residual; stub body).
            let sa_buf = outer_buf.clone();
            let shrink_array = rtyper.lowlevel_helper_function_with_builder(
                "ll_shrink_array".to_string(),
                vec![outer_buf.clone(), LowLevelType::Signed],
                outer_buf.clone(),
                move |_r, _a, _res| {
                    build_ll_shrink_array_stub_helper_graph("ll_shrink_array", sa_buf)
                },
            )?;
            let shrink_array_fn = functionptr_const(rtyper, &shrink_array)?;

            // shrink_final_fn = ll_shrink_final(ll_builder)
            let sf_builder = outer_builder.clone();
            let sf_buf = outer_buf.clone();
            let shrink_final = rtyper.lowlevel_helper_function_with_builder(
                "ll_shrink_final".to_string(),
                vec![outer_builder.clone()],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    build_ll_shrink_final_helper_graph(
                        "ll_shrink_final",
                        sf_builder,
                        sf_buf,
                        shrink_array_fn,
                    )
                },
            )?;
            let shrink_final_fn = functionptr_const(rtyper, &shrink_final)?;

            build_ll_build_helper_graph(
                "ll_build",
                outer_builder.clone(),
                outer_buf.clone(),
                fold_pieces_fn,
                shrink_final_fn,
            )
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

/// RPython `BaseStringBuilderRepr.rtype_bool(self, hop)`
/// (`rbuilder.py:52-55`):
///
/// ```python
/// def rtype_bool(self, hop):
///     vlist = hop.inputargs(self)
///     hop.exception_cannot_occur()
///     return hop.gendirectcall(self.ll_bool, *vlist)
/// ```
fn rtype_builder_bool(
    self_repr: &dyn Repr,
    hop: &HighLevelOp,
    builder_ptr_lltype: LowLevelType,
) -> RTypeResult {
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(self_repr)])?;
    hop.exception_cannot_occur()?;
    let ptr_for_builder = builder_ptr_lltype.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        "ll_bool".to_string(),
        vec![builder_ptr_lltype],
        LowLevelType::Bool,
        move |_rtyper, _args, _result| build_ll_bool_helper_graph("ll_bool", ptr_for_builder),
    )?;
    hop.gendirectcall(&helper, vlist)
}

/// RPython `AbstractStringBuilderRepr.rtyper_new(self, hop)`
/// (`rtyper/rbuilder.py`):
///
/// ```python
/// def rtyper_new(self, hop):
///     if len(hop.args_v) == 0:
///         v_arg = hop.inputconst(lltype.Signed, INIT_SIZE)
///     else:
///         v_arg = hop.inputarg(lltype.Signed, 0)
///     hop.exception_cannot_occur()
///     return hop.gendirectcall(self.ll_new, v_arg)
/// ```
///
/// Shared body for the StringBuilderRepr / UnicodeBuilderRepr constructor
/// lowering. `self.ll_new` resolves to the concrete `ll_new` helper graph
/// whose whole callee closure is materialised here: `ll_min` (via the
/// hardcoded `lowlevel_helper_function` builder) and `mallocstr` /
/// `mallocunicode` (via `build_ll_mallocstr_helper_graph`), both baked in
/// as `direct_call` callee constants by [`functionptr_const`]. The
/// helper-graph cache keys stay distinct across STR / UNICODE because the
/// `ll_new` result lltype and the malloc helper name differ.
fn rtype_builder_new(
    hop: &HighLevelOp,
    builder_ptr_lltype: LowLevelType,
    builder_struct: LowLevelType,
    buf_lltype: LowLevelType,
    malloc_name: &'static str,
) -> RTypeResult {
    use crate::translator::rtyper::lltypesystem::rstr::build_ll_mallocstr_helper_graph;
    use crate::translator::rtyper::rbuilder::INIT_SIZE;
    let v_arg = if hop.nb_args() == 0 {
        Hlvalue::Constant(HighLevelOp::inputconst(
            ConvertedTo::LowLevelType(&LowLevelType::Signed),
            &ConstValue::Int(INIT_SIZE),
        )?)
    } else {
        hop.inputarg(ConvertedTo::LowLevelType(&LowLevelType::Signed), 0)?
    };
    hop.exception_cannot_occur()?;

    let ptr_for_builder = builder_ptr_lltype.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        "ll_new".to_string(),
        vec![LowLevelType::Signed],
        builder_ptr_lltype,
        move |rtyper, _args, _result| {
            let min = rtyper.lowlevel_helper_function(
                "ll_min",
                vec![LowLevelType::Unsigned, LowLevelType::Unsigned],
                LowLevelType::Unsigned,
            )?;
            let min_fn = functionptr_const(rtyper, &min)?;

            let malloc_buf = buf_lltype.clone();
            let malloc = rtyper.lowlevel_helper_function_with_builder(
                malloc_name.to_string(),
                vec![LowLevelType::Signed],
                buf_lltype.clone(),
                move |_r, _a, _res| build_ll_mallocstr_helper_graph(malloc_name, malloc_buf),
            )?;
            let mallocfn = functionptr_const(rtyper, &malloc)?;

            build_ll_new_helper_graph(
                "ll_new",
                ptr_for_builder.clone(),
                builder_struct.clone(),
                buf_lltype.clone(),
                min_fn,
                mallocfn,
            )
        },
    )?;
    hop.gendirectcall(&helper, vec![v_arg])
}

impl Default for StringBuilderRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for StringBuilderRepr {
    /// RPython `StringBuilderRepr.lowleveltype = lltype.Ptr(STRINGBUILDER)`.
    fn lowleveltype(&self) -> &LowLevelType {
        &STRINGBUILDERPTR
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "StringBuilderRepr"
    }

    fn rtype_bool(&self, hop: &HighLevelOp) -> RTypeResult {
        rtype_builder_bool(self, hop, STRINGBUILDERPTR.clone())
    }

    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        match method_name {
            "getlength" => rtype_builder_getlength(self, hop, STRINGBUILDERPTR.clone()),
            "append" => {
                // `SomeStringBuilder.method_append` accepts only `SomeString`
                // or `SomeChar` (rstring.py) — a byte char, never a unicode
                // code point.
                let is_char = matches!(
                    hop.args_s.borrow().get(1),
                    Some(crate::annotator::model::SomeValue::Char(_))
                );
                if is_char {
                    let char_repr = crate::translator::rtyper::rstr::char_repr();
                    rtype_builder_append_char(
                        self,
                        char_repr.as_ref(),
                        hop,
                        STRINGBUILDERPTR.clone(),
                        STRINGPIECEPTR.clone(),
                        STRINGPIECE.clone(),
                        STRPTR.clone(),
                        LowLevelType::Char,
                        "mallocstr",
                    )
                } else {
                    let string_repr = crate::translator::rtyper::rstr::string_repr();
                    rtype_builder_append(
                        self,
                        string_repr.as_ref(),
                        hop,
                        STRINGBUILDERPTR.clone(),
                        STRINGPIECEPTR.clone(),
                        STRINGPIECE.clone(),
                        STRPTR.clone(),
                        "mallocstr",
                        "copystrcontent",
                    )
                }
            }
            "append_slice" => {
                let string_repr = crate::translator::rtyper::rstr::string_repr();
                rtype_builder_append_slice(
                    self,
                    string_repr.as_ref(),
                    hop,
                    STRINGBUILDERPTR.clone(),
                    STRINGPIECEPTR.clone(),
                    STRINGPIECE.clone(),
                    STRPTR.clone(),
                    "mallocstr",
                    "copystrcontent",
                )
            }
            "build" => rtype_builder_build(
                self,
                hop,
                STRINGBUILDERPTR.clone(),
                STRINGPIECEPTR.clone(),
                STRPTR.clone(),
                "mallocstr",
                "copystrcontent",
            ),
            _ => Err(TyperError::message(format!(
                "missing {}.rtype_method_{method_name}",
                self.class_name()
            ))),
        }
    }
}

/// RPython `class UnicodeBuilderRepr(BaseStringBuilderRepr)`.
#[derive(Debug)]
pub struct UnicodeBuilderRepr {
    state: ReprState,
}

impl UnicodeBuilderRepr {
    pub fn new() -> Self {
        UnicodeBuilderRepr {
            state: ReprState::new(),
        }
    }

    /// RPython `UnicodeBuilderRepr.basetp = UNICODE`.
    pub fn basetp(&self) -> &'static LowLevelType {
        &crate::translator::rtyper::lltypesystem::rstr::UNICODE
    }

    /// RPython `AbstractStringBuilderRepr.rtyper_new(self, hop)`
    /// (`rtyper/rbuilder.py`), inherited by `UnicodeBuilderRepr`.
    pub fn rtyper_new(&self, hop: &HighLevelOp) -> RTypeResult {
        rtype_builder_new(
            hop,
            UNICODEBUILDERPTR.clone(),
            UNICODEBUILDER.clone(),
            UNICODEPTR.clone(),
            "mallocunicode",
        )
    }
}

impl Default for UnicodeBuilderRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for UnicodeBuilderRepr {
    /// RPython `UnicodeBuilderRepr.lowleveltype = lltype.Ptr(UNICODEBUILDER)`.
    fn lowleveltype(&self) -> &LowLevelType {
        &UNICODEBUILDERPTR
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "UnicodeBuilderRepr"
    }

    fn rtype_bool(&self, hop: &HighLevelOp) -> RTypeResult {
        rtype_builder_bool(self, hop, UNICODEBUILDERPTR.clone())
    }

    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        match method_name {
            "getlength" => rtype_builder_getlength(self, hop, UNICODEBUILDERPTR.clone()),
            "append" => {
                // `SomeUnicodeBuilder.method_append` accepts only
                // `SomeUnicodeString` or `SomeUnicodeCodePoint` (rstring.py) —
                // a unicode code point, never a byte char.
                let is_char = matches!(
                    hop.args_s.borrow().get(1),
                    Some(crate::annotator::model::SomeValue::UnicodeCodePoint(_))
                );
                if is_char {
                    let char_repr = crate::translator::rtyper::rstr::unichar_repr();
                    rtype_builder_append_char(
                        self,
                        char_repr.as_ref(),
                        hop,
                        UNICODEBUILDERPTR.clone(),
                        UNICODEPIECEPTR.clone(),
                        UNICODEPIECE.clone(),
                        UNICODEPTR.clone(),
                        LowLevelType::UniChar,
                        "mallocunicode",
                    )
                } else {
                    let string_repr = crate::translator::rtyper::rstr::unicode_repr();
                    rtype_builder_append(
                        self,
                        string_repr.as_ref(),
                        hop,
                        UNICODEBUILDERPTR.clone(),
                        UNICODEPIECEPTR.clone(),
                        UNICODEPIECE.clone(),
                        UNICODEPTR.clone(),
                        "mallocunicode",
                        "copyunicodecontent",
                    )
                }
            }
            "append_slice" => {
                let string_repr = crate::translator::rtyper::rstr::unicode_repr();
                rtype_builder_append_slice(
                    self,
                    string_repr.as_ref(),
                    hop,
                    UNICODEBUILDERPTR.clone(),
                    UNICODEPIECEPTR.clone(),
                    UNICODEPIECE.clone(),
                    UNICODEPTR.clone(),
                    "mallocunicode",
                    "copyunicodecontent",
                )
            }
            "build" => rtype_builder_build(
                self,
                hop,
                UNICODEBUILDERPTR.clone(),
                UNICODEPIECEPTR.clone(),
                UNICODEPTR.clone(),
                "mallocunicode",
                "copyunicodecontent",
            ),
            _ => Err(TyperError::message(format!(
                "missing {}.rtype_method_{method_name}",
                self.class_name()
            ))),
        }
    }
}

/// RPython `stringbuilder_repr = StringBuilderRepr()`.
pub fn stringbuilder_repr() -> Arc<StringBuilderRepr> {
    static REPR: OnceLock<Arc<StringBuilderRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(StringBuilderRepr::new()))
        .clone()
}

/// RPython `unicodebuilder_repr = UnicodeBuilderRepr()`.
pub fn unicodebuilder_repr() -> Arc<UnicodeBuilderRepr> {
    static REPR: OnceLock<Arc<UnicodeBuilderRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(UnicodeBuilderRepr::new()))
        .clone()
}

/// The pyre adaptation of the RPython `rlib.rstring` ExtRegistryEntry
/// `specialize_call`, which lowers the builder ctor to
/// `hop.r_result.rtyper_new(hop)`.  The result repr — a
/// `StringBuilderRepr` or `UnicodeBuilderRepr`, chosen by `makerepr`
/// from the `SomeStringBuilder` / `SomeUnicodeBuilder` annotation —
/// selects which `ll_new` helper graph the emitted direct call binds.
pub fn rtype_newstringbuilder(hop: &HighLevelOp) -> RTypeResult {
    let r_result = hop
        .r_result
        .borrow()
        .clone()
        .ok_or_else(|| TyperError::message("rtype_newstringbuilder: r_result missing"))?;
    let any_r: &dyn std::any::Any = r_result.as_ref();
    if let Some(r) = any_r.downcast_ref::<StringBuilderRepr>() {
        return r.rtyper_new(hop);
    }
    if let Some(r) = any_r.downcast_ref::<UnicodeBuilderRepr>() {
        return r.rtyper_new(hop);
    }
    Err(TyperError::message(
        "rtype_newstringbuilder: hop.r_result is not a builder repr",
    ))
}

#[cfg(test)]
mod tests {
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};
    use crate::translator::rtyper::rmodel::Repr;

    #[test]
    fn stringpiece_prev_piece_points_back_to_stringpiece() {
        let LowLevelType::ForwardReference(piece_fwd) = super::STRINGPIECE.clone() else {
            panic!("STRINGPIECE must be a GcForwardReference");
        };
        let Some(LowLevelType::Struct(piece)) = piece_fwd.resolved() else {
            panic!("STRINGPIECE must resolve to a struct");
        };
        let Some(LowLevelType::Ptr(prev_ptr)) = piece.getattr_field_type("prev_piece") else {
            panic!("STRINGPIECE.prev_piece must be Ptr");
        };
        assert!(matches!(prev_ptr.TO, PtrTarget::ForwardReference(_)));
        assert_eq!(
            LowLevelType::ForwardReference(piece_fwd),
            LowLevelType::from(prev_ptr.TO)
        );
    }

    #[test]
    fn stringbuilder_fields_match_rpython_shape() {
        let LowLevelType::Struct(builder) = super::STRINGBUILDER.clone() else {
            panic!("STRINGBUILDER must be GcStruct");
        };
        assert_eq!(
            builder._names,
            vec![
                "current_buf",
                "current_pos",
                "current_end",
                "total_size",
                "extra_pieces",
            ]
        );
        assert_eq!(
            builder.getattr_field_type("current_pos"),
            Some(LowLevelType::Signed)
        );
        assert_eq!(
            builder.getattr_field_type("current_end"),
            Some(LowLevelType::Signed)
        );
        assert_eq!(
            builder.getattr_field_type("total_size"),
            Some(LowLevelType::Signed)
        );
    }

    #[test]
    fn repr_singletons_have_distinct_lowleveltypes() {
        assert_ne!(
            super::stringbuilder_repr().lowleveltype(),
            super::unicodebuilder_repr().lowleveltype()
        );
    }

    #[test]
    fn builder_reprs_impl_repr_trait_and_coerce_to_dyn() {
        let sb = super::stringbuilder_repr();
        assert_eq!(sb.class_name(), "StringBuilderRepr");
        assert_eq!(sb.lowleveltype(), &*super::STRINGBUILDERPTR);

        let ub = super::unicodebuilder_repr();
        assert_eq!(ub.class_name(), "UnicodeBuilderRepr");
        assert_eq!(ub.lowleveltype(), &*super::UNICODEBUILDERPTR);

        // The reprs must be usable as `Arc<dyn Repr>` so the rtyper's
        // `inputarg` / `gendirectcall` machinery can consume them.
        let _dyn_sb: std::sync::Arc<dyn Repr> = super::stringbuilder_repr();
        let _dyn_ub: std::sync::Arc<dyn Repr> = super::unicodebuilder_repr();
    }

    #[test]
    fn runtime_helper_surface_is_explicitly_deferred() {
        let err = super::ll_append().expect_err("append helper is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_append"));

        let err = super::ll_build().expect_err("build helper is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_build"));

        let err = super::ll_append_multiple_char().expect_err("multiple-char helper is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_append_multiple_char"));
    }

    #[test]
    fn build_ll_getlength_reads_fields_and_returns_signed_length() {
        use super::Hlvalue;
        let helper =
            super::build_ll_getlength_helper_graph("ll_getlength", super::STRINGBUILDERPTR.clone())
                .expect("build_ll_getlength_helper_graph");
        assert_eq!(helper.func.name, "ll_getlength");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        // total_size - (current_end - current_pos)
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            ops,
            vec!["getfield", "getfield", "int_sub", "getfield", "int_sub"]
        );
        assert_eq!(startblock.inputargs.len(), 1);
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Signed)
        );
    }

    #[test]
    fn build_ll_bool_compares_pointer_against_null_and_returns_bool() {
        use super::Hlvalue;
        let helper = super::build_ll_bool_helper_graph("ll_bool", super::STRINGBUILDERPTR.clone())
            .expect("build_ll_bool_helper_graph");
        assert_eq!(helper.func.name, "ll_bool");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        // ll_builder != nullptr(TO)
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["ptr_ne"]);
        assert_eq!(startblock.inputargs.len(), 1);
        // second arg is the null pointer constant of the builder's own type.
        let Hlvalue::Constant(null_arg) = &startblock.operations[0].args[1] else {
            panic!("ptr_ne second arg must be a null Constant");
        };
        assert_eq!(null_arg.value, super::ConstValue::None);
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Bool));
    }

    fn dummy_funcptr_const() -> super::Constant {
        super::Constant::with_concretetype(super::ConstValue::None, LowLevelType::Void)
    }

    #[test]
    fn build_ll_new_clamps_size_mallocs_builder_and_inits_fields() {
        use super::Hlvalue;
        let helper = super::build_ll_new_helper_graph(
            "ll_new",
            super::STRINGBUILDERPTR.clone(),
            super::STRINGBUILDER.clone(),
            super::STRPTR.clone(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_new_helper_graph");
        assert_eq!(helper.func.name, "ll_new");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        // intmask(min(r_uint(init_size), 1280)); malloc; mallocfn; 4 setfields
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            ops,
            vec![
                "cast_int_to_uint",
                "direct_call",
                "cast_uint_to_int",
                "malloc",
                "direct_call",
                "setfield",
                "setfield",
                "setfield",
                "setfield",
            ]
        );
        assert_eq!(startblock.inputargs.len(), 1);
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(super::STRINGBUILDERPTR.clone())
        );
    }

    #[test]
    fn build_ll_shrink_final_reads_pos_shrinks_buf_and_updates_fields() {
        use super::Hlvalue;
        let helper = super::build_ll_shrink_final_helper_graph(
            "ll_shrink_final",
            super::STRINGBUILDERPTR.clone(),
            super::STRPTR.clone(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_shrink_final_helper_graph");
        assert_eq!(helper.func.name, "ll_shrink_final");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        // final_size = current_pos; buf = shrink(current_buf, final_size); 3 setfields
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            ops,
            vec![
                "getfield",
                "getfield",
                "direct_call",
                "setfield",
                "setfield",
                "setfield",
            ]
        );
        assert_eq!(startblock.inputargs.len(), 1);
        // Void return.
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    fn walk_ops(start: &crate::flowspace::model::BlockRef) -> (usize, Vec<String>) {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![start.clone()];
        let mut all_ops: Vec<String> = Vec::new();
        let mut count = 0usize;
        while let Some(b) = stack.pop() {
            if !seen.insert(std::rc::Rc::as_ptr(&b) as usize) {
                continue;
            }
            count += 1;
            let bb = b.borrow();
            for op in &bb.operations {
                all_ops.push(op.opname.clone());
            }
            for link in &bb.exits {
                if let Some(t) = link.borrow().target.clone() {
                    stack.push(t);
                }
            }
        }
        (count, all_ops)
    }

    #[test]
    fn build_ll__ll_append_multiple_char_two_char_write_loops() {
        use super::Hlvalue;
        let helper = super::build_ll__ll_append_multiple_char_helper_graph(
            "_ll_append_multiple_char",
            super::STRINGBUILDERPTR.clone(),
            LowLevelType::Char,
            super::STRPTR.clone(), // chars array ptr placeholder
            super::STRPTR.clone(),
            dummy_funcptr_const(),
        )
        .expect("build_ll__ll_append_multiple_char_helper_graph");
        assert_eq!(helper.func.name, "_ll_append_multiple_char");
        let inner = helper.graph.borrow();

        // header: part1 = current_end - current_pos; if times > part1.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getfield", "getfield", "int_sub", "int_gt"]);
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);

        let (count, all_ops) = walk_ops(&inner.startblock);
        // start, setup, l1_header, l1_body, grow_tail, after, l2_header,
        // l2_body, returnblock
        assert_eq!(count, 9);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        assert_eq!(n("getfield"), 5);
        assert_eq!(n("int_sub"), 2);
        assert_eq!(n("int_gt"), 1);
        assert_eq!(n("int_lt"), 2); // two loop headers
        assert_eq!(n("getsubstruct"), 2); // buf.chars in each loop body
        assert_eq!(n("setarrayitem"), 2); // char writes
        assert_eq!(n("int_add"), 3); // two i+=1 + end=pos+times
        assert_eq!(n("direct_call"), 1); // ll_grow_by
        assert_eq!(n("setfield"), 1); // current_pos = end

        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    #[test]
    fn build_ll_jit_try_append_multiple_char_guards_and_fills_small_constant_size() {
        use super::Hlvalue;
        let helper = super::build_ll_jit_try_append_multiple_char_helper_graph(
            "ll_jit_try_append_multiple_char",
            super::STRINGBUILDERPTR.clone(),
            LowLevelType::Char,
            super::STRPTR.clone(), // chars array ptr placeholder
            super::STRPTR.clone(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_jit_try_append_multiple_char_helper_graph");
        assert_eq!(helper.func.name, "ll_jit_try_append_multiple_char");
        let inner = helper.graph.borrow();

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["direct_call"]); // jit.isconstant(size)
        assert_eq!(startblock.inputargs.len(), 3);
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);

        let (count, all_ops) = walk_ops(&inner.startblock);
        // start, size==0, pos-const, end-const, fit, <=16, setup,
        // loop-header, loop-body, size==1, append-one, returnblock
        assert_eq!(count, 12);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        assert_eq!(n("direct_call"), 4); // 3 jit.isconstant calls + ll_append_char
        assert_eq!(n("int_eq"), 2); // size == 0, size == 1
        assert_eq!(n("getfield"), 3); // current_pos, current_end, current_buf
        assert_eq!(n("int_sub"), 1);
        assert_eq!(n("int_le"), 2);
        assert_eq!(n("int_add"), 2); // stop = pos + size; pos += 1
        assert_eq!(n("setfield"), 1);
        assert_eq!(n("int_lt"), 1);
        assert_eq!(n("getsubstruct"), 1);
        assert_eq!(n("setarrayitem"), 1);

        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Bool));
    }

    #[test]
    fn build_ll_append_charpsize_copies_grows_and_merges_tail() {
        use super::Hlvalue;
        let helper = super::build_ll_append_charpsize_helper_graph(
            "ll_append_charpsize",
            super::STRINGBUILDERPTR.clone(),
            crate::translator::rtyper::lltypesystem::rffi::CCHARP.clone(),
            super::STRPTR.clone(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_append_charpsize_helper_graph");
        assert_eq!(helper.func.name, "ll_append_charpsize");
        let inner = helper.graph.borrow();

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getfield", "getfield", "int_sub", "int_gt"]);
        assert_eq!(startblock.inputargs.len(), 3);
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);

        let (count, all_ops) = walk_ops(&inner.startblock);
        // start, overrun body, shared tail, returnblock
        assert_eq!(count, 4);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        assert_eq!(n("getfield"), 5); // start pos/end, body buf, tail pos/buf
        assert_eq!(n("int_sub"), 2); // part1 and size -= part1
        assert_eq!(n("int_gt"), 1);
        assert_eq!(n("direct_call"), 3); // first copy, grow_by, final copy
        assert_eq!(n("direct_ptradd"), 1);
        assert_eq!(n("int_add"), 1);
        assert_eq!(n("setfield"), 1);

        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    #[test]
    fn build_ll_jit_try_append_slice_guards_copies_and_unrolls_size_arms() {
        use super::Hlvalue;
        let size_specialized: Vec<_> = (2..=10)
            .map(|n| (n, dummy_funcptr_const(), dummy_funcptr_const()))
            .collect();
        let helper = super::build_ll_jit_try_append_slice_helper_graph(
            "ll_jit_try_append_slice",
            super::STRINGBUILDERPTR.clone(),
            super::STRPTR.clone(),
            super::STRPTR.clone(), // chars array ptr placeholder
            LowLevelType::Char,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            &size_specialized,
        )
        .expect("build_ll_jit_try_append_slice_helper_graph");
        assert_eq!(helper.func.name, "ll_jit_try_append_slice");
        let inner = helper.graph.borrow();

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["direct_call"]); // jit.isconstant(size)
        assert_eq!(startblock.inputargs.len(), 4);
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);

        let (count, all_ops) = walk_ops(&inner.startblock);
        // 12 non-unrolled blocks + 9 explicit size arms with 5 blocks each.
        assert_eq!(count, 57);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        assert_eq!(n("direct_call"), 31); // 4 base calls + 9*(isconstant + func0 + funcstart)
        assert_eq!(n("int_eq"), 20); // size 0/1 + 9 size checks + 9 start==0 checks
        assert_eq!(n("getfield"), 3); // current_pos, current_end, current_buf
        assert_eq!(n("int_sub"), 1);
        assert_eq!(n("int_le"), 2);
        assert_eq!(n("int_lt"), 1);
        assert_eq!(n("getsubstruct"), 3); // loop read/write + size==1 read
        assert_eq!(n("getarrayitem"), 2); // loop read + size==1 read
        assert_eq!(n("setarrayitem"), 1);
        assert_eq!(n("int_add"), 3); // stop, pos += 1, start += 1
        assert_eq!(n("setfield"), 1);

        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Bool));
    }

    #[test]
    fn build_ll_append_branches_on_we_are_jitted_symbolic() {
        use super::Hlvalue;
        let helper = super::build_ll_append_helper_graph(
            "ll_append",
            super::STRINGBUILDERPTR.clone(),
            super::STRPTR.clone(),
            super::STRPTR.clone(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_append_helper_graph");
        assert_eq!(helper.func.name, "ll_append");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        // startblock only branches; no ops.
        assert!(startblock.operations.is_empty());
        assert_eq!(startblock.exits.len(), 2);
        // exitswitch is the identity-bearing we_are_jitted SpecTag const.
        match &startblock.exitswitch {
            Some(Hlvalue::Constant(c)) => assert_eq!(
                c.value,
                super::ConstValue::SpecTag(super::WE_ARE_JITTED_TAG_ID)
            ),
            other => panic!("exitswitch must be the SpecTag constant, got {other:?}"),
        }
        drop(startblock);
        let (count, all_ops) = walk_ops(&inner.startblock);
        // start, jit arm, no-jit arm, returnblock.
        assert_eq!(count, 4);
        // jit arm: ll_jit_append; no-jit arm: getsubstruct+getarraysize+_ll_append.
        assert_eq!(
            all_ops
                .iter()
                .filter(|o| o.as_str() == "direct_call")
                .count(),
            2
        );
        assert_eq!(
            all_ops
                .iter()
                .filter(|o| o.as_str() == "getsubstruct")
                .count(),
            1
        );
        assert_eq!(
            all_ops
                .iter()
                .filter(|o| o.as_str() == "getarraysize")
                .count(),
            1
        );
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    #[test]
    fn build_ll_append_char_conditional_grows_then_writes_char() {
        use super::Hlvalue;
        let helper = super::build_ll_append_char_helper_graph(
            "ll_append_char",
            super::STRINGBUILDERPTR.clone(),
            LowLevelType::Char,
            super::STRPTR.clone(), // chars array ptr placeholder
            super::STRPTR.clone(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_append_char_helper_graph");
        assert_eq!(helper.func.name, "ll_append_char");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(
            ops,
            vec![
                "getfield", // current_pos (for ==)
                "getfield", // current_end
                "int_eq",   // current_pos == current_end
            ]
        );
        match &startblock.exitswitch {
            Some(Hlvalue::Constant(c)) => assert_eq!(
                c.value,
                super::ConstValue::SpecTag(super::WE_ARE_JITTED_TAG_ID)
            ),
            other => panic!("exitswitch must be the SpecTag constant, got {other:?}"),
        }
        assert_eq!(startblock.exits.len(), 2);
        let jit_exit = startblock.exits[0].borrow();
        assert!(
            matches!(
                &jit_exit.exitcase,
                Some(Hlvalue::Constant(c)) if c.value == super::ConstValue::Bool(true)
            ),
            "true (jitted) link is first"
        );
        let jit_block = jit_exit.target.as_ref().unwrap().clone();
        drop(jit_exit);
        drop(startblock);
        let jit_borrow = jit_block.borrow();
        let jit_ops: Vec<&str> = jit_borrow
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(jit_ops, vec!["jit_conditional_call"]);
        drop(jit_borrow);
        let (count, all_ops) = walk_ops(&inner.startblock);
        // start, jit, interp-if, interp-direct, tail, returnblock.
        assert_eq!(count, 6);
        assert_eq!(
            all_ops
                .iter()
                .filter(|o| o.as_str() == "jit_conditional_call")
                .count(),
            1
        );
        assert_eq!(
            all_ops
                .iter()
                .filter(|o| o.as_str() == "direct_call")
                .count(),
            1
        );
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    #[test]
    fn build_ll_append_slice_branches_on_we_are_jitted_symbolic() {
        use super::Hlvalue;
        let helper = super::build_ll_append_slice_helper_graph(
            "ll_append_slice",
            super::STRINGBUILDERPTR.clone(),
            super::STRPTR.clone(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_append_slice_helper_graph");
        assert_eq!(helper.func.name, "ll_append_slice");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        assert!(startblock.operations.is_empty());
        assert_eq!(startblock.exits.len(), 2);
        match &startblock.exitswitch {
            Some(Hlvalue::Constant(c)) => assert_eq!(
                c.value,
                super::ConstValue::SpecTag(super::WE_ARE_JITTED_TAG_ID)
            ),
            other => panic!("exitswitch must be the SpecTag constant, got {other:?}"),
        }
        assert_eq!(startblock.inputargs.len(), 4);
        drop(startblock);
        let (count, all_ops) = walk_ops(&inner.startblock);
        // start, jit arm, no-jit arm, returnblock.
        assert_eq!(count, 4);
        // jit arm: ll_jit_append_slice; no-jit arm: int_sub + _ll_append.
        assert_eq!(
            all_ops
                .iter()
                .filter(|o| o.as_str() == "direct_call")
                .count(),
            2
        );
        assert_eq!(
            all_ops.iter().filter(|o| o.as_str() == "int_sub").count(),
            1
        );
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    #[test]
    fn build_ll_append_multiple_char_nested_jit_guard_shares_fallback() {
        use super::Hlvalue;
        let helper = super::build_ll_append_multiple_char_helper_graph(
            "ll_append_multiple_char",
            super::STRINGBUILDERPTR.clone(),
            LowLevelType::Char,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_append_multiple_char_helper_graph");
        assert_eq!(helper.func.name, "ll_append_multiple_char");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        assert!(startblock.operations.is_empty());
        assert_eq!(startblock.exits.len(), 2);
        match &startblock.exitswitch {
            Some(Hlvalue::Constant(c)) => assert_eq!(
                c.value,
                super::ConstValue::SpecTag(super::WE_ARE_JITTED_TAG_ID)
            ),
            other => panic!("exitswitch must be the SpecTag constant, got {other:?}"),
        }
        drop(startblock);
        // start, jit-guard, shared fallback, returnblock — the fallback
        // block is shared by both false edges, so the walk sees 4 blocks.
        let (count, all_ops) = walk_ops(&inner.startblock);
        assert_eq!(count, 4);
        // jit-guard direct_call + fallback direct_call = 2.
        assert_eq!(
            all_ops
                .iter()
                .filter(|o| o.as_str() == "direct_call")
                .count(),
            2
        );
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    #[test]
    fn build_ll_jit_append_tries_slice_then_falls_back_to_res0() {
        use super::Hlvalue;
        let helper = super::build_ll_jit_append_helper_graph(
            "ll_jit_append",
            super::STRINGBUILDERPTR.clone(),
            super::STRPTR.clone(),
            super::STRPTR.clone(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_jit_append_helper_graph");
        assert_eq!(helper.func.name, "ll_jit_append");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["getsubstruct", "getarraysize", "direct_call"]);
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);
        let (count, all_ops) = walk_ops(&inner.startblock);
        assert_eq!(count, 3); // start, fallback, returnblock
        assert_eq!(
            all_ops
                .iter()
                .filter(|o| o.as_str() == "direct_call")
                .count(),
            2
        );
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    #[test]
    fn build_ll_jit_append_slice_tries_slice_then_falls_back_to_res_slice() {
        use super::Hlvalue;
        let helper = super::build_ll_jit_append_slice_helper_graph(
            "ll_jit_append_slice",
            super::STRINGBUILDERPTR.clone(),
            super::STRPTR.clone(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_jit_append_slice_helper_graph");
        assert_eq!(helper.func.name, "ll_jit_append_slice");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["int_sub", "direct_call"]);
        assert_eq!(startblock.inputargs.len(), 4);
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);
        let (count, all_ops) = walk_ops(&inner.startblock);
        assert_eq!(count, 3);
        assert_eq!(
            all_ops
                .iter()
                .filter(|o| o.as_str() == "direct_call")
                .count(),
            2
        );
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    #[test]
    fn build_ll_grow_and_append_fastpath_and_copy_grow_slowpath() {
        use super::Hlvalue;
        let helper = super::build_ll_grow_and_append_helper_graph(
            "ll_grow_and_append",
            super::STRINGBUILDERPTR.clone(),
            super::STRINGPIECEPTR.clone(),
            super::STRINGPIECE.clone(),
            super::STRPTR.clone(),
            super::STRPTR.clone(), // chars array ptr placeholder
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_grow_and_append_helper_graph");
        assert_eq!(helper.func.name, "ll_grow_and_append");
        let inner = helper.graph.borrow();

        // header: size > 1280 test, first of the four short-circuit conditions.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["int_gt"]);
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);

        let (count, all_ops) = walk_ops(&inner.startblock);
        // b0, b1, b2, b3, fast_check, fast_body, slow, returnblock. The fast
        // path splits: fast_check holds the trailing ovfcheck (its overflow
        // edge falls to slow, `except OverflowError: pass`), fast_body the
        // malloc/link. The exceptblock is unreachable (overflow does not raise).
        assert_eq!(count, 8);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        assert_eq!(n("int_gt"), 1);
        assert_eq!(n("int_eq"), 3); // pos==0, start==0, size==len
        assert_eq!(n("getsubstruct"), 1);
        assert_eq!(n("getarraysize"), 1);
        assert_eq!(n("int_add_ovf"), 1); // fast_check: total_size + size
        assert_eq!(n("malloc"), 1); // fast_body: new piece
        assert_eq!(n("getfield"), 7);
        assert_eq!(n("setfield"), 5); // 4 fast + 1 slow (current_pos)
        assert_eq!(n("int_sub"), 2); // slow: part1, size-part1
        assert_eq!(n("int_add"), 1); // slow: start+part1
        assert_eq!(n("direct_call"), 3); // slow: copy, grow_by, copy

        // Exactly one block is a canraise ovfcheck (fast_check); its overflow
        // edge targets a block with operations (slow) carrying four data args,
        // not the [class, instance] extravars -> no MemoryError raise here.
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![inner.startblock.clone()];
        let mut ovf_blocks = 0usize;
        while let Some(b) = stack.pop() {
            if !seen.insert(std::rc::Rc::as_ptr(&b) as usize) {
                continue;
            }
            let bb = b.borrow();
            if bb.canraise() {
                ovf_blocks += 1;
                let exc = bb.exits[1].borrow();
                let exc_target = exc.target.clone().expect("ovf exc edge has a target");
                assert!(
                    !exc_target.borrow().operations.is_empty(),
                    "overflow falls to the slow path, not a bare raise block"
                );
                assert_eq!(
                    exc.args.len(),
                    4,
                    "overflow carries the four slow-path inputs"
                );
            }
            for link in &bb.exits {
                if let Some(t) = link.borrow().target.clone() {
                    stack.push(t);
                }
            }
        }
        assert_eq!(ovf_blocks, 1);

        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    #[test]
    fn build_ll_grow_by_ovf_arith_mallocs_piece_and_relinks_buffer() {
        use super::Hlvalue;
        let helper = super::build_ll_grow_by_helper_graph(
            "ll_grow_by",
            super::STRINGBUILDERPTR.clone(),
            super::STRINGPIECEPTR.clone(),
            super::STRINGPIECE.clone(),
            super::STRPTR.clone(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_grow_by_helper_graph");
        assert_eq!(helper.func.name, "ll_grow_by");
        let inner = helper.graph.borrow();

        // Each `ovfcheck` is the trailing op of its own block so the
        // OverflowError -> raise MemoryError edge can attach. block_a
        // (startblock): total_size read once, then `needed + total_size`.
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["getfield", "int_add_ovf"]);
        assert_eq!(startblock.inputargs.len(), 2);
        // c_last_exception switch: normal fall-through + OverflowError catch.
        assert_eq!(startblock.exits.len(), 2);
        assert!(startblock.canraise());
        assert_eq!(startblock.exits[0].borrow().exitcase, None);
        drop(startblock);

        let (count, all_ops) = walk_ops(&inner.startblock);
        // block_a, block_b, block_c, block_d, handler, returnblock, exceptblock
        assert_eq!(count, 7);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        assert_eq!(n("int_add_ovf"), 3); // one ovfcheck per block
        assert_eq!(n("int_and"), 1); // & ~63
        assert_eq!(n("direct_call"), 1); // mallocfn
        assert_eq!(n("malloc"), 1); // old_piece
        assert_eq!(n("getfield"), 3); // total_size, current_buf, extra_pieces
        assert_eq!(n("setfield"), 7); // buf, prev_piece + 5 builder fields

        // The OverflowError handler has no ops and raises MemoryError via a
        // single link to the graph exceptblock carrying [class, instance].
        let except_ptr = std::rc::Rc::as_ptr(&inner.exceptblock) as usize;
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![inner.startblock.clone()];
        let mut raises = 0usize;
        while let Some(b) = stack.pop() {
            if !seen.insert(std::rc::Rc::as_ptr(&b) as usize) {
                continue;
            }
            let bb = b.borrow();
            for link in &bb.exits {
                let l = link.borrow();
                if let Some(t) = l.target.clone() {
                    if std::rc::Rc::as_ptr(&t) as usize == except_ptr {
                        assert!(bb.operations.is_empty(), "handler has no ops");
                        assert_eq!(l.args.len(), 2, "MemoryError raise carries class+instance");
                        raises += 1;
                    }
                    stack.push(t);
                }
            }
        }
        assert_eq!(raises, 1, "exactly one MemoryError raise edge");

        // Data-flow guard (not just op-name counts): in block_d the mallocfn
        // allocation size must be the SAME Variable written to `current_end`
        // (allocated size == recorded buffer end, matching `rbuilder.py:104`
        // `new_string = mallocfn(needed)` and `:113` `current_end = needed`),
        // and `total_size` must be a distinct value (`total_new`, not the
        // rounded `needed`). Swapping which variant feeds mallocfn / current_end
        // / total_size is a buffer-size regression op-name counts cannot see.
        let mut block_d = None;
        let mut seen_d = std::collections::HashSet::new();
        let mut stack_d = vec![inner.startblock.clone()];
        while let Some(b) = stack_d.pop() {
            if !seen_d.insert(std::rc::Rc::as_ptr(&b) as usize) {
                continue;
            }
            if b.borrow()
                .operations
                .iter()
                .any(|o| o.opname == "direct_call")
            {
                block_d = Some(b.clone());
            }
            let exits: Vec<_> = b.borrow().exits.clone();
            for link in &exits {
                if let Some(t) = link.borrow().target.clone() {
                    stack_d.push(t);
                }
            }
        }
        let block_d = block_d.expect("block_d holds the mallocfn direct_call");
        let bd = block_d.borrow();
        let malloc_size = bd
            .operations
            .iter()
            .find(|o| o.opname == "direct_call")
            .map(|o| o.args[1].clone())
            .expect("mallocfn(size) direct_call");
        let field_value = |field: &str| {
            bd.operations
                .iter()
                .find(|o| o.opname == "setfield" && o.args[1] == super::void_field_const(field))
                .map(|o| o.args[2].clone())
                .unwrap_or_else(|| panic!("block_d setfield {field}"))
        };
        let current_end = field_value("current_end");
        let total_size = field_value("total_size");
        assert_eq!(
            malloc_size, current_end,
            "mallocfn size must be the same value written to current_end"
        );
        assert_ne!(
            current_end, total_size,
            "current_end (rounded needed) must differ from total_size (total_new)"
        );
        assert_eq!(
            field_value("current_pos"),
            super::constant_with_lltype(super::ConstValue::Int(0), LowLevelType::Signed),
            "current_pos initialised to 0"
        );
        drop(bd);

        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    #[test]
    fn helper_builder_converts_ll_grow_by_overflow_llexitcase_to_vtable() {
        // `RPythonTyper.specialize_block` → `_convert_link` (`rtyper.py`):
        // `close_ovfcheck_block` OverflowError links go through
        // `rclass.get_type_repr.convert_const`, so `llexitcase` is the
        // low-level vtable (`ConstValue::LLPtr`), not the host class.
        use super::{ConstValue, Hlvalue, STRINGBUILDERPTR, STRINGPIECE, STRINGPIECEPTR, STRPTR};
        let (_ann, rtyper) = setup_rtyper();
        let helper = rtyper
            .lowlevel_helper_function_with_builder(
                "ll_grow_by".to_string(),
                vec![STRINGBUILDERPTR.clone(), LowLevelType::Signed],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    super::build_ll_grow_by_helper_graph(
                        "ll_grow_by",
                        STRINGBUILDERPTR.clone(),
                        STRINGPIECEPTR.clone(),
                        STRINGPIECE.clone(),
                        STRPTR.clone(),
                        dummy_funcptr_const(),
                    )
                },
            )
            .expect("ll_grow_by helper");
        let pygraph = helper.graph.expect("helper graph");
        let inner = pygraph.graph.borrow();
        let mut converted = 0usize;
        for block in inner.iterblocks() {
            if !block.borrow().canraise() {
                continue;
            }
            let exits = block.borrow().exits.clone();
            assert_eq!(exits.len(), 2, "ovfcheck has normal + OverflowError exits");
            match &exits[1].borrow().llexitcase {
                Some(Hlvalue::Constant(c)) => {
                    assert!(
                        matches!(c.value, ConstValue::LLPtr(_)),
                        "OverflowError llexitcase must be LLPtr vtable, got {:?}",
                        c.value
                    );
                    converted += 1;
                }
                other => panic!("expected Constant llexitcase, got {other:?}"),
            }
            match &exits[1].borrow().exitcase {
                Some(Hlvalue::Constant(c)) => {
                    assert!(
                        matches!(c.value, ConstValue::HostObject(_)),
                        "exitcase stays the host OverflowError class, got {:?}",
                        c.value
                    );
                }
                other => panic!("expected host-class exitcase, got {other:?}"),
            }
        }
        assert!(
            converted >= 1,
            "ll_grow_by must convert at least one OverflowError llexitcase"
        );
    }

    #[test]
    fn ll_grow_by_overflow_raises_memoryerror_as_llexception() {
        // Overflowing `ll_grow_by` takes the OverflowError handler
        // (`except OverflowError: raise MemoryError`) instead of
        // reporting mixed vtable/host-class representations.
        use std::any::Any;
        use std::rc::Rc;

        use super::{STRINGBUILDER, STRINGBUILDERPTR, STRINGPIECE, STRINGPIECEPTR, STRPTR};
        use crate::translator::rtyper::llinterp::{FrameError, LLInterpreter, LLValue};
        use crate::translator::rtyper::lltypesystem::lltype::{
            LowLevelValue, MallocFlavor, malloc,
        };
        use crate::translator::rtyper::rtyper::functionptr_const;

        let (ann, rtyper) = setup_rtyper();
        let helper = rtyper
            .lowlevel_helper_function_with_builder(
                "ll_grow_by".to_string(),
                vec![STRINGBUILDERPTR.clone(), LowLevelType::Signed],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    super::build_ll_grow_by_helper_graph(
                        "ll_grow_by",
                        STRINGBUILDERPTR.clone(),
                        STRINGPIECEPTR.clone(),
                        STRINGPIECE.clone(),
                        STRPTR.clone(),
                        dummy_funcptr_const(),
                    )
                },
            )
            .expect("ll_grow_by helper");
        let _c_func = functionptr_const(&rtyper, &helper).expect("functionptr const");
        let pygraph = helper.graph.expect("helper graph");
        let mut builder = malloc(STRINGBUILDER.clone(), None, MallocFlavor::Gc, true)
            .expect("malloc STRINGBUILDER");
        builder
            .setattr("total_size", LowLevelValue::Signed(1))
            .expect("total_size");
        let interp = Rc::new(LLInterpreter::new(rtyper.clone(), false, None));
        let err = interp
            .eval_graph(
                pygraph.graph.clone() as Rc<dyn Any>,
                vec![
                    Rc::new(LowLevelValue::Ptr(Box::new(builder))) as Rc<dyn Any>,
                    Rc::new(i64::MAX) as Rc<dyn Any>,
                ],
                false,
            )
            .expect_err("overflowing ll_grow_by raises");
        match err {
            FrameError::LLException { etype, evalue } => {
                let LLValue::Ptr(etype_ptr) = etype else {
                    panic!("etype must be the low-level vtable Ptr, got {etype:?}");
                };
                assert!(
                    matches!(evalue, LLValue::Ptr(_)),
                    "evalue must be the low-level instance Ptr, got {evalue:?}"
                );
                let ed = rtyper.exceptiondata().expect("exceptiondata");
                let memerr = crate::annotator::exception::standard_exception_classes()
                    .into_iter()
                    .find(|cls| cls.qualname() == "MemoryError")
                    .expect("MemoryError");
                let expected = crate::translator::rtyper::rmodel::inputconst(
                    ed.r_exception_type.as_ref(),
                    &super::ConstValue::HostObject(memerr),
                )
                .expect("MemoryError vtable");
                let super::ConstValue::LLPtr(expected_ptr) = expected.value else {
                    panic!(
                        "expected MemoryError vtable LLPtr, got {:?}",
                        expected.value
                    );
                };
                assert_eq!(
                    *etype_ptr, *expected_ptr,
                    "raised etype must be the MemoryError vtable"
                );
            }
            FrameError::Task(e) => panic!(
                "OverflowError handler must be taken, not a task error: {}",
                e.message
            ),
        }
        let _keep = ann;
    }

    #[test]
    fn ll_grow_by_exceptblock_link_carries_llptr_memoryerror_constants() {
        // `insert_link_conversions` → `inputconst(r_a2, a1.value)` for
        // exceptblock inputargs (`exceptiondata.r_exception_type` /
        // `r_exception_value`). After helper registration the MemoryError
        // raise link carries `ConstValue::LLPtr` in args[0] and args[1].
        use super::{ConstValue, Hlvalue, STRINGBUILDERPTR, STRINGPIECE, STRINGPIECEPTR, STRPTR};
        let (_ann, rtyper) = setup_rtyper();
        let helper = rtyper
            .lowlevel_helper_function_with_builder(
                "ll_grow_by".to_string(),
                vec![STRINGBUILDERPTR.clone(), LowLevelType::Signed],
                LowLevelType::Void,
                move |_r, _a, _res| {
                    super::build_ll_grow_by_helper_graph(
                        "ll_grow_by",
                        STRINGBUILDERPTR.clone(),
                        STRINGPIECEPTR.clone(),
                        STRINGPIECE.clone(),
                        STRPTR.clone(),
                        dummy_funcptr_const(),
                    )
                },
            )
            .expect("ll_grow_by helper");
        let pygraph = helper.graph.expect("helper graph");
        let inner = pygraph.graph.borrow();
        let exceptblock = inner.exceptblock.clone();
        let mut converted = 0usize;
        for block in inner.iterblocks() {
            let exits = block.borrow().exits.clone();
            for link in exits {
                let targets_except = match link.borrow().target.as_ref() {
                    Some(t) => std::rc::Rc::ptr_eq(t, &exceptblock),
                    None => false,
                };
                if !targets_except {
                    continue;
                }
                let args = &link.borrow().args;
                assert!(args.len() >= 2, "exceptblock link carries etype and evalue");
                match args[0].as_ref() {
                    Some(Hlvalue::Constant(c)) => {
                        assert!(
                            matches!(c.value, ConstValue::LLPtr(_)),
                            "exceptblock args[0] must be LLPtr vtable, got {:?}",
                            c.value
                        );
                    }
                    other => panic!("expected Constant args[0], got {other:?}"),
                }
                match args[1].as_ref() {
                    Some(Hlvalue::Constant(c)) => {
                        assert!(
                            matches!(c.value, ConstValue::LLPtr(_)),
                            "exceptblock args[1] must be LLPtr instance, got {:?}",
                            c.value
                        );
                    }
                    other => panic!("expected Constant args[1], got {other:?}"),
                }
                converted += 1;
            }
        }
        assert!(
            converted >= 1,
            "ll_grow_by must convert at least one exceptblock MemoryError link"
        );
    }

    #[test]
    fn build_ll_append_res0_computes_len_then_calls_ll_append() {
        use super::Hlvalue;
        let helper = super::build_ll_append_res0_helper_graph(
            "ll_append_res0",
            super::STRINGBUILDERPTR.clone(),
            super::STRPTR.clone(),
            super::STRPTR.clone(), // chars array ptr placeholder
            dummy_funcptr_const(),
        )
        .expect("build_ll_append_res0_helper_graph");
        assert_eq!(helper.func.name, "ll_append_res0");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["getsubstruct", "getarraysize", "direct_call"]);
        assert_eq!(startblock.inputargs.len(), 2);
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    #[test]
    fn build_ll_append_res_slice_subtracts_then_calls_ll_append() {
        use super::Hlvalue;
        let helper = super::build_ll_append_res_slice_helper_graph(
            "ll_append_res_slice",
            super::STRINGBUILDERPTR.clone(),
            super::STRPTR.clone(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_append_res_slice_helper_graph");
        assert_eq!(helper.func.name, "ll_append_res_slice");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(ops, vec!["int_sub", "direct_call"]);
        assert_eq!(startblock.inputargs.len(), 4);
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    #[test]
    fn build_ll_append_sized_calls_ll_append_with_constant_size() {
        use super::Hlvalue;

        for (name, start_is_zero, expected_inputargs) in [
            ("ll_append_0_5", true, 2usize),
            ("ll_append_start_5", false, 3usize),
        ] {
            let helper = super::build_ll_append_sized_helper_graph(
                name,
                super::STRINGBUILDERPTR.clone(),
                super::STRPTR.clone(),
                start_is_zero,
                5,
                dummy_funcptr_const(),
            )
            .expect("build_ll_append_sized_helper_graph");
            assert_eq!(helper.func.name, name);
            let inner = helper.graph.borrow();
            let (count, all_ops) = walk_ops(&inner.startblock);
            assert_eq!(count, 2);
            assert_eq!(all_ops, vec!["direct_call".to_string()]);

            let startblock = inner.startblock.borrow();
            assert_eq!(startblock.inputargs.len(), expected_inputargs);
            assert_eq!(startblock.operations[0].args.len(), 5);
            drop(startblock);

            let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
                panic!("returnblock inputarg must be a Variable");
            };
            assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
        }
    }

    #[test]
    fn build_ll__ll_append_branches_grow_vs_inline_copy() {
        use super::Hlvalue;
        let helper = super::build_ll__ll_append_helper_graph(
            "_ll_append",
            super::STRINGBUILDERPTR.clone(),
            super::STRPTR.clone(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll__ll_append_helper_graph");
        assert_eq!(helper.func.name, "_ll_append");
        let inner = helper.graph.borrow();

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getfield", "getfield", "int_sub", "int_lt"]);
        assert_eq!(startblock.inputargs.len(), 4);
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);

        let (count, all_ops) = walk_ops(&inner.startblock);
        // start, grow, copy, returnblock
        assert_eq!(count, 4);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        assert_eq!(n("getfield"), 3); // current_pos, current_end, current_buf
        assert_eq!(n("int_sub"), 1);
        assert_eq!(n("int_lt"), 1);
        assert_eq!(n("int_add"), 1);
        assert_eq!(n("setfield"), 1);
        assert_eq!(n("direct_call"), 2); // grow_and_append, copy_string_contents

        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    #[test]
    fn build_ll_fold_pieces_assembles_fastpath_and_copy_loop_cfg() {
        use super::Hlvalue;
        let helper = super::build_ll_fold_pieces_helper_graph(
            "ll_fold_pieces",
            super::STRINGBUILDERPTR.clone(),
            super::STRINGPIECEPTR.clone(),
            super::STRPTR.clone(),
            super::STRPTR.clone(), // chars array ptr placeholder
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_fold_pieces_helper_graph");
        assert_eq!(helper.func.name, "ll_fold_pieces");
        let inner = helper.graph.borrow();

        // header: getlength call, read+null extra_pieces, current_pos==0 test.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(
            start_ops,
            vec![
                "direct_call", // ll_getlength
                "getfield",    // extra_pieces
                "setfield",    // extra_pieces = null
                "getfield",    // current_pos
                "int_eq",
            ]
        );
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);

        let (count, all_ops) = walk_ops(&inner.startblock);
        // start, cond2, fast, slow, loop, loop_next, returnblock
        assert_eq!(count, 7);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        assert_eq!(n("direct_call"), 3); // ll_getlength, mallocfn, copy_string_contents
        assert_eq!(n("getfield"), 8);
        assert_eq!(n("setfield"), 9); // extra_pieces=null + 4 fast + 4 slow
        assert_eq!(n("int_eq"), 1);
        assert_eq!(n("int_sub"), 1); // dst -= piece_lgt
        assert_eq!(n("ptr_nonzero"), 2); // not prev_piece, not extra
        assert_eq!(n("getsubstruct"), 1); // len(piece.chars)
        assert_eq!(n("getarraysize"), 1);

        // Void return.
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    #[test]
    fn build_ll_build_branches_extra_then_shrink_and_returns_current_buf() {
        use super::Hlvalue;
        let helper = super::build_ll_build_helper_graph(
            "ll_build",
            super::STRINGBUILDERPTR.clone(),
            super::STRPTR.clone(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_build_helper_graph");
        assert_eq!(helper.func.name, "ll_build");
        let inner = helper.graph.borrow();

        // start: getfield extra_pieces; ptr_nonzero; 2-way branch.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getfield", "ptr_nonzero"]);
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);

        // Walk every reachable block, tallying op frequencies.
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![inner.startblock.clone()];
        let mut all_ops: Vec<String> = Vec::new();
        let mut count = 0usize;
        while let Some(b) = stack.pop() {
            if !seen.insert(std::rc::Rc::as_ptr(&b) as usize) {
                continue;
            }
            count += 1;
            let bb = b.borrow();
            for op in &bb.operations {
                all_ops.push(op.opname.clone());
            }
            for link in &bb.exits {
                if let Some(t) = link.borrow().target.clone() {
                    stack.push(t);
                }
            }
        }
        // start, fold, elif, shrink, ret, returnblock
        assert_eq!(count, 6);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        assert_eq!(n("getfield"), 4); // extra_pieces, current_pos, total_size, current_buf
        assert_eq!(n("ptr_nonzero"), 1);
        assert_eq!(n("int_ne"), 1);
        assert_eq!(n("direct_call"), 2); // fold + shrink arms

        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(super::STRPTR.clone())
        );
    }

    #[test]
    fn build_ll_arrayclear_zeroes_each_slot_in_a_bare_element_loop() {
        use super::Hlvalue;
        let helper = super::build_ll_arrayclear_helper_graph("ll_arrayclear", LowLevelType::Signed)
            .expect("build_ll_arrayclear_helper_graph");
        assert_eq!(helper.func.name, "ll_arrayclear");
        let inner = helper.graph.borrow();

        // start: length = len(p); i = 0 (a single unconditional exit).
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getarraysize"]);
        assert_eq!(startblock.inputargs.len(), 1); // p
        assert_eq!(startblock.exits.len(), 1);
        drop(startblock);

        let (count, all_ops) = walk_ops(&inner.startblock);
        // start, cond, body, returnblock.
        assert_eq!(count, 4);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        assert_eq!(n("getarraysize"), 1);
        assert_eq!(n("int_lt"), 1); // loop header
        assert_eq!(n("setarrayitem"), 1); // p[i] = ZERO
        assert_eq!(n("int_add"), 1); // i += 1
        assert_eq!(n("gc_writebarrier"), 0); // NULL store needs no barrier
        assert_eq!(n("raw_memclear"), 0); // fast path deferred

        // Void return.
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    #[test]
    fn build_ll_arrayfill_barriers_once_then_fills_each_slot() {
        use super::Hlvalue;
        let helper = super::build_ll_arrayfill_helper_graph("ll_arrayfill", LowLevelType::Signed)
            .expect("build_ll_arrayfill_helper_graph");
        assert_eq!(helper.func.name, "ll_arrayfill");
        let inner = helper.graph.borrow();

        // start: length = len(p); one gc_writebarrier(p) before the loop.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getarraysize", "gc_writebarrier"]);
        assert_eq!(startblock.inputargs.len(), 2); // p, item
        assert_eq!(startblock.exits.len(), 1);
        drop(startblock);

        let (count, all_ops) = walk_ops(&inner.startblock);
        // start, cond, body, returnblock.
        assert_eq!(count, 4);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        assert_eq!(n("getarraysize"), 1);
        assert_eq!(n("gc_writebarrier"), 1); // one barrier covers all slots
        assert_eq!(n("int_lt"), 1); // loop header
        assert_eq!(n("bare_setarrayitem"), 1); // p[i] = item, barrier-free
        assert_eq!(n("setarrayitem"), 0); // uses the bare store
        assert_eq!(n("int_add"), 1); // i += 1

        // Void return.
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    #[test]
    fn jit_specialized_size_table_matches_upstream_range() {
        assert_eq!(super::MAX_N, 10);
        assert_eq!(
            super::make_func_for_size(2),
            ("ll_append_0_2".into(), "ll_append_start_2".into(), 2)
        );
        assert_eq!(super::unroll_func_for_size.len(), 9);
        assert_eq!(super::unroll_func_for_size[0].2, 2);
        assert_eq!(super::unroll_func_for_size.last().unwrap().2, super::MAX_N);
    }

    /// Builds a unary `HighLevelOp` whose sole argument carries `repr` /
    /// `builder_ptr_lltype`, mirroring `rstr.rs`'s `build_string_unary_hop`.
    /// The `SomePtr` s-binding is non-constant so `inputarg` routes through
    /// `convertvar`, which short-circuits (same repr) and yields the arg
    /// variable unchanged.
    fn build_builder_unary_hop(
        rtyper: std::rc::Rc<crate::translator::rtyper::rtyper::RPythonTyper>,
        llops: std::rc::Rc<std::cell::RefCell<crate::translator::rtyper::rtyper::LowLevelOpList>>,
        opname: &str,
        builder_ptr_lltype: LowLevelType,
        result_lltype: LowLevelType,
        repr: std::sync::Arc<dyn Repr>,
    ) -> crate::translator::rtyper::rtyper::HighLevelOp {
        use crate::flowspace::model::{Hlvalue, SpaceOperation, Variable};
        let LowLevelType::Ptr(inner) = builder_ptr_lltype.clone() else {
            panic!("builder lltype must be a Ptr");
        };
        let v_arg = Variable::new();
        v_arg.set_concretetype(Some(builder_ptr_lltype));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(result_lltype));
        let hop = crate::translator::rtyper::rtyper::HighLevelOp::new(
            rtyper,
            SpaceOperation::new(
                opname.to_string(),
                vec![Hlvalue::Variable(v_arg)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops,
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Ptr(
                crate::translator::rtyper::lltypesystem::lltype::SomePtr::new(*inner),
            ));
        hop.args_r.borrow_mut().push(Some(repr));
        hop
    }

    fn setup_rtyper() -> (
        std::rc::Rc<crate::annotator::annrpython::RPythonAnnotator>,
        std::rc::Rc<crate::translator::rtyper::rtyper::RPythonTyper>,
    ) {
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(crate::translator::rtyper::rtyper::RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        // `ann` is returned so the caller keeps it alive — RPythonTyper
        // holds only a Weak upgrade to the annotator.
        (ann, rtyper)
    }

    fn assert_single_direct_call_to(
        llops: &std::rc::Rc<std::cell::RefCell<crate::translator::rtyper::rtyper::LowLevelOpList>>,
        helper_name: &str,
    ) {
        use crate::flowspace::model::Hlvalue;
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1, "expected one direct_call");
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr, got {:?}", ops.ops[0].args[0]);
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains(helper_name),
            "expected funcptr '{helper_name}' in {dbg}"
        );
    }

    /// rbuilder.py — `rtype_method_getlength` lowers to a single
    /// `direct_call` against the `ll_getlength` helper graph.
    #[test]
    fn stringbuilder_rtype_method_getlength_emits_direct_call() {
        use crate::flowspace::model::Hlvalue;
        let (_ann, rtyper) = setup_rtyper();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(
            crate::translator::rtyper::rtyper::LowLevelOpList::new(rtyper.clone(), None),
        ));
        let hop = build_builder_unary_hop(
            rtyper.clone(),
            llops.clone(),
            "getlength",
            super::STRINGBUILDERPTR.clone(),
            LowLevelType::Signed,
            super::stringbuilder_repr() as std::sync::Arc<dyn Repr>,
        );
        let result = super::stringbuilder_repr()
            .rtype_method("getlength", &hop)
            .unwrap_or_else(|err| panic!("rtype_method getlength: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        assert_single_direct_call_to(&llops, "ll_getlength");
    }

    /// SPIKE (L4-first) — `rtype_method("build")` lowers to a single
    /// `direct_call` against the `ll_build` helper graph. Proves the
    /// method-dispatch pipeline reaches `rtype_builder_build`; the spike
    /// helper is swapped for the full fold_pieces+shrink tree in L2-full.
    #[test]
    fn stringbuilder_rtype_method_build_emits_direct_call() {
        use crate::flowspace::model::Hlvalue;
        let (_ann, rtyper) = setup_rtyper();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(
            crate::translator::rtyper::rtyper::LowLevelOpList::new(rtyper.clone(), None),
        ));
        let hop = build_builder_unary_hop(
            rtyper.clone(),
            llops.clone(),
            "build",
            super::STRINGBUILDERPTR.clone(),
            super::STRPTR.clone(),
            super::stringbuilder_repr() as std::sync::Arc<dyn Repr>,
        );
        let result = super::stringbuilder_repr()
            .rtype_method("build", &hop)
            .unwrap_or_else(|err| panic!("rtype_method build: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        assert_single_direct_call_to(&llops, "ll_build");
    }

    /// SPIKE (L4-first) — `rtype_method("append")` (builder, str) lowers
    /// to a single `direct_call` against the `ll_append` helper graph.
    /// Proves the binary method-dispatch pipeline reaches
    /// `rtype_builder_append`; the spike helper is swapped for the full
    /// jit_append/_ll_append/grow tree in L2-full.
    #[test]
    fn stringbuilder_rtype_method_append_emits_direct_call() {
        use crate::flowspace::model::{Hlvalue, SpaceOperation, Variable};
        let (_ann, rtyper) = setup_rtyper();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(
            crate::translator::rtyper::rtyper::LowLevelOpList::new(rtyper.clone(), None),
        ));
        // args: (builder receiver, str piece).
        let v_builder = Variable::new();
        v_builder.set_concretetype(Some(super::STRINGBUILDERPTR.clone()));
        let v_piece = Variable::new();
        v_piece.set_concretetype(Some(super::STRPTR.clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));
        let hop = crate::translator::rtyper::rtyper::HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "append".to_string(),
                vec![
                    Hlvalue::Variable(v_builder.clone()),
                    Hlvalue::Variable(v_piece.clone()),
                ],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        {
            let LowLevelType::Ptr(inner) = super::STRINGBUILDERPTR.clone() else {
                panic!("builder lltype must be a Ptr");
            };
            let mut s = hop.args_s.borrow_mut();
            s.push(crate::annotator::model::SomeValue::Ptr(
                crate::translator::rtyper::lltypesystem::lltype::SomePtr::new(*inner),
            ));
            s.push(crate::annotator::model::SomeValue::String(
                crate::annotator::model::SomeString::new(false, false),
            ));
        }
        {
            let mut r = hop.args_r.borrow_mut();
            r.push(Some(super::stringbuilder_repr() as std::sync::Arc<dyn Repr>));
            r.push(Some(
                crate::translator::rtyper::rstr::string_repr() as std::sync::Arc<dyn Repr>
            ));
        }
        let result = super::stringbuilder_repr()
            .rtype_method("append", &hop)
            .unwrap_or_else(|err| panic!("rtype_method append: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_)) | None));
        assert_single_direct_call_to(&llops, "ll_append");

        // `lowlevel_helper_function_with_builder` is eager, so lowering
        // `append` materialises the WHOLE nested callee tree, not just the
        // top-level `ll_append`. Assert every sub-helper graph was minted +
        // registered — a silently-empty or cache-key-collided sub-builder
        // would pass the single-direct_call check above but fail here.
        let graph_names: std::collections::HashSet<String> = _ann
            .translator
            .graphs
            .borrow()
            .iter()
            .map(|g| g.borrow().name.clone())
            .collect();
        for expected in [
            "ll_append",
            "ll_append_res0",
            "_ll_append",
            "ll_grow_and_append",
            "ll_grow_by",
            "ll_copy_string_contents",
            "mallocstr",
        ] {
            assert!(
                graph_names.contains(expected),
                "append helper tree missing `{expected}`; minted graphs = {graph_names:?}"
            );
        }
    }

    /// `rtype_method("append_slice")` (builder, str, start, end) lowers to a
    /// single `direct_call` against `ll_append_slice` and eagerly mints its
    /// whole nested callee tree (res_slice + the shared `_ll_append` chain).
    #[test]
    fn stringbuilder_rtype_method_append_slice_emits_direct_call() {
        use super::{ConstValue, constant_with_lltype};
        use crate::flowspace::model::{Hlvalue, SpaceOperation, Variable};
        let (_ann, rtyper) = setup_rtyper();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(
            crate::translator::rtyper::rtyper::LowLevelOpList::new(rtyper.clone(), None),
        ));
        // args: (builder receiver, str piece, start, end); start/end are
        // Signed constants so `inputarg` binds them via `inputconst`.
        let v_builder = Variable::new();
        v_builder.set_concretetype(Some(super::STRINGBUILDERPTR.clone()));
        let v_piece = Variable::new();
        v_piece.set_concretetype(Some(super::STRPTR.clone()));
        let c_start = constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed);
        let c_end = constant_with_lltype(ConstValue::Int(5), LowLevelType::Signed);
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));
        let hop = crate::translator::rtyper::rtyper::HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "append_slice".to_string(),
                vec![
                    Hlvalue::Variable(v_builder.clone()),
                    Hlvalue::Variable(v_piece.clone()),
                    c_start,
                    c_end,
                ],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        {
            let LowLevelType::Ptr(inner) = super::STRINGBUILDERPTR.clone() else {
                panic!("builder lltype must be a Ptr");
            };
            let mut s = hop.args_s.borrow_mut();
            s.push(crate::annotator::model::SomeValue::Ptr(
                crate::translator::rtyper::lltypesystem::lltype::SomePtr::new(*inner),
            ));
            s.push(crate::annotator::model::SomeValue::String(
                crate::annotator::model::SomeString::new(false, false),
            ));
            s.push(crate::annotator::model::SomeValue::Integer(
                crate::annotator::model::SomeInteger::new(true, false),
            ));
            s.push(crate::annotator::model::SomeValue::Integer(
                crate::annotator::model::SomeInteger::new(true, false),
            ));
        }
        {
            let mut r = hop.args_r.borrow_mut();
            r.push(Some(super::stringbuilder_repr() as std::sync::Arc<dyn Repr>));
            r.push(Some(
                crate::translator::rtyper::rstr::string_repr() as std::sync::Arc<dyn Repr>
            ));
            // start/end are constants; `inputarg` never reads their source repr.
            r.push(None);
            r.push(None);
        }
        let result = super::stringbuilder_repr()
            .rtype_method("append_slice", &hop)
            .unwrap_or_else(|err| panic!("rtype_method append_slice: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_)) | None));
        assert_single_direct_call_to(&llops, "ll_append_slice");

        let graph_names: std::collections::HashSet<String> = _ann
            .translator
            .graphs
            .borrow()
            .iter()
            .map(|g| g.borrow().name.clone())
            .collect();
        for expected in [
            "ll_append_slice",
            "ll_append_res_slice",
            "_ll_append",
            "ll_grow_and_append",
            "ll_grow_by",
            "ll_copy_string_contents",
            "mallocstr",
        ] {
            assert!(
                graph_names.contains(expected),
                "append_slice helper tree missing `{expected}`; minted graphs = {graph_names:?}"
            );
        }
    }

    /// rbuilder.py — `rtype_bool` lowers to a single `direct_call`
    /// against the `ll_bool` helper graph.
    #[test]
    fn stringbuilder_rtype_bool_emits_direct_call() {
        use crate::flowspace::model::Hlvalue;
        let (_ann, rtyper) = setup_rtyper();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(
            crate::translator::rtyper::rtyper::LowLevelOpList::new(rtyper.clone(), None),
        ));
        let hop = build_builder_unary_hop(
            rtyper.clone(),
            llops.clone(),
            "bool",
            super::STRINGBUILDERPTR.clone(),
            LowLevelType::Bool,
            super::stringbuilder_repr() as std::sync::Arc<dyn Repr>,
        );
        let result = super::stringbuilder_repr()
            .rtype_bool(&hop)
            .unwrap_or_else(|err| panic!("rtype_bool: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        assert_single_direct_call_to(&llops, "ll_bool");
    }

    /// Builds a nullary `HighLevelOp` (constructor call with no arguments)
    /// whose result carries `builder_ptr_lltype`. `rtyper_new` reads
    /// `hop.nb_args() == 0` and materialises the `INIT_SIZE` const arg.
    fn build_builder_nullary_hop(
        rtyper: std::rc::Rc<crate::translator::rtyper::rtyper::RPythonTyper>,
        llops: std::rc::Rc<std::cell::RefCell<crate::translator::rtyper::rtyper::LowLevelOpList>>,
        builder_ptr_lltype: LowLevelType,
    ) -> crate::translator::rtyper::rtyper::HighLevelOp {
        use crate::flowspace::model::{Hlvalue, SpaceOperation, Variable};
        let v_result = Variable::new();
        v_result.set_concretetype(Some(builder_ptr_lltype));
        crate::translator::rtyper::rtyper::HighLevelOp::new(
            rtyper,
            SpaceOperation::new(
                "simple_call".to_string(),
                Vec::new(),
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops,
        )
    }

    /// rtyper/rbuilder.py — `rtyper_new` on a zero-argument
    /// constructor lowers to a single `direct_call` against the `ll_new`
    /// helper graph (the `INIT_SIZE` default feeds the const arg).
    #[test]
    fn stringbuilder_rtyper_new_zero_args_emits_direct_call() {
        use crate::flowspace::model::Hlvalue;
        let (_ann, rtyper) = setup_rtyper();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(
            crate::translator::rtyper::rtyper::LowLevelOpList::new(rtyper.clone(), None),
        ));
        let hop = build_builder_nullary_hop(
            rtyper.clone(),
            llops.clone(),
            super::STRINGBUILDERPTR.clone(),
        );
        let result = super::stringbuilder_repr()
            .rtyper_new(&hop)
            .unwrap_or_else(|err| panic!("rtyper_new: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        assert_single_direct_call_to(&llops, "ll_new");
    }

    /// rtyper/rbuilder.py — `rtyper_new` on a `with_capacity(n)` constructor
    /// (`len(hop.args_v) != 0`) threads the size operand into `ll_new`
    /// instead of the `INIT_SIZE` default, so the emitted `direct_call`
    /// carries the requested size (200), not the default.
    #[test]
    fn stringbuilder_rtyper_new_with_size_threads_size_into_direct_call() {
        use crate::flowspace::model::{Hlvalue, SpaceOperation, Variable};
        let (_ann, rtyper) = setup_rtyper();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(
            crate::translator::rtyper::rtyper::LowLevelOpList::new(rtyper.clone(), None),
        ));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(super::STRINGBUILDERPTR.clone()));
        // A `Constant` size arg lets `HighLevelOp.inputarg` short-circuit to
        // `inputconst` without an `args_r` binding — the single operand is all
        // `rtype_builder_new`'s `nb_args() != 0` branch needs.
        let size = super::constant_with_lltype(super::ConstValue::Int(200), LowLevelType::Signed);
        let hop = crate::translator::rtyper::rtyper::HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "newstringbuilder".to_string(),
                vec![size],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        let result = super::stringbuilder_repr()
            .rtyper_new(&hop)
            .unwrap_or_else(|err| panic!("rtyper_new: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        assert_single_direct_call_to(&llops, "ll_new");
        // `direct_call(ll_new_funcptr, size)` — the size operand (args[1]) is
        // the threaded 200, not the `INIT_SIZE` default.
        let ops = llops.borrow();
        let Hlvalue::Constant(size_arg) = &ops.ops[0].args[1] else {
            panic!(
                "ll_new size arg must be a Constant, got {:?}",
                ops.ops[0].args[1]
            );
        };
        let dbg = format!("{:?}", size_arg.value);
        assert!(dbg.contains("200"), "expected threaded size 200 in {dbg}");
    }

    /// The `newstringbuilder` ctor op flows through the rtyper's real
    /// `translate_operation` dispatch: the result builder repr's
    /// `rtyper_new` lowers it to a single `direct_call` against `ll_new`.
    #[test]
    fn translate_operation_newstringbuilder_routes_to_rtyper_new() {
        use crate::flowspace::model::{Hlvalue, SpaceOperation, Variable};
        let (_ann, rtyper) = setup_rtyper();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(
            crate::translator::rtyper::rtyper::LowLevelOpList::new(rtyper.clone(), None),
        ));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(super::STRINGBUILDERPTR.clone()));
        let hop = crate::translator::rtyper::rtyper::HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "newstringbuilder".to_string(),
                Vec::new(),
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        *hop.r_result.borrow_mut() = Some(super::stringbuilder_repr() as std::sync::Arc<dyn Repr>);
        let result = rtyper
            .translate_operation(&hop)
            .unwrap_or_else(|err| panic!("translate_operation newstringbuilder: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        assert_single_direct_call_to(&llops, "ll_new");
    }

    /// The `UnicodeBuilderRepr` constructor lowers to its own `ll_new`
    /// (distinct helper-graph cache key via the UNICODE result lltype).
    #[test]
    fn unicodebuilder_rtyper_new_zero_args_emits_direct_call() {
        use crate::flowspace::model::Hlvalue;
        let (_ann, rtyper) = setup_rtyper();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(
            crate::translator::rtyper::rtyper::LowLevelOpList::new(rtyper.clone(), None),
        ));
        let hop = build_builder_nullary_hop(
            rtyper.clone(),
            llops.clone(),
            super::UNICODEBUILDERPTR.clone(),
        );
        let result = super::unicodebuilder_repr()
            .rtyper_new(&hop)
            .unwrap_or_else(|err| panic!("rtyper_new: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        assert_single_direct_call_to(&llops, "ll_new");
    }

    /// rbuilder.py mirror for UnicodeBuilderRepr — same shape,
    /// distinct builder pointer lltype (`Ptr(UNICODEBUILDER)`).
    #[test]
    fn unicodebuilder_rtype_method_getlength_emits_direct_call() {
        use crate::flowspace::model::Hlvalue;
        let (_ann, rtyper) = setup_rtyper();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(
            crate::translator::rtyper::rtyper::LowLevelOpList::new(rtyper.clone(), None),
        ));
        let hop = build_builder_unary_hop(
            rtyper.clone(),
            llops.clone(),
            "getlength",
            super::UNICODEBUILDERPTR.clone(),
            LowLevelType::Signed,
            super::unicodebuilder_repr() as std::sync::Arc<dyn Repr>,
        );
        let result = super::unicodebuilder_repr()
            .rtype_method("getlength", &hop)
            .unwrap_or_else(|err| panic!("rtype_method getlength: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        assert_single_direct_call_to(&llops, "ll_getlength");
    }

    /// UnicodeBuilder mirror of the StringBuilder `append_slice` surface: the
    /// same shared `rtype_builder_append_slice` body, threaded with the UNICODE
    /// pointer / piece lltypes and the `mallocunicode` malloc helper.
    #[test]
    fn unicodebuilder_rtype_method_append_slice_emits_direct_call() {
        use super::{ConstValue, constant_with_lltype};
        use crate::flowspace::model::{Hlvalue, SpaceOperation, Variable};
        let (_ann, rtyper) = setup_rtyper();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(
            crate::translator::rtyper::rtyper::LowLevelOpList::new(rtyper.clone(), None),
        ));
        let v_builder = Variable::new();
        v_builder.set_concretetype(Some(super::UNICODEBUILDERPTR.clone()));
        let v_piece = Variable::new();
        v_piece.set_concretetype(Some(super::UNICODEPTR.clone()));
        let c_start = constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed);
        let c_end = constant_with_lltype(ConstValue::Int(5), LowLevelType::Signed);
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));
        let hop = crate::translator::rtyper::rtyper::HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "append_slice".to_string(),
                vec![
                    Hlvalue::Variable(v_builder.clone()),
                    Hlvalue::Variable(v_piece.clone()),
                    c_start,
                    c_end,
                ],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        {
            let LowLevelType::Ptr(inner) = super::UNICODEBUILDERPTR.clone() else {
                panic!("builder lltype must be a Ptr");
            };
            let mut s = hop.args_s.borrow_mut();
            s.push(crate::annotator::model::SomeValue::Ptr(
                crate::translator::rtyper::lltypesystem::lltype::SomePtr::new(*inner),
            ));
            s.push(crate::annotator::model::SomeValue::UnicodeString(
                crate::annotator::model::SomeUnicodeString::new(false, false),
            ));
            s.push(crate::annotator::model::SomeValue::Integer(
                crate::annotator::model::SomeInteger::new(true, false),
            ));
            s.push(crate::annotator::model::SomeValue::Integer(
                crate::annotator::model::SomeInteger::new(true, false),
            ));
        }
        {
            let mut r = hop.args_r.borrow_mut();
            r.push(Some(
                super::unicodebuilder_repr() as std::sync::Arc<dyn Repr>
            ));
            r.push(Some(
                crate::translator::rtyper::rstr::unicode_repr() as std::sync::Arc<dyn Repr>
            ));
            r.push(None);
            r.push(None);
        }
        let result = super::unicodebuilder_repr()
            .rtype_method("append_slice", &hop)
            .unwrap_or_else(|err| panic!("rtype_method append_slice: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_)) | None));
        assert_single_direct_call_to(&llops, "ll_append_slice");

        let graph_names: std::collections::HashSet<String> = _ann
            .translator
            .graphs
            .borrow()
            .iter()
            .map(|g| g.borrow().name.clone())
            .collect();
        for expected in [
            "ll_append_slice",
            "ll_append_res_slice",
            "_ll_append",
            "ll_grow_and_append",
            "ll_grow_by",
            "ll_copy_string_contents",
            "mallocunicode",
        ] {
            assert!(
                graph_names.contains(expected),
                "append_slice helper tree missing `{expected}`; minted graphs = {graph_names:?}"
            );
        }
    }

    /// rbuilder.py mirror for UnicodeBuilderRepr.
    #[test]
    fn unicodebuilder_rtype_bool_emits_direct_call() {
        use crate::flowspace::model::Hlvalue;
        let (_ann, rtyper) = setup_rtyper();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(
            crate::translator::rtyper::rtyper::LowLevelOpList::new(rtyper.clone(), None),
        ));
        let hop = build_builder_unary_hop(
            rtyper.clone(),
            llops.clone(),
            "bool",
            super::UNICODEBUILDERPTR.clone(),
            LowLevelType::Bool,
            super::unicodebuilder_repr() as std::sync::Arc<dyn Repr>,
        );
        let result = super::unicodebuilder_repr()
            .rtype_bool(&hop)
            .unwrap_or_else(|err| panic!("rtype_bool: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        assert_single_direct_call_to(&llops, "ll_bool");
    }
}
