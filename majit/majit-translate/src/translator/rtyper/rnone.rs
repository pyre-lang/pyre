//! RPython `rpython/rtyper/rnone.py` — `NoneRepr` + `none_repr`
//! singleton + `ll_none_hash` + `rtype_is_None` helper.
//!
//! Upstream rnone.py (84 LOC) covers five surfaces:
//!
//! | upstream | Rust mirror |
//! |---|---|
//! | `class NoneRepr(Repr)` (`rnone.py`) | [`NoneRepr`] |
//! | `none_repr = NoneRepr()` singleton (`rnone.py`) | [`none_repr`] |
//! | `SomeNone.rtyper_makerepr / rtyper_makekey` (`rnone.py`) | wired in [`super::rmodel::rtyper_makerepr`] + [`super::rmodel::rtyper_makekey`] |
//! | `ll_none_hash` (`rnone.py`) | [`ll_none_hash`] |
//! | `rtype_is_None(robj1, rnone2, hop, pos=0)` (`rnone.py`) | [`rtype_is_None`] |
//!
//! ## Wired through the pairtype dispatcher
//!
//! * `class __extend__(pairtype(Repr, NoneRepr))` + `pairtype(NoneRepr,
//!   Repr)` `convert_from_to` / `rtype_is_` (rnone.py) — wired via
//!   [`super::pairtype::pair_convert_from_to`] +
//!   [`super::pairtype::pair_rtype_is_`] dispatchers calling
//!   [`pair_any_none_convert_from_to`], [`pair_none_any_convert_from_to`],
//!   [`pair_any_none_rtype_is_`], and [`pair_none_any_rtype_is_`]. The
//!   freestanding [`rtype_is_None`] helper used to be the entire surface
//!   while the dispatcher was unported; it is still public so callers
//!   that already hold concrete reprs can skip the double-dispatch.
//! * The `SmallFunctionSetPBCRepr` branch (rnone.py) is wired
//!   through [`super::pairtype::ReprClassId::SmallFunctionSetPBCRepr`]
//!   plus [`Repr::pbc_s_pbc`] — `can_be_None` true emits
//!   `char_eq(v1, '\000')`, false folds to `inputconst(Bool, False)`.
//!
//! ## Deferred to follow-up commits
//!
//! * `get_ll_eq_function` / `get_ll_hash_function` /
//!   `get_ll_fasthash_function` (rnone.py) — the [`Repr`] trait
//!   does not yet carry these slots; they land with the `rdict.py`
//!   port that consumes them.
//! * `Address` branch of [`rtype_is_None`] (rnone.py) — requires
//!   `lltype.Address` + `adr_eq` + `robj1.null_instance()` which have no
//!   Rust counterpart (`rpython/rtyper/lltypesystem/llmemory.py` is
//!   unported). Marked with a structured TyperError placeholder.

use std::sync::Arc;
use std::sync::OnceLock;

use crate::flowspace::model::{ConstValue, Constant, Hlvalue};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::lltypesystem::rstr::{STRPTR, const_str_cache_llstr};
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult, HighLevelOp};

/// RPython `class NoneRepr(Repr)` (`rnone.py`).
///
/// ```python
/// class NoneRepr(Repr):
///     lowleveltype = Void
///     def rtype_bool(self, hop):
///         return Constant(False, Bool)
///     def none_call(self, hop):
///         raise TyperError("attempt to call constant None")
///     def ll_str(self, none):
///         return llstr("None")
///     def get_ll_eq_function(self):
///         return None
///     def get_ll_hash_function(self):
///         return ll_none_hash
///     get_ll_fasthash_function = get_ll_hash_function
///     rtype_simple_call = none_call
///     rtype_call_args = none_call
/// ```
#[derive(Debug)]
pub struct NoneRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl NoneRepr {
    pub fn new() -> Self {
        NoneRepr {
            state: ReprState::new(),
            lltype: LowLevelType::Void,
        }
    }
}

impl Default for NoneRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for NoneRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "NoneRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::NoneRepr
    }

    /// RPython `NoneRepr.get_ll_eq_function(self)` (`rnone.py`):
    ///
    /// ```python
    /// def get_ll_eq_function(self):
    ///     return None
    /// ```
    ///
    /// `NoneRepr.lowleveltype` is `Void`, so `None == None` is the
    /// only possible comparison; the primitive `int_eq` on Void
    /// reduces to constant `True` at the typer level. Returning
    /// `None` instructs callers to use the inline primitive op
    /// (`gen_eq_function`'s `eq_funcs[i] or operator.eq` fallback).
    fn get_ll_eq_function(
        &self,
        _rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        Ok(None)
    }

    /// RPython `NoneRepr.get_ll_hash_function(self)` (`rnone.py`):
    ///
    /// ```python
    /// def get_ll_hash_function(self):
    ///     return ll_none_hash
    ///
    /// def ll_none_hash(_):
    ///     return 0
    /// ```
    ///
    /// Synthesizes a single-block helper graph that ignores its `Void`
    /// inputarg and returns the constant `Signed 0`. Used by
    /// [`super::rtuple::gen_hash_function`] when a tuple item Repr is
    /// `NoneRepr` (e.g. `Optional[int]` items dispatched as `None`).
    fn get_ll_hash_function(
        &self,
        rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        let name = "ll_none_hash".to_string();
        rtyper
            .lowlevel_helper_function_with_builder(
                name.clone(),
                vec![LowLevelType::Void],
                LowLevelType::Signed,
                move |_rtyper, _args, _result| build_ll_none_hash_helper_graph(&name),
            )
            .map(Some)
    }

    /// rnone.py:28
    fn get_ll_fasthash_function(
        &self,
        rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        self.get_ll_hash_function(rtyper)
    }

    /// RPython `NoneRepr.rtype_bool(self, hop)` (`rnone.py`):
    /// `return Constant(False, Bool)`.
    ///
    /// Upstream returns a bare `Constant` with `concretetype=Bool`; no
    /// `genop` is emitted. The pyre form mirrors that by returning a
    /// `Hlvalue::Constant` carrying a typed `Constant`.
    fn rtype_bool(&self, _hop: &HighLevelOp) -> RTypeResult {
        Ok(Some(Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Bool(false),
            LowLevelType::Bool,
        ))))
    }

    /// RPython `Repr.rtype_str` (`rmodel.py`) calls
    /// `hop.gendirectcall(self.ll_str, v_self)`. `NoneRepr.ll_str`
    /// (`rnone.py`) is `return llstr("None")`.
    fn rtype_str(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_none_str".to_string(),
            vec![LowLevelType::Void],
            STRPTR.clone(),
            |_rtyper, _args, _result| build_ll_none_str_helper_graph("ll_none_str"),
        )?;
        hop.gendirectcall(&helper, vlist)
    }

    /// RPython `NoneRepr.none_call(self, hop)` aliased via
    /// `rtype_simple_call = none_call` (`rnone.py`):
    /// `raise TyperError("attempt to call constant None")`.
    fn rtype_simple_call(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(TyperError::message("attempt to call constant None"))
    }

    /// RPython `NoneRepr.rtype_call_args = none_call` (`rnone.py`).
    fn rtype_call_args(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(TyperError::message("attempt to call constant None"))
    }
}

/// RPython `none_repr = NoneRepr()` (`rnone.py`).
///
/// Upstream keeps a module-global singleton so `r is none_repr` ==
/// identity comparisons work throughout the typer. Pyre matches via an
/// `Arc<NoneRepr>` cached in a `OnceLock` and exposes `Arc::ptr_eq` as
/// the identity predicate (the same adaptation used by
/// [`super::rmodel::impossible_repr`]).
pub fn none_repr() -> Arc<NoneRepr> {
    static REPR: OnceLock<Arc<NoneRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(NoneRepr::new())).clone()
}

/// RPython `ll_none_hash(_)` (`rnone.py`):
/// `return 0`.
///
/// Returns the canonical None hash value. Upstream uses this as
/// `NoneRepr.get_ll_hash_function`'s return value; pyre exposes it as
/// a freestanding helper for the `rdict.py` port to consume directly
/// when that lands.
pub fn ll_none_hash(_: &ConstValue) -> i64 {
    0
}

/// Synthesizes the `ll_none_hash(_)` helper graph (`rnone.py`):
/// single block, ignores the `Void` inputarg, returns `Signed 0`.
///
/// Used by [`NoneRepr::get_ll_hash_function`] via
/// [`super::rtyper::RPythonTyper::lowlevel_helper_function_with_builder`].
pub(crate) fn build_ll_none_hash_helper_graph(
    name: &str,
) -> Result<crate::flowspace::pygraph::PyGraph, TyperError> {
    use crate::flowspace::model::{Block, BlockRefExt, FunctionGraph, GraphFunc, Hlvalue, Link};
    use crate::translator::rtyper::rtyper::{
        constant_with_lltype, helper_pygraph_from_graph, variable_with_lltype,
    };

    let arg = variable_with_lltype("_", LowLevelType::Void);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg)]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let zero_const = constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed);
    startblock.closeblock(vec![
        Link::new(vec![zero_const], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["_".to_string()],
        func,
    ))
}

/// Synthesizes `NoneRepr.ll_str` (`rnone.py`): `return llstr("None")`.
///
/// Single block, ignores the `Void` inputarg, returns the prebuilt
/// `'None'` `Ptr(STR)` (`annlowlevel.llstr`).
pub(crate) fn build_ll_none_str_helper_graph(
    name: &str,
) -> Result<crate::flowspace::pygraph::PyGraph, TyperError> {
    use crate::flowspace::model::{Block, BlockRefExt, FunctionGraph, GraphFunc, Hlvalue, Link};
    use crate::translator::rtyper::rtyper::{
        constant_with_lltype, helper_pygraph_from_graph, variable_with_lltype,
    };

    let arg = variable_with_lltype("_", LowLevelType::Void);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg)]);
    let return_var = variable_with_lltype("result", STRPTR.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let none_ptr = const_str_cache_llstr(b"None").map_err(TyperError::message)?;
    let none_const = constant_with_lltype(ConstValue::LLPtr(Box::new(none_ptr)), STRPTR.clone());
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
        vec!["_".to_string()],
        func,
    ))
}

/// RPython `rtype_is_None(robj1, rnone2, hop, pos=0)`
/// (`rnone.py`).
///
/// ```python
/// def rtype_is_None(robj1, rnone2, hop, pos=0):
///     if isinstance(robj1.lowleveltype, Ptr):
///         v1 = hop.inputarg(robj1, pos)
///         return hop.genop('ptr_iszero', [v1], resulttype=Bool)
///     elif robj1.lowleveltype == Address:
///         v1 = hop.inputarg(robj1, pos)
///         cnull = hop.inputconst(Address, robj1.null_instance())
///         return hop.genop('adr_eq', [v1, cnull], resulttype=Bool)
///     elif robj1 == none_repr:
///         return hop.inputconst(Bool, True)
///     elif isinstance(robj1, SmallFunctionSetPBCRepr):
///         if robj1.s_pbc.can_be_None:
///             v1 = hop.inputarg(robj1, pos)
///             return hop.genop('char_eq', [v1, inputconst(Char, '\000')],
///                              resulttype=Bool)
///         else:
///             return inputconst(Bool, False)
///     else:
///         raise TyperError('rtype_is_None of %r' % (robj1))
/// ```
///
/// `pos` defaults to 0 upstream; Rust callers pass `0` for
/// `pairtype(Repr, NoneRepr).rtype_is_` and `1` for
/// `pairtype(NoneRepr, Repr).rtype_is_` (`rnone.py`).
#[allow(non_snake_case)]
pub fn rtype_is_None(
    robj1: &dyn Repr,
    _rnone2: &NoneRepr,
    hop: &HighLevelOp,
    pos: usize,
) -> RTypeResult {
    // upstream: `isinstance(robj1.lowleveltype, Ptr)`.
    if matches!(robj1.lowleveltype(), LowLevelType::Ptr(_)) {
        let v1 = hop.inputarg(robj1, pos)?;
        return Ok(hop.genop(
            "ptr_iszero",
            vec![v1],
            GenopResult::LLType(LowLevelType::Bool),
        ));
    }
    // upstream: `elif robj1.lowleveltype == Address`
    // (`rnone.py:65-67`). `Address` is the lltype primitive defined
    // by `llmemory.py NULL = fakeaddress(None)`; pyre carries it
    // via `LowLevelType::Address` and its `_defl()` yields the
    // `_address::Null` default. Construct the NULL Address constant
    // through the same `_defl()` path that
    // `rpbc::null_address_constant` uses, then emit `adr_eq(v1, c_null)`.
    if matches!(robj1.lowleveltype(), LowLevelType::Address) {
        let v1 = hop.inputarg(robj1, pos)?;
        let null_value = match LowLevelType::Address._defl() {
            crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Address(addr) => {
                ConstValue::LLAddress(addr)
            }
            other => {
                return Err(TyperError::message(format!(
                    "rtype_is_None: Address._defl() must yield LowLevelValue::Address, got {other:?}",
                )));
            }
        };
        let c_null = Hlvalue::Constant(Constant::with_concretetype(
            null_value,
            LowLevelType::Address,
        ));
        return Ok(hop.genop(
            "adr_eq",
            vec![v1, c_null],
            GenopResult::LLType(LowLevelType::Bool),
        ));
    }

    // upstream: `elif robj1 == none_repr: return inputconst(Bool, True)`.
    // Upstream relies on `is`-based comparison (`r is none_repr`,
    // `rnone.py`). Pyre can't use Arc::ptr_eq here because the
    // signature takes `&dyn Repr` (no Arc handle surfaces at the
    // helpers' API boundary); the class-id tag is the equivalent
    // identity comparison since `NoneRepr` is instantiated only via
    // the `none_repr()` singleton.
    if matches!(
        robj1.repr_class_id(),
        super::pairtype::ReprClassId::NoneRepr
    ) {
        return HighLevelOp::inputconst(&LowLevelType::Bool, &ConstValue::Bool(true))
            .map(|c| Some(Hlvalue::Constant(c)));
    }
    // upstream rnone.py:76-82:
    //   elif isinstance(robj1, SmallFunctionSetPBCRepr):
    //       if robj1.s_pbc.can_be_None:
    //           v1 = hop.inputarg(robj1, pos)
    //           return hop.genop('char_eq', [v1, inputconst(Char, '\000')],
    //                            resulttype=Bool)
    //       else:
    //           return inputconst(Bool, False)
    if matches!(
        robj1.repr_class_id(),
        super::pairtype::ReprClassId::SmallFunctionSetPBCRepr
    ) {
        let s_pbc = robj1.pbc_s_pbc().ok_or_else(|| {
            TyperError::message("rtype_is_None: SmallFunctionSetPBCRepr missing pbc_s_pbc accessor")
        })?;
        if s_pbc.can_be_none {
            let v1 = hop.inputarg(robj1, pos)?;
            let c_zero = Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::byte_str(vec![0u8]),
                LowLevelType::Char,
            ));
            return Ok(hop.genop(
                "char_eq",
                vec![v1, c_zero],
                GenopResult::LLType(LowLevelType::Bool),
            ));
        } else {
            return HighLevelOp::inputconst(&LowLevelType::Bool, &ConstValue::Bool(false))
                .map(|c| Some(Hlvalue::Constant(c)));
        }
    }

    // upstream: `else: raise TyperError('rtype_is_None of %r' % (robj1))`.
    Err(TyperError::message(format!(
        "rtype_is_None of {}",
        robj1.repr_string()
    )))
}

// ____________________________________________________________
// pairtype helpers — dispatched from [`super::pairtype`].
//
// Upstream defines `class __extend__(pairtype(Repr, NoneRepr))` +
// `pairtype(NoneRepr, Repr)` at `rnone.py`. Pyre splits the two
// directions into four freestanding helpers that the central
// pair-dispatchers in `super::pairtype` call. The helper signatures
// follow `tool/pairtype.py`: a result of `Ok(None)` means "NotImplemented"
// so the dispatcher keeps walking pair_mro; `Ok(Some(v))` commits this
// arm.

/// RPython `pairtype(Repr, NoneRepr).convert_from_to` (rnone.py):
///
/// ```python
/// def convert_from_to((r_from, _), v, llops):
///     return inputconst(Void, None)
/// ```
///
/// Produces a `Constant(None)` pinned to `Void` — upstream uses this
/// to discard a value whose target repr is `NoneRepr` (i.e. the caller
/// wants the `None` sentinel and the source repr is irrelevant).
pub fn pair_any_none_convert_from_to(
    _r_from: &dyn Repr,
    _r_to: &dyn Repr,
    _v: &Hlvalue,
    _llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    Ok(Some(Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ))))
}

/// RPython `pairtype(NoneRepr, Repr).convert_from_to` (rnone.py):
///
/// ```python
/// def convert_from_to((_, r_to), v, llops):
///     return inputconst(r_to, None)
/// ```
///
/// Produces a `Constant(None)` pinned to the target repr's lowleveltype
/// — upstream uses this to synthesise the `None` default for any
/// nullable target (e.g. a Ptr, an Address, …).
pub fn pair_none_any_convert_from_to(
    _r_from: &dyn Repr,
    r_to: &dyn Repr,
    _v: &Hlvalue,
    _llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    let constant = crate::translator::rtyper::rmodel::inputconst(r_to, &ConstValue::None)?;
    Ok(Some(Hlvalue::Constant(constant)))
}

/// RPython `pairtype(Repr, NoneRepr).rtype_is_` (rnone.py):
///
/// ```python
/// def rtype_is_((robj1, rnone2), hop):
///     if hop.s_result.is_constant():
///         return hop.inputconst(Bool, hop.s_result.const)
///     return rtype_is_None(robj1, rnone2, hop, pos=0)
/// ```
pub fn pair_any_none_rtype_is_(
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &HighLevelOp,
) -> Result<Option<Hlvalue>, TyperError> {
    if let Some(folded) = rtype_is_constant_fold(hop)? {
        return Ok(Some(folded));
    }
    // The second side of the pair must be a NoneRepr — fetch the
    // singleton since the NoneRepr type itself carries no per-instance
    // state the helper needs (`rnone.py` uses only `rnone2` as
    // a type tag, never reads its fields).
    let none_side = none_repr();
    let _ = r2; // keeps signature parity with upstream pairtype helper
    rtype_is_None(r1, &none_side, hop, 0)
}

/// RPython `pairtype(NoneRepr, Repr).rtype_is_` (rnone.py):
///
/// ```python
/// def rtype_is_((rnone1, robj2), hop):
///     if hop.s_result.is_constant():
///         return hop.inputconst(Bool, hop.s_result.const)
///     return rtype_is_None(robj2, rnone1, hop, pos=1)
/// ```
pub fn pair_none_any_rtype_is_(
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &HighLevelOp,
) -> Result<Option<Hlvalue>, TyperError> {
    if let Some(folded) = rtype_is_constant_fold(hop)? {
        return Ok(Some(folded));
    }
    let none_side = none_repr();
    let _ = r1;
    rtype_is_None(r2, &none_side, hop, 1)
}

/// Constant-fold shared between `pairtype(Repr, NoneRepr).rtype_is_`
/// and `pairtype(NoneRepr, Repr).rtype_is_` (`rnone.py`):
/// `if hop.s_result.is_constant(): return hop.inputconst(Bool, const)`.
fn rtype_is_constant_fold(hop: &HighLevelOp) -> Result<Option<Hlvalue>, TyperError> {
    let s_result = hop.s_result.borrow();
    let Some(s) = s_result.as_ref() else {
        return Ok(None);
    };
    let Some(value) = s.const_() else {
        return Ok(None);
    };
    match value {
        ConstValue::Bool(_) => {
            let c = HighLevelOp::inputconst(&LowLevelType::Bool, value)?;
            Ok(Some(Hlvalue::Constant(c)))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::translator::rtyper::rmodel::{impossible_repr, setupstate};
    use crate::translator::rtyper::rtyper::RPythonTyper;

    #[test]
    fn none_repr_lowleveltype_is_void_and_repr_string_matches_upstream() {
        // rnone.py — `lowleveltype = Void`.
        let r = NoneRepr::new();
        assert_eq!(r.lowleveltype(), &LowLevelType::Void);
        // rmodel.py `<%s %s>` formatter — `<NoneRepr Void>`.
        assert_eq!(r.repr_string(), "<NoneRepr Void>");
        // rmodel.py compact_repr — "NoneRepr" → "NoneR" replacement
        // followed by short_name.
        assert_eq!(r.compact_repr(), "NoneR Void");
    }

    #[test]
    fn none_repr_singleton_returns_same_arc() {
        // rnone.py — `none_repr = NoneRepr()` module-global; pyre
        // mirrors via an `Arc<NoneRepr>` cached in OnceLock.
        let a = none_repr();
        let b = none_repr();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn none_repr_singleton_is_distinct_from_impossible_repr() {
        // rnone.py:33 vs rmodel.py — both are `Void`-typed Repr
        // singletons but have different Python classes (NoneRepr vs
        // VoidRepr); pyre preserves the identity distinction.
        let n = none_repr();
        let i = impossible_repr();
        let n_erased = n.clone() as Arc<dyn Repr>;
        let i_erased = i.clone() as Arc<dyn Repr>;
        assert!(!Arc::ptr_eq(&n_erased, &i_erased));
    }

    #[test]
    fn setup_on_none_repr_reaches_finished_state() {
        // rmodel.py setup state machine — NoneRepr inherits the default
        // `_setup_repr` (no-op) so `setup()` should transition directly
        // NOTINITIALIZED → FINISHED.
        let r = NoneRepr::new();
        assert_eq!(r.state().get(), setupstate::NotInitialized);
        r.setup().expect("NoneRepr.setup() should succeed");
        assert_eq!(r.state().get(), setupstate::Finished);
    }

    /// rnone.py — `NoneRepr.get_ll_hash_function` returns the
    /// `ll_none_hash` function. The synthesized helper graph has a
    /// single block with no operations and a closeblock returning the
    /// constant `Signed 0` directly.
    #[test]
    fn none_repr_get_ll_hash_function_synthesizes_zero_returning_helper() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r = none_repr();

        let llfn = r
            .get_ll_hash_function(&rtyper)
            .expect("get_ll_hash_function(None)")
            .expect("returns Some helper");
        assert_eq!(llfn.name, "ll_none_hash");
        assert_eq!(llfn.args, vec![LowLevelType::Void]);
        assert_eq!(llfn.result, LowLevelType::Signed);

        let graph = llfn.graph.as_ref().expect("helper carries a graph");
        let inner = graph.graph.borrow();
        let startblock = inner.startblock.borrow();
        assert!(
            startblock.operations.is_empty(),
            "ll_none_hash has no operations, got {:?}",
            startblock.operations
        );
        assert_eq!(startblock.exits.len(), 1);
        let exit = startblock.exits[0].borrow();
        assert_eq!(exit.args.len(), 1);
        match exit.args[0].as_ref() {
            Some(Hlvalue::Constant(c)) => {
                assert_eq!(c.value, ConstValue::Int(0));
                assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Signed));
            }
            other => panic!("expected Constant Signed 0, got {other:?}"),
        }
    }

    #[test]
    fn ll_none_hash_returns_zero() {
        // rnone.py — `return 0`.
        assert_eq!(ll_none_hash(&ConstValue::None), 0);
        // Upstream `ll_none_hash` ignores its argument; pyre keeps the
        // same shape. Any ConstValue should produce 0.
        assert_eq!(ll_none_hash(&ConstValue::Int(42)), 0);
    }

    /// rnone.py — `NoneRepr.ll_str` returns `llstr("None")`. The helper
    /// graph ignores its Void argument and closeblock-returns that
    /// prebuilt `Ptr(STR)`.
    #[test]
    fn none_repr_ll_str_helper_returns_prebuilt_none_string() {
        use crate::translator::rtyper::lltypesystem::rstr::prebuilt_str_bytes_and_hash;

        let graph = build_ll_none_str_helper_graph("ll_none_str").expect("ll_none_str graph");
        let inner = graph.graph.borrow();
        let startblock = inner.startblock.borrow();
        assert!(startblock.operations.is_empty());
        assert_eq!(startblock.exits.len(), 1);
        let exit = startblock.exits[0].borrow();
        match exit.args[0].as_ref() {
            Some(Hlvalue::Constant(c)) => {
                assert_eq!(c.concretetype.as_ref(), Some(&*STRPTR));
                let ConstValue::LLPtr(p) = &c.value else {
                    panic!("expected Ptr(STR), got {:?}", c.value);
                };
                let (bytes, _) = prebuilt_str_bytes_and_hash(p).expect("prebuilt STR chars");
                assert_eq!(bytes, b"None");
            }
            other => panic!("expected Constant Ptr(STR) 'None', got {other:?}"),
        }
    }

    #[test]
    fn rtype_simple_call_on_none_raises_typer_error() {
        // rnone.py + rnone.py — `rtype_simple_call = none_call`
        // raises `TyperError("attempt to call constant None")`.
        let r = NoneRepr::new();
        let hop = dummy_hop();
        let err = r.rtype_simple_call(&hop).unwrap_err();
        assert!(err.to_string().contains("attempt to call constant None"));
    }

    #[test]
    fn rtype_call_args_on_none_raises_same_typer_error() {
        // rnone.py — `rtype_call_args = none_call`.
        let r = NoneRepr::new();
        let hop = dummy_hop();
        let err = r.rtype_call_args(&hop).unwrap_err();
        assert!(err.to_string().contains("attempt to call constant None"));
    }

    #[test]
    fn rtyper_getrepr_on_some_none_returns_the_singleton() {
        // rnone.py — SomeNone.rtyper_makerepr returns the module-
        // global `none_repr`. Verify the full dispatch chain
        // `rtyper.getrepr → rtyper_makerepr → none_repr()` keeps that
        // singleton identity intact.
        use crate::annotator::model::{SomeNone, SomeValue};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let s_none = SomeValue::None_(SomeNone::new());
        let r = rtyper.getrepr(&s_none).expect("getrepr(SomeNone)");
        let expected = none_repr() as Arc<dyn Repr>;
        assert!(Arc::ptr_eq(&r, &expected));

        // rtyper.py:54-57 cache sanity — the same SomeNone key must
        // dedupe to the same Arc on a second lookup.
        let r2 = rtyper.getrepr(&s_none).expect("second getrepr(SomeNone)");
        assert!(Arc::ptr_eq(&r, &r2));
    }

    #[test]
    fn rtype_bool_on_none_returns_constant_false() {
        // rnone.py — `return Constant(False, Bool)`. Pyre
        // returns `Some(Hlvalue::Constant(...))` carrying the typed
        // constant directly (no genop emitted).
        let r = NoneRepr::new();
        let hop = dummy_hop();
        let Some(Hlvalue::Constant(c)) = r.rtype_bool(&hop).expect("rtype_bool should succeed")
        else {
            panic!("rtype_bool should return a Constant");
        };
        assert_eq!(c.value, ConstValue::Bool(false));
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Bool));
    }

    fn dummy_hop() -> HighLevelOp {
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let spaceop = SpaceOperation::new(
            OpKind::SimpleCall.opname(),
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(RefCell::new(
            crate::translator::rtyper::rtyper::LowLevelOpList::new(rtyper.clone(), None),
        ));
        HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops)
    }
}
