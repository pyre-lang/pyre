//! RPython `rpython/rtyper/rfloat.py` — `FloatRepr` + `SingleFloatRepr`
//! + `LongFloatRepr` + singleton + `SomeFloat/SingleFloat/LongFloat`
//!   dispatch.
//!
//! Upstream rfloat.py (174 LOC) covers three Reprs:
//!
//! | upstream | Rust mirror |
//! |---|---|
//! | `class FloatRepr(Repr)` (`rfloat.py`) | [`FloatRepr`] |
//! | `float_repr = FloatRepr()` singleton (`rfloat.py`) | [`float_repr`] |
//! | `class SingleFloatRepr(Repr)` (`rfloat.py`) | [`SingleFloatRepr`] |
//! | `class LongFloatRepr(Repr)` (`rfloat.py`) | [`LongFloatRepr`] |
//! | `SomeFloat/SomeSingleFloat/SomeLongFloat rtyper_make*` (`rfloat.py`) | wired in [`super::rmodel::rtyper_makerepr`] + [`super::rmodel::rtyper_makekey`] |
//! | `pairtype(FloatRepr, FloatRepr).rtype_add/sub/mul/truediv` (`rfloat.py`) | dispatched via [`super::pairtype::pair_rtype_add`] etc. → [`rtype_template`] |
//! | `pairtype(FloatRepr, FloatRepr).rtype_eq/ne/lt/le/gt/ge/is_` (`rfloat.py`) | dispatched via [`super::pairtype::pair_rtype_eq`] etc. → [`rtype_compare_template`] |
//! | `_rtype_template(hop, func)` / `_rtype_compare_template(hop, func)` helpers (`rfloat.py`) | [`rtype_template`] / [`rtype_compare_template`] |
//!
//! ## Deferred to follow-up commits
//!
//! * `get_ll_{gt,lt,ge,le}_function` (`rfloat.py:21-25`) — require
//!   trait slots for ordering helpers. Used by `rdict.py` /
//!   `rordereddict.py` when those land; no current consumer. (`get_ll_eq`
//!   and `get_ll_hash` are landed on the [`Repr`] trait.)
//! * `ll_str(self, f)` (`rfloat.py`) — depends on
//!   `rpython/rlib/rfloat.py:formatd` + `annlowlevel.llstr` which are
//!   not ported.

use std::sync::Arc;
use std::sync::OnceLock;

use crate::flowspace::model::{
    Block, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{
    ConvertedTo, GenopResult, HighLevelOp, constant_with_lltype, helper_pygraph_from_graph,
    variable_with_lltype,
};

/// RPython `class FloatRepr(Repr)` (`rfloat.py`).
///
/// ```python
/// class FloatRepr(Repr):
///     lowleveltype = Float
///
///     def convert_const(self, value):
///         if not isinstance(value, (int, base_int, float)):  # can be bool too
///             raise TyperError("not a float: %r" % (value,))
///         return float(value)
///
///     def rtype_bool(_, hop):
///         vlist = hop.inputargs(Float)
///         return hop.genop('float_is_true', vlist, resulttype=Bool)
///
///     def rtype_neg(_, hop):
///         vlist = hop.inputargs(Float)
///         return hop.genop('float_neg', vlist, resulttype=Float)
///
///     def rtype_pos(_, hop):
///         vlist = hop.inputargs(Float)
///         return vlist[0]
///
///     def rtype_abs(_, hop):
///         vlist = hop.inputargs(Float)
///         return hop.genop('float_abs', vlist, resulttype=Float)
///
///     def rtype_int(_, hop):
///         vlist = hop.inputargs(Float)
///         hop.exception_cannot_occur()
///         return hop.genop('cast_float_to_int', vlist, resulttype=Signed)
///
///     def rtype_float(_, hop):
///         vlist = hop.inputargs(Float)
///         hop.exception_cannot_occur()
///         return vlist[0]
/// ```
#[derive(Debug)]
pub struct FloatRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl FloatRepr {
    pub fn new() -> Self {
        FloatRepr {
            state: ReprState::new(),
            lltype: LowLevelType::Float,
        }
    }
}

impl Default for FloatRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for FloatRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "FloatRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::FloatRepr
    }

    /// RPython `FloatRepr.get_ll_eq_function(self)` (`rfloat.py`):
    ///
    /// ```python
    /// def get_ll_eq_function(self):
    ///     return None
    /// ```
    ///
    /// Returning `None` instructs callers (`gen_eq_function` /
    /// `rtype_contains`) to fall back to the primitive `float_eq`
    /// inline op via `eq_funcs[i] or operator.eq`.
    fn get_ll_eq_function(
        &self,
        _rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        Ok(None)
    }

    /// RPython `FloatRepr.get_ll_hash_function(self)` (`rfloat.py`):
    ///
    /// ```python
    /// def get_ll_hash_function(self):
    ///     return _hash_float
    /// ```
    ///
    /// Synthesizes the multi-block `_hash_float(f) -> Signed` helper
    /// graph porting `rlib/objectmodel.py`. See
    /// [`build_ll_hash_float_helper_graph`] for the layout.
    fn get_ll_hash_function(
        &self,
        rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        let name = "_hash_float".to_string();
        rtyper
            .lowlevel_helper_function_with_builder(
                name.clone(),
                vec![LowLevelType::Float],
                LowLevelType::Signed,
                move |_rtyper, _args, _result| build_ll_hash_float_helper_graph(&name),
            )
            .map(Some)
    }

    /// RPython `FloatRepr.convert_const(self, value)` (`rfloat.py`):
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if not isinstance(value, (int, base_int, float)):  # can be bool too
    ///         raise TyperError("not a float: %r" % (value,))
    ///     return float(value)
    /// ```
    ///
    /// `base_int` covers RPython's `r_uint` / `r_longlong` etc. from
    /// `rlib/rarithmetic.py`; `r_uint` is `ConstValue::UInt`.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        let as_float = match value {
            ConstValue::Float(bits) => *bits,
            ConstValue::Int(i) => (*i as f64).to_bits(),
            ConstValue::UInt(i) => (*i as f64).to_bits(),
            ConstValue::Bool(b) => (if *b { 1.0_f64 } else { 0.0_f64 }).to_bits(),
            other => {
                return Err(TyperError::message(format!("not a float: {other:?}")));
            }
        };
        Ok(Constant::with_concretetype(
            ConstValue::Float(as_float),
            LowLevelType::Float,
        ))
    }

    /// RPython `FloatRepr.rtype_bool(_, hop)` (`rfloat.py`):
    /// `return hop.genop('float_is_true', [v], resulttype=Bool)`.
    fn rtype_bool(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Float)])?;
        Ok(hop.genop(
            "float_is_true",
            vlist,
            GenopResult::LLType(LowLevelType::Bool),
        ))
    }

    /// RPython `FloatRepr.rtype_neg(_, hop)` (`rfloat.py`):
    /// `return hop.genop('float_neg', [v], resulttype=Float)`.
    fn rtype_neg(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Float)])?;
        Ok(hop.genop("float_neg", vlist, GenopResult::LLType(LowLevelType::Float)))
    }

    /// RPython `FloatRepr.rtype_pos(_, hop)` (`rfloat.py`):
    /// `return vlist[0]` — identity pass-through after inputargs coerces.
    fn rtype_pos(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Float)])?;
        Ok(vlist.into_iter().next())
    }

    /// RPython `FloatRepr.rtype_abs(_, hop)` (`rfloat.py`):
    /// `return hop.genop('float_abs', [v], resulttype=Float)`.
    fn rtype_abs(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Float)])?;
        Ok(hop.genop("float_abs", vlist, GenopResult::LLType(LowLevelType::Float)))
    }

    /// RPython `FloatRepr.rtype_int(_, hop)` (`rfloat.py`):
    ///
    /// ```python
    /// def rtype_int(_, hop):
    ///     vlist = hop.inputargs(Float)
    ///     # int(x) never raises in RPython, you need to use
    ///     # rarithmetic.ovfcheck_float_to_int() if you want this
    ///     hop.exception_cannot_occur()
    ///     return hop.genop('cast_float_to_int', vlist, resulttype=Signed)
    /// ```
    fn rtype_int(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Float)])?;
        hop.exception_cannot_occur()?;
        Ok(hop.genop(
            "cast_float_to_int",
            vlist,
            GenopResult::LLType(LowLevelType::Signed),
        ))
    }

    /// RPython `FloatRepr.rtype_float(_, hop)` (`rfloat.py`):
    /// identity pass-through plus `exception_cannot_occur`.
    fn rtype_float(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Float)])?;
        hop.exception_cannot_occur()?;
        Ok(vlist.into_iter().next())
    }
}

/// RPython `float_repr = FloatRepr()` (`rfloat.py`).
pub fn float_repr() -> Arc<FloatRepr> {
    static REPR: OnceLock<Arc<FloatRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(FloatRepr::new())).clone()
}

/// RPython `_hash_float(f)` (`rlib/objectmodel.py`):
///
/// ```python
/// def _hash_float(f):
///     if not isfinite(f):
///         if math.isinf(f):
///             if f < 0.0:
///                 return -271828
///             else:
///                 return 314159
///         else: #isnan(f):
///             return 0
///     v, expo = math.frexp(f)
///     v *= TAKE_NEXT
///     hipart = int(v)
///     v = (v - float(hipart)) * TAKE_NEXT
///     x = hipart + int(v) + (expo << 15)
///     return intmask(x)
/// TAKE_NEXT = float(2**31)
/// ```
///
/// **Subnormal handling**: IEEE 754 subnormal floats (`exp_raw == 0`
/// with non-zero mantissa) need their leading-bit position recovered
/// because the encoded exponent does not capture it. Pyre's
/// lloperation surface has no `int_clz` op, but `cast_longlong_to_float`
/// exists and is exact for any 52-bit integer (fits in f64's 53-bit
/// mantissa). Casting `mantissa_lo` to f64 and reading back its IEEE
/// biased exponent yields `1023 + B` where `B` is the leading bit
/// position — equivalent to a CLZ result without a CLZ helper graph.
/// `block_finite_nonzero` thus branches on `exp_raw == 0` and
/// converges on a normalized `(expo_ll, mantissa_lo, sign_bit)`
/// triple in `block_join` before running the hash arithmetic.
///
/// Synthesizes an 8-block graph:
///
/// - **start**: `diff = float_sub(f, f); isfin = float_eq(diff, 0.0)`.
///   `(f - f) == 0.0` is the JIT-friendly `isfinite(f)` shape from
///   `ll_math.py`. exitswitch on `isfin`: True →
///   `block_finite`, False → `block_not_finite`.
/// - **block_not_finite**: `is_self = float_eq(f, f)` distinguishes
///   `inf` from `NaN` (NaN ≠ NaN). exitswitch: True → `block_inf`,
///   False → return constant 0 (NaN case).
/// - **block_inf**: `is_neg = float_lt(f, 0.0)`. exitswitch: True →
///   return constant `-271828`, False → return constant `314159`.
/// - **block_finite**: `is_zero = float_eq(f, 0.0)`. The bit-
///   manipulation `decompose_float` formula returns `(0.5, -1022)`
///   for `f == 0.0` instead of `(0.0, 0)`, so the zero case must
///   short-circuit to constant 0 (matches `_hash_float(0.0) == 0`
///   under upstream's `math.frexp` semantics).
/// - **block_finite_nonzero**: extract `bits`, `bits_high`,
///   `exp_raw`, `mantissa_lo`, `sign_bit` from the float bit
///   pattern; branch on `is_subnormal = (exp_raw == 0)`.
/// - **block_subnormal**: round-trip `mantissa_lo` through
///   `cast_longlong_to_float`/`convert_float_bytes_to_longlong` to
///   recover `B = leading_bit_position(mantissa_lo)`. Compute
///   `expo_ll = B - 1073` and shift `mantissa_lo` left by `(52 - B)`
///   so the leading 1 falls at bit 52 (becomes the IEEE 754
///   implicit-bit slot when masked to bits 0..51).
/// - **block_normal**: `expo_ll = exp_raw - 1022`. Mantissa already
///   carries the implicit-bit-equivalent fraction in bits 0..51, so
///   it passes through unchanged.
/// - **block_join**: assemble `mantissa_bits = mantissa_lo | (1022 <<
///   52) | sign_bit`, decode to `mantissa`, truncate `expo_ll` to
///   `expo`, then run PyPy's `_hash_float` arithmetic
///   (`hipart + int(v) + (expo << 15)`). Pure existing-op chain; no
///   backend lowering changes required.
///
/// `TAKE_NEXT = 2147483648.0` (`2**31`); inlined as a Float constant.
/// `intmask(x)` is identity on a 64-bit Signed host.
pub(crate) fn build_ll_hash_float_helper_graph(name: &str) -> Result<PyGraph, TyperError> {
    // Block layout (created up-front so closeblock can reference
    // successors).
    let f_start = variable_with_lltype("f", LowLevelType::Float);
    let startblock = Block::shared(vec![Hlvalue::Variable(f_start.clone())]);

    let f_nf = variable_with_lltype("f_nf", LowLevelType::Float);
    let block_not_finite = Block::shared(vec![Hlvalue::Variable(f_nf.clone())]);

    let f_inf = variable_with_lltype("f_inf", LowLevelType::Float);
    let block_inf = Block::shared(vec![Hlvalue::Variable(f_inf.clone())]);

    let f_fin = variable_with_lltype("f_fin", LowLevelType::Float);
    let block_finite = Block::shared(vec![Hlvalue::Variable(f_fin.clone())]);

    let f_nz = variable_with_lltype("f_nz", LowLevelType::Float);
    let block_finite_nonzero = Block::shared(vec![Hlvalue::Variable(f_nz.clone())]);

    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let float_zero = || constant_with_lltype(ConstValue::float(0.0), LowLevelType::Float);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);

    // start block: isfinite check via `(f - f) == 0.0`.
    let diff = variable_with_lltype("diff", LowLevelType::Float);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "float_sub",
        vec![
            Hlvalue::Variable(f_start.clone()),
            Hlvalue::Variable(f_start.clone()),
        ],
        Hlvalue::Variable(diff.clone()),
    ));
    let isfin = variable_with_lltype("isfin", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "float_eq",
        vec![Hlvalue::Variable(diff), float_zero()],
        Hlvalue::Variable(isfin.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(isfin));
    let start_true_link = Link::new(
        vec![Hlvalue::Variable(f_start.clone())],
        Some(block_finite.clone()),
        Some(bool_true()),
    )
    .into_ref();
    let start_false_link = Link::new(
        vec![Hlvalue::Variable(f_start)],
        Some(block_not_finite.clone()),
        Some(bool_false()),
    )
    .into_ref();
    startblock.closeblock(vec![start_true_link, start_false_link]);

    // block_not_finite: distinguish inf from NaN via `f == f`.
    let is_self = variable_with_lltype("is_self", LowLevelType::Bool);
    block_not_finite
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "float_eq",
            vec![
                Hlvalue::Variable(f_nf.clone()),
                Hlvalue::Variable(f_nf.clone()),
            ],
            Hlvalue::Variable(is_self.clone()),
        ));
    block_not_finite.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_self));
    let nf_true_link = Link::new(
        vec![Hlvalue::Variable(f_nf.clone())],
        Some(block_inf.clone()),
        Some(bool_true()),
    )
    .into_ref();
    let nf_false_link = Link::new(
        vec![signed_const(0)],
        Some(graph.returnblock.clone()),
        Some(bool_false()),
    )
    .into_ref();
    block_not_finite.closeblock(vec![nf_true_link, nf_false_link]);

    // block_inf: branch on sign.
    let is_neg = variable_with_lltype("is_neg", LowLevelType::Bool);
    block_inf.borrow_mut().operations.push(SpaceOperation::new(
        "float_lt",
        vec![Hlvalue::Variable(f_inf), float_zero()],
        Hlvalue::Variable(is_neg.clone()),
    ));
    block_inf.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_neg));
    let inf_neg_link = Link::new(
        vec![signed_const(-271828)],
        Some(graph.returnblock.clone()),
        Some(bool_true()),
    )
    .into_ref();
    let inf_pos_link = Link::new(
        vec![signed_const(314159)],
        Some(graph.returnblock.clone()),
        Some(bool_false()),
    )
    .into_ref();
    block_inf.closeblock(vec![inf_neg_link, inf_pos_link]);

    // block_finite: short-circuit `f == 0.0` to return 0 since the
    // bit-manipulation `decompose_float` formula returns (0.5, -1022)
    // for f == 0.0 (the formula's output is `frexp` for non-zero
    // values only). _hash_float(0.0) is canonically 0.
    let is_zero = variable_with_lltype("is_zero", LowLevelType::Bool);
    block_finite
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "float_eq",
            vec![Hlvalue::Variable(f_fin.clone()), float_zero()],
            Hlvalue::Variable(is_zero.clone()),
        ));
    block_finite.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_zero));
    let fin_zero_link = Link::new(
        vec![signed_const(0)],
        Some(graph.returnblock.clone()),
        Some(bool_true()),
    )
    .into_ref();
    let fin_nonzero_link = Link::new(
        vec![Hlvalue::Variable(f_fin)],
        Some(block_finite_nonzero.clone()),
        Some(bool_false()),
    )
    .into_ref();
    block_finite.closeblock(vec![fin_zero_link, fin_nonzero_link]);

    // block_finite_nonzero: extract IEEE 754 components, then
    // branch on `exp_raw == 0` (subnormal) so the two flavors
    // converge on a normalized `(expo_ll, mantissa_lo, sign_bit)`
    // before reconstructing the [0.5, 1)-mantissa float for the hash
    // arithmetic.
    //
    // `bits = f.to_bits()` (rebinding f64 ↔ i64).
    let bits = variable_with_lltype("bits", LowLevelType::SignedLongLong);
    block_finite_nonzero
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "convert_float_bytes_to_longlong",
            vec![Hlvalue::Variable(f_nz)],
            Hlvalue::Variable(bits.clone()),
        ));
    let llong_const =
        |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::SignedLongLong);
    // `bits_high = bits >> 52`.
    let bits_high = variable_with_lltype("bits_high", LowLevelType::SignedLongLong);
    block_finite_nonzero
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "llong_rshift",
            vec![Hlvalue::Variable(bits.clone()), llong_const(52)],
            Hlvalue::Variable(bits_high.clone()),
        ));
    // `exp_raw = bits_high & 0x7ff`.
    let exp_raw = variable_with_lltype("exp_raw", LowLevelType::SignedLongLong);
    block_finite_nonzero
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "llong_and",
            vec![Hlvalue::Variable(bits_high), llong_const(0x7ff)],
            Hlvalue::Variable(exp_raw.clone()),
        ));
    // `mantissa_lo = bits & 0x000f_ffff_ffff_ffff`.
    let mantissa_lo = variable_with_lltype("mantissa_lo", LowLevelType::SignedLongLong);
    block_finite_nonzero
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "llong_and",
            vec![
                Hlvalue::Variable(bits.clone()),
                llong_const(0x000f_ffff_ffff_ffff),
            ],
            Hlvalue::Variable(mantissa_lo.clone()),
        ));
    // `sign_bit = bits & 0x8000_0000_0000_0000` — preserve IEEE 754
    // sign bit so `math.frexp(-x)` returns a negative mantissa.
    let sign_bit = variable_with_lltype("sign_bit", LowLevelType::SignedLongLong);
    block_finite_nonzero
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "llong_and",
            vec![Hlvalue::Variable(bits), llong_const(i64::MIN)],
            Hlvalue::Variable(sign_bit.clone()),
        ));

    // Branch: subnormal iff `exp_raw == 0`. Normalized
    // `(expo_ll, mantissa_lo, sign_bit)` triple converges in `block_join`.
    let is_subnormal = variable_with_lltype("is_subnormal", LowLevelType::Bool);
    block_finite_nonzero
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "llong_eq",
            vec![Hlvalue::Variable(exp_raw.clone()), llong_const(0)],
            Hlvalue::Variable(is_subnormal.clone()),
        ));
    block_finite_nonzero.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_subnormal));

    // Block layout for the post-extraction phase.
    let mantissa_lo_for_sub = variable_with_lltype("mantissa_lo", LowLevelType::SignedLongLong);
    let sign_bit_for_sub = variable_with_lltype("sign_bit", LowLevelType::SignedLongLong);
    let block_subnormal = Block::shared(vec![
        Hlvalue::Variable(mantissa_lo_for_sub.clone()),
        Hlvalue::Variable(sign_bit_for_sub.clone()),
    ]);

    let exp_raw_for_normal = variable_with_lltype("exp_raw", LowLevelType::SignedLongLong);
    let mantissa_lo_for_normal = variable_with_lltype("mantissa_lo", LowLevelType::SignedLongLong);
    let sign_bit_for_normal = variable_with_lltype("sign_bit", LowLevelType::SignedLongLong);
    let block_normal = Block::shared(vec![
        Hlvalue::Variable(exp_raw_for_normal.clone()),
        Hlvalue::Variable(mantissa_lo_for_normal.clone()),
        Hlvalue::Variable(sign_bit_for_normal.clone()),
    ]);

    let expo_ll_for_join = variable_with_lltype("expo_ll", LowLevelType::SignedLongLong);
    let mantissa_lo_for_join = variable_with_lltype("mantissa_lo", LowLevelType::SignedLongLong);
    let sign_bit_for_join = variable_with_lltype("sign_bit", LowLevelType::SignedLongLong);
    let block_join = Block::shared(vec![
        Hlvalue::Variable(expo_ll_for_join.clone()),
        Hlvalue::Variable(mantissa_lo_for_join.clone()),
        Hlvalue::Variable(sign_bit_for_join.clone()),
    ]);

    let to_subnormal = Link::new(
        vec![
            Hlvalue::Variable(mantissa_lo.clone()),
            Hlvalue::Variable(sign_bit.clone()),
        ],
        Some(block_subnormal.clone()),
        Some(bool_true()),
    )
    .into_ref();
    let to_normal = Link::new(
        vec![
            Hlvalue::Variable(exp_raw),
            Hlvalue::Variable(mantissa_lo),
            Hlvalue::Variable(sign_bit),
        ],
        Some(block_normal.clone()),
        Some(bool_false()),
    )
    .into_ref();
    block_finite_nonzero.closeblock(vec![to_subnormal, to_normal]);

    // block_subnormal: recover leading-bit position B via the
    // IEEE 754 round-trip trick. `cast_longlong_to_float` is exact for
    // the 52-bit mantissa (< 2^53), so the resulting f64's biased
    // exponent (`exp = 1023 + B`) directly yields B without a
    // CLZ helper graph. Then expo_ll = B - 1073 and shift the leading
    // bit out so the lower fraction lines up with IEEE 754 bits 0..51.
    //
    // RPython upstream relies on host `math.frexp` for subnormal
    // exponents (rlib/objectmodel.py:639); pyre's lloperation surface
    // has no `math.frexp` op, so this trick is the equivalent.
    let mantissa_lo_as_float = variable_with_lltype("mantissa_lo_f", LowLevelType::Float);
    block_subnormal
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "cast_longlong_to_float",
            vec![Hlvalue::Variable(mantissa_lo_for_sub.clone())],
            Hlvalue::Variable(mantissa_lo_as_float.clone()),
        ));
    let mantissa_lo_bits = variable_with_lltype("mantissa_lo_bits", LowLevelType::SignedLongLong);
    block_subnormal
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "convert_float_bytes_to_longlong",
            vec![Hlvalue::Variable(mantissa_lo_as_float)],
            Hlvalue::Variable(mantissa_lo_bits.clone()),
        ));
    let mantissa_lo_exp_high =
        variable_with_lltype("mantissa_lo_exp_high", LowLevelType::SignedLongLong);
    block_subnormal
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "llong_rshift",
            vec![Hlvalue::Variable(mantissa_lo_bits), llong_const(52)],
            Hlvalue::Variable(mantissa_lo_exp_high.clone()),
        ));
    let mantissa_lo_exp = variable_with_lltype("mantissa_lo_exp", LowLevelType::SignedLongLong);
    block_subnormal
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "llong_and",
            vec![Hlvalue::Variable(mantissa_lo_exp_high), llong_const(0x7ff)],
            Hlvalue::Variable(mantissa_lo_exp.clone()),
        ));
    // `B = mantissa_lo_exp - 1023` (B in [0, 51]).
    let b_pos = variable_with_lltype("b_pos", LowLevelType::SignedLongLong);
    block_subnormal
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "llong_sub",
            vec![Hlvalue::Variable(mantissa_lo_exp), llong_const(1023)],
            Hlvalue::Variable(b_pos.clone()),
        ));
    // `expo_ll = B - 1073` — yields -1073..-1022 across full subnormal range.
    let expo_ll_sub = variable_with_lltype("expo_ll", LowLevelType::SignedLongLong);
    block_subnormal
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "llong_sub",
            vec![Hlvalue::Variable(b_pos.clone()), llong_const(1073)],
            Hlvalue::Variable(expo_ll_sub.clone()),
        ));
    // `shift_amount = 52 - B` (in [1, 52]). Shift mantissa_lo so the
    // leading-1 lands at bit 52, then mask to the IEEE 754 fraction
    // field (bits 0..51) — the leading-1 then becomes the implicit bit.
    let shift_amount = variable_with_lltype("shift_amount", LowLevelType::SignedLongLong);
    block_subnormal
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "llong_sub",
            vec![llong_const(52), Hlvalue::Variable(b_pos)],
            Hlvalue::Variable(shift_amount.clone()),
        ));
    let mantissa_lo_shifted =
        variable_with_lltype("mantissa_lo_shifted", LowLevelType::SignedLongLong);
    block_subnormal
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "llong_lshift",
            vec![
                Hlvalue::Variable(mantissa_lo_for_sub),
                Hlvalue::Variable(shift_amount),
            ],
            Hlvalue::Variable(mantissa_lo_shifted.clone()),
        ));
    let mantissa_lo_normalized_sub =
        variable_with_lltype("mantissa_lo_norm", LowLevelType::SignedLongLong);
    block_subnormal
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "llong_and",
            vec![
                Hlvalue::Variable(mantissa_lo_shifted),
                llong_const(0x000f_ffff_ffff_ffff),
            ],
            Hlvalue::Variable(mantissa_lo_normalized_sub.clone()),
        ));
    block_subnormal.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(expo_ll_sub),
                Hlvalue::Variable(mantissa_lo_normalized_sub),
                Hlvalue::Variable(sign_bit_for_sub),
            ],
            Some(block_join.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_normal: `expo_ll = exp_raw - 1022`. Normal floats
    // already encode (1 + frac/2^52) at bias 1023, so mantissa_lo
    // passes through unchanged.
    let expo_ll_normal = variable_with_lltype("expo_ll", LowLevelType::SignedLongLong);
    block_normal
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "llong_sub",
            vec![Hlvalue::Variable(exp_raw_for_normal), llong_const(1022)],
            Hlvalue::Variable(expo_ll_normal.clone()),
        ));
    block_normal.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(expo_ll_normal),
                Hlvalue::Variable(mantissa_lo_for_normal),
                Hlvalue::Variable(sign_bit_for_normal),
            ],
            Some(block_join.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_join: assemble the [0.5, 1)-mantissa float and run
    // PyPy's `_hash_float` arithmetic on it.
    //
    // `mantissa_bits = mantissa_lo_normalized | (1022 << 52) | sign_bit`.
    let mantissa_unsigned = variable_with_lltype("mantissa_unsigned", LowLevelType::SignedLongLong);
    block_join.borrow_mut().operations.push(SpaceOperation::new(
        "llong_or",
        vec![
            Hlvalue::Variable(mantissa_lo_for_join),
            llong_const(0x3fe0_0000_0000_0000),
        ],
        Hlvalue::Variable(mantissa_unsigned.clone()),
    ));
    let mantissa_bits = variable_with_lltype("mantissa_bits", LowLevelType::SignedLongLong);
    block_join.borrow_mut().operations.push(SpaceOperation::new(
        "llong_or",
        vec![
            Hlvalue::Variable(mantissa_unsigned),
            Hlvalue::Variable(sign_bit_for_join),
        ],
        Hlvalue::Variable(mantissa_bits.clone()),
    ));
    // `mantissa = f64::from_bits(mantissa_bits)`.
    let mantissa = variable_with_lltype("mantissa", LowLevelType::Float);
    block_join.borrow_mut().operations.push(SpaceOperation::new(
        "convert_longlong_bytes_to_float",
        vec![Hlvalue::Variable(mantissa_bits)],
        Hlvalue::Variable(mantissa.clone()),
    ));
    // Truncate `expo_ll` (SignedLongLong) → `expo` (Signed) for use
    // with `int_lshift`. On 64-bit hosts both lltypes are 64-bit so
    // this is a value-preserving cast for the small exponent range.
    let expo = variable_with_lltype("expo", LowLevelType::Signed);
    block_join.borrow_mut().operations.push(SpaceOperation::new(
        "truncate_longlong_to_int",
        vec![Hlvalue::Variable(expo_ll_for_join)],
        Hlvalue::Variable(expo.clone()),
    ));

    // `TAKE_NEXT = 2**31 = 2147483648.0`.
    let take_next = || constant_with_lltype(ConstValue::float(2147483648.0), LowLevelType::Float);
    // `v *= TAKE_NEXT`.
    let v_scaled = variable_with_lltype("v_scaled", LowLevelType::Float);
    block_join.borrow_mut().operations.push(SpaceOperation::new(
        "float_mul",
        vec![Hlvalue::Variable(mantissa), take_next()],
        Hlvalue::Variable(v_scaled.clone()),
    ));
    // `hipart = int(v)`.
    let hipart = variable_with_lltype("hipart", LowLevelType::Signed);
    block_join.borrow_mut().operations.push(SpaceOperation::new(
        "cast_float_to_int",
        vec![Hlvalue::Variable(v_scaled.clone())],
        Hlvalue::Variable(hipart.clone()),
    ));
    // `v = (v - float(hipart)) * TAKE_NEXT`.
    let hipart_f = variable_with_lltype("hipart_f", LowLevelType::Float);
    block_join.borrow_mut().operations.push(SpaceOperation::new(
        "cast_int_to_float",
        vec![Hlvalue::Variable(hipart.clone())],
        Hlvalue::Variable(hipart_f.clone()),
    ));
    let v_diff = variable_with_lltype("v_diff", LowLevelType::Float);
    block_join.borrow_mut().operations.push(SpaceOperation::new(
        "float_sub",
        vec![Hlvalue::Variable(v_scaled), Hlvalue::Variable(hipart_f)],
        Hlvalue::Variable(v_diff.clone()),
    ));
    let v_final = variable_with_lltype("v_final", LowLevelType::Float);
    block_join.borrow_mut().operations.push(SpaceOperation::new(
        "float_mul",
        vec![Hlvalue::Variable(v_diff), take_next()],
        Hlvalue::Variable(v_final.clone()),
    ));
    let v_final_int = variable_with_lltype("v_final_int", LowLevelType::Signed);
    block_join.borrow_mut().operations.push(SpaceOperation::new(
        "cast_float_to_int",
        vec![Hlvalue::Variable(v_final)],
        Hlvalue::Variable(v_final_int.clone()),
    ));
    // `expo << 15`.
    let expo_shifted = variable_with_lltype("expo_shifted", LowLevelType::Signed);
    block_join.borrow_mut().operations.push(SpaceOperation::new(
        "int_lshift",
        vec![Hlvalue::Variable(expo), signed_const(15)],
        Hlvalue::Variable(expo_shifted.clone()),
    ));
    // `x = hipart + int(v) + (expo << 15)`.
    let sum1 = variable_with_lltype("sum1", LowLevelType::Signed);
    block_join.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![Hlvalue::Variable(hipart), Hlvalue::Variable(v_final_int)],
        Hlvalue::Variable(sum1.clone()),
    ));
    let sum2 = variable_with_lltype("sum2", LowLevelType::Signed);
    block_join.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![Hlvalue::Variable(sum1), Hlvalue::Variable(expo_shifted)],
        Hlvalue::Variable(sum2.clone()),
    ));
    // `intmask(x)` is identity on a 64-bit Signed host.
    block_join.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(sum2)],
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
        vec!["f".to_string()],
        func,
    ))
}

// ____________________________________________________________
// pairtype(FloatRepr, FloatRepr) — rfloat.py.

pub fn rtype_template(hop: &HighLevelOp, func: &str) -> RTypeResult {
    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::Float),
        ConvertedTo::LowLevelType(&LowLevelType::Float),
    ])?;
    let opname = format!("float_{func}");
    Ok(hop.genop(&opname, vlist, GenopResult::LLType(LowLevelType::Float)))
}

pub fn rtype_compare_template(hop: &HighLevelOp, func: &str) -> RTypeResult {
    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::Float),
        ConvertedTo::LowLevelType(&LowLevelType::Float),
    ])?;
    let opname = format!("float_{func}");
    Ok(hop.genop(&opname, vlist, GenopResult::LLType(LowLevelType::Bool)))
}

// ____________________________________________________________
// SingleFloatRepr / LongFloatRepr — `rfloat.py`.
//
// Upstream creates a fresh `SingleFloatRepr()` / `LongFloatRepr()` per
// `rtyper_makerepr` call (no module-global singleton). Pyre mirrors by
// instantiating a fresh `Arc<SingleFloatRepr>` / `Arc<LongFloatRepr>`
// in [`super::rmodel::rtyper_makerepr`] per SomeSingleFloat /
// SomeLongFloat.

/// RPython `class SingleFloatRepr(Repr)` (`rfloat.py`).
#[derive(Debug)]
pub struct SingleFloatRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl SingleFloatRepr {
    pub fn new() -> Self {
        SingleFloatRepr {
            state: ReprState::new(),
            lltype: LowLevelType::SingleFloat,
        }
    }
}

impl Default for SingleFloatRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for SingleFloatRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "SingleFloatRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::SingleFloatRepr
    }

    /// RPython `SingleFloatRepr.rtype_float(self, hop)`
    /// (`rfloat.py:153-158`):
    ///
    /// ```python
    /// def rtype_float(self, hop):
    ///     v, = hop.inputargs(lltype.SingleFloat)
    ///     hop.exception_cannot_occur()
    ///     return hop.genop('cast_primitive', [v], resulttype=lltype.Float)
    /// ```
    fn rtype_float(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::SingleFloat)])?;
        hop.exception_cannot_occur()?;
        Ok(hop.genop(
            "cast_primitive",
            vlist,
            GenopResult::LLType(LowLevelType::Float),
        ))
    }
}

/// RPython `class LongFloatRepr(Repr)` (`rfloat.py`).
#[derive(Debug)]
pub struct LongFloatRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl LongFloatRepr {
    pub fn new() -> Self {
        LongFloatRepr {
            state: ReprState::new(),
            lltype: LowLevelType::LongFloat,
        }
    }
}

impl Default for LongFloatRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for LongFloatRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "LongFloatRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::LongFloatRepr
    }

    /// RPython `LongFloatRepr.rtype_float(self, hop)`
    /// (`rfloat.py`): symmetrical to [`SingleFloatRepr::rtype_float`]
    /// but over `LongFloat → Float`.
    fn rtype_float(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::LongFloat)])?;
        hop.exception_cannot_occur()?;
        Ok(hop.genop(
            "cast_primitive",
            vlist,
            GenopResult::LLType(LowLevelType::Float),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::translator::rtyper::rtyper::RPythonTyper;

    #[test]
    fn float_repr_lowleveltype_and_repr_string_match_upstream() {
        // rfloat.py — `lowleveltype = Float`.
        let r = FloatRepr::new();
        assert_eq!(r.lowleveltype(), &LowLevelType::Float);
        assert_eq!(r.repr_string(), "<FloatRepr Float>");
        assert_eq!(r.compact_repr(), "FloatR Float");
    }

    #[test]
    fn float_repr_singleton_returns_same_arc() {
        // rfloat.py — `float_repr = FloatRepr()` module-global.
        let a = float_repr();
        let b = float_repr();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn convert_const_accepts_int_float_bool() {
        // rfloat.py — `isinstance(value, (int, base_int, float))`
        // admits all numeric kinds. Pyre normalises to `ConstValue::Float`.
        let r = FloatRepr::new();
        let c_int = r.convert_const(&ConstValue::Int(42)).unwrap();
        let ConstValue::Float(bits) = c_int.value else {
            panic!("expected Float after int coercion");
        };
        assert_eq!(f64::from_bits(bits), 42.0);

        let c_bool = r.convert_const(&ConstValue::Bool(true)).unwrap();
        let ConstValue::Float(bits) = c_bool.value else {
            panic!("expected Float after bool coercion");
        };
        assert_eq!(f64::from_bits(bits), 1.0);

        let c_float = r.convert_const(&ConstValue::Float(3.25_f64.to_bits()));
        let c = c_float.unwrap();
        assert_eq!(c.value, ConstValue::Float(3.25_f64.to_bits()));
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Float));
    }

    /// rfloat.py — `FloatRepr.get_ll_hash_function` returns
    /// `_hash_float`. The synthesized helper carries the Float→Signed
    /// signature and a 4-block CFG (start, not_finite, inf, finite).
    #[test]
    fn float_repr_get_ll_hash_function_synthesizes_multi_block_helper() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r = float_repr();

        let llfn = r
            .get_ll_hash_function(&rtyper)
            .expect("get_ll_hash_function(Float)")
            .expect("returns Some helper");
        assert_eq!(llfn.name, "_hash_float");
        assert_eq!(llfn.args, vec![LowLevelType::Float]);
        assert_eq!(llfn.result, LowLevelType::Signed);

        let graph = llfn.graph.as_ref().expect("helper carries a graph");
        let inner = graph.graph.borrow();

        // start block: float_sub + float_eq + exitswitch on isfin.
        let start = inner.startblock.borrow();
        let start_ops: Vec<&str> = start
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["float_sub", "float_eq"]);
        assert!(start.exitswitch.is_some());
        assert_eq!(start.exits.len(), 2);
    }

    /// rlib/objectmodel.py:639 — the finite path inlines RustPython's
    /// `decompose_float` (bit manipulation via
    /// `convert_float_bytes_to_longlong` + llong shift/and/or +
    /// `convert_longlong_bytes_to_float`) and then PyPy's `_hash_float`
    /// arithmetic. The block_finite block branches on `f == 0.0`
    /// (returning constant 0 for the zero case since the bit-formula
    /// returns `(0.5, -1022)` for input `0.0` — divergent from
    /// `math.frexp(0.0) == (0.0, 0)`); the False branch points at
    /// block_finite_nonzero where the full arithmetic lives.
    #[test]
    fn float_repr_hash_helper_finite_path_inlines_decompose_float_bit_manip() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r = float_repr();

        let llfn = r.get_ll_hash_function(&rtyper).unwrap().unwrap();
        let graph = llfn.graph.as_ref().unwrap();
        let inner = graph.graph.borrow();

        // Drill into block_finite via start[True] exit.
        let start = inner.startblock.borrow();
        let true_link = start.exits[0].borrow();
        let block_finite = true_link.target.as_ref().expect("true link target").clone();
        drop(true_link);
        drop(start);

        // block_finite: short-circuit `f == 0.0` to constant 0.
        let finite = block_finite.borrow();
        let finite_ops: Vec<&str> = finite
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(finite_ops, vec!["float_eq"]);
        assert_eq!(finite.exits.len(), 2);

        // block_finite_nonzero is the False branch — extracts IEEE
        // components and branches on `exp_raw == 0`.
        let nonzero_link = finite.exits[1].borrow();
        let block_nz = nonzero_link.target.as_ref().expect("False target").clone();
        drop(nonzero_link);
        drop(finite);

        let nz = block_nz.borrow();
        let nz_ops: Vec<&str> = nz.operations.iter().map(|op| op.opname.as_str()).collect();
        assert_eq!(
            nz_ops,
            vec![
                "convert_float_bytes_to_longlong",
                "llong_rshift",
                "llong_and", // exp_raw
                "llong_and", // mantissa_lo
                "llong_and", // sign_bit
                "llong_eq",  // is_subnormal
            ]
        );
        assert_eq!(nz.exits.len(), 2);

        // block_subnormal — True branch — IEEE round-trip trick to
        // recover leading-bit position; closes to block_join with the
        // normalized triple.
        let sub_link = nz.exits[0].borrow();
        let block_sub = sub_link.target.as_ref().expect("subnormal target").clone();
        drop(sub_link);
        let sub = block_sub.borrow();
        let sub_ops: Vec<&str> = sub.operations.iter().map(|op| op.opname.as_str()).collect();
        assert_eq!(
            sub_ops,
            vec![
                "cast_longlong_to_float",          // mantissa_lo as f64
                "convert_float_bytes_to_longlong", // back to bit pattern
                "llong_rshift",                    // (>> 52)
                "llong_and",                       // & 0x7ff -> 1023 + B
                "llong_sub",                       // B = .. - 1023
                "llong_sub",                       // expo_ll = B - 1073
                "llong_sub",                       // shift_amount = 52 - B
                "llong_lshift",                    // mantissa_lo << shift_amount
                "llong_and",                       // & 0x000fffffffffffff
            ]
        );

        // block_normal — False branch — `expo_ll = exp_raw - 1022`.
        let norm_link = nz.exits[1].borrow();
        let block_norm = norm_link.target.as_ref().expect("normal target").clone();
        drop(norm_link);
        drop(nz);
        let norm = block_norm.borrow();
        let norm_ops: Vec<&str> = norm
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(norm_ops, vec!["llong_sub"]);

        // block_join — both branches converge here. Single exit; ops
        // assemble mantissa_bits then run PyPy's _hash_float arithmetic.
        let to_join = norm.exits[0].borrow();
        let block_join = to_join.target.as_ref().expect("join target").clone();
        drop(to_join);
        drop(norm);
        let join = block_join.borrow();
        let join_ops: Vec<&str> = join
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            join_ops,
            vec![
                "llong_or", // mantissa_unsigned
                "llong_or", // mantissa_bits = mantissa_unsigned | sign_bit
                "convert_longlong_bytes_to_float",
                "truncate_longlong_to_int", // expo
                // _hash_float arithmetic (rlib/objectmodel.py).
                "float_mul",
                "cast_float_to_int",
                "cast_int_to_float",
                "float_sub",
                "float_mul",
                "cast_float_to_int",
                "int_lshift",
                "int_add",
                "int_add",
            ]
        );
    }

    #[test]
    fn convert_const_rejects_non_numeric() {
        // rfloat.py:15-16 — `raise TyperError("not a float: %r" % ...)`.
        let r = FloatRepr::new();
        let err = r.convert_const(&ConstValue::byte_str("pi")).unwrap_err();
        assert!(err.to_string().contains("not a float"));
    }

    #[test]
    fn single_float_repr_lowleveltype_matches_upstream() {
        // rfloat.py — `lowleveltype = lltype.SingleFloat`.
        let r = SingleFloatRepr::new();
        assert_eq!(r.lowleveltype(), &LowLevelType::SingleFloat);
        assert_eq!(r.repr_string(), "<SingleFloatRepr SingleFloat>");
    }

    #[test]
    fn long_float_repr_lowleveltype_matches_upstream() {
        // rfloat.py — `lowleveltype = lltype.LongFloat`.
        let r = LongFloatRepr::new();
        assert_eq!(r.lowleveltype(), &LowLevelType::LongFloat);
        assert_eq!(r.repr_string(), "<LongFloatRepr LongFloat>");
    }

    #[test]
    fn rtyper_getrepr_on_some_float_returns_the_singleton() {
        // rfloat.py — `SomeFloat.rtyper_makerepr` returns
        // `float_repr` (module-global). Verify the full dispatch chain.
        use crate::annotator::model::{SomeFloat, SomeValue};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let s_float = SomeValue::Float(SomeFloat::new());
        let r = rtyper.getrepr(&s_float).expect("getrepr(SomeFloat)");
        let expected = float_repr() as Arc<dyn Repr>;
        assert!(Arc::ptr_eq(&r, &expected));

        let r2 = rtyper.getrepr(&s_float).expect("second getrepr(SomeFloat)");
        assert!(Arc::ptr_eq(&r, &r2));
    }

    #[test]
    fn rtyper_getrepr_on_some_single_float_allocates_per_call_repr() {
        // rfloat.py — `SomeSingleFloat.rtyper_makerepr`
        // returns `SingleFloatRepr()` (fresh per call — no module-
        // global singleton). But `getrepr` caches the first produced
        // instance per key, so two lookups return the same Arc.
        use crate::annotator::model::{SomeSingleFloat, SomeValue};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let s_single = SomeValue::SingleFloat(SomeSingleFloat::new());
        let r = rtyper.getrepr(&s_single).expect("getrepr(SomeSingleFloat)");
        assert_eq!(r.lowleveltype(), &LowLevelType::SingleFloat);
        let r2 = rtyper
            .getrepr(&s_single)
            .expect("second getrepr(SomeSingleFloat)");
        // rtyper.py:54-57: cached Arc identity preserved across lookups.
        assert!(Arc::ptr_eq(&r, &r2));
    }
}
