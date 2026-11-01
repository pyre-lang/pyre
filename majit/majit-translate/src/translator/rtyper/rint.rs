//! RPython `rpython/rtyper/rint.py` — `IntegerRepr` + integer repr
//! singletons + `SomeInteger` dispatch.
//!
//! ## Scope of this port
//!
//! Upstream rint.py is 675 LOC. This commit lands:
//!
//! | upstream | Rust mirror |
//! |---|---|
//! | `class IntegerRepr(FloatRepr)` (`rint.py`) | [`IntegerRepr`] |
//! | `IntegerRepr.__init__` + `opprefix` (`rint.py`) | [`IntegerRepr::new`] / [`IntegerRepr::opprefix`] |
//! | `IntegerRepr.convert_const` (`rint.py`) | [`IntegerRepr::convert_const`] |
//! | `rtype_bool` (`rint.py`) | [`IntegerRepr::rtype_bool`] |
//! | `rtype_abs` (`rint.py`) | [`IntegerRepr::rtype_abs`] |
//! | `rtype_invert` (`rint.py`) | [`IntegerRepr::rtype_invert`] |
//! | `rtype_neg` (`rint.py`) | [`IntegerRepr::rtype_neg`] |
//! | `rtype_pos` (`rint.py`) | [`IntegerRepr::rtype_pos`] |
//! | `rtype_int` (`rint.py`) | [`IntegerRepr::rtype_int`] |
//! | `rtype_float` (`rint.py`) | [`IntegerRepr::rtype_float`] |
//! | `getintegerrepr(lltype, prefix=None)` (`rint.py`) | [`getintegerrepr`] |
//! | Standard integer singletons `signed_repr` / `unsigned_repr` / `signedlonglong_repr` / `unsignedlonglong_repr` (`rint.py`) | [`signed_repr`] / [`unsigned_repr`] / [`signedlonglong_repr`] / [`unsignedlonglong_repr`] |
//! | `SomeInteger.rtyper_makerepr` / `rtyper_makekey` (`rint.py`) | wired in [`super::rmodel::rtyper_makerepr`] / [`super::rmodel::rtyper_makekey`] |
//! | `pairtype(IntegerRepr, IntegerRepr).rtype_add/sub/mul/floordiv/mod/and_/or_/xor/lshift/rshift` + `_ovf` siblings (`rint.py`) | dispatched via [`super::pairtype::pair_rtype_add`] etc. → [`rtype_template`] / [`rtype_call_helper`] |
//! | `pairtype(IntegerRepr, IntegerRepr).rtype_eq/ne/lt/le/gt/ge/is_` (`rint.py`) | dispatched via [`super::pairtype::pair_rtype_eq`] etc. → [`rtype_compare_template`] |
//! | `pairtype(IntegerRepr, IntegerRepr).convert_from_to` (`rint.py`) | [`pair_integer_integer_convert_from_to`] |
//! | `pairtype(IntegerRepr, FloatRepr).convert_from_to` (`rint.py`) | [`pair_integer_float_convert_from_to`] |
//! | `pairtype(FloatRepr, IntegerRepr).convert_from_to` (`rint.py`) | [`pair_float_integer_convert_from_to`] |
//! | `_rtype_template` / `_rtype_compare_template` / `_rtype_call_helper` + `rtype_add_ovf` (`rint.py`) | [`rtype_template`] / [`rtype_compare_template`] / [`rtype_call_helper`] / [`rtype_add_ovf`] |
//!
//! ## Deferred to follow-up commits
//!
//! * Remaining `pairtype(IntegerRepr, IntegerRepr)` helpers outside the
//!   currently landed arithmetic/comparison/direct-call subset
//!   (`rint.py`), e.g. `divmod` and low-level helper execution
//!   details that require the wider rtyper/annlowlevel port.
//! * Full exception-class matching for `_rtype_call_helper` and
//!   `rtype_chr` / `rtype_unichr` (`has_implicit_exception(ValueError)`,
//!   `ZeroDivisionError`, etc.) — the method structure is in place, but
//!   `rtyper.py` still needs `exceptiondata.py` parity.
//! * `rtype_hex` / `rtype_oct` / `rtype_bin` (`rint.py`) — require
//!   `lltypesystem/ll_str.py` (`ll_int2hex/oct/bin`).
//! * `ll_str` / `ll_hash_int` / `ll_hash_long_long` / `ll_eq_shortint`
//!   (`rint.py`) — land with `rdict.py` (needs
//!   `get_ll_{eq,hash}_function`) + `annlowlevel.llstr`.
//! * `get_ll_{eq,ge,gt,lt,le,hash,fasthash,dummyval}_function`
//!   (`rint.py`) — require trait slots absent from pyre's `Repr`.

#![allow(non_camel_case_types)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::annotator::model::{SomeValue, unionof};
use crate::flowspace::model::{ConstValue, Constant, Hlvalue};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, LowLevelValue};
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult, HighLevelOp, LowLevelOpList};

/// RPython `class IntegerRepr(FloatRepr)` (`rint.py`).
///
/// ```python
/// class IntegerRepr(FloatRepr):
///     def __init__(self, lowleveltype, opprefix):
///         self.lowleveltype = lowleveltype
///         self._opprefix = opprefix
///         self.as_int = self
///
///     @property
///     def opprefix(self):
///         if self._opprefix is None:
///             raise TyperError("arithmetic not supported on %r, its size is too small"
///                              % self.lowleveltype)
///         return self._opprefix
/// ```
///
/// The `FloatRepr` base class provides `rtype_float` / `convert_const`
/// fall-through (`rfloat.py:11-58`); pyre implements `IntegerRepr`
/// directly on the [`Repr`] trait rather than subclassing
/// `FloatRepr` — structural subtyping is not used downstream (no caller
/// relies on `isinstance(r, FloatRepr)` for an integer repr).
///
/// The upstream `self.as_int = self` attribute is not ported: it exists
/// only so that `BoolRepr(IntegerRepr)` can override it to
/// `signed_repr`. The Rust port implements `BoolRepr` as a standalone
/// `Repr` (see [`super::rbool`]) which makes the `as_int` indirection
/// unnecessary.
#[derive(Debug)]
pub struct IntegerRepr {
    state: ReprState,
    lltype: LowLevelType,
    /// RPython `self._opprefix` (`rint.py:21`). `None` for the
    /// small-size short-int reprs that upstream builds via
    /// `build_number(None, S)` for `S in {SignedShort, UnsignedShort,
    /// SignedChar, UnsignedChar}`; those raise a TyperError when any
    /// arithmetic `rtype_*` method reaches `opprefix()`.
    opprefix: Option<&'static str>,
}

impl IntegerRepr {
    pub fn new(lowleveltype: LowLevelType, opprefix: Option<&'static str>) -> Self {
        IntegerRepr {
            state: ReprState::new(),
            lltype: lowleveltype,
            opprefix,
        }
    }

    /// RPython `IntegerRepr.opprefix` property (`rint.py`).
    pub fn opprefix(&self) -> Result<&'static str, TyperError> {
        self.opprefix.ok_or_else(|| {
            TyperError::message(format!(
                "arithmetic not supported on {:?}, its size is too small",
                self.lltype
            ))
        })
    }

    fn s_result_unsigned(&self, hop: &HighLevelOp) -> bool {
        // upstream: `hop.s_result.unsigned` — SomeInteger carries the
        // flag. When `s_result` isn't a SomeInteger (should not happen
        // for the integer rtype_* paths) default to `false` to match
        // the signed-integer code paths.
        hop.s_result
            .borrow()
            .as_ref()
            .and_then(|s| match s {
                SomeValue::Integer(i) => Some(i.unsigned),
                _ => None,
            })
            .unwrap_or(false)
    }

    /// Convert `lowleveltype._defl()` to the `ConstValue` carrier that
    /// `contains_value` admits. `UnsignedLongLongLong._defl` is
    /// `LowLevelValue::UnsignedLongLongLong(0)` (`r_ulonglonglong(0)`),
    /// which must become `UInt128(0)`, not `UInt(0)`.
    fn defl_as_const(lltype: &LowLevelType) -> Result<ConstValue, TyperError> {
        match (lltype, lltype._defl()) {
            (LowLevelType::UnsignedLongLongLong, LowLevelValue::UnsignedLongLongLong(_)) => {
                Ok(ConstValue::UInt128(0))
            }
            (LowLevelType::SignedLongLongLong, LowLevelValue::SignedLongLongLong(_)) => {
                Ok(ConstValue::Int128(0))
            }
            (_, LowLevelValue::Unsigned(n)) => Ok(ConstValue::UInt(n)),
            (_, LowLevelValue::Signed(n)) => Ok(ConstValue::Int(n)),
            (_, other) => Err(TyperError::message(format!(
                "IntegerRepr.rtype_neg: unexpected lowleveltype._defl {other:?}"
            ))),
        }
    }
}

impl Repr for IntegerRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "IntegerRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::IntegerRepr
    }

    /// RPython `IntegerRepr.get_ll_eq_function(self)` (`rint.py`):
    ///
    /// ```python
    /// def get_ll_eq_function(self):
    ///     if getattr(self, '_opprefix', '?') is None:
    ///         return ll_eq_shortint
    ///     return None
    /// ```
    ///
    /// `_opprefix is None` is the "shortint" branch (RPython narrow
    /// integers like `r_singleshort` whose `==` requires explicit
    /// `intmask` widening). Pyre's primitive lltypes (`Signed`, `Bool`,
    /// `Unsigned`, long-long variants) all carry an `opprefix`, so the
    /// shortint branch is never taken; returning `None` instructs the
    /// caller (`gen_eq_function` / `rtype_contains`) to fall back to
    /// the primitive `int_eq`/`uint_eq`/... inline op via
    /// `eq_funcs[i] or operator.eq`.
    fn get_ll_eq_function(
        &self,
        _rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        Ok(None)
    }

    /// RPython `IntegerRepr.get_ll_hash_function(self)` (`rint.py`):
    ///
    /// ```python
    /// def get_ll_hash_function(self):
    ///     if (sys.maxint == 2147483647 and
    ///         self.lowleveltype in (SignedLongLong, UnsignedLongLong)):
    ///         return ll_hash_long_long
    ///     return ll_hash_int
    ///
    /// def ll_hash_int(n):
    ///     return intmask(n)
    /// ```
    ///
    /// Pyre runs on 64-bit hosts (`sys.maxint == 2**63 - 1`); the
    /// `ll_hash_long_long` branch is never taken. `ll_hash_int(n) =
    /// intmask(n)` is identity for `Signed` and a width-narrowing cast
    /// for the wider primitive widths. Synthesizes a single-block helper
    /// graph per width so [`gen_hash_function`] can dispatch via
    /// `direct_call`.
    fn get_ll_hash_function(
        &self,
        rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        let lltype = self.lltype.clone();
        let name = format!("ll_hash_int_{}", lltype.short_name());
        rtyper
            .lowlevel_helper_function_with_builder(
                name.clone(),
                vec![lltype.clone()],
                LowLevelType::Signed,
                move |_rtyper, args, _result| build_ll_hash_int_helper_graph(&name, &args[0]),
            )
            .map(Some)
    }

    /// rint.py:198
    fn get_ll_fasthash_function(
        &self,
        rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        self.get_ll_hash_function(rtyper)
    }

    /// RPython `IntegerRepr.convert_const(self, value)` (`rint.py`):
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if isinstance(value, objectmodel.Symbolic):
    ///         return value
    ///     T = typeOf(value)
    ///     if isinstance(T, Number) or T is Bool:
    ///         return cast_primitive(self.lowleveltype, value)
    ///     raise TyperError("not an integer: %r" % (value,))
    /// ```
    ///
    /// Pyre maps `Bool` through the same primitive cast: `True` becomes
    /// integer `1`, `False` becomes integer `0`. The `Symbolic` branch
    /// (rint.py `if isinstance(value, objectmodel.Symbolic): return
    /// value`) carries `llmemory` address offsets through unchanged; the
    /// concrete byte size is resolved only at code emission.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        let converted = match value {
            ConstValue::AddressOffset(_) => value.clone(),
            ConstValue::InheritanceId { .. } => value.clone(),
            ConstValue::Int(n) => match self.lltype {
                // `cast_primitive(Unsigned, int)` → `r_uint`
                // (`lltype.Unsigned = build_number("Unsigned", r_uint)`).
                LowLevelType::Unsigned | LowLevelType::UnsignedLongLong => {
                    ConstValue::UInt(*n as u64)
                }
                // `IntegerRepr.convert_const` → `cast_primitive` →
                // `Number._cast`: `r_longlonglong(value)` /
                // `r_ulonglonglong(value)`.
                LowLevelType::SignedLongLongLong => ConstValue::Int128(*n as i128),
                LowLevelType::UnsignedLongLongLong => ConstValue::UInt128(*n as i128 as u128),
                _ => value.clone(),
            },
            ConstValue::UInt(n) => match self.lltype {
                LowLevelType::Signed | LowLevelType::SignedLongLong => ConstValue::Int(*n as i64),
                // `cast_primitive` to SignedLongLongLong yields a positive
                // `r_longlonglong` (`IntegerRepr.convert_const`).
                LowLevelType::SignedLongLongLong => ConstValue::Int128(*n as i128),
                LowLevelType::Unsigned | LowLevelType::UnsignedLongLong => ConstValue::UInt(*n),
                LowLevelType::UnsignedLongLongLong => ConstValue::UInt128(*n as u128),
                _ => value.clone(),
            },
            // `IntegerRepr.convert_const` → `cast_primitive` →
            // `Number._cast` (`intmask` / `r_uint` / `longlonglongmask` /
            // `r_ulonglonglong`).
            ConstValue::Int128(_) | ConstValue::UInt128(_) => {
                let (low128, fits_i128) = match value {
                    ConstValue::Int128(n) => (*n as u128, true),
                    ConstValue::UInt128(u) => (*u, *u <= i128::MAX as u128),
                    _ => unreachable!(),
                };
                match self.lltype {
                    LowLevelType::Signed
                    | LowLevelType::Unsigned
                    | LowLevelType::SignedLongLong
                    | LowLevelType::UnsignedLongLong => {
                        // `intmask` / `r_uint` mask to `LONG_BIT`
                        // (`target_signed_size * 8`), `longlongmask` /
                        // `r_ulonglong` to 64 bits.
                        let bits = match self.lltype {
                            LowLevelType::Signed | LowLevelType::Unsigned => {
                                crate::layout::target_signed_size() as u32 * 8
                            }
                            _ => 64,
                        };
                        let shift = 128 - bits;
                        if matches!(
                            self.lltype,
                            LowLevelType::Signed | LowLevelType::SignedLongLong
                        ) {
                            ConstValue::Int((((low128 << shift) as i128) >> shift) as i64)
                        } else {
                            ConstValue::UInt(((low128 << shift) >> shift) as u64)
                        }
                    }
                    LowLevelType::SignedLongLongLong if fits_i128 => {
                        ConstValue::Int128(low128 as i128)
                    }
                    LowLevelType::SignedLongLongLong => {
                        return Err(TyperError::message(format!(
                            "OverflowError: {low128} does not fit in signed 128-bit integer"
                        )));
                    }
                    LowLevelType::UnsignedLongLongLong => ConstValue::UInt128(low128),
                    _ => value.clone(),
                }
            }
            ConstValue::Bool(flag) => match self.lltype {
                LowLevelType::Unsigned | LowLevelType::UnsignedLongLong => {
                    ConstValue::UInt(u64::from(*flag))
                }
                LowLevelType::SignedLongLongLong => ConstValue::Int128(i128::from(*flag)),
                LowLevelType::UnsignedLongLongLong => ConstValue::UInt128(u128::from(*flag)),
                _ => ConstValue::Int(i64::from(*flag)),
            },
            _ => return Err(TyperError::message(format!("not an integer: {value:?}"))),
        };
        Ok(Constant::with_concretetype(converted, self.lltype.clone()))
    }

    /// RPython `IntegerRepr.rtype_bool(self, hop)` (`rint.py`):
    ///
    /// ```python
    /// def rtype_bool(self, hop):
    ///     assert self is self.as_int   # rtype_is_true() is overridden in BoolRepr
    ///     vlist = hop.inputargs(self)
    ///     return hop.genop(self.opprefix + 'is_true', vlist, resulttype=Bool)
    /// ```
    fn rtype_bool(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        let opname = format!("{}is_true", self.opprefix()?);
        Ok(hop.genop(&opname, vlist, GenopResult::LLType(LowLevelType::Bool)))
    }

    /// RPython `IntegerRepr.rtype_abs(self, hop)` (`rint.py`):
    ///
    /// ```python
    /// def rtype_abs(self, hop):
    ///     self = self.as_int
    ///     vlist = hop.inputargs(self)
    ///     if hop.s_result.unsigned:
    ///         return vlist[0]
    ///     else:
    ///         return hop.genop(self.opprefix + 'abs', vlist, resulttype=self)
    /// ```
    fn rtype_abs(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        if self.s_result_unsigned(hop) {
            return Ok(vlist.into_iter().next());
        }
        let opname = format!("{}abs", self.opprefix()?);
        Ok(hop.genop(&opname, vlist, GenopResult::LLType(self.lltype.clone())))
    }

    /// RPython `IntegerRepr.rtype_abs_ovf(self, hop)` (`rint.py`).
    fn rtype_abs_ovf(&self, hop: &HighLevelOp) -> RTypeResult {
        if self.s_result_unsigned(hop) {
            return Err(TyperError::message("forbidden uint_abs_ovf"));
        }
        rtype_call_helper(hop, "abs_ovf".to_string(), &[])
    }

    /// RPython `IntegerRepr.rtype_invert(self, hop)` (`rint.py`):
    ///
    /// ```python
    /// def rtype_invert(self, hop):
    ///     self = self.as_int
    ///     vlist = hop.inputargs(self)
    ///     return hop.genop(self.opprefix + 'invert', vlist, resulttype=self)
    /// ```
    fn rtype_invert(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        let opname = format!("{}invert", self.opprefix()?);
        Ok(hop.genop(&opname, vlist, GenopResult::LLType(self.lltype.clone())))
    }

    /// RPython `IntegerRepr.rtype_neg(self, hop)` (`rint.py`):
    ///
    /// ```python
    /// def rtype_neg(self, hop):
    ///     self = self.as_int
    ///     vlist = hop.inputargs(self)
    ///     if hop.s_result.unsigned:
    ///         # implement '-r_uint(x)' with unsigned subtraction '0 - x'
    ///         zero = self.lowleveltype._defl()
    ///         vlist.insert(0, hop.inputconst(self.lowleveltype, zero))
    ///         return hop.genop(self.opprefix + 'sub', vlist, resulttype=self)
    ///     else:
    ///         return hop.genop(self.opprefix + 'neg', vlist, resulttype=self)
    /// ```
    fn rtype_neg(&self, hop: &HighLevelOp) -> RTypeResult {
        let mut vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        if self.s_result_unsigned(hop) {
            // `IntegerRepr.rtype_neg`: `zero = self.lowleveltype._defl()`.
            let zero_value = Self::defl_as_const(&self.lltype)?;
            let zero = HighLevelOp::inputconst(&self.lltype, &zero_value)?;
            vlist.insert(0, Hlvalue::Constant(zero));
            let opname = format!("{}sub", self.opprefix()?);
            return Ok(hop.genop(&opname, vlist, GenopResult::LLType(self.lltype.clone())));
        }
        let opname = format!("{}neg", self.opprefix()?);
        Ok(hop.genop(&opname, vlist, GenopResult::LLType(self.lltype.clone())))
    }

    /// RPython `IntegerRepr.rtype_neg_ovf(self, hop)` (`rint.py`).
    fn rtype_neg_ovf(&self, hop: &HighLevelOp) -> RTypeResult {
        if self.s_result_unsigned(hop) {
            hop.exception_cannot_occur()?;
            return self.rtype_neg(hop);
        }
        rtype_call_helper(hop, "neg_ovf".to_string(), &[])
    }

    /// RPython `IntegerRepr.rtype_pos(self, hop)` (`rint.py`):
    /// identity pass-through after `inputargs(self)`.
    fn rtype_pos(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        Ok(vlist.into_iter().next())
    }

    /// RPython `IntegerRepr.rtype_int(self, hop)` (`rint.py`):
    ///
    /// ```python
    /// def rtype_int(self, hop):
    ///     if self.lowleveltype in (Unsigned, UnsignedLongLong):
    ///         raise TyperError("use intmask() instead of int(r_uint(...))")
    ///     vlist = hop.inputargs(Signed)
    ///     hop.exception_cannot_occur()
    ///     return vlist[0]
    /// ```
    fn rtype_int(&self, hop: &HighLevelOp) -> RTypeResult {
        if matches!(
            self.lltype,
            LowLevelType::Unsigned | LowLevelType::UnsignedLongLong
        ) {
            return Err(TyperError::message(
                "use intmask() instead of int(r_uint(...))",
            ));
        }
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Signed)])?;
        hop.exception_cannot_occur()?;
        Ok(vlist.into_iter().next())
    }

    /// RPython `IntegerRepr.rtype_float(_, hop)` (`rint.py`):
    /// coerce to Float and return — identity over the coerced Float value.
    fn rtype_float(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Float)])?;
        hop.exception_cannot_occur()?;
        Ok(vlist.into_iter().next())
    }

    /// RPython `IntegerRepr.rtype_chr(_, hop)` (`rint.py`).
    fn rtype_chr(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Signed)])?;
        if hop.has_implicit_exception("ValueError") {
            hop.exception_is_here()?;
            let llfunc = lowlevel_helper_from_hop(
                hop,
                "ll_check_chr",
                vec![LowLevelType::Signed],
                LowLevelType::Void,
            )?;
            let _ = hop.gendirectcall(&llfunc, vec![vlist[0].clone()])?;
        } else {
            hop.exception_cannot_occur()?;
        }
        Ok(hop.genop(
            "cast_int_to_char",
            vlist,
            GenopResult::LLType(LowLevelType::Char),
        ))
    }

    /// RPython `IntegerRepr.rtype_unichr(_, hop)` (`rint.py`).
    fn rtype_unichr(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Signed)])?;
        if hop.has_implicit_exception("ValueError") {
            hop.exception_is_here()?;
            let llfunc = lowlevel_helper_from_hop(
                hop,
                "ll_check_unichr",
                vec![LowLevelType::Signed],
                LowLevelType::Void,
            )?;
            let _ = hop.gendirectcall(&llfunc, vec![vlist[0].clone()])?;
        } else {
            hop.exception_cannot_occur()?;
        }
        Ok(hop.genop(
            "cast_int_to_unichar",
            vlist,
            GenopResult::LLType(LowLevelType::UniChar),
        ))
    }

    /// `IntegerRepr` inherits `Repr.rtype_str` (`rmodel.py`), which
    /// calls `hop.gendirectcall(self.ll_str, v_self)`. `IntegerRepr.ll_str`
    /// (`rint.py`, `@jit.elidable`) is `ll_int2dec(i)`. The default
    /// `Repr::rtype_str` raises `MissingRTypeOperation`, so this override
    /// supplies the per-width `ll_int2dec` helper graph and calls it.
    fn rtype_str(&self, hop: &HighLevelOp) -> RTypeResult {
        use crate::translator::rtyper::lltypesystem::rstr::{
            STRPTR, build_ll_int2dec_helper_graph,
        };
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        // The helper graph's input block is Signed for the signed
        // specialisation and Unsigned for the unsigned one; that must equal
        // this repr's lowleveltype (the type `gendirectcall` passes and the
        // declared helper signature). The 64-bit-target int reprs are exactly
        // Signed/Unsigned; the long-long widths keep the default raise.
        let (name, signed_input) = match &self.lltype {
            LowLevelType::Unsigned => ("ll_uint2dec", false),
            LowLevelType::Signed => ("ll_int2dec", true),
            _ => return Err(self.missing_rtype_operation("str")),
        };
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            name.to_string(),
            vec![self.lltype.clone()],
            STRPTR.clone(),
            move |_rtyper, _args, _result| build_ll_int2dec_helper_graph(name, signed_input),
        )?;
        hop.gendirectcall(&helper, vlist)
    }
}

// ____________________________________________________________
// Helper graph synthesis — `ll_hash_int(n) = intmask(n)` (`rint.py`).

/// RPython `ll_hash_int(n) = intmask(n)` (`rint.py`).
///
/// `rbuiltin.py` `rtype_intmask` does `hop.inputargs(lltype.Signed)`,
/// which converts the argument with `IntegerRepr.convert_from_to`
/// (`pairtype(IntegerRepr, IntegerRepr)` in `rint.py`): Bool →
/// `cast_bool_to_int` (`rbool.py`); Unsigned → `cast_uint_to_int`;
/// SignedLongLong → `truncate_longlong_to_int`; every other pair →
/// `cast_primitive` with a Signed result Variable. Synthesizes a
/// single-block helper graph that emits that same op. Suitable as a
/// `direct_call` target from [`super::rtuple::gen_hash_function`].
///
/// Reused from [`super::rbool::BoolRepr::get_ll_hash_function`] —
/// upstream `BoolRepr(IntegerRepr)` inherits `get_ll_hash_function`
/// directly via Python's MRO; the Rust port reaches the same
/// `ll_hash_int` graph by both Repr's `get_ll_hash_function` calling
/// this builder.
pub(crate) fn build_ll_hash_int_helper_graph(
    name: &str,
    lltype: &LowLevelType,
) -> Result<crate::flowspace::pygraph::PyGraph, TyperError> {
    use crate::flowspace::model::{Block, FunctionGraph, GraphFunc, Hlvalue, Link, SpaceOperation};
    use crate::translator::rtyper::rtyper::{helper_pygraph_from_graph, variable_with_lltype};

    let arg = variable_with_lltype("n", lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // `rtype_intmask` / `IntegerRepr.convert_from_to`: identity for
    // `Signed` (already the target). Bool MUST emit `cast_bool_to_int`
    // — without it the return link would carry a `Bool`-typed Variable
    // into a `Signed`-typed return slot (`gen_hash_function` mixes the
    // result with `int_mul`/`int_xor`).
    let return_value = if lltype == &LowLevelType::Signed {
        Hlvalue::Variable(arg)
    } else {
        let cast_op = match lltype {
            LowLevelType::Bool => "cast_bool_to_int",
            LowLevelType::Unsigned => "cast_uint_to_int",
            LowLevelType::SignedLongLong => "truncate_longlong_to_int",
            LowLevelType::UnsignedLongLong
            | LowLevelType::SignedLongLongLong
            | LowLevelType::UnsignedLongLongLong => "cast_primitive",
            other => {
                return Err(TyperError::message(format!(
                    "ll_hash_int: no cast-to-Signed op for {other:?}"
                )));
            }
        };
        let cast_var = variable_with_lltype("hashed", LowLevelType::Signed);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            cast_op,
            vec![Hlvalue::Variable(arg)],
            Hlvalue::Variable(cast_var.clone()),
        ));
        Hlvalue::Variable(cast_var)
    };

    use crate::flowspace::model::BlockRefExt;
    startblock.closeblock(vec![
        Link::new(vec![return_value], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["n".to_string()],
        func,
    ))
}

// ____________________________________________________________
// Standard integer singletons — `rint.py:193-198`.

/// RPython `signed_repr = getintegerrepr(Signed, 'int_')` (`rint.py`).
pub fn signed_repr() -> Arc<IntegerRepr> {
    static REPR: OnceLock<Arc<IntegerRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_"))))
        .clone()
}

/// RPython `unsigned_repr = getintegerrepr(Unsigned, 'uint_')` (`rint.py`).
pub fn unsigned_repr() -> Arc<IntegerRepr> {
    static REPR: OnceLock<Arc<IntegerRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(IntegerRepr::new(LowLevelType::Unsigned, Some("uint_"))))
        .clone()
}

/// RPython `signedlonglong_repr = getintegerrepr(SignedLongLong, 'llong_')`
/// (`rint.py`).
pub fn signedlonglong_repr() -> Arc<IntegerRepr> {
    static REPR: OnceLock<Arc<IntegerRepr>> = OnceLock::new();
    REPR.get_or_init(|| {
        Arc::new(IntegerRepr::new(
            LowLevelType::SignedLongLong,
            Some("llong_"),
        ))
    })
    .clone()
}

/// RPython `signedlonglonglong_repr =
/// getintegerrepr(SignedLongLongLong, 'lllong_')` (`rint.py`).
pub fn signedlonglonglong_repr() -> Arc<IntegerRepr> {
    static REPR: OnceLock<Arc<IntegerRepr>> = OnceLock::new();
    REPR.get_or_init(|| {
        Arc::new(IntegerRepr::new(
            LowLevelType::SignedLongLongLong,
            Some("lllong_"),
        ))
    })
    .clone()
}

/// RPython `unsignedlonglong_repr = getintegerrepr(UnsignedLongLong, 'ullong_')`
/// (`rint.py`).
pub fn unsignedlonglong_repr() -> Arc<IntegerRepr> {
    static REPR: OnceLock<Arc<IntegerRepr>> = OnceLock::new();
    REPR.get_or_init(|| {
        Arc::new(IntegerRepr::new(
            LowLevelType::UnsignedLongLong,
            Some("ullong_"),
        ))
    })
    .clone()
}

/// RPython `unsignedlonglonglong_repr =
/// getintegerrepr(UnsignedLongLongLong, 'ulllong_')` (`rint.py`).
pub fn unsignedlonglonglong_repr() -> Arc<IntegerRepr> {
    static REPR: OnceLock<Arc<IntegerRepr>> = OnceLock::new();
    REPR.get_or_init(|| {
        Arc::new(IntegerRepr::new(
            LowLevelType::UnsignedLongLongLong,
            Some("ulllong_"),
        ))
    })
    .clone()
}

fn integer_opprefix_for(lltype: &LowLevelType) -> Result<&'static str, TyperError> {
    match lltype {
        LowLevelType::Signed => Ok("int_"),
        LowLevelType::Unsigned => Ok("uint_"),
        LowLevelType::SignedLongLong => Ok("llong_"),
        LowLevelType::SignedLongLongLong => Ok("lllong_"),
        LowLevelType::UnsignedLongLong => Ok("ullong_"),
        LowLevelType::UnsignedLongLongLong => Ok("ulllong_"),
        other => Err(TyperError::message(format!(
            "arithmetic not supported on {other:?}, its size is too small"
        ))),
    }
}

fn integer_repr_for_lltype(lltype: &LowLevelType) -> Result<Arc<IntegerRepr>, TyperError> {
    match lltype {
        LowLevelType::Signed => Ok(signed_repr()),
        LowLevelType::Unsigned => Ok(unsigned_repr()),
        LowLevelType::SignedLongLong => Ok(signedlonglong_repr()),
        LowLevelType::SignedLongLongLong => Ok(signedlonglonglong_repr()),
        LowLevelType::UnsignedLongLong => Ok(unsignedlonglong_repr()),
        LowLevelType::UnsignedLongLongLong => Ok(unsignedlonglonglong_repr()),
        other => Err(TyperError::message(format!(
            "no IntegerRepr singleton for {other:?}"
        ))),
    }
}

fn hop_r_result(hop: &HighLevelOp) -> Result<Arc<dyn Repr>, TyperError> {
    hop.r_result
        .borrow()
        .clone()
        .ok_or_else(|| TyperError::message("HighLevelOp.r_result is not set"))
}

fn s_integer_flags(hop: &HighLevelOp, index: usize) -> (bool, bool) {
    hop.args_s
        .borrow()
        .get(index)
        .and_then(|s| match s {
            SomeValue::Integer(i) => Some((i.nonneg, i.unsigned)),
            _ => None,
        })
        .unwrap_or((false, false))
}

fn s_result_integer_flags(hop: &HighLevelOp) -> (bool, bool) {
    hop.s_result
        .borrow()
        .as_ref()
        .and_then(|s| match s {
            SomeValue::Integer(i) => Some((i.nonneg, i.unsigned)),
            _ => None,
        })
        .unwrap_or((false, false))
}

fn op_appendix(exc_cls_name: &str) -> Option<&'static str> {
    match exc_cls_name {
        "OverflowError" => Some("ovf"),
        "IndexError" => Some("idx"),
        "KeyError" => Some("key"),
        "ZeroDivisionError" => Some("zer"),
        "ValueError" => Some("val"),
        _ => None,
    }
}

fn all_integer_args_nonneg(hop: &HighLevelOp) -> bool {
    hop.args_s
        .borrow()
        .iter()
        .all(|s_arg| matches!(s_arg, SomeValue::Integer(i) if i.nonneg))
}

fn has_nonnegargs_helper(funcname: &str) -> bool {
    matches!(funcname, "ll_int_py_div" | "ll_int_py_mod")
}

fn hlvalue_concretetype(value: &Hlvalue) -> Option<LowLevelType> {
    match value {
        Hlvalue::Variable(v) => v.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    }
}

// ____________________________________________________________
// pairtype(IntegerRepr, X) conversions — rint.py.

pub fn pair_integer_integer_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    // rint.py:204-213: four fast paths, then generic cast_primitive.
    let opname = match (r_from.lowleveltype(), r_to.lowleveltype()) {
        (LowLevelType::Signed, LowLevelType::Unsigned) => Some("cast_int_to_uint"),
        (LowLevelType::Unsigned, LowLevelType::Signed) => Some("cast_uint_to_int"),
        (LowLevelType::Signed, LowLevelType::SignedLongLong) => Some("cast_int_to_longlong"),
        (LowLevelType::SignedLongLong, LowLevelType::Signed) => Some("truncate_longlong_to_int"),
        // A machine word widened to longlong. `cast_primitive` is only
        // ported for two word-sized integers; these are the widen ops
        // `jtransform.py` `_int_to_int_cast` emits instead.
        (LowLevelType::Unsigned, LowLevelType::UnsignedLongLong) => Some("cast_uint_to_ulonglong"),
        (LowLevelType::Signed, LowLevelType::UnsignedLongLong) => Some("cast_int_to_ulonglong"),
        (LowLevelType::Unsigned, LowLevelType::SignedLongLong) => Some("cast_uint_to_longlong"),
        _ => None,
    };
    if let Some(opname) = opname {
        return Ok(llops
            .genop(
                opname,
                vec![v.clone()],
                GenopResult::LLType(r_to.lowleveltype().clone()),
            )
            .map(Hlvalue::Variable));
    }
    // rint.py:214: anything not in the fast-path table emits
    // cast_primitive with the target lowleveltype as resulttype.
    Ok(llops
        .genop(
            "cast_primitive",
            vec![v.clone()],
            GenopResult::LLType(r_to.lowleveltype().clone()),
        )
        .map(Hlvalue::Variable))
}

pub fn pair_integer_float_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    if r_to.lowleveltype() != &LowLevelType::Float {
        return Ok(None);
    }
    let opname = match r_from.lowleveltype() {
        LowLevelType::Unsigned => "cast_uint_to_float",
        LowLevelType::Signed => "cast_int_to_float",
        LowLevelType::SignedLongLong => "cast_longlong_to_float",
        LowLevelType::UnsignedLongLong => "cast_ulonglong_to_float",
        _ => return Ok(None),
    };
    Ok(llops
        .genop(
            opname,
            vec![v.clone()],
            GenopResult::LLType(LowLevelType::Float),
        )
        .map(Hlvalue::Variable))
}

pub fn pair_float_integer_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    if r_from.lowleveltype() != &LowLevelType::Float {
        return Ok(None);
    }
    let opname = match r_to.lowleveltype() {
        LowLevelType::Unsigned => "cast_float_to_uint",
        LowLevelType::Signed => "cast_float_to_int",
        LowLevelType::SignedLongLong => "cast_float_to_longlong",
        LowLevelType::UnsignedLongLong => "cast_float_to_ulonglong",
        _ => return Ok(None),
    };
    Ok(llops
        .genop(
            opname,
            vec![v.clone()],
            GenopResult::LLType(r_to.lowleveltype().clone()),
        )
        .map(Hlvalue::Variable))
}

// ____________________________________________________________
// pairtype(IntegerRepr, IntegerRepr) — rint.py.

pub fn rtype_template(hop: &HighLevelOp, func: &str) -> RTypeResult {
    let r_result = hop_r_result(hop)?;
    let repr: Arc<dyn Repr> = if r_result.lowleveltype() == &LowLevelType::Bool {
        signed_repr()
    } else {
        r_result.clone()
    };
    let repr2: Arc<dyn Repr> = if matches!(func, "lshift" | "rshift") {
        signed_repr()
    } else {
        repr.clone()
    };
    let vlist = hop.inputargs(vec![
        ConvertedTo::Repr(repr.as_ref()),
        ConvertedTo::Repr(repr2.as_ref()),
    ])?;
    let prefix = integer_opprefix_for(repr.lowleveltype())?;
    let opname = format!("{prefix}{func}");

    if func.contains("_ovf") || func.starts_with("py_mod") || func.starts_with("py_div") {
        match opname.as_str() {
            "int_add_ovf" | "int_add_nonneg_ovf" | "int_sub_ovf" | "int_mul_ovf" => {
                hop.has_implicit_exception("OverflowError");
                hop.exception_is_here()?;
            }
            _ => {
                return Err(TyperError::message(format!(
                    "{func:?} should not be used here any more"
                )));
            }
        }
    } else {
        hop.exception_cannot_occur()?;
    }

    let v_res = hop
        .genop(&opname, vlist, GenopResult::Repr(repr.clone()))
        .ok_or_else(|| TyperError::message(format!("{opname} unexpectedly returned Void")))?;
    hop.llops
        .borrow_mut()
        .convertvar(v_res, repr.as_ref(), r_result.as_ref())
        .map(Some)
}

pub fn rtype_add_ovf(hop: &HighLevelOp) -> RTypeResult {
    let mut func = "add_ovf";
    let r_result = hop_r_result(hop)?;
    if integer_opprefix_for(r_result.lowleveltype())? == "int_" {
        let (arg1_nonneg, _) = s_integer_flags(hop, 0);
        let (arg2_nonneg, _) = s_integer_flags(hop, 1);
        if arg2_nonneg {
            func = "add_nonneg_ovf";
        } else if arg1_nonneg {
            let copied = hop.copy();
            copied.swap_fst_snd_args();
            return rtype_template(&copied, "add_nonneg_ovf");
        }
    }
    rtype_template(hop, func)
}

pub fn rtype_call_helper(
    hop: &HighLevelOp,
    mut func: String,
    implicit_excs: &[&str],
) -> RTypeResult {
    let mut any_implicit_exception = false;
    if func.ends_with("_ovf") {
        let (_, unsigned_result) = s_result_integer_flags(hop);
        if unsigned_result {
            return Err(TyperError::message(format!("forbidden unsigned {func}")));
        }
        hop.has_implicit_exception("OverflowError");
        any_implicit_exception = true;
    }

    for implicit_exc in implicit_excs {
        if hop.has_implicit_exception(implicit_exc) {
            let appendix = op_appendix(implicit_exc).ok_or_else(|| {
                TyperError::message(format!("unknown implicit exception {implicit_exc}"))
            })?;
            func.push('_');
            func.push_str(appendix);
            any_implicit_exception = true;
        }
    }

    if !any_implicit_exception && !func.starts_with("py_mod") && !func.starts_with("py_div") {
        return rtype_template(hop, &func);
    }

    let repr = hop_r_result(hop)?;
    if repr.lowleveltype() == &LowLevelType::Bool {
        return Err(TyperError::message(
            "_rtype_call_helper result repr must not be Bool",
        ));
    }
    let vlist = if matches!(func.as_str(), "abs_ovf" | "neg_ovf") {
        hop.inputargs(vec![ConvertedTo::Repr(repr.as_ref())])?
    } else if func.starts_with("lshift") || func.starts_with("rshift") {
        let signed = signed_repr();
        hop.inputargs(vec![
            ConvertedTo::Repr(repr.as_ref()),
            ConvertedTo::Repr(signed.as_ref()),
        ])?
    } else {
        hop.inputargs(vec![
            ConvertedTo::Repr(repr.as_ref()),
            ConvertedTo::Repr(repr.as_ref()),
        ])?
    };

    if any_implicit_exception {
        hop.exception_is_here()?;
    } else {
        hop.exception_cannot_occur()?;
    }

    let prefix = integer_opprefix_for(repr.lowleveltype())?;
    let mut funcname = format!("ll_{prefix}{func}");
    if all_integer_args_nonneg(hop) {
        let nonneg_funcname = format!("{funcname}_nonnegargs");
        if has_nonnegargs_helper(&funcname) {
            funcname = nonneg_funcname;
        }
    }
    let arg_types = vlist
        .iter()
        .map(|v| {
            hlvalue_concretetype(v)
                .ok_or_else(|| TyperError::message("gendirectcall argument missing concretetype"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let llfunc = lowlevel_helper_from_hop(hop, funcname, arg_types, repr.lowleveltype().clone())?;
    let v_result = hop
        .gendirectcall(&llfunc, vlist)?
        .ok_or_else(|| TyperError::message("direct_call unexpectedly returned Void"))?;
    if hlvalue_concretetype(&v_result).as_ref() != Some(repr.lowleveltype()) {
        return Err(TyperError::message(format!(
            "direct_call result type mismatch: expected {}, got {:?}",
            repr.lowleveltype().short_name(),
            hlvalue_concretetype(&v_result)
        )));
    }
    Ok(Some(v_result))
}

fn lowlevel_helper_from_hop(
    hop: &HighLevelOp,
    name: impl Into<String>,
    args: Vec<LowLevelType>,
    result: LowLevelType,
) -> Result<crate::translator::rtyper::rtyper::LowLevelFunction, TyperError> {
    hop.rtyper.lowlevel_helper_function(name, args, result)
}

pub fn rtype_compare_template(hop: &HighLevelOp, func: &str) -> RTypeResult {
    let (nonneg1, unsigned1) = s_integer_flags(hop, 0);
    let (nonneg2, unsigned2) = s_integer_flags(hop, 1);
    if (unsigned1 || unsigned2) && (!nonneg1 || !nonneg2) {
        return Err(TyperError::message(
            "comparing a signed and an unsigned number",
        ));
    }
    let (s_int1, s_int2) = {
        let args_s = hop.args_s.borrow();
        let s_int1 = args_s
            .first()
            .cloned()
            .ok_or_else(|| TyperError::message("missing left integer annotation"))?;
        let s_int2 = args_s
            .get(1)
            .cloned()
            .ok_or_else(|| TyperError::message("missing right integer annotation"))?;
        (s_int1, s_int2)
    };
    let s_union =
        unionof([&s_int1, &s_int2]).map_err(|err| TyperError::message(err.to_string()))?;
    let r_union = hop.rtyper.getrepr(&s_union)?;
    let repr = if r_union.lowleveltype() == &LowLevelType::Bool {
        // RPython: `rtyper.getrepr(unionof(...)).as_int`.  BoolRepr sets
        // `as_int = signed_repr` in rbool.py.
        signed_repr()
    } else {
        integer_repr_for_lltype(r_union.lowleveltype())?
    };
    let vlist = hop.inputargs(vec![
        ConvertedTo::Repr(repr.as_ref()),
        ConvertedTo::Repr(repr.as_ref()),
    ])?;
    hop.exception_is_here()?;
    let prefix = integer_opprefix_for(repr.lowleveltype())?;
    let opname = format!("{prefix}{func}");
    Ok(hop.genop(&opname, vlist, GenopResult::LLType(LowLevelType::Bool)))
}

pub const INT_BITS_1: u32 = i64::BITS - 1;
pub const LLONG_BITS_1: u32 = i64::BITS - 1;
pub const LLLONG_BITS_1: u32 = i128::BITS - 1;
pub const INT_MIN: i64 = i64::MIN;

fn zero_division(message: &str) -> TyperError {
    TyperError::message(message)
}

fn overflow_error(message: &str) -> TyperError {
    TyperError::message(message)
}

/// RPython `ll_int_py_div(x, y)` (`rint.py`).
pub fn ll_int_py_div(x: i64, y: i64) -> i64 {
    let r = x.wrapping_div(y);
    let p = r.wrapping_mul(y);
    let u = if y < 0 {
        p.wrapping_sub(x)
    } else {
        x.wrapping_sub(p)
    };
    r.wrapping_add(u >> INT_BITS_1)
}

pub fn ll_int_py_div_nonnegargs(x: i64, y: i64) -> i64 {
    let r = x.wrapping_div(y);
    assert!(r >= 0, "int_py_div_nonnegargs(): one arg is negative");
    r
}

pub fn ll_int_py_div_zer(x: i64, y: i64) -> Result<i64, TyperError> {
    if y == 0 {
        return Err(zero_division("integer division"));
    }
    Ok(ll_int_py_div(x, y))
}

pub fn ll_int_py_div_ovf(x: i64, y: i64) -> Result<i64, TyperError> {
    if x == INT_MIN && y == -1 {
        return Err(overflow_error("integer division"));
    }
    Ok(ll_int_py_div(x, y))
}

pub fn ll_int_py_div_ovf_zer(x: i64, y: i64) -> Result<i64, TyperError> {
    if y == 0 {
        return Err(zero_division("integer division"));
    }
    ll_int_py_div_ovf(x, y)
}

pub fn ll_uint_py_div(x: u64, y: u64) -> u64 {
    x / y
}

pub fn ll_uint_py_div_zer(x: u64, y: u64) -> Result<u64, TyperError> {
    if y == 0 {
        return Err(zero_division("unsigned integer division"));
    }
    Ok(ll_uint_py_div(x, y))
}

pub fn ll_llong_py_div(x: i64, y: i64) -> i64 {
    ll_int_py_div(x, y)
}

pub fn ll_llong_py_div_zer(x: i64, y: i64) -> Result<i64, TyperError> {
    ll_int_py_div_zer(x, y)
}

pub fn ll_ullong_py_div(x: u64, y: u64) -> u64 {
    ll_uint_py_div(x, y)
}

pub fn ll_ullong_py_div_zer(x: u64, y: u64) -> Result<u64, TyperError> {
    ll_uint_py_div_zer(x, y)
}

pub fn ll_lllong_py_div(x: i128, y: i128) -> i128 {
    let r = x.wrapping_div(y);
    let p = r.wrapping_mul(y);
    let u = if y < 0 {
        p.wrapping_sub(x)
    } else {
        x.wrapping_sub(p)
    };
    r.wrapping_add(u >> LLLONG_BITS_1)
}

pub fn ll_lllong_py_div_zer(x: i128, y: i128) -> Result<i128, TyperError> {
    if y == 0 {
        return Err(zero_division("longlonglong division"));
    }
    Ok(ll_lllong_py_div(x, y))
}

pub fn ll_ulllong_py_div(x: u128, y: u128) -> u128 {
    x / y
}

pub fn ll_ulllong_py_div_zer(x: u128, y: u128) -> Result<u128, TyperError> {
    if y == 0 {
        return Err(zero_division("unsigned longlonglong division"));
    }
    Ok(ll_ulllong_py_div(x, y))
}

pub fn ll_int_py_mod(x: i64, y: i64) -> i64 {
    let r = x.wrapping_rem(y);
    let u = if y < 0 { -r } else { r };
    r.wrapping_add(y & (u >> INT_BITS_1))
}

pub fn ll_int_py_mod_nonnegargs(x: i64, y: i64) -> i64 {
    let r = x.wrapping_rem(y);
    assert!(r >= 0, "int_py_mod_nonnegargs(): one arg is negative");
    r
}

pub fn ll_int_py_mod_zer(x: i64, y: i64) -> Result<i64, TyperError> {
    if y == 0 {
        return Err(zero_division("integer modulo"));
    }
    Ok(ll_int_py_mod(x, y))
}

pub fn ll_int_py_mod_ovf(x: i64, y: i64) -> Result<i64, TyperError> {
    if x == INT_MIN && y == -1 {
        return Err(overflow_error("integer modulo"));
    }
    Ok(ll_int_py_mod(x, y))
}

pub fn ll_int_py_mod_ovf_zer(x: i64, y: i64) -> Result<i64, TyperError> {
    if y == 0 {
        return Err(zero_division("integer modulo"));
    }
    ll_int_py_mod_ovf(x, y)
}

pub fn ll_uint_py_mod(x: u64, y: u64) -> u64 {
    x % y
}

pub fn ll_uint_py_mod_zer(x: u64, y: u64) -> Result<u64, TyperError> {
    if y == 0 {
        return Err(zero_division("unsigned integer modulo"));
    }
    Ok(ll_uint_py_mod(x, y))
}

pub fn ll_llong_py_mod(x: i64, y: i64) -> i64 {
    ll_int_py_mod(x, y)
}

pub fn ll_llong_py_mod_zer(x: i64, y: i64) -> Result<i64, TyperError> {
    ll_int_py_mod_zer(x, y)
}

pub fn ll_ullong_py_mod(x: u64, y: u64) -> u64 {
    ll_uint_py_mod(x, y)
}

pub fn ll_ullong_py_mod_zer(x: u64, y: u64) -> Result<u64, TyperError> {
    ll_uint_py_mod_zer(x, y)
}

pub fn ll_lllong_py_mod(x: i128, y: i128) -> i128 {
    let r = x.wrapping_rem(y);
    let u = if y < 0 { -r } else { r };
    r.wrapping_add(y & (u >> LLLONG_BITS_1))
}

pub fn ll_lllong_py_mod_zer(x: i128, y: i128) -> Result<i128, TyperError> {
    if y == 0 {
        return Err(zero_division("longlonglong modulo"));
    }
    Ok(ll_lllong_py_mod(x, y))
}

pub fn ll_ulllong_py_mod(x: u128, y: u128) -> u128 {
    x % y
}

pub fn ll_ulllong_py_mod_zer(x: u128, y: u128) -> Result<u128, TyperError> {
    if y == 0 {
        return Err(zero_division("unsigned longlonglong modulo"));
    }
    Ok(ll_ulllong_py_mod(x, y))
}

pub fn ll_int_lshift_ovf(x: i64, y: u32) -> Result<i64, TyperError> {
    let result = x
        .checked_shl(y)
        .ok_or_else(|| overflow_error("x<<y loosing bits or changing sign"))?;
    if result.checked_shr(y).unwrap_or_default() != x {
        return Err(overflow_error("x<<y loosing bits or changing sign"));
    }
    Ok(result)
}

pub fn ll_int_neg_ovf(x: i64) -> Result<i64, TyperError> {
    if x == INT_MIN {
        return Err(overflow_error("integer negation"));
    }
    Ok(-x)
}

pub fn ll_int_abs_ovf(x: i64) -> Result<i64, TyperError> {
    if x == INT_MIN {
        return Err(overflow_error("integer absolute value"));
    }
    Ok(x.abs())
}

pub fn ll_hash_int(n: i64) -> i64 {
    n
}

pub fn ll_hash_long_long(n: i64) -> i64 {
    n.wrapping_add(9_i64.wrapping_mul(n >> 32))
}

pub fn ll_eq_shortint(n: i64, m: i64) -> bool {
    n == m
}

pub fn ll_check_chr(n: i64) -> Result<(), TyperError> {
    if (0..=255).contains(&n) {
        Ok(())
    } else {
        Err(TyperError::message("ValueError"))
    }
}

pub fn ll_check_unichr(n: i64) -> Result<(), TyperError> {
    if (0..=0x10ffff).contains(&n) {
        Ok(())
    } else {
        Err(TyperError::message("ValueError"))
    }
}

/// RPython `_integer_reprs = {}` + `getintegerrepr(lltype, prefix=None)`
/// (`rint.py`).
///
/// Returns the cached singleton for each lltype. The standard integer
/// lltypes use their module-level singleton accessors, and any other
/// integer-like lltype is kept in the fallback cache just like
/// upstream's `_integer_reprs` dictionary.
pub fn getintegerrepr(lltype: &LowLevelType) -> Arc<IntegerRepr> {
    match lltype {
        LowLevelType::Signed => signed_repr(),
        LowLevelType::Unsigned => unsigned_repr(),
        LowLevelType::SignedLongLong => signedlonglong_repr(),
        LowLevelType::SignedLongLongLong => signedlonglonglong_repr(),
        LowLevelType::UnsignedLongLong => unsignedlonglong_repr(),
        LowLevelType::UnsignedLongLongLong => unsignedlonglonglong_repr(),
        other => {
            static INTEGER_REPRS: OnceLock<Mutex<HashMap<LowLevelType, Arc<IntegerRepr>>>> =
                OnceLock::new();
            let cache = INTEGER_REPRS.get_or_init(|| Mutex::new(HashMap::new()));
            let mut cache = cache.lock();
            cache
                .entry(other.clone())
                .or_insert_with(|| Arc::new(IntegerRepr::new(other.clone(), None)))
                .clone()
        }
    }
}

// RPython `SomeInteger.rtyper_makerepr(self, rtyper)` (`rint.py`)
// inlines into [`super::rmodel::rtyper_makerepr`]'s `SomeValue::Integer`
// arm — `build_number(None, knowntype)` + `getintegerrepr(lltype)`.
// Centralising the dispatch there keeps the `rtyper.reprs` cache key
// computation (`rtyper_makekey`) and the Repr construction adjacent,
// matching the upstream `rtyper.py getrepr` pattern.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::model::{
        KnownType, SomeChar, SomeInteger, SomeUnicodeCodePoint, SomeValue,
    };
    use crate::flowspace::model::{SpaceOperation, Variable};
    use crate::translator::rtyper::rmodel::setupstate;
    use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};
    use std::cell::RefCell;
    use std::rc::Rc;

    fn unary_integer_hop(
        opname: &str,
        rtyper: &Rc<RPythonTyper>,
        llops: Rc<RefCell<LowLevelOpList>>,
        arg_type: LowLevelType,
        result_type: LowLevelType,
        s_arg: SomeValue,
        s_result: SomeValue,
    ) -> HighLevelOp {
        let v_arg = Variable::new();
        v_arg.set_concretetype(Some(arg_type.clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(result_type));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                opname.to_string(),
                vec![Hlvalue::Variable(v_arg)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops,
        );
        let r_arg = getintegerrepr(&arg_type) as Arc<dyn Repr>;
        let r_result = getintegerrepr(&arg_type) as Arc<dyn Repr>;
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().push(s_arg);
        hop.args_r.borrow_mut().push(Some(r_arg));
        *hop.s_result.borrow_mut() = Some(s_result);
        *hop.r_result.borrow_mut() = Some(r_result);
        hop
    }

    #[test]
    fn signed_repr_matches_rint_singleton_shape() {
        let r = signed_repr();
        assert_eq!(r.lowleveltype(), &LowLevelType::Signed);
        assert_eq!(r.opprefix().unwrap(), "int_");
        assert!(Arc::ptr_eq(&r, &signed_repr()));
    }

    #[test]
    fn unsigned_and_longlong_singletons_match_upstream_opprefixes() {
        // rint.py:193-198.
        assert_eq!(unsigned_repr().opprefix().unwrap(), "uint_");
        assert_eq!(signedlonglong_repr().opprefix().unwrap(), "llong_");
        assert_eq!(unsignedlonglong_repr().opprefix().unwrap(), "ullong_");
    }

    #[test]
    fn low_level_integer_helpers_match_python_div_mod_and_guards() {
        assert_eq!(INT_BITS_1, 63);
        assert_eq!(INT_MIN, i64::MIN);

        assert_eq!(ll_int_py_div(-3, 2), -2);
        assert_eq!(ll_int_py_div(3, -2), -2);
        assert_eq!(ll_int_py_mod(-3, 2), 1);
        assert_eq!(ll_int_py_mod(3, -2), -1);

        assert!(ll_int_py_div_zer(1, 0).is_err());
        assert!(ll_int_py_mod_zer(1, 0).is_err());
        assert!(ll_int_py_div_ovf(INT_MIN, -1).is_err());
        assert!(ll_int_neg_ovf(INT_MIN).is_err());
        assert!(ll_int_abs_ovf(INT_MIN).is_err());

        assert_eq!(ll_uint_py_div(7, 3), 2);
        assert_eq!(ll_uint_py_mod(7, 3), 1);
        assert_eq!(ll_hash_int(42), 42);
        assert!(ll_eq_shortint(3, 3));
        assert!(ll_check_chr(255).is_ok());
        assert!(ll_check_chr(256).is_err());
        assert!(ll_check_unichr(0x10ffff).is_ok());
        assert!(ll_check_unichr(0x110000).is_err());
    }

    #[test]
    fn opprefix_on_unknown_size_raises_typer_error() {
        // rint.py:24-29 — a repr with `_opprefix = None` raises on
        // `self.opprefix`. Upstream builds these via
        // `build_number(None, SignedShort)` etc. Pyre reproduces the
        // error path without requiring the short-int lltype.
        let r = IntegerRepr::new(LowLevelType::Signed, None);
        let err = r.opprefix().unwrap_err();
        assert!(err.to_string().contains("arithmetic not supported"));
    }

    #[test]
    fn integer_repr_convert_const_accepts_int_and_bool() {
        let r = signed_repr();
        let c = r.convert_const(&ConstValue::Int(7)).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Signed));
        let c = r.convert_const(&ConstValue::Bool(true)).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Signed));

        // `cast_primitive(Unsigned, int)` → `r_uint` (`rint.convert_const`).
        let u = unsigned_repr();
        let c = u.convert_const(&ConstValue::Int(7)).unwrap();
        assert_eq!(c.value, ConstValue::UInt(7));
        assert_eq!(c.concretetype, Some(LowLevelType::Unsigned));
        let c = u.convert_const(&ConstValue::UInt(7)).unwrap();
        assert_eq!(c.value, ConstValue::UInt(7));
        let c = u.convert_const(&ConstValue::Bool(true)).unwrap();
        assert_eq!(c.value, ConstValue::UInt(1));
    }

    #[test]
    fn integer_repr_convert_const_passes_address_offset_through() {
        // rint.py:26-27 — `if isinstance(value, objectmodel.Symbolic):
        // return value`. llmemory address offsets carry through unchanged.
        let r = signed_repr();
        let offset = ConstValue::AddressOffset(
            crate::translator::rtyper::lltypesystem::llmemory::AddressOffset::ItemOffset(
                crate::translator::rtyper::lltypesystem::llmemory::ItemOffset {
                    TYPE: LowLevelType::Signed,
                    repeat: 1,
                },
            ),
        );
        let c = r.convert_const(&offset).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Signed));
        assert_eq!(c.value, offset);
    }

    #[test]
    fn integer_repr_convert_const_rejects_non_integer() {
        // rint.py:37 — `raise TyperError("not an integer: %r" % ...)`.
        let r = signed_repr();
        let err = r.convert_const(&ConstValue::byte_str("x")).unwrap_err();
        assert!(err.to_string().contains("not an integer"));
    }

    #[test]
    fn integer_repr_convert_const_preserves_full_width_carriers() {
        for (repr, value, expected_lltype) in [
            (
                signedlonglonglong_repr(),
                ConstValue::Int128(i128::MIN),
                LowLevelType::SignedLongLongLong,
            ),
            (
                unsignedlonglonglong_repr(),
                ConstValue::UInt128(u128::MAX),
                LowLevelType::UnsignedLongLongLong,
            ),
        ] {
            let converted = repr
                .convert_const(&value)
                .expect("wide Number constant converts");
            assert_eq!(converted.value, value);
            assert_eq!(converted.concretetype, Some(expected_lltype));
        }
        // `cast_primitive` of `r_uint` onto SignedLongLongLong is a
        // positive `r_longlonglong` (`IntegerRepr.convert_const`).
        let converted = signedlonglonglong_repr()
            .convert_const(&ConstValue::Int(5))
            .expect("Int converts to SignedLongLongLong");
        assert_eq!(converted.value, ConstValue::Int128(5));
        let converted = unsignedlonglonglong_repr()
            .convert_const(&ConstValue::Int(5))
            .expect("Int converts to UnsignedLongLongLong");
        assert_eq!(converted.value, ConstValue::UInt128(5));
        let converted = signedlonglonglong_repr()
            .convert_const(&ConstValue::UInt(7))
            .expect("UInt converts to SignedLongLongLong");
        assert_eq!(converted.value, ConstValue::Int128(7));
        let converted = signedlonglonglong_repr()
            .convert_const(&ConstValue::UInt(1u64 << 63))
            .expect("UInt(1<<63) converts to positive Int128");
        assert_eq!(converted.value, ConstValue::Int128(1i128 << 63));
    }

    #[test]
    fn integer_repr_convert_const_casts_128bit_via_number_cast() {
        // `IntegerRepr.convert_const` → `cast_primitive` → `Number._cast`.
        let converted = signed_repr()
            .convert_const(&ConstValue::Int128((1i128 << 64) + 5))
            .expect("intmask of Int128((1<<64)+5)");
        assert_eq!(converted.value, ConstValue::Int(5));
        let converted = signed_repr()
            .convert_const(&ConstValue::Int128(-1))
            .expect("intmask of Int128(-1)");
        assert_eq!(converted.value, ConstValue::Int(-1));

        let converted = unsigned_repr()
            .convert_const(&ConstValue::UInt128(u128::MAX))
            .expect("r_uint of UInt128(u128::MAX)");
        assert_eq!(converted.value, ConstValue::UInt(u64::MAX));

        let converted = unsignedlonglonglong_repr()
            .convert_const(&ConstValue::Int128(-1))
            .expect("r_ulonglonglong of Int128(-1)");
        assert_eq!(converted.value, ConstValue::UInt128(u128::MAX));

        let err = signedlonglonglong_repr()
            .convert_const(&ConstValue::UInt128(u128::MAX))
            .expect_err("longlonglongmask of UInt128(u128::MAX)");
        assert!(err.to_string().contains("OverflowError"));
        let converted = signedlonglonglong_repr()
            .convert_const(&ConstValue::UInt128(5))
            .expect("longlonglongmask of UInt128(5)");
        assert_eq!(converted.value, ConstValue::Int128(5));
    }

    #[test]
    fn wide_integer_conversions_and_binops_use_rpython_lllong_families() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));

        // rint.py convert_from_to — word-sized inputs promoted for an rbigint
        // LONG_TYPE/ULONG_TYPE operation go through the generic
        // `cast_primitive` arm with the exact 128-bit result lltype.
        for (from, to, source_lltype, expected_lltype) in [
            (
                signed_repr(),
                signedlonglonglong_repr(),
                LowLevelType::Signed,
                LowLevelType::SignedLongLongLong,
            ),
            (
                unsigned_repr(),
                unsignedlonglonglong_repr(),
                LowLevelType::Unsigned,
                LowLevelType::UnsignedLongLongLong,
            ),
        ] {
            let mut llops = LowLevelOpList::new(rtyper.clone(), None);
            let source = Variable::new();
            source.set_concretetype(Some(source_lltype));
            let converted = pair_integer_integer_convert_from_to(
                from.as_ref(),
                to.as_ref(),
                &Hlvalue::Variable(source),
                &mut llops,
            )
            .expect("word-to-wide conversion rtypes")
            .expect("word-to-wide conversion returns a value");
            assert_eq!(llops.ops.len(), 1);
            assert_eq!(llops.ops[0].opname, "cast_primitive");
            assert_eq!(
                hlvalue_concretetype(&converted),
                Some(expected_lltype),
                "cast_primitive must retain the exact wide result type"
            );
        }

        // rint.py _rtype_template — arithmetic itself selects the result repr's
        // `lllong_` / `ulllong_` prefix, never the word-sized `int_` family.
        for (known, repr, expected_opname) in [
            (
                KnownType::LongLongLong,
                signedlonglonglong_repr(),
                "lllong_add",
            ),
            (
                KnownType::ULongLongLong,
                unsignedlonglonglong_repr(),
                "ulllong_add",
            ),
        ] {
            let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
            let v_left = Variable::new();
            v_left.set_concretetype(Some(repr.lowleveltype().clone()));
            let v_right = Variable::new();
            v_right.set_concretetype(Some(repr.lowleveltype().clone()));
            let v_result = Variable::new();
            v_result.set_concretetype(Some(repr.lowleveltype().clone()));
            let hop = HighLevelOp::new(
                rtyper.clone(),
                SpaceOperation::new(
                    "add",
                    vec![Hlvalue::Variable(v_left), Hlvalue::Variable(v_right)],
                    Hlvalue::Variable(v_result),
                ),
                Vec::new(),
                llops.clone(),
            );
            let s_integer = SomeValue::Integer(SomeInteger::new_with_knowntype(false, known));
            let repr_dyn = repr.clone() as Arc<dyn Repr>;
            hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
            hop.args_s
                .borrow_mut()
                .extend([s_integer.clone(), s_integer.clone()]);
            hop.args_r
                .borrow_mut()
                .extend([Some(repr_dyn.clone()), Some(repr_dyn.clone())]);
            *hop.s_result.borrow_mut() = Some(s_integer);
            *hop.r_result.borrow_mut() = Some(repr_dyn);

            let result = rtype_template(&hop, "add")
                .expect("wide add rtypes")
                .expect("wide add returns a value");
            assert_eq!(
                hlvalue_concretetype(&result),
                Some(repr.lowleveltype().clone())
            );
            let ops = llops.borrow();
            assert_eq!(ops.ops.len(), 1);
            assert_eq!(ops.ops[0].opname, expected_opname);
        }
    }

    #[test]
    fn getintegerrepr_on_standard_lltypes_returns_cached_singletons() {
        // rint.py cache — `_integer_reprs[lltype]` returns the
        // same IntegerRepr instance on repeated lookups for the
        // standard integer lltypes.
        assert!(Arc::ptr_eq(
            &getintegerrepr(&LowLevelType::Signed),
            &signed_repr()
        ));
        assert!(Arc::ptr_eq(
            &getintegerrepr(&LowLevelType::Unsigned),
            &unsigned_repr()
        ));
        assert!(Arc::ptr_eq(
            &getintegerrepr(&LowLevelType::SignedLongLong),
            &signedlonglong_repr()
        ));
        assert!(Arc::ptr_eq(
            &getintegerrepr(&LowLevelType::SignedLongLongLong),
            &signedlonglonglong_repr()
        ));
        assert!(Arc::ptr_eq(
            &getintegerrepr(&LowLevelType::UnsignedLongLong),
            &unsignedlonglong_repr()
        ));
        assert!(Arc::ptr_eq(
            &getintegerrepr(&LowLevelType::UnsignedLongLongLong),
            &unsignedlonglonglong_repr()
        ));
    }

    /// rint.py — `IntegerRepr.get_ll_hash_function` returns
    /// `ll_hash_int`. For Signed (host word), `intmask(n) = n`; the
    /// synthesized helper graph should be a single block with no
    /// operations, returning the inputarg directly.
    #[test]
    fn integer_repr_get_ll_hash_function_signed_synthesizes_identity_helper() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r = signed_repr();

        let llfn = r
            .get_ll_hash_function(&rtyper)
            .expect("get_ll_hash_function(Signed) should not error")
            .expect("get_ll_hash_function(Signed) returns Some helper");
        assert_eq!(llfn.name, "ll_hash_int_Signed");
        assert_eq!(llfn.args, vec![LowLevelType::Signed]);
        assert_eq!(llfn.result, LowLevelType::Signed);

        let graph = llfn.graph.as_ref().expect("helper carries a graph");
        let inner = graph.graph.borrow();
        let startblock = inner.startblock.borrow();
        assert!(
            startblock.operations.is_empty(),
            "Signed identity has no operations, got {:?}",
            startblock.operations
        );
        assert_eq!(
            startblock.exits.len(),
            1,
            "single closeblock link to returnblock"
        );
        let exit = startblock.exits[0].borrow();
        assert_eq!(exit.args.len(), 1);
        assert!(
            matches!(exit.args[0].as_ref(), Some(Hlvalue::Variable(_))),
            "identity returns the inputarg directly"
        );
    }

    /// rint.py get_ll_hash_function — wider widths emit a single `cast_*_to_int` op.
    #[test]
    fn integer_repr_get_ll_hash_function_unsigned_emits_cast_uint_to_int() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r = unsigned_repr();

        let llfn = r
            .get_ll_hash_function(&rtyper)
            .expect("get_ll_hash_function(Unsigned)")
            .expect("returns Some helper");
        assert_eq!(llfn.name, "ll_hash_int_Unsigned");
        assert_eq!(llfn.args, vec![LowLevelType::Unsigned]);

        let graph = llfn.graph.as_ref().expect("helper carries a graph");
        let inner = graph.graph.borrow();
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.operations.len(), 1);
        assert_eq!(startblock.operations[0].opname, "cast_uint_to_int");
    }

    /// `rtype_intmask` / `IntegerRepr.convert_from_to`: widths without a
    /// dedicated cast-to-Signed op emit `cast_primitive`.
    #[test]
    fn integer_repr_get_ll_hash_function_wide_emits_convert_from_to_ops() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        for (r, expected) in [
            (signedlonglong_repr(), "truncate_longlong_to_int"),
            (unsignedlonglong_repr(), "cast_primitive"),
            (signedlonglonglong_repr(), "cast_primitive"),
            (unsignedlonglonglong_repr(), "cast_primitive"),
        ] {
            let llfn = r
                .get_ll_hash_function(&rtyper)
                .expect("get_ll_hash_function")
                .expect("returns Some helper");
            let graph = llfn.graph.as_ref().expect("helper carries a graph");
            let inner = graph.graph.borrow();
            let startblock = inner.startblock.borrow();
            assert_eq!(startblock.operations.len(), 1, "{}", llfn.name);
            assert_eq!(startblock.operations[0].opname, expected, "{}", llfn.name);
        }
    }

    /// `lowlevel_helper_function_with_builder` caches by
    /// `(name, args, result)` so repeated calls return the same
    /// graph — important so multiple tuple shapes share helpers.
    #[test]
    fn integer_repr_get_ll_hash_function_signed_caches_helper_graph() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r = signed_repr();
        let a = r
            .get_ll_hash_function(&rtyper)
            .expect("first call")
            .expect("Some");
        let b = r
            .get_ll_hash_function(&rtyper)
            .expect("second call")
            .expect("Some");
        let ga = a.graph.as_ref().expect("a.graph");
        let gb = b.graph.as_ref().expect("b.graph");
        assert!(Rc::ptr_eq(&ga.graph, &gb.graph), "cached graph identity");
    }

    #[test]
    fn setup_on_integer_repr_reaches_finished_state() {
        let r = signed_repr();
        // Singleton may have already been set up by earlier tests.
        if matches!(r.state().get(), setupstate::NotInitialized) {
            r.setup().expect("IntegerRepr.setup() should succeed");
        }
        assert_eq!(r.state().get(), setupstate::Finished);
    }

    #[test]
    fn rtyper_getrepr_on_some_integer_returns_lltype_matched_singleton() {
        // rint.py — `SomeInteger.rtyper_makerepr` resolves via
        // `build_number(None, knowntype) → getintegerrepr(lltype)`.
        // Verify the end-to-end dispatch keeps cached-Arc identity.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);

        let s_int = SomeValue::Integer(SomeInteger::new(false, false));
        let r = rtyper.getrepr(&s_int).expect("getrepr(SomeInteger Int)");
        assert_eq!(r.lowleveltype(), &LowLevelType::Signed);
        let expected = signed_repr() as Arc<dyn Repr>;
        assert!(Arc::ptr_eq(&r, &expected));

        // rtyper.py:54-57 cache sanity — second lookup for the same
        // knowntype key dedupes to the same Arc.
        let r2 = rtyper
            .getrepr(&s_int)
            .expect("second getrepr(SomeInteger Int)");
        assert!(Arc::ptr_eq(&r, &r2));

        // rint.py rtyper_makekey — key is `(class, knowntype)`; r_uint keys
        // resolve to the distinct `unsigned_repr` singleton.
        let s_uint = SomeValue::Integer(SomeInteger::new(false, true));
        let r_uint = rtyper.getrepr(&s_uint).expect("getrepr(SomeInteger Ruint)");
        let expected_uint = unsigned_repr() as Arc<dyn Repr>;
        assert!(Arc::ptr_eq(&r_uint, &expected_uint));

        let s_lllong = SomeValue::Integer(SomeInteger::new_with_knowntype(
            false,
            KnownType::LongLongLong,
        ));
        let r_lllong = rtyper
            .getrepr(&s_lllong)
            .expect("getrepr(SomeInteger LongLongLong)");
        let expected_lllong = signedlonglonglong_repr() as Arc<dyn Repr>;
        assert!(Arc::ptr_eq(&r_lllong, &expected_lllong));

        let s_ulllong = SomeValue::Integer(SomeInteger::new_with_knowntype(
            false,
            KnownType::ULongLongLong,
        ));
        let r_ulllong = rtyper
            .getrepr(&s_ulllong)
            .expect("getrepr(SomeInteger ULongLongLong)");
        let expected_ulllong = unsignedlonglonglong_repr() as Arc<dyn Repr>;
        assert!(Arc::ptr_eq(&r_ulllong, &expected_ulllong));
    }

    #[test]
    fn rtype_compare_template_uses_union_repr_as_int_for_bool_args() {
        // rint.py:611 — `rtyper.getrepr(unionof(s_int1, s_int2)).as_int`.
        // For BoolRepr, `as_int` is `signed_repr`, so bool comparisons
        // cast both operands to Signed and emit `int_eq`.
        use crate::annotator::model::SomeBool;
        use crate::translator::rtyper::rbool::bool_repr;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let v_left = Variable::new();
        v_left.set_concretetype(Some(LowLevelType::Bool));
        let v_right = Variable::new();
        v_right.set_concretetype(Some(LowLevelType::Bool));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Bool));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "eq".to_string(),
                vec![Hlvalue::Variable(v_left), Hlvalue::Variable(v_right)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        let r_bool = bool_repr() as Arc<dyn Repr>;
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Bool(SomeBool::new()),
            SomeValue::Bool(SomeBool::new()),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_bool.clone()), Some(r_bool)]);
        *hop.s_result.borrow_mut() = Some(SomeValue::Bool(SomeBool::new()));
        *hop.r_result.borrow_mut() = Some(bool_repr() as Arc<dyn Repr>);

        let result = rtype_compare_template(&hop, "eq").expect("bool comparison should rtype");
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 3);
        assert_eq!(ops.ops[0].opname, "cast_bool_to_int");
        assert_eq!(ops.ops[1].opname, "cast_bool_to_int");
        assert_eq!(ops.ops[2].opname, "int_eq");
    }

    #[test]
    fn rtype_abs_ovf_signed_uses_ll_int_abs_ovf_direct_call() {
        // rint.py rtype_abs_ovf + rint.py _rtype_call_helper.
        use crate::translator::rtyper::lltypesystem::lltype::_ptr_obj;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = unary_integer_hop(
            "abs_ovf",
            &rtyper,
            llops.clone(),
            LowLevelType::Signed,
            LowLevelType::Signed,
            SomeValue::Integer(SomeInteger::new(false, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        );

        let result = signed_repr()
            .rtype_abs_ovf(&hop)
            .expect("abs_ovf should rtype")
            .expect("abs_ovf should return a value");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert!(ops._called_exception_is_here_or_cannot_occur);
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c_func) = &ops.ops[0].args[0] else {
            panic!("direct_call first arg must be function constant");
        };
        let ConstValue::LLPtr(ptr) = &c_func.value else {
            panic!("direct_call first arg must be an ll pointer");
        };
        let _ptr_obj::Func(func) = ptr._obj().expect("function pointer must be concrete") else {
            panic!("direct_call first arg must point to a function");
        };
        assert_eq!(func._name, "ll_int_abs_ovf");
    }

    #[test]
    fn rtype_neg_ovf_unsigned_delegates_to_unsigned_neg_without_exception() {
        // rint.py rtype_neg_ovf — unsigned neg_ovf is supported and becomes
        // the same `0 - x` lowering as rtype_neg.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = unary_integer_hop(
            "neg_ovf",
            &rtyper,
            llops.clone(),
            LowLevelType::Unsigned,
            LowLevelType::Unsigned,
            SomeValue::Integer(SomeInteger::new(false, true)),
            SomeValue::Integer(SomeInteger::new(false, true)),
        );

        let result = unsigned_repr()
            .rtype_neg_ovf(&hop)
            .expect("unsigned neg_ovf should rtype")
            .expect("unsigned neg_ovf should return a value");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert!(ops._called_exception_is_here_or_cannot_occur);
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "uint_sub");
        assert_eq!(ops.ops[0].args.len(), 2);
    }

    #[test]
    fn rtype_neg_unsignedlonglonglong_defl_zero_passes_contains_value() {
        let lltype = LowLevelType::UnsignedLongLongLong;
        let zero = IntegerRepr::defl_as_const(&lltype).expect("unsigned 128-bit _defl");
        assert_eq!(zero, ConstValue::UInt128(0));
        assert!(lltype.contains_value(&zero));
    }

    #[test]
    fn rtype_chr_emits_cast_int_to_char_like_rint() {
        // rint.py rtype_chr. With no exceptionlinks, has_implicit_exception
        // returns false and the check helper is not emitted.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = unary_integer_hop(
            "chr",
            &rtyper,
            llops.clone(),
            LowLevelType::Signed,
            LowLevelType::Char,
            SomeValue::Integer(SomeInteger::new(false, false)),
            SomeValue::Char(SomeChar::new(false)),
        );

        let result = signed_repr()
            .rtype_chr(&hop)
            .expect("chr should rtype")
            .expect("chr should return a value");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert!(ops._called_exception_is_here_or_cannot_occur);
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "cast_int_to_char");
    }

    #[test]
    fn rtype_unichr_emits_cast_int_to_unichar_like_rint() {
        // rint.py. Mirrors rtype_chr with UniChar result type.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = unary_integer_hop(
            "unichr",
            &rtyper,
            llops.clone(),
            LowLevelType::Signed,
            LowLevelType::UniChar,
            SomeValue::Integer(SomeInteger::new(false, false)),
            SomeValue::UnicodeCodePoint(SomeUnicodeCodePoint::new(false)),
        );

        let result = signed_repr()
            .rtype_unichr(&hop)
            .expect("unichr should rtype")
            .expect("unichr should return a value");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert!(ops._called_exception_is_here_or_cannot_occur);
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "cast_int_to_unichar");
    }

    /// `self.0 & other.0 == other.0` where both words are the integer a
    /// single-field newtype erased to (`tyref_transparent_inner_value_type`).
    /// A `u16` shell is `r_uint` (`uint_and` + `uint_eq`, `rint.py`
    /// `integer_opprefix_for`). A signed shell is `int_and` + `int_eq`.
    #[test]
    fn erased_integer_pos0_and_eq_annotates_bool_and_rtypes_without_getattr() {
        use crate::annotator::model::s_uint;
        use crate::flowspace::model::{
            Block, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, Link,
        };

        fn assert_shell(got: &Option<SomeValue>, shell: &SomeValue) {
            let SomeValue::Integer(expect) = shell else {
                panic!("shell is not an integer: {shell:?}");
            };
            let Some(SomeValue::Integer(got)) = got else {
                panic!("positional field was not the integer shell: {got:?}");
            };
            assert_eq!(got.unsigned, expect.unsigned);
            assert_eq!(got.nonneg, expect.nonneg);
            assert_eq!(got.base.knowntype, expect.base.knowntype);
            assert!(got.base.const_box.is_none());
        }

        fn contains_body(shell: SomeValue, and_op: &str, eq_op: &str) {
            let ann = RPythonAnnotator::new(None, None, None, false);
            let rtyper = Rc::new(RPythonTyper::new(&ann));
            rtyper
                .initialize_exceptiondata()
                .expect("initialize_exceptiondata");

            let self_v = Variable::named("self_bits");
            let other_v = Variable::named("other_bits");
            let startblock = Block::shared(vec![
                Hlvalue::Variable(self_v.clone()),
                Hlvalue::Variable(other_v.clone()),
            ]);
            let graph = FunctionGraph::new("erased_contains", startblock.clone());
            let self_0 = Variable::named("self_0");
            let other_0 = Variable::named("other_0");
            let masked = Variable::named("masked");
            let result = Variable::named("r");
            let attr = Hlvalue::Constant(Constant::new(ConstValue::byte_str("__pos_0")));
            startblock.borrow_mut().operations.extend([
                SpaceOperation::new(
                    "getattr",
                    vec![Hlvalue::Variable(self_v), attr.clone()],
                    Hlvalue::Variable(self_0.clone()),
                ),
                SpaceOperation::new(
                    "getattr",
                    vec![Hlvalue::Variable(other_v), attr],
                    Hlvalue::Variable(other_0.clone()),
                ),
                SpaceOperation::new(
                    "and_",
                    vec![
                        Hlvalue::Variable(self_0.clone()),
                        Hlvalue::Variable(other_0.clone()),
                    ],
                    Hlvalue::Variable(masked.clone()),
                ),
                SpaceOperation::new(
                    "eq",
                    vec![
                        Hlvalue::Variable(masked),
                        Hlvalue::Variable(other_0.clone()),
                    ],
                    Hlvalue::Variable(result.clone()),
                ),
            ]);
            startblock.closeblock(vec![
                Link::new(
                    vec![Hlvalue::Variable(result)],
                    Some(graph.returnblock.clone()),
                    None,
                )
                .into_ref(),
            ]);
            let graph = graph.into_ref();

            let annotated = ann
                .build_graph_types(&graph, &[Some(shell.clone()), Some(shell.clone())], true)
                .expect("erased contains must annotate");
            assert!(
                matches!(annotated, Some(SomeValue::Bool(_))),
                "self.0 & other.0 == other.0 is SomeBool, got {annotated:?}"
            );
            let s_self_0 = ann.annotation(&Hlvalue::Variable(self_0));
            let s_other_0 = ann.annotation(&Hlvalue::Variable(other_0));
            assert_shell(&s_self_0, &shell);
            assert_shell(&s_other_0, &shell);

            rtyper
                .specialize_more_blocks()
                .expect("erased contains must rtype");
            let names: Vec<String> = startblock
                .borrow()
                .operations
                .iter()
                .map(|op| op.opname.clone())
                .collect();
            assert!(
                names.iter().any(|op| op == and_op) && names.iter().any(|op| op == eq_op),
                "expected {and_op} + {eq_op}, ops={names:?}"
            );
            assert!(
                names.iter().all(|op| op != "getattr" && op != "getfield"),
                "positional field must not survive as a field read, ops={names:?}"
            );
            assert!(
                names.iter().any(|op| op == "same_as"),
                "rtyper renames the field read with same_as, ops={names:?}"
            );
        }

        // u16 / unsigned newtype. `ValueType::Unsigned` is `s_uint`.
        contains_body(s_uint(), "uint_and", "uint_eq");
        // Signed integer shell. `integer_opprefix_for(Signed)` is `int_`.
        contains_body(
            SomeValue::Integer(SomeInteger::default()),
            "int_and",
            "int_eq",
        );

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let flags = Variable::named("flags");
        let startblock = Block::shared(vec![Hlvalue::Variable(flags.clone())]);
        let graph = FunctionGraph::new("erased_pos0", startblock.clone());
        let word = Variable::named("word");
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getattr",
            vec![
                Hlvalue::Variable(flags),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("__pos_0"))),
            ],
            Hlvalue::Variable(word.clone()),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(word)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let graph = graph.into_ref();
        let mut constant = SomeInteger::new(false, true);
        constant.base.const_box = Some(Constant::new(ConstValue::UInt(4)));
        let annotated = ann
            .build_graph_types(&graph, &[Some(SomeValue::Integer(constant))], true)
            .expect("constant positional field must annotate");
        match annotated {
            Some(SomeValue::Integer(i)) => {
                assert!(i.unsigned && i.nonneg);
                assert_eq!(i.base.knowntype, KnownType::Ruint);
                assert!(matches!(
                    i.base.const_box.as_ref().map(|c| &c.value),
                    Some(ConstValue::UInt(4))
                ));
            }
            other => panic!("constant __pos_0 must stay the unsigned integer, got {other:?}"),
        }
        rtyper
            .specialize_more_blocks()
            .expect("constant positional field must rtype");
        let names: Vec<String> = startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert!(
            names.iter().all(|op| op != "getattr") && names.iter().any(|op| op == "same_as"),
            "constant field read lowers to same_as, ops={names:?}"
        );
    }

    /// `SomeBool` walks Integer before Object. The identity is the integer
    /// shell only; a bool has no `__pos_0`.
    #[test]
    fn bool_pos0_getattr_is_not_erased_integer_identity() {
        use crate::annotator::model::SomeBool;
        use crate::flowspace::model::{
            Block, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, Link,
        };

        let ann = RPythonAnnotator::new(None, None, None, false);
        let bit = Variable::named("bit");
        let startblock = Block::shared(vec![Hlvalue::Variable(bit.clone())]);
        let graph = FunctionGraph::new("bool_pos0", startblock.clone());
        let out = Variable::named("out");
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getattr",
            vec![
                Hlvalue::Variable(bit),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("__pos_0"))),
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
        let graph = graph.into_ref();
        let err = ann
            .build_graph_types(&graph, &[Some(SomeValue::Bool(SomeBool::new()))], true)
            .expect_err("bool __pos_0 must not annotate");
        let text = err.to_string();
        assert!(
            text.contains("__pos_0") && text.contains("Bool"),
            "bool field read must stay missing, got {text}"
        );
    }
}
