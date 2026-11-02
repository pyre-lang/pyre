//! RPython `rpython/rtyper/rbuilder.py` parity module.
//!
//! Upstream splits string-builder rtyping into this abstract method
//! surface and a concrete lltypesystem implementation in
//! `rpython/rtyper/lltypesystem/rbuilder.py`. The lltypesystem half is
//! still pending in pyre, so this module records the exact abstract
//! `AbstractStringBuilderRepr` method names first. That keeps callers
//! and future ports using the upstream names instead of inventing local
//! aliases.
//!
//! Deferred port: upstream `AbstractStringBuilderRepr` carries method
//! *bodies* (`rtyper_new`, `rtype_method_append`, `rtype_method_build`,
//! `rtype_bool`, `convert_const`, …) that `hop.gendirectcall` into
//! `self.ll_new` / `self.ll_append` / `self.ll_build` / `self.ll_bool`.
//! Those low-level helpers are the explicitly-deferred stubs in
//! `lltypesystem/rbuilder.rs` (`builder_runtime_deferred`), so the method
//! bodies cannot be ported faithfully yet. Converge by landing the
//! `lltypesystem/rbuilder.rs` runtime helpers, then porting the
//! `rtype_method_*` bodies onto this repr.

use crate::flowspace::model::{ConstValue, Constant};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::rmodel::RTypeResult;
use crate::translator::rtyper::rtyper::HighLevelOp;

/// RPython `rpython.rlib.rstring.INIT_SIZE`.
///
/// `AbstractStringBuilderRepr.rtyper_new` uses this default when the
/// high-level constructor receives no explicit initial size.
pub const INIT_SIZE: i64 = 100;

/// RPython `class AbstractStringBuilderRepr(Repr)`.
///
/// The concrete low-level fields (`ll_new`, `ll_append`, `ll_build`,
/// etc.) are supplied by `lltypesystem/rbuilder.py` upstream. Pyre
/// exposes this marker before that concrete repr lands so module and
/// class names already line up with RPython.
#[derive(Debug, Default)]
pub struct AbstractStringBuilderRepr;

/// Upstream method names on `AbstractStringBuilderRepr`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(test)]
enum StringBuilderMethod {
    /// `rtyper_new`
    RtyperNew,
    /// `rtype_method_append`
    Append,
    /// `rtype_method_append_slice`
    AppendSlice,
    /// `rtype_method_append_multiple_char`
    AppendMultipleChar,
    /// `rtype_method_append_charpsize`
    AppendCharpsize,
    /// `rtype_method_getlength`
    Getlength,
    /// `rtype_method_build`
    Build,
    /// `rtype_bool`
    RtypeBool,
    /// `convert_const`
    ConvertConst,
}

#[cfg(test)]
impl StringBuilderMethod {
    /// RPython method suffix used by `BuiltinMethodRepr`.
    #[cfg(test)]
    pub const fn as_method_name(self) -> Option<&'static str> {
        match self {
            StringBuilderMethod::RtyperNew
            | StringBuilderMethod::RtypeBool
            | StringBuilderMethod::ConvertConst => None,
            StringBuilderMethod::Append => Some("append"),
            StringBuilderMethod::AppendSlice => Some("append_slice"),
            StringBuilderMethod::AppendMultipleChar => Some("append_multiple_char"),
            StringBuilderMethod::AppendCharpsize => Some("append_charpsize"),
            StringBuilderMethod::Getlength => Some("getlength"),
            StringBuilderMethod::Build => Some("build"),
        }
    }

    /// RPython method name on `AbstractStringBuilderRepr`.
    #[cfg(test)]
    pub const fn as_upstream_name(self) -> &'static str {
        match self {
            StringBuilderMethod::RtyperNew => "rtyper_new",
            StringBuilderMethod::Append => "rtype_method_append",
            StringBuilderMethod::AppendSlice => "rtype_method_append_slice",
            StringBuilderMethod::AppendMultipleChar => "rtype_method_append_multiple_char",
            StringBuilderMethod::AppendCharpsize => "rtype_method_append_charpsize",
            StringBuilderMethod::Getlength => "rtype_method_getlength",
            StringBuilderMethod::Build => "rtype_method_build",
            StringBuilderMethod::RtypeBool => "rtype_bool",
            StringBuilderMethod::ConvertConst => "convert_const",
        }
    }

    /// Backward-compatible name for the upstream method surface.
    #[cfg(test)]
    pub const fn as_rtype_method_name(self) -> &'static str {
        self.as_upstream_name()
    }
}

/// Upstream method table from `rbuilder.py:7-58`.
#[cfg(test)]
const STRING_BUILDER_METHODS: [StringBuilderMethod; 9] = [
    StringBuilderMethod::RtyperNew,
    StringBuilderMethod::Append,
    StringBuilderMethod::AppendSlice,
    StringBuilderMethod::AppendMultipleChar,
    StringBuilderMethod::AppendCharpsize,
    StringBuilderMethod::Getlength,
    StringBuilderMethod::Build,
    StringBuilderMethod::RtypeBool,
    StringBuilderMethod::ConvertConst,
];

impl AbstractStringBuilderRepr {
    fn deferred_method(name: &str) -> TyperError {
        TyperError::missing_rtype_operation(format!(
            "rbuilder.AbstractStringBuilderRepr.{name} deferred until \
             lltypesystem/rbuilder runtime helpers land"
        ))
    }

    /// RPython `AbstractStringBuilderRepr.rtyper_new(self, hop)`.
    pub fn rtyper_new(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(Self::deferred_method("rtyper_new"))
    }

    /// RPython `AbstractStringBuilderRepr.rtype_method_append(self, hop)`.
    pub fn rtype_method_append(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(Self::deferred_method("rtype_method_append"))
    }

    /// RPython `AbstractStringBuilderRepr.rtype_method_append_slice(self, hop)`.
    pub fn rtype_method_append_slice(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(Self::deferred_method("rtype_method_append_slice"))
    }

    /// RPython
    /// `AbstractStringBuilderRepr.rtype_method_append_multiple_char(self, hop)`.
    pub fn rtype_method_append_multiple_char(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(Self::deferred_method("rtype_method_append_multiple_char"))
    }

    /// RPython
    /// `AbstractStringBuilderRepr.rtype_method_append_charpsize(self, hop)`.
    pub fn rtype_method_append_charpsize(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(Self::deferred_method("rtype_method_append_charpsize"))
    }

    /// RPython `AbstractStringBuilderRepr.rtype_method_getlength(self, hop)`.
    pub fn rtype_method_getlength(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(Self::deferred_method("rtype_method_getlength"))
    }

    /// RPython `AbstractStringBuilderRepr.rtype_method_build(self, hop)`.
    pub fn rtype_method_build(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(Self::deferred_method("rtype_method_build"))
    }

    /// RPython `AbstractStringBuilderRepr.rtype_bool(self, hop)`.
    pub fn rtype_bool(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(Self::deferred_method("rtype_bool"))
    }

    /// RPython `AbstractStringBuilderRepr.convert_const(self, value)`.
    pub fn convert_const(&self, _value: &ConstValue) -> Result<Constant, TyperError> {
        Err(Self::deferred_method("convert_const"))
    }

    /// Resolve either an exact upstream method name or a
    /// `BuiltinMethodRepr.methodname` suffix to the upstream method arm.
    #[cfg(test)]
    fn method_from_name(method_name: &str) -> Result<StringBuilderMethod, TyperError> {
        STRING_BUILDER_METHODS
            .iter()
            .copied()
            .find(|method| {
                method.as_upstream_name() == method_name
                    || method.as_method_name() == Some(method_name)
            })
            .ok_or_else(|| {
                TyperError::message(format!(
                    "missing AbstractStringBuilderRepr method {method_name}"
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AbstractStringBuilderRepr, INIT_SIZE, STRING_BUILDER_METHODS, StringBuilderMethod,
    };

    #[test]
    fn init_size_matches_rlib_rstring_default() {
        assert_eq!(INIT_SIZE, 100);
    }

    #[test]
    fn abstract_string_builder_method_names_match_rpython_surface() {
        let names: Vec<_> = STRING_BUILDER_METHODS
            .iter()
            .map(|method| method.as_rtype_method_name())
            .collect();
        assert_eq!(
            names,
            vec![
                "rtyper_new",
                "rtype_method_append",
                "rtype_method_append_slice",
                "rtype_method_append_multiple_char",
                "rtype_method_append_charpsize",
                "rtype_method_getlength",
                "rtype_method_build",
                "rtype_bool",
                "convert_const",
            ]
        );
    }

    #[test]
    fn method_from_name_resolves_builtin_method_suffixes() {
        assert_eq!(
            AbstractStringBuilderRepr::method_from_name("append").unwrap(),
            StringBuilderMethod::Append
        );
        assert_eq!(
            AbstractStringBuilderRepr::method_from_name("append_charpsize").unwrap(),
            StringBuilderMethod::AppendCharpsize
        );
        assert!(AbstractStringBuilderRepr::method_from_name("extend").is_err());
    }

    #[test]
    fn method_from_name_resolves_exact_upstream_names() {
        assert_eq!(
            AbstractStringBuilderRepr::method_from_name("rtyper_new").unwrap(),
            StringBuilderMethod::RtyperNew
        );
        assert_eq!(
            AbstractStringBuilderRepr::method_from_name("rtype_bool").unwrap(),
            StringBuilderMethod::RtypeBool
        );
        assert_eq!(
            AbstractStringBuilderRepr::method_from_name("convert_const").unwrap(),
            StringBuilderMethod::ConvertConst
        );
    }
}
