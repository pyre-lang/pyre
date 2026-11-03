//! The immutable-field ranks of `rpython/rtyper/rclass.py`.

use serde::{Deserialize, Serialize};

/// RPython `rpython/rtyper/rclass.py` — `IR_IMMUTABLE` / `IR_IMMUTABLE_ARRAY`
/// / `IR_QUASIIMMUTABLE` / `IR_QUASIIMMUTABLE_ARRAY`.  Parsed from
/// `_immutable_fields_` string literals (`rclass.py _parse_field_list`):
///
/// * `"field"`       → `Immutable`
/// * `"field?"`      → `QuasiImmutable`
/// * `"field[*]"`    → `ImmutableArray`
/// * `"field?[*]"`   → `QuasiImmutableArray`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ImmutableRank {
    #[default]
    Immutable,
    QuasiImmutable,
    ImmutableArray,
    QuasiImmutableArray,
}

impl ImmutableRank {
    /// Parse a single RPython-style `_immutable_fields_` entry.  Returns the
    /// bare field name and its rank.  Suffix precedence matches
    /// `rclass.py:649-661`: `?[*]` → `[*]` → `?` → plain.
    pub fn parse(entry: &str) -> (String, Self) {
        if let Some(stripped) = entry.strip_suffix("?[*]") {
            (stripped.to_string(), Self::QuasiImmutableArray)
        } else if let Some(stripped) = entry.strip_suffix("[*]") {
            (stripped.to_string(), Self::ImmutableArray)
        } else if let Some(stripped) = entry.strip_suffix('?') {
            (stripped.to_string(), Self::QuasiImmutable)
        } else {
            (entry.to_string(), Self::Immutable)
        }
    }

    /// RPython `ImmutableRanking.pure` flag — `rclass.py`.  True for
    /// `IR_IMMUTABLE` / `IR_IMMUTABLE_ARRAY`; false for the quasi variants
    /// (they pin via guard, not via pure flag).
    pub fn is_immutable(self) -> bool {
        matches!(self, Self::Immutable | Self::ImmutableArray)
    }

    /// True for `IR_QUASIIMMUTABLE` / `IR_QUASIIMMUTABLE_ARRAY` —
    /// `jtransform.py Transformer.rewrite_op_getfield immut in (IR_QUASIIMMUTABLE, IR_QUASIIMMUTABLE_ARRAY)`.
    pub fn is_quasi_immutable(self) -> bool {
        matches!(self, Self::QuasiImmutable | Self::QuasiImmutableArray)
    }

    /// True for `IR_IMMUTABLE_ARRAY` / `IR_QUASIIMMUTABLE_ARRAY` —
    /// `rclass.py rank in (IR_QUASIIMMUTABLE_ARRAY, IR_IMMUTABLE_ARRAY)`.
    pub fn is_array(self) -> bool {
        matches!(self, Self::ImmutableArray | Self::QuasiImmutableArray)
    }

    /// Suffix `rclass.py _parse_field_list` peels off an `_immutable_fields_`
    /// entry: `""` / `"?"` / `"[*]"` / `"?[*]"`.
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Immutable => "",
            Self::QuasiImmutable => "?",
            Self::ImmutableArray => "[*]",
            Self::QuasiImmutableArray => "?[*]",
        }
    }
}
