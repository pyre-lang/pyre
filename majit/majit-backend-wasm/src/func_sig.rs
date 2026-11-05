//! Packed encoding of a residual callee's wasm function type.
//!
//! Shared by the host `env.jit_func_sig` import and the guest compiler.
//!
//! Encoding:
//! - `0` = unknown (empty slot, slot 0, non-func ref, more than
//!   [`crate::codegen::MAX_CALL_ARGS`] params, more than one result, or a
//!   param/result type other than i32/i64/f32/f64).
//! - otherwise bit 63 = 1 (known), bits 0..=4 = nparams, bits 5..=7 = result
//!   kind (0 none, 1 i32, 2 i64, 3 f32, 4 f64), bits 8+2i = param i kind
//!   (0 i32, 1 i64, 2 f32, 3 f64).

use crate::codegen::MAX_CALL_ARGS;

const KNOWN_BIT: i64 = 1_i64 << 63;
const NPARAMS_MASK: i64 = 0x1f;
const RESULT_SHIFT: u32 = 5;
const RESULT_MASK: i64 = 0x7;
const PARAM_SHIFT: u32 = 8;
const PARAM_BITS: u32 = 2;
const PARAM_MASK: i64 = 0x3;

/// One wasm value type the residual-call oracle can name.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum FuncSigVal {
    I32 = 0,
    I64 = 1,
    F32 = 2,
    F64 = 3,
}

impl FuncSigVal {
    fn from_param_bits(bits: u8) -> Option<Self> {
        match bits {
            0 => Some(Self::I32),
            1 => Some(Self::I64),
            2 => Some(Self::F32),
            3 => Some(Self::F64),
            _ => None,
        }
    }

    fn from_result_kind(kind: u8) -> Option<Option<Self>> {
        match kind {
            0 => Some(None),
            1 => Some(Some(Self::I32)),
            2 => Some(Some(Self::I64)),
            3 => Some(Some(Self::F32)),
            4 => Some(Some(Self::F64)),
            _ => None,
        }
    }

    fn result_kind(result: Option<Self>) -> u8 {
        match result {
            None => 0,
            Some(Self::I32) => 1,
            Some(Self::I64) => 2,
            Some(Self::F32) => 3,
            Some(Self::F64) => 4,
        }
    }
}

/// A residual callee's wasm signature, or the absence of one.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WasmSig {
    pub params: Vec<FuncSigVal>,
    pub result: Option<FuncSigVal>,
}

impl WasmSig {
    pub fn has_f32(&self) -> bool {
        self.result == Some(FuncSigVal::F32) || self.params.contains(&FuncSigVal::F32)
    }
}

/// Pack `params`/`result` into the `jit_func_sig` i64, or `0` if unencodable.
pub fn encode_func_sig(params: &[FuncSigVal], result: Option<FuncSigVal>) -> i64 {
    if params.len() > MAX_CALL_ARGS {
        return 0;
    }
    let nparams = params.len() as i64;
    let mut bits =
        KNOWN_BIT | nparams | (i64::from(FuncSigVal::result_kind(result)) << RESULT_SHIFT);
    for (i, param) in params.iter().enumerate() {
        bits |= i64::from(*param as u8) << (PARAM_SHIFT + PARAM_BITS * i as u32);
    }
    bits
}

/// Unpack a `jit_func_sig` i64. `0` and any ill-formed word are unknown.
pub fn decode_func_sig(bits: i64) -> Option<WasmSig> {
    if bits == 0 || bits & KNOWN_BIT == 0 {
        return None;
    }
    let nparams = (bits & NPARAMS_MASK) as usize;
    if nparams > MAX_CALL_ARGS {
        return None;
    }
    let result_kind = ((bits >> RESULT_SHIFT) & RESULT_MASK) as u8;
    let result = FuncSigVal::from_result_kind(result_kind)?;
    let mut params = Vec::with_capacity(nparams);
    for i in 0..nparams {
        let raw = ((bits >> (PARAM_SHIFT + PARAM_BITS * i as u32)) & PARAM_MASK) as u8;
        params.push(FuncSigVal::from_param_bits(raw)?);
    }
    Some(WasmSig { params, result })
}

#[cfg(test)]
mod tests {
    use super::*;

    const KNOWN: i64 = 1_i64 << 63;

    #[test]
    fn codec_golden_values() {
        assert_eq!(encode_func_sig(&[], None), KNOWN);
        assert_eq!(
            encode_func_sig(&[FuncSigVal::I64, FuncSigVal::I64], Some(FuncSigVal::I64)),
            KNOWN | 2 | (2 << 5) | (1 << 8) | (1 << 10)
        );
        assert_eq!(
            encode_func_sig(&[FuncSigVal::I32, FuncSigVal::I32], Some(FuncSigVal::I32)),
            KNOWN | 2 | (1 << 5)
        );
        assert_eq!(
            encode_func_sig(&[FuncSigVal::I64], None),
            KNOWN | 1 | (1 << 8)
        );
        assert_eq!(
            encode_func_sig(&[FuncSigVal::I64], Some(FuncSigVal::I64)),
            KNOWN | 1 | (2 << 5) | (1 << 8)
        );
        assert_eq!(
            encode_func_sig(&[FuncSigVal::F32], Some(FuncSigVal::I64)),
            KNOWN | 1 | (2 << 5) | (2 << 8)
        );
        assert_eq!(
            encode_func_sig(&[FuncSigVal::I64; 17], Some(FuncSigVal::I64)),
            0
        );
        assert_eq!(decode_func_sig(0), None);
        assert_eq!(
            decode_func_sig(KNOWN | 2 | (2 << 5) | (1 << 8) | (1 << 10)),
            Some(WasmSig {
                params: vec![FuncSigVal::I64, FuncSigVal::I64],
                result: Some(FuncSigVal::I64),
            })
        );
    }

    #[test]
    fn codec_roundtrip() {
        let cases: &[(&[FuncSigVal], Option<FuncSigVal>)] = &[
            (&[], None),
            (&[FuncSigVal::I32], Some(FuncSigVal::I32)),
            (&[FuncSigVal::I64, FuncSigVal::F64], Some(FuncSigVal::F64)),
            (&[FuncSigVal::F32, FuncSigVal::F32], None),
        ];
        for &(params, result) in cases {
            let bits = encode_func_sig(params, result);
            assert_ne!(bits, 0);
            let decoded = decode_func_sig(bits).expect("roundtrip");
            assert_eq!(decoded.params, params);
            assert_eq!(decoded.result, result);
        }
    }
}
