//! Crate-level test helpers for building green keys from their greens.

/// Typed greenkey whose single Int green is `raw`.
pub(crate) fn gk(raw: i64) -> majit_ir::GreenKey {
    majit_ir::GreenKey::new(vec![raw])
}

/// Two-Int key beginning with `first` whose `get_uhash` is `target`.
///
/// Used when the number is a cell key `mint_cell_key` produced, or a
/// one-Int key's hash that a later typed key must mint beside:
/// `warmstate::tests` minted-key cases, `pyjitpl` retrace squatters, and
/// `jitdriver::tests` `resolve_cell_key_reads_the_entry_driver`.
pub(crate) fn gk_with_uhash(target: u64, first: i64) -> majit_ir::GreenKey {
    use majit_ir::{GREEN_UHASH_MULT, GREEN_UHASH_SEED, GreenType, green_uhash_step};
    let mut inverse: u64 = 1;
    for _ in 0..6 {
        inverse = inverse.wrapping_mul(2u64.wrapping_sub(GREEN_UHASH_MULT.wrapping_mul(inverse)));
    }
    debug_assert_eq!(GREEN_UHASH_MULT.wrapping_mul(inverse), 1);
    let prefix = green_uhash_step(GREEN_UHASH_SEED, GreenType::Int, first);
    let key = majit_ir::GreenKey::new(vec![first, (prefix ^ target.wrapping_mul(inverse)) as i64]);
    debug_assert_eq!(key.get_uhash(), target);
    key
}
