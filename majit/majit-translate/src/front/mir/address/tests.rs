/// Widening integer storage keeps every address bit (`rbuiltin.py` `gen_cast`).
#[test]
fn int_scalar_width_covers_word_accepts_widening_integer() {
    assert!(super::int_scalar_width_covers_word("U64", 8));
    assert!(super::int_scalar_width_covers_word("I64", 8));
    assert!(super::int_scalar_width_covers_word("U128", 8));
    assert!(!super::int_scalar_width_covers_word("U32", 8));
    assert!(super::int_scalar_width_covers_word("U64", 4));
    assert!(super::int_scalar_width_covers_word("U32", 4));
    assert!(!super::int_scalar_width_covers_word("U16", 4));
}
