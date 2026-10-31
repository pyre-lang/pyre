//! `_immutable_fields_` must survive the emit-site layout table.
//!
//! The declaration reaches the proc-macro path as a string on the struct, and
//! the layout table is the only channel between that string and the field
//! descrs the optimizer reads.  It carried offsets, widths and pointer-ness and
//! dropped immutability on the floor.
//!
//! Registration order is the interesting part.  A struct's layout accumulates
//! across every emit site that touches it, and not all of them name a declaring
//! struct — a `new_struct` for a transient allocation names none.  So these
//! cover a declaration arriving first, arriving last, and never arriving.

use majit_metainterp::JitCodeBuilder;

/// Distinct per test: the layout registry is per-builder, but the conflict
/// registry these registrations also feed is process-global.
const TID_RANKS: u64 = 0x494D_4D55_5401;
const TID_UNDECLARED: u64 = 0x494D_4D55_5402;
const TID_LATE: u64 = 0x494D_4D55_5403;
const TID_NEVER: u64 = 0x494D_4D55_5404;

const NODE: &[(usize, bool, &str, usize, bool, bool)] = &[
    (0, false, "kind", 1, false, false),
    (2, false, "marked", 1, false, false),
    (8, true, "left", 8, false, false),
    (16, false, "version", 8, true, false),
];

fn field<'a>(
    builder: &'a JitCodeBuilder,
    type_id: u64,
    name: &str,
) -> &'a majit_jitcode::jitcode::BhFieldSpec {
    builder
        .struct_size_spec(type_id)
        .unwrap_or_else(|| panic!("no layout registered for {type_id:#x}"))
        .all_fielddescrs
        .iter()
        .find(|f| f.name == name)
        .unwrap_or_else(|| panic!("{name} missing from the registered layout"))
}

#[test]
fn declared_ranks_land_on_the_field_specs() {
    let mut builder = JitCodeBuilder::new();
    builder.register_struct_layout(24, TID_RANKS, false, false, NODE, "kind,left,version?");

    assert!(field(&builder, TID_RANKS, "kind").is_immutable);
    assert!(field(&builder, TID_RANKS, "left").is_immutable);
}

#[test]
fn a_quasi_immutable_entry_is_recorded_as_quasi() {
    // `rclass.py _parse_field_list`'s `?` suffix. Declared (any rank) ⇒
    // immutable, `descr.py get_field_descr`; the `?` also stamps
    // `is_quasi_immutable`. `jtransform.py rewrite_op_getfield` still emits
    // `record_quasiimmut_field` before the pure read.
    let mut builder = JitCodeBuilder::new();
    builder.register_struct_layout(24, TID_RANKS + 16, false, false, NODE, "version?");

    let version = field(&builder, TID_RANKS + 16, "version");
    assert!(version.is_quasi_immutable);
    assert!(
        version.is_immutable,
        "a declared `?` rank stamps is_immutable (`descr.py get_field_descr`)",
    );
}

#[test]
fn an_undeclared_field_stays_mutable() {
    let mut builder = JitCodeBuilder::new();
    builder.register_struct_layout(24, TID_UNDECLARED, false, false, NODE, "kind,left");

    let marked = field(&builder, TID_UNDECLARED, "marked");
    assert!(
        !marked.is_immutable,
        "the mark is the mutable state; folding its read is wrong code",
    );
    assert!(!marked.is_quasi_immutable);
}

#[test]
fn a_declaration_arriving_after_an_undeclared_site_still_applies() {
    // The registration order that would otherwise lose it: a site with no
    // declaration registers the whole layout first, so every offset is already
    // known and the second registration has nothing new to append.
    let mut builder = JitCodeBuilder::new();
    builder.register_struct_layout(24, TID_LATE, false, false, NODE, "");
    assert!(!field(&builder, TID_LATE, "left").is_immutable);

    builder.register_struct_layout(24, TID_LATE, false, false, NODE, "kind,left");
    assert!(
        field(&builder, TID_LATE, "left").is_immutable,
        "emit sites lower in an arbitrary order; the declaration must not \
         depend on arriving first",
    );
    assert!(field(&builder, TID_LATE, "kind").is_immutable);
    assert!(!field(&builder, TID_LATE, "marked").is_immutable);
}

#[test]
fn a_later_site_without_a_declaration_does_not_clear_one() {
    // The mirror image, and the reason the merge only ever raises: a site that
    // names no declaration is silent about the struct, not a claim that its
    // fields are mutable.
    let mut builder = JitCodeBuilder::new();
    builder.register_struct_layout(24, TID_NEVER, false, false, NODE, "kind,left");
    builder.register_struct_layout(24, TID_NEVER, false, false, NODE, "");

    assert!(field(&builder, TID_NEVER, "left").is_immutable);
    assert!(field(&builder, TID_NEVER, "kind").is_immutable);
}

#[test]
fn a_struct_that_declares_nothing_has_no_immutable_field() {
    let mut builder = JitCodeBuilder::new();
    builder.register_struct_layout(24, TID_NEVER + 16, false, false, NODE, "");

    for (_, _, name, _, _, _) in NODE {
        assert!(
            !field(&builder, TID_NEVER + 16, name).is_immutable,
            "{name} was declared immutable by nothing",
        );
    }
}

/// The declaration is inert until the *minted descr* carries it.
///
/// `BhFieldSpec` is a way station.  What the optimizer reads is the
/// `CanonicalBhDescr::Field` the emit site mints, and that site resolved its
/// width, flag and signedness off the registered layout while writing
/// immutability as a literal `false` — so the whole chain could be correct and
/// still fold nothing.
mod minted_descr {
    use super::*;
    use majit_metainterp::jitcode::CanonicalBhDescr;

    const TID_MINT: u64 = 0x494D_4D55_5410;

    fn immutability_of(field_name: &str, declaration: &str) -> (bool, bool) {
        let mut builder = JitCodeBuilder::new();
        builder.ensure_i_regs(2);
        builder.ensure_r_regs(2);
        builder.register_struct_layout(24, TID_MINT, false, false, NODE, declaration);
        // The access is what mints the descr; registering the layout alone
        // does not.
        builder.getfield_gc_i(0, 1, 2, TID_MINT, field_name);
        let jitcode = builder.finish();
        jitcode
            .exec
            .descrs
            .iter()
            .find_map(|entry| match entry.as_bh_descr() {
                Some(CanonicalBhDescr::Field {
                    name,
                    is_immutable,
                    is_quasi_immutable,
                    ..
                }) if name == field_name => Some((*is_immutable, *is_quasi_immutable)),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no Field descr minted for {field_name}"))
    }

    #[test]
    fn a_declared_field_mints_an_immutable_descr() {
        assert_eq!(
            immutability_of("marked", "marked"),
            (true, false),
            "the declaration must reach the descr the emit site mints, not just \
             the layout spec it is resolved against",
        );
    }

    #[test]
    fn an_undeclared_field_mints_a_mutable_descr() {
        assert_eq!(immutability_of("marked", "kind,left"), (false, false));
    }

    #[test]
    fn a_quasi_declared_field_mints_a_quasi_descr() {
        // Declared (any rank) ⇒ immutable, `descr.py get_field_descr`; the `?`
        // also stamps `is_quasi_immutable`.
        assert_eq!(immutability_of("marked", "marked?"), (true, true));
    }
}
