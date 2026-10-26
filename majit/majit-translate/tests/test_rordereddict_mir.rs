//! Native-source JIT hints for rpython/rtyper/lltypesystem/rordereddict.py.

use majit_charon_reader::Llbc;
use majit_translate::front::llbc_hints::harvest_hints_from_llbcs;

#[test]
#[ignore = "requires pre-extracted pyre-object.ullbc"]
fn dict_compaction_keeps_upstream_opaque_boundary() {
    let llbc = Llbc::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    ))
    .expect("extract pyre-object before testing its JIT hints");
    let functions: Vec<_> = llbc
        .iter_local_fns()
        .filter(|function| {
            let path = function.item_meta.name_path();
            path.contains("::rordereddict::") && path.ends_with("::remove_deleted_items")
        })
        .collect();
    assert_eq!(
        functions.len(),
        1,
        "test must reach the native compaction body"
    );
    let hints = harvest_hints_from_llbcs(std::slice::from_ref(&llbc));
    let compaction: Vec<_> = hints
        .iter()
        .filter(|(path, _)| {
            path.starts_with("rordereddict::") && path.ends_with("::remove_deleted_items")
        })
        .collect();
    assert_eq!(
        compaction.len(),
        1,
        "ll_dict_remove_deleted_items is @jit.dont_look_inside upstream"
    );
    assert!(
        compaction[0]
            .1
            .iter()
            .any(|hint| hint == "dont_look_inside")
    );
    assert!(
        !compaction[0].1.iter().any(|hint| hint == "not_rpython"),
        "opaque to the JIT is not excluded from translation"
    );
}
