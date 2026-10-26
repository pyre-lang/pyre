//! The type ids `gc-root-reachability` widens its scan with are read off the
//! artefact's own spelling of `[T]` and of a hash-consed type.
//!
//! `pyre-object` holds `&[PyObjectRef]` locals (`pin_roots` takes one), so
//! an empty answer here means the slice or the `{"Value": [id, body]}`
//! spelling went unread, not that the crate has none.

use majit_charon_reader::Llbc;
use majit_translate::memory::gctransform::liveness;

const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/llbc/pyre-object.ullbc"
);

#[test]
fn a_gc_pointer_slice_is_found_in_the_object_crate() {
    if !std::path::Path::new(OBJECT_LLBC).is_file() {
        eprintln!("skipping: {OBJECT_LLBC} is missing; run `python3 scripts/extract-llbc.py`");
        return;
    }
    let llbc = Llbc::load(OBJECT_LLBC).expect("load llbc");
    let gc_tys = liveness::gc_ptr_type_ids(&llbc);
    assert!(!gc_tys.is_empty(), "no PyObjectRef type id");
    assert!(
        !liveness::gc_slice_type_ids(&llbc, &gc_tys).is_empty(),
        "no &[PyObjectRef] type id"
    );
}
