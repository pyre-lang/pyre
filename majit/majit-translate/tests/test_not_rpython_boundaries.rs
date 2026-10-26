//! Host-only object-space setup must stay outside the translated graph set.
//!
//! PyPy marks `ObjSpace.make_builtins`, `_codecs.Module.__init__`, and
//! `register_builtin_error_handlers` with `@not_rpython`.  In particular, the
//! error-handler function table is a translation-time PBC registry, not a
//! runtime tuple whose Rust function pointers need an `InstanceRepr`.

use majit_charon_reader::Llbc;
use majit_translate::front::llbc_hints::harvest_hints_from_llbcs;

const INTERPRETER_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/llbc/pyre-interpreter.ullbc"
);

#[test]
#[ignore = "loads the full interpreter LLBC artifact"]
fn codec_handlers_are_registered_at_the_not_rpython_startup_boundary() {
    let llbc = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter.ullbc");
    let hints = harvest_hints_from_llbcs(std::slice::from_ref(&llbc));

    for path in [
        "module::_codecs::register_builtin_error_handlers",
        "importing::install_builtin_modules",
        "importing::init_sys_path",
    ] {
        assert!(
            hints
                .get(path)
                .is_some_and(|values| values.iter().any(|hint| hint == "not_rpython")),
            "{path} must retain PyPy's host-only startup boundary: {hints:?}"
        );
    }
}
