//! `Option<*mut T>` from an integer, driven through the codewriter.
//!
//! A raw pointer has no niche (`Option<&T>` / `Option<NonNull<T>>` do), so
//! both `FUNC.RESULT` (`call.py` `get_jitcode_calldescr`) and the CFG
//! return (`flatten.py` `make_return` via `history.getkind`) must use the
//! same Option representation. `mmap_handle` is this shape on Windows.

use majit_charon_reader::Llbc;
use majit_translate::codewriter::call::CallControl;
use majit_translate::codewriter::codewriter::CodeWriter;
use majit_translate::codewriter::jitcode::JitCode;
use majit_translate::front::mir::build_semantic_program_from_llbc;
use majit_translate::{CallPath, GraphTransformConfig};
use std::sync::OnceLock;

const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../charon-corpus/corpus.ullbc");

fn load_corpus() -> &'static Llbc {
    static LLBC: OnceLock<Llbc> = OnceLock::new();
    LLBC.get_or_init(|| Llbc::load(CORPUS).expect("load corpus.ullbc"))
}

fn assemble(leaf: &str) {
    let llbc = load_corpus();
    let program = build_semantic_program_from_llbc(llbc).expect("build corpus SemanticProgram");
    let func = program
        .functions
        .iter()
        .find(|func| func.name == leaf)
        .unwrap_or_else(|| panic!("{leaf} in SemanticProgram"));
    let mut graph = func.graph().clone();
    // `lib.rs` stamps `SemanticFunction.return_type` onto the graph so
    // `call.py` `get_jitcode_calldescr` reads `FUNC.RESULT` off the callee.
    graph.return_type = func.return_type.clone();
    let path = CallPath::from_segments(["charon_corpus", leaf]);
    let mut callcontrol = CallControl::new();
    callcontrol.set_struct_fields(program.struct_fields.clone());
    callcontrol.register_function_graph(path.clone(), graph.clone());
    let mut codewriter = CodeWriter::new();
    let jitcode = std::sync::Arc::new(JitCode::new(leaf));
    codewriter.transform_graph_to_jitcode(
        &graph,
        &path,
        &mut callcontrol,
        &GraphTransformConfig::default(),
        &jitcode,
        false,
        0,
    );
}

#[test]
fn option_raw_c_void_from_int_assembles() {
    assemble("option_raw_c_void_from_int");
}

#[test]
fn option_raw_struct_from_int_assembles() {
    assemble("option_raw_struct_from_int");
}
