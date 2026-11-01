//! Build native-only library assets outside `pyre-interpreter`.
//! `compile_crate_ecis` compiles `_cffi_backend` `parse_c_type.py`'s ECI.

#[cfg(feature = "wasm_vfs")]
use std::path::Path;

/// Pure-Python files for the browser VFS.  Seeded from the modules a
/// Pyodide-style playground actually imports (`re`, `json`, `datetime`,
/// `random`, `pathlib`, `dataclasses`, `typing`, …) and their import
/// closure, minus heavy optional trees (`asyncio`, `pdb`, `email`, `http`,
/// `ssl`, `ctypes`, `unittest`, `tarfile`, …) and the modules that need
/// them (`urllib.request`).  C-level dependencies
/// (`_sre`, `_json`, `_random`, `_abc`, `_weakref`, `itertools`,
/// `_collections`, `_thread`) are builtin; the wrappers they back
/// (`operator.py`, `codecs.py`, `json/{decoder,encoder,scanner}.py`)
/// still have to be embedded. The `importlib` bootstrap sources are
/// embedded so startup installs `_frozen_importlib` and imports go
/// through `sys.meta_path`, and the VFS answers the directory listings
/// `FileFinder` makes.
#[cfg(feature = "wasm_vfs")]
const STDLIB_CLOSURE: &[&str] = &[
    "__future__.py",
    "_collections_abc.py",
    "_colorize.py",
    "_compat_pickle.py",
    "_opcode_metadata.py",
    "_py_abc.py",
    "_py_warnings.py",
    "_pydatetime.py",
    "_pydecimal.py",
    "_strptime.py",
    "_threading_local.py",
    "_weakrefset.py",
    "abc.py",
    "annotationlib.py",
    "ast.py",
    "base64.py",
    "bisect.py",
    "calendar.py",
    "codecs.py",
    "codeop.py",
    "collections/__init__.py",
    "contextlib.py",
    "contextvars.py",
    "copy.py",
    "copyreg.py",
    "csv.py",
    "dataclasses.py",
    "datetime.py",
    "decimal.py",
    "dis.py",
    "encodings/__init__.py",
    "encodings/aliases.py",
    "encodings/ascii.py",
    "encodings/latin_1.py",
    "encodings/utf_8.py",
    "enum.py",
    "fnmatch.py",
    "fractions.py",
    "functools.py",
    "genericpath.py",
    "getopt.py",
    "gettext.py",
    "glob.py",
    "heapq.py",
    "html/__init__.py",
    "html/entities.py",
    "importlib/__init__.py",
    "importlib/_abc.py",
    "importlib/_bootstrap.py",
    "importlib/_bootstrap_external.py",
    "importlib/machinery.py",
    "importlib/util.py",
    "inspect.py",
    "io.py",
    "ipaddress.py",
    "json/__init__.py",
    "json/decoder.py",
    "json/encoder.py",
    "json/scanner.py",
    "keyword.py",
    "linecache.py",
    "locale.py",
    "logging/__init__.py",
    "ntpath.py",
    "numbers.py",
    "opcode.py",
    "operator.py",
    "os.py",
    "pathlib/__init__.py",
    "pathlib/_os.py",
    "pickle.py",
    "posixpath.py",
    "pprint.py",
    "random.py",
    "re/__init__.py",
    "re/_casefix.py",
    "re/_compiler.py",
    "re/_constants.py",
    "re/_parser.py",
    "reprlib.py",
    "shutil.py",
    "stat.py",
    "statistics.py",
    "string/__init__.py",
    "struct.py",
    "tempfile.py",
    "textwrap.py",
    "threading.py",
    "token.py",
    "tokenize.py",
    "traceback.py",
    "types.py",
    "typing.py",
    "urllib/__init__.py",
    "urllib/error.py",
    "urllib/parse.py",
    "urllib/response.py",
    "warnings.py",
    "weakref.py",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    majit_eci::build::compile_crate_ecis(&majit_eci::ExternalCompilationInfo::default());
    #[cfg(feature = "wasm_vfs")]
    if std::env::var_os("CARGO_FEATURE_WASM_VFS").is_some() {
        build_stdlib_vfs();
    }
}

/// Pack the browser wasm stdlib closure beside the native decoder that owns
/// its format.  `pyre-interpreter` only embeds this finished asset and parses
/// its Python-facing VFS records; neither LZ4 nor the source-file walk belongs
/// to the frequently re-extracted runtime crate.
#[cfg(feature = "wasm_vfs")]
fn build_stdlib_vfs() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let stdlib_root = Path::new(&manifest_dir).join("../../lib-python/3");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");

    // [count: u32 LE], followed by repeated
    // [name_len: u32 LE][name][src_len: u32 LE][source].
    let mut raw = Vec::new();
    raw.extend_from_slice(&(STDLIB_CLOSURE.len() as u32).to_le_bytes());
    for rel in STDLIB_CLOSURE {
        let path = stdlib_root.join(rel);
        println!("cargo:rerun-if-changed={}", path.display());
        let source = std::fs::read(&path)
            .unwrap_or_else(|err| panic!("wasm_vfs: cannot read {}: {err}", path.display()));
        let name = rel.as_bytes();
        raw.extend_from_slice(&(name.len() as u32).to_le_bytes());
        raw.extend_from_slice(name);
        raw.extend_from_slice(&(source.len() as u32).to_le_bytes());
        raw.extend_from_slice(&source);
    }

    let compressed = lz4_flex::block::compress_prepend_size(&raw);
    let blob_path = Path::new(&out_dir).join("stdlib_vfs.lz4");
    std::fs::write(&blob_path, compressed)
        .unwrap_or_else(|err| panic!("wasm_vfs: cannot write {}: {err}", blob_path.display()));
}
