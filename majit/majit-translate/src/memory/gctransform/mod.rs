//! `rpython/memory/gctransform/` — the GC root transformer.
//!
//! Upstream inserts the shadow-stack save/reload bracket automatically:
//! `framework.py` brackets every collecting operation, `shadowstack.py`
//! emits the `gc_push_roots` / `gc_pop_roots` pair, `shadowcolor.py` colours
//! the saves like registers, and `postprocess_double_check` verifies the
//! placement.  Application code therefore carries no rooting at all —
//! `pypy/objspace/std/listobject.py descr_index` names its receiver and its
//! needle as plain locals.
//!
//! [`shadowstack`] now owns the same graph-wide postprocessing hook as PyPy:
//! the backend invokes it once over the translator's complete graph set, so
//! application functions carry no per-function attribute.  It applies
//! [`shadowcolor`] after the framework transformer has emitted
//! `gc_push_roots` / `gc_pop_roots` markers.
//!
//! [`framework`] now supplies the marker-insertion half for translated
//! flowspace graphs: `CollectAnalyzer` classifies each operation and
//! `get_livevars_for_roots` derives its root set from graph liveness. Native
//! interpreter paths compiled directly by rustc still retain their existing
//! source `push_roots` brackets, and [`liveness`] checks those brackets over
//! the extracted ULLBC; the graph transform and the native audit therefore
//! cover their respective execution paths without a source-level colouring
//! approximation.
pub mod framework;
pub(crate) mod instantiate;
pub mod liveness;
pub mod shadowcolor;
pub mod shadowstack;
