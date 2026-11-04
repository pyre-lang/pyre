//! RPython `rpython/translator/translator.py` — `TranslationContext`
//! skeleton.
//!
//! Upstream `TranslationContext` is a large driver object carrying
//! configuration, the annotator, the flowgraph cache, entry-point
//! bookkeeping, and translation-phase state. Only the subset the
//! annotator port currently consumes is declared here; additional
//! fields land as the driver calls them.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::{Rc, Weak};

use crate::annotator::annrpython::RPythonAnnotator;
use crate::annotator::policy::AnnotatorPolicy;
use crate::flowspace::model::{BlockKey, GraphKey, GraphRef, HostObject, checkgraph};
use crate::flowspace::objspace::build_flow;
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::rtyper::RPythonTyper;
use crate::translator::simplify;

/// RPython `translator.config.translation` — subset of upstream's
/// `rpython.config.translationoption.get_combined_translation_config`
/// output that the annotator driver actually reads. Fields land
/// incrementally as `self.translator.config.translation.*` accesses
/// manifest. The full upstream `OptionDescription` tree is ported in
/// [`crate::config::translationoption`] (with
/// [`crate::config::translationoption::translation_optiondescription`]
/// mirroring `translationoption.py`); this struct stays the
/// flat bool-shaped subset until consumers route through the option
/// tree.
#[derive(Clone, Debug)]
pub struct TranslationOptions {
    /// RPython `config.translation.verbose` (FLOWING_FLAGS default
    /// `False`).
    pub verbose: bool,
    /// RPython `config.translation.list_comprehension_operations`
    /// (FLOWING_FLAGS default `False`).
    pub list_comprehension_operations: bool,
    /// RPython `config.translation.keepgoing` — consumed by
    /// `TranslationContext.buildannotator()` upstream.
    pub keepgoing: bool,
    pub sandbox: bool,
    /// RPython `config.translation.check_str_without_nul` (upstream
    /// default `False`). Copied into `TLS.check_str_without_nul` by
    /// `RPythonAnnotator.build_types` (annrpython.py).
    pub check_str_without_nul: bool,
    /// RPython `config.translation.taggedpointers` (upstream default
    /// `False`). Consumed by `rclass.buildinstancerepr`
    /// (`rpython/rtyper/rclass.py`) to gate the
    /// `TaggedInstanceRepr` path for `UnboxedValue` subclasses.
    pub taggedpointers: bool,
    /// RPython `config.translation.withsmallfuncsets` (upstream
    /// default `0`, i.e. disabled). Gate read by
    /// `rpython/rtyper/rpbc.py small_cand` — when the PBC's
    /// desc-set size is strictly less than `withsmallfuncsets`,
    /// `SmallFunctionSetPBCRepr` becomes a candidate instead of
    /// `FunctionsPBCRepr`.
    pub withsmallfuncsets: usize,
    /// RPython `config.translation.rweakref` (upstream default `True`).
    /// Gate for RPython-level weakref support in the backend.
    pub rweakref: bool,
}

impl Default for TranslationOptions {
    fn default() -> Self {
        Self {
            verbose: false,
            list_comprehension_operations: false,
            keepgoing: false,
            sandbox: false,
            check_str_without_nul: false,
            taggedpointers: false,
            withsmallfuncsets: 0,
            rweakref: true,
        }
    }
}

/// RPython `translator.config` — outer `Config` holder. The annotator
/// driver reaches into it as `self.translator.config.translation.X`;
/// the Rust port preserves that access path via nested structs.
#[derive(Clone, Debug, Default)]
pub struct TranslationConfig {
    /// RPython root-level `config.translating` BoolOption installed by
    /// `get_combined_translation_config(translating=...)` at
    /// `rpython/config/translationoption.py`. Upstream's
    /// `TranslationContext.__init__` (translator.py) passes
    /// `translating=True`, so every `TranslationContext` created
    /// without an explicit config observes
    /// `config.translating == True`.
    pub translating: bool,
    /// RPython `config.translation` OptionGroup.
    pub translation: TranslationOptions,
}

impl TranslationConfig {
    /// Reads the typed-struct's seven `translation.<name>` fields out
    /// of the schema-driven [`crate::config::config::Config`] used by
    /// `TranslationDriver.config`. Mirrors the read path upstream's
    /// `TranslationContext.__init__` takes when handed
    /// `config=self.driver.config` at `interactive.py:19` /
    /// `driver.py`: each `config.translation.<name>` access is a
    /// `__getattr__` dispatch into the same `_cfgimpl_values` dict the
    /// driver's overrides write into.
    ///
    /// Errors out if the schema layout drifts (missing key, wrong
    /// variant) — that's a structural bug, not a runtime input error,
    /// so surfacing it via `Result::Err` rather than defaulting keeps
    /// the call site honest.
    pub fn from_rc_config(
        config: &std::rc::Rc<crate::config::config::Config>,
    ) -> Result<Self, crate::config::config::ConfigError> {
        use crate::config::config::{ConfigError, ConfigValue, OptionValue};
        fn get_bool(
            config: &std::rc::Rc<crate::config::config::Config>,
            path: &str,
        ) -> Result<bool, ConfigError> {
            match config.get(path)? {
                ConfigValue::Value(OptionValue::Bool(b)) => Ok(b),
                ConfigValue::Value(OptionValue::None) => Ok(false),
                other => Err(ConfigError::Generic(format!(
                    "TranslationConfig::from_rc_config: {path} expected Bool, got {other:?}"
                ))),
            }
        }
        fn get_int(
            config: &std::rc::Rc<crate::config::config::Config>,
            path: &str,
        ) -> Result<i64, ConfigError> {
            match config.get(path)? {
                ConfigValue::Value(OptionValue::Int(i)) => Ok(i),
                ConfigValue::Value(OptionValue::None) => Ok(0),
                other => Err(ConfigError::Generic(format!(
                    "TranslationConfig::from_rc_config: {path} expected Int, got {other:?}"
                ))),
            }
        }
        Ok(TranslationConfig {
            translating: get_bool(config, "translating")?,
            translation: TranslationOptions {
                verbose: get_bool(config, "translation.verbose")?,
                list_comprehension_operations: get_bool(
                    config,
                    "translation.list_comprehension_operations",
                )?,
                keepgoing: get_bool(config, "translation.keepgoing")?,
                sandbox: get_bool(config, "translation.sandbox")?,
                check_str_without_nul: get_bool(config, "translation.check_str_without_nul")?,
                taggedpointers: get_bool(config, "translation.taggedpointers")?,
                withsmallfuncsets: get_int(config, "translation.withsmallfuncsets")? as usize,
                rweakref: get_bool(config, "translation.rweakref")?,
            },
        })
    }
}

/// Rust carrier for upstream `**flowing_flags` in
/// `TranslationContext.__init__`. `None` means the caller did not pass
/// the flag; `Some(bool)` means it should overwrite the matching
/// `config.translation.*` field.
#[derive(Clone, Debug, Default)]
pub struct FlowingFlags {
    pub verbose: Option<bool>,
    pub list_comprehension_operations: Option<bool>,
}

/// Minimal carrier for upstream `get_platform(config)` result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Platform;

// RPython `exceptiontransform.py` is intentionally not ported as a
// standalone module. Pyre lowers Rust `Result` / `?` into exceptional
// graph exits directly, so exception-transform semantics should land in
// that Rust lowering path instead of adding an `ExceptionTransformer`
// placeholder surface.

/// RPython `get_combined_translation_config(translating=True)`
/// (translationoption.py). The Rust port exposes the
/// `translating=True` call that `TranslationContext.__init__` uses
/// when `config is None`; upstream callers that need
/// `translating=False` pre-construct a [`TranslationConfig`] and pass
/// it explicitly.
fn get_combined_translation_config() -> TranslationConfig {
    TranslationConfig {
        translating: true,
        ..TranslationConfig::default()
    }
}

fn get_platform(_config: &TranslationConfig) -> Platform {
    Platform
}

/// Key for [`TranslationContext::callgraph`]. Matches upstream's
/// Python dict key `(caller_graph, callee_graph, position_tag)` at
/// translator.py, where `position_tag = (parent_block, parent_index)`.
/// All three components carry pointer-identity semantics; Rust uses
/// `GraphKey` / `BlockKey` for the object handles.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CallGraphKey {
    pub caller: GraphKey,
    pub callee: GraphKey,
    pub tag_block: BlockKey,
    pub tag_index: usize,
}

/// Value for [`TranslationContext::callgraph`]. Upstream stores
/// `(caller_graph, callee_graph)` (translator.py), preserving the
/// GraphRef handles that outlive individual call-site traversals.
#[derive(Clone)]
pub struct CallGraphEdge {
    pub caller: GraphRef,
    pub callee: GraphRef,
}

/// RPython `class TranslationContext` (translator.py).
///
/// Held by [`RPythonAnnotator`]; fields land incrementally as the
/// annotator driver's `self.translator.*` calls manifest.
pub struct TranslationContext {
    /// RPython `self.config`. Upstream defaults to
    /// `get_combined_translation_config(translating=True)` when `None`
    /// is passed to `__init__`; the Rust port mirrors that via
    /// `get_combined_translation_config()`.
    pub config: TranslationConfig,
    /// RPython `self.annotator = None` (translator.py).
    ///
    /// Upstream stores a direct back-reference to the live
    /// `RPythonAnnotator`. Rust keeps the same lookup surface but uses
    /// `Weak` to avoid the refcount cycle that Python's GC collects.
    pub annotator: RefCell<Option<Weak<RPythonAnnotator>>>,
    /// RPython `self.rtyper = None` (translator.py).
    pub rtyper: RefCell<Option<Rc<RPythonTyper>>>,
    /// RPython `self.platform = get_platform(config)` (translator.py).
    pub platform: Platform,
    /// RPython `translator.frozen`, set by `driver.task_database_c`
    /// before the C database builder walks annotated graphs.
    pub frozen: Cell<bool>,
    // `self.exceptiontransformer = None` (translator.py) stays absent:
    // exception-transform behavior belongs in pyre's Rust lowering path.
    /// RPython `self.graphs = []` — every flow graph known to the
    /// translator. `RPythonAnnotator.complete()` iterates this to force
    /// annotation of each return variable.
    pub graphs: RefCell<Vec<GraphRef>>,
    /// RPython `self.callgraph = {}` (translator.py).
    /// `{opaque_tag: (caller-graph, callee-graph)}` — keyed by
    /// `(caller, callee, tag)` triple (translator.py:66). Populated
    /// every time the annotator's `recursivecall` records a non-None
    /// `whence` tag.
    pub callgraph: RefCell<HashMap<CallGraphKey, CallGraphEdge>>,
    /// RPython `self._prebuilt_graphs = {}` (translator.py).
    pub _prebuilt_graphs: RefCell<HashMap<HostObject, Rc<PyGraph>>>,
    /// Per-`HostObject` lift-failure record drained from
    /// `cutover::populate_call_registry_from_call_graphs`'s Pass 2
    /// failure arm.  When the eager pre-pass cannot lift a callee
    /// into a `PyGraph`, the error is stashed here (keyed by the
    /// callee's `HostObject` identity) so a later `buildflowgraph`
    /// fallback surfaces the actual lift error instead of the
    /// generic "missing code object" message.  Preserves the
    /// producer-side failure point analogous to upstream's lazy
    /// `getdesc → newfuncdesc → buildgraph` chain
    /// (`rpython/annotator/bookkeeper.py`) where the
    /// original exception propagates.
    pub _lift_errors: RefCell<HashMap<HostObject, String>>,
    /// RPython `self.entry_point_graph`. Set by
    /// `RPythonAnnotator.build_types(main_entry_point=True)`.
    pub entry_point_graph: RefCell<Option<GraphRef>>,
}

impl TranslationContext {
    pub fn new() -> Self {
        Self::with_config_and_flowing_flags(None, FlowingFlags::default())
    }

    /// RPython `TranslationContext.__init__(self, config=None,
    /// **flowing_flags)` (translator.py).
    pub fn with_config_and_flowing_flags(
        config: Option<TranslationConfig>,
        flowing_flags: FlowingFlags,
    ) -> Self {
        // upstream: `if config is None: config =
        // get_combined_translation_config(translating=True)`.
        let mut config = config.unwrap_or_else(get_combined_translation_config);
        // upstream: `for attr in ['verbose',
        // 'list_comprehension_operations']: if attr in flowing_flags:
        // setattr(config.translation, attr, flowing_flags[attr])`.
        if let Some(verbose) = flowing_flags.verbose {
            config.translation.verbose = verbose;
        }
        if let Some(list_ops) = flowing_flags.list_comprehension_operations {
            config.translation.list_comprehension_operations = list_ops;
        }
        let platform = get_platform(&config);
        TranslationContext {
            config,
            platform,
            frozen: Cell::new(false),
            annotator: RefCell::new(None),
            rtyper: RefCell::new(None),
            graphs: RefCell::new(Vec::new()),
            callgraph: RefCell::new(HashMap::new()),
            _prebuilt_graphs: RefCell::new(HashMap::new()),
            _lift_errors: RefCell::new(HashMap::new()),
            entry_point_graph: RefCell::new(None),
        }
    }

    /// RPython `translator.annotator = self` assignment performed in
    /// `RPythonAnnotator.__init__` (annrpython.py) and
    /// `TranslationContext.buildannotator()` (translator.py).
    pub fn set_annotator(&self, annotator: Weak<RPythonAnnotator>) {
        *self.annotator.borrow_mut() = Some(annotator);
    }

    pub fn annotator(&self) -> Option<Rc<RPythonAnnotator>> {
        self.annotator.borrow().as_ref().and_then(Weak::upgrade)
    }

    /// RPython `TranslationContext.buildannotator()` (translator.py).
    pub fn buildannotator(
        self: &Rc<Self>,
        policy: Option<AnnotatorPolicy>,
    ) -> Rc<RPythonAnnotator> {
        if self.annotator().is_some() {
            panic!("ValueError: we already have an annotator");
        }
        let annotator = RPythonAnnotator::new_with_translator(
            Some(Rc::clone(self)),
            policy,
            None,
            self.config.translation.keepgoing,
        );
        self.set_annotator(Rc::downgrade(&annotator));
        annotator
    }

    /// RPython `TranslationContext.buildflowgraph()` (translator.py).
    pub fn buildflowgraph(&self, func: HostObject, mute_dot: bool) -> Result<Rc<PyGraph>, String> {
        let graph_func = func
            .user_function()
            .cloned()
            .ok_or_else(|| format!("buildflowgraph() expects a function, got {:?}", func))?;

        if let Some(pygraph) = self._prebuilt_graphs.borrow_mut().remove(&func) {
            return Ok(pygraph);
        }

        // If the lift of this host's lowered body
        // (`FunctionDesc::source_template`) recorded a failure,
        // surface that error instead of the generic "missing code
        // object" fallback below.  Mirrors
        // `bookkeeper.getdesc → newfuncdesc → FunctionDesc.buildgraph`
        // where the original construction error propagates from the
        // lazy `cachedgraph` consumer.
        if let Some(lift_err) = self._lift_errors.borrow().get(&func).cloned() {
            return Err(format!(
                "buildflowgraph({}): source lift failed (recorded \
                 lazy-failure error): {lift_err}",
                graph_func.name,
            ));
        }

        let code = graph_func
            .code
            .as_ref()
            .ok_or_else(|| format!("buildflowgraph({}): missing code object", graph_func.name))?;

        // Fail-loud: a GraphFunc whose `HostCode.co_code` is empty carries
        // no lowered bytecode, so `build_flow` below would silently produce
        // an empty / wrong graph. This occurs when the MIR front-end
        // registered a metadata-only carrier for a host whose body it could
        // not lower. Surface the failure here — `buildflowgraph(callee)` is
        // exactly the call site upstream raises FlowingError when a
        // function's analysis cannot proceed (`flowcontext.py`).
        if code.co_code.is_empty() {
            return Err(format!(
                "buildflowgraph({}): `HostCode.co_code` is empty — the MIR \
                 front-end produced a metadata-only carrier with no lowered \
                 body for this host, so build_flow would yield an empty / \
                 wrong graph.",
                graph_func.name
            ));
        }

        let graph = build_flow(graph_func.clone()).map_err(|err| format!("{err:?}"))?;
        simplify::simplify_graph(&graph, None);
        if self.config.translation.list_comprehension_operations {
            simplify::detect_list_comprehension(&graph);
        }
        if !self.config.translation.verbose && !mute_dot {
            // upstream emits `log.dot()` here. The logger plumbing
            // is not ported yet, so preserve only the control-flow
            // gating.
        }
        let pygraph = Rc::new(PyGraph {
            graph: graph.into_ref(),
            func: graph_func.clone(),
            signature: RefCell::new(code.signature.clone()),
            defaults: RefCell::new(Some(graph_func.defaults.clone())),
            access_directly: std::cell::Cell::new(false),
        });
        self.graphs.borrow_mut().push(pygraph.graph.clone());
        Ok(pygraph)
    }

    /// RPython `translator.rtyper = self` assignment performed in
    /// `TranslationContext.buildrtyper()` (translator.py).  Mirror
    /// of [`Self::set_annotator`] for the rtyper slot.
    pub fn set_rtyper(&self, rtyper: Rc<RPythonTyper>) {
        *self.rtyper.borrow_mut() = Some(rtyper);
    }

    /// RPython `TranslationContext.buildrtyper()` (translator.py).
    pub fn buildrtyper(&self) -> Rc<RPythonTyper> {
        let annotator = self.annotator().expect("ValueError: no annotator");
        if self.rtyper.borrow().is_some() {
            panic!("ValueError: we already have an rtyper");
        }
        let rtyper = Rc::new(RPythonTyper::new(&annotator));
        // Upstream `rtyper.py`: `self.exceptiondata = ExceptionData(self)`
        // is part of `RPythonTyper.__init__`. The Rust port defers the
        // initialisation so the rtyper can be wrapped in `Rc<Self>`
        // before populating; call it now so callers observe the same
        // post-`buildrtyper()` invariants upstream guarantees.
        rtyper
            .initialize_exceptiondata()
            .expect("RPythonTyper::initialize_exceptiondata");
        self.set_rtyper(rtyper.clone());
        rtyper
    }

    pub fn rtyper(&self) -> Option<Rc<RPythonTyper>> {
        self.rtyper.borrow().as_ref().cloned()
    }

    /// RPython `TranslationContext.checkgraphs()` (translator.py).
    pub fn checkgraphs(&self) {
        for graph in self.graphs.borrow().iter() {
            checkgraph(&graph.borrow());
        }
    }

    /// RPython `translator.getexceptiontransformer()` at
    /// `rpython/translator/translator.py`.
    ///
    /// Upstream:
    /// ```python
    /// def getexceptiontransformer(self):
    ///     if self.rtyper is None:
    ///         raise ValueError("no rtyper")
    ///     if self.exceptiontransformer is not None:
    ///         return self.exceptiontransformer
    ///     from rpython.translator.exceptiontransform import ExceptionTransformer
    ///     self.exceptiontransformer = ExceptionTransformer(self)
    ///     return self.exceptiontransformer
    /// ```
    ///
    /// The `rtyper is None` guard at `:87-88` is mirrored exactly. The
    /// lazy creation at `:91-92` (`ExceptionTransformer(self)`) stays
    /// absent because pyre represents exception edges in the Rust lowering
    /// pipeline. For now the method returns `Ok(None)` so callers route the
    /// `None` through to downstream consumers (`genc.py:92` stores the
    /// value verbatim).
    pub fn getexceptiontransformer(
        &self,
    ) -> Result<
        Option<std::rc::Rc<dyn std::any::Any>>,
        crate::translator::tool::taskengine::TaskError,
    > {
        if self.rtyper.borrow().is_none() {
            return Err(crate::translator::tool::taskengine::TaskError {
                message: "translator.py:88 ValueError: no rtyper".to_string(),
            });
        }
        Ok(None)
    }

    /// Upstream `genc.py:128 for obj in exports.EXPORTS_obj2name.keys():
    /// db.getcontainernode(obj)`.
    ///
    /// `rpython/rlib/exports.py` is not yet ported; the local stub
    /// returns an empty iterator so the upstream loop body matches
    /// line-by-line and runs zero times. Convergence path = port
    /// `rlib/exports.py` and have its `EXPORTS_obj2name` populated by
    /// the same `@export` decorator the upstream uses.
    pub fn exports_obj2name_keys(&self) -> Vec<std::rc::Rc<dyn std::any::Any>> {
        Vec::new()
    }

    /// Upstream `genc.py exports.clear()`. Stub no-op until
    /// `rlib/exports.py` lands locally.
    pub fn clear_exports(&self) {}

    /// Upstream `genc.py for ll_func in db.translator._call_at_startup`.
    ///
    /// `_call_at_startup` is registered upstream by code paths that
    /// have not yet been ported (rffi tooling, GC startup hooks).
    /// Returns an empty Vec so the loop body matches upstream and
    /// degrades to zero iterations. Convergence path = land the
    /// `_call_at_startup` slot on `TranslationContext` once those
    /// upstream code paths land locally.
    pub fn call_at_startup(&self) -> Vec<std::rc::Rc<dyn std::any::Any>> {
        Vec::new()
    }

    /// RPython `TranslationContext.update_call_graph(caller_graph,
    /// callee_graph, position_tag)` (translator.py).
    ///
    /// ```python
    /// def update_call_graph(self, caller_graph, callee_graph, position_tag):
    ///     key = caller_graph, callee_graph, position_tag
    ///     self.callgraph[key] = caller_graph, callee_graph
    /// ```
    ///
    /// Upstream dedupes by the full key triple; re-recording the same
    /// (caller, callee, tag) overwrites the value (same graph refs).
    pub fn update_call_graph(&self, caller: &GraphRef, callee: &GraphRef, tag: (BlockKey, usize)) {
        let (tag_block, tag_index) = tag;
        let key = CallGraphKey {
            caller: GraphKey::of(caller),
            callee: GraphKey::of(callee),
            tag_block,
            tag_index,
        };
        let edge = CallGraphEdge {
            caller: caller.clone(),
            callee: callee.clone(),
        };
        self.callgraph.borrow_mut().insert(key, edge);
    }
}

impl Default for TranslationContext {
    fn default() -> Self {
        Self::new()
    }
}

/// RPython `graphof(translator, func)` (translator.py).
///
/// The upstream helper also accepts a `FunctionGraph` and returns it
/// unchanged. Rust call sites already have the typed `GraphRef` in that case,
/// so this helper covers the host-function lookup path: find the one graph in
/// `translator.graphs` whose attached `FunctionGraph.func` is the same
/// `GraphFunc` as `func`.
pub fn graphof(translator: &TranslationContext, func: &HostObject) -> GraphRef {
    let Some(target) = func.user_function() else {
        panic!("graphof() expects a user function, got {func:?}");
    };
    let result: Vec<GraphRef> = translator
        .graphs
        .borrow()
        .iter()
        .filter(|graph| {
            graph
                .borrow()
                .func
                .as_ref()
                .is_some_and(|graph_func| graph_func == target)
        })
        .cloned()
        .collect();
    assert_eq!(result.len(), 1);
    result.into_iter().next().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::argument::Signature;
    use crate::flowspace::bytecode::HostCode;
    use crate::flowspace::model::{
        Block, ConstValue, Constant, FunctionGraph, GraphFunc, HostObject,
    };
    use std::cell::RefCell as StdRefCell;
    use std::rc::Rc;

    fn mk_graph(name: &'static str) -> GraphRef {
        let block = Rc::new(StdRefCell::new(Block::new(vec![])));
        Rc::new(StdRefCell::new(FunctionGraph::new(name, block)))
    }

    fn host_func(name: &str, argnames: &[&str]) -> HostObject {
        let code = HostCode {
            id: HostCode::fresh_identity(),
            co_name: name.to_string(),
            co_filename: "<test>".to_string(),
            co_firstlineno: 1,
            co_nlocals: argnames.len() as u32,
            co_argcount: argnames.len() as u32,
            co_stacksize: 0,
            co_flags: crate::flowspace::objspace::CoFlags::NEWLOCALS.bits(),
            co_code: rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
            co_varnames: argnames.iter().map(|arg| (*arg).to_string()).collect(),
            co_freevars: Vec::new(),
            co_cellvars: Vec::new(),
            consts: Vec::new(),
            names: Vec::new(),
            co_lnotab: Vec::new(),
            exceptiontable: Vec::new().into_boxed_slice(),
            signature: Signature::new(
                argnames.iter().map(|arg| (*arg).to_string()).collect(),
                None,
                None,
            ),
        };
        let func = GraphFunc::from_host_code(
            code,
            Constant::new(ConstValue::Dict(Default::default())),
            vec![],
        );
        HostObject::new_user_function(func)
    }

    #[test]
    fn update_call_graph_records_caller_callee_tag_triple() {
        let ctx = TranslationContext::new();
        let caller = mk_graph("caller");
        let callee = mk_graph("callee");
        let block = caller.borrow().startblock.clone();
        let tag = (BlockKey::of(&block), 7);
        ctx.update_call_graph(&caller, &callee, tag);
        assert_eq!(ctx.callgraph.borrow().len(), 1);
        let key = CallGraphKey {
            caller: GraphKey::of(&caller),
            callee: GraphKey::of(&callee),
            tag_block: BlockKey::of(&block),
            tag_index: 7,
        };
        assert!(ctx.callgraph.borrow().contains_key(&key));
    }

    #[test]
    fn update_call_graph_dedupes_on_full_triple() {
        let ctx = TranslationContext::new();
        let caller = mk_graph("caller");
        let callee = mk_graph("callee");
        let block = caller.borrow().startblock.clone();
        let tag = (BlockKey::of(&block), 0);
        ctx.update_call_graph(&caller, &callee, tag.clone());
        ctx.update_call_graph(&caller, &callee, tag);
        assert_eq!(ctx.callgraph.borrow().len(), 1);
    }

    #[test]
    fn update_call_graph_distinct_tags_record_distinct_edges() {
        let ctx = TranslationContext::new();
        let caller = mk_graph("caller");
        let callee = mk_graph("callee");
        let block = caller.borrow().startblock.clone();
        ctx.update_call_graph(&caller, &callee, (BlockKey::of(&block), 1));
        ctx.update_call_graph(&caller, &callee, (BlockKey::of(&block), 2));
        assert_eq!(ctx.callgraph.borrow().len(), 2);
    }

    #[test]
    fn buildannotator_records_shared_identity() {
        let ctx = Rc::new(TranslationContext::new());
        let ann = ctx.buildannotator(None);
        let stored = ctx.annotator().expect("annotator should be installed");
        assert!(Rc::ptr_eq(&ann, &stored));
        assert!(Rc::ptr_eq(&ann.translator, &ctx));
    }

    #[test]
    fn implicit_annotator_backlink_drops_with_annotator_rc() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let translator = ann.translator.clone();
        assert!(translator.annotator().is_some());
        drop(ann);
        assert!(translator.annotator().is_none());
    }

    #[test]
    fn buildannotator_backlink_drops_when_annotator_is_gone() {
        let ctx = Rc::new(TranslationContext::new());
        {
            let _ann = ctx.buildannotator(None);
            assert!(ctx.annotator().is_some());
        }
        assert!(ctx.annotator().is_none());
    }

    #[test]
    fn buildannotator_uses_translation_keepgoing() {
        let mut ctx = TranslationContext::new();
        ctx.config.translation.keepgoing = true;
        let ctx = Rc::new(ctx);
        let ann = ctx.buildannotator(None);
        assert!(ann.keepgoing);
    }

    #[test]
    fn buildannotator_rejects_second_annotator() {
        let ctx = Rc::new(TranslationContext::new());
        let _ann = ctx.buildannotator(None);
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = ctx.buildannotator(None);
        }));
        assert!(err.is_err());
    }

    #[test]
    fn new_context_starts_with_sandbox_disabled() {
        let ctx = TranslationContext::new();
        assert!(!ctx.config.translation.sandbox);
    }

    #[test]
    fn new_context_initializes_prebuilt_graph_cache() {
        // RPython `TranslationContext.__init__` (translator.py)
        // seeds `self._prebuilt_graphs = {}`. Keep only the cache
        // field that is actually consumed by the current Rust port.
        let ctx = TranslationContext::new();
        assert!(ctx._prebuilt_graphs.borrow().is_empty());
    }

    #[test]
    fn default_construction_sets_translating_true() {
        // Upstream `TranslationContext.__init__` (translator.py)
        // calls `get_combined_translation_config(translating=True)`
        // whenever `config is None`. The Rust port mirrors this via
        // the `get_combined_translation_config` helper.
        let ctx = TranslationContext::new();
        assert!(ctx.config.translating);
    }

    #[test]
    fn explicit_config_preserves_caller_provided_translating() {
        // When the caller passes their own config,
        // `TranslationContext.__init__` stores it verbatim. The Rust
        // port mirrors this: `translating=false` survives the
        // constructor.
        let caller_config = TranslationConfig {
            translating: false,
            ..TranslationConfig::default()
        };
        let ctx = TranslationContext::with_config_and_flowing_flags(
            Some(caller_config),
            FlowingFlags::default(),
        );
        assert!(!ctx.config.translating);
    }

    #[test]
    fn flowing_flags_override_translation_subfields() {
        let ctx = TranslationContext::with_config_and_flowing_flags(
            None,
            FlowingFlags {
                verbose: Some(true),
                list_comprehension_operations: Some(true),
            },
        );
        assert!(ctx.config.translation.verbose);
        assert!(ctx.config.translation.list_comprehension_operations);
    }

    #[test]
    fn checkgraphs_walks_registered_graphs() {
        use crate::flowspace::model::{BlockRefExt, ConstValue, Constant, Hlvalue, Link};
        let ctx = TranslationContext::new();
        let graph = mk_graph("checked");
        {
            let g = graph.borrow();
            let link = Rc::new(StdRefCell::new(Link::new(
                vec![Hlvalue::Constant(Constant::new(ConstValue::Int(1)))],
                Some(g.returnblock.clone()),
                None,
            )));
            g.startblock.closeblock(vec![link]);
        }
        ctx.graphs.borrow_mut().push(graph);
        ctx.checkgraphs();
    }

    #[test]
    fn buildflowgraph_prebuilt_graph_is_not_registered() {
        let ctx = TranslationContext::new();
        let func = host_func("f", &[]);
        let prebuilt = Rc::new(PyGraph {
            graph: mk_graph("prebuilt"),
            func: host_func("f", &[])
                .user_function()
                .cloned()
                .expect("host_func should build a user function"),
            signature: RefCell::new(Signature::new(Vec::new(), None, None)),
            defaults: RefCell::new(None),
            access_directly: std::cell::Cell::new(true),
        });
        ctx._prebuilt_graphs
            .borrow_mut()
            .insert(func.clone(), prebuilt.clone());

        let pygraph = ctx.buildflowgraph(func, false).expect("prebuilt graph");

        assert!(Rc::ptr_eq(&pygraph, &prebuilt));
        assert!(ctx.graphs.borrow().is_empty());
        assert!(pygraph.defaults.borrow().is_none());
        assert!(pygraph.access_directly.get());
    }

    #[test]
    fn buildflowgraph_rejects_non_function_before_prebuilt_cache_lookup() {
        let ctx = TranslationContext::new();
        let module = HostObject::new_module("pkg.not_a_function");
        let prebuilt = Rc::new(PyGraph {
            graph: mk_graph("prebuilt"),
            func: host_func("f", &[])
                .user_function()
                .cloned()
                .expect("host_func should build a user function"),
            signature: RefCell::new(Signature::new(Vec::new(), None, None)),
            defaults: RefCell::new(None),
            access_directly: std::cell::Cell::new(true),
        });
        ctx._prebuilt_graphs
            .borrow_mut()
            .insert(module.clone(), prebuilt);

        let err = match ctx.buildflowgraph(module.clone(), false) {
            Ok(_) => panic!("non-functions must be rejected before cache lookup"),
            Err(err) => err,
        };

        assert!(err.contains("buildflowgraph() expects a function"));
        assert!(ctx._prebuilt_graphs.borrow().contains_key(&module));
    }
}
