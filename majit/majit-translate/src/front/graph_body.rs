//! On-demand construction of a funcobj's lowered body.
//!
//! `translator.py buildflowgraph(func)` builds a function's flow graph
//! from the function object when a consumer first asks for it, and
//! `description.py FunctionDesc.cachedgraph` is that consumer: a
//! cache miss builds, a hit returns.  Nothing in upstream builds every
//! reachable function's graph up front — `translator.graphs` *is* the set
//! that demand has already reached.
//!
//! Pyre's funcobjs are Charon-extracted `FunDecl`s rather than Python
//! function objects, so the analogue of "build the flow graph from the
//! function object" is "lower the `FunDecl` from the LLBC it came from".
//! That requires the LLBC set to stay alive past the whole-program build,
//! which is what a [`GraphBodyProvider`] owns.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use majit_charon_reader::Llbc;

use crate::codewriter::call::{DeclaredFuncObj, FuncObjDeclarations};
use crate::front::mir::{
    self, CrateLowering, CrateLoweringState, DeclBuildError, GraphStamp, LowerError,
};
use crate::front::semantic::{SemanticFunction, SemanticProgram};
use crate::model::{FunctionGraph, GraphKey, LazyGraph};

/// Where a funcobj's body comes from: the LLBC that carries it and the
/// Charon `def_id` that indexes it there (`Llbc::fn_by_id`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "no funcobj records its body source yet")
)]
pub(crate) struct GraphBodySource {
    pub llbc_index: u32,
    pub def_id: u64,
}

/// Owns the extracted LLBC set and everything else the lowering reads, so
/// a funcobj's body can be built after the whole-program pass has run.
/// Each funcobj's [`LazyGraph`] holds its crate and the tables, and builds
/// its body from them on first demand.
pub(crate) struct GraphBodyProvider {
    crates: Vec<Rc<ProvidedCrate>>,
    tables: Rc<ProviderTables>,
}

/// What every crate's lowering reads besides its own artefact and state.
///
/// The three `HostStaticAddrs` tables and the error-carrier spec are held
/// owned because [`crate::HostStaticAddrs`] borrows all of them from the
/// caller's frame; the borrowed view is rebuilt per lowering call.
struct ProviderTables {
    jitdriver_receiver_roots: Vec<String>,
    pytypes: Vec<(String, i64)>,
    pytypes_by_struct: Vec<(String, i64)>,
    refs: Vec<(String, i64)>,
    int_values: Vec<(String, i64)>,
    error_carrier: crate::OwnedErrorCarrierSpec,
    scalar_field_stores: Vec<OwnedScalarFieldStore>,
    /// The funcobj hint attributes harvested across the whole input; every
    /// crate's headers read them ([`CrateLoweringState`]).
    func_hints: HashMap<String, Vec<String>>,
    /// Where the funcobjs declared apart from a crate's program go: the
    /// clause specializations, each declared when the body whose call names
    /// it is built (`specialize.py default_specialize` runs as the annotator
    /// reaches the call).
    declarations: FuncObjDeclarations,
    /// Where every crate logs a declaration whose body did not lower; the
    /// pipeline reports it once the bodies it builds are built.
    skipped: mir::LoweringSkips,
}

/// One lowered crate: its artefact and the lowering state its decls were
/// lowered with.
struct ProvidedCrate {
    llbc: Llbc,
    state: CrateLoweringState,
}

/// Owned mirror of [`crate::ScalarFieldStore`], held for the reason the
/// carrier spec above it is: a demanded body has to lower the declared
/// boundaries the whole-program pass lowered, and the declaration borrows
/// from the caller's frame.
#[derive(Debug)]
struct OwnedScalarFieldStore {
    function_path: String,
    owner_root: String,
    field: String,
    bank: crate::ScalarBank,
}

impl OwnedScalarFieldStore {
    fn own(store: &crate::ScalarFieldStore<'_>) -> Self {
        Self {
            function_path: store.function_path.to_string(),
            owner_root: store.owner_root.to_string(),
            field: store.field.to_string(),
            bank: store.bank,
        }
    }

    fn borrowed(&self) -> crate::ScalarFieldStore<'_> {
        crate::ScalarFieldStore {
            function_path: &self.function_path,
            owner_root: &self.owner_root,
            field: &self.field,
            bank: self.bank,
        }
    }
}

impl GraphBodyProvider {
    pub(crate) fn new(
        static_addrs: crate::HostStaticAddrs<'_>,
        jitdriver_receiver_roots: &[String],
        func_hints: HashMap<String, Vec<String>>,
        declarations: FuncObjDeclarations,
        skipped: mir::LoweringSkips,
    ) -> Self {
        let own = |rows: &[(&str, i64)]| -> Vec<(String, i64)> {
            rows.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
        };
        let tables = ProviderTables {
            jitdriver_receiver_roots: jitdriver_receiver_roots.to_vec(),
            pytypes: own(static_addrs.pytypes),
            pytypes_by_struct: own(static_addrs.pytypes_by_struct),
            refs: own(static_addrs.refs),
            int_values: own(static_addrs.int_values),
            error_carrier: crate::OwnedErrorCarrierSpec::own(static_addrs.error_carrier),
            scalar_field_stores: static_addrs
                .scalar_field_stores
                .iter()
                .map(OwnedScalarFieldStore::own)
                .collect(),
            func_hints,
            declarations,
            skipped,
        };
        Self {
            crates: Vec::new(),
            tables: Rc::new(tables),
        }
    }

    /// Declare the funcobjs of one already-linked artefact and keep it. No
    /// body is built here: each is built when something first asks for its
    /// graph (`description.py FunctionDesc.getgraphs` / `cachedgraph`), and
    /// building one declares the clause specializations it names. The
    /// caller applied
    /// `discover_transparent_scalar_kinds` and `discover_foldable_const_lits`
    /// across the whole set first, and `cross_tombstoned_leaves` is the
    /// duplicate-leaf verdict across that set.
    pub(crate) fn lower_prelinked_crate(
        &mut self,
        llbc: Llbc,
        module_paths: &[&str],
        cross_tombstoned_leaves: &HashSet<String>,
        gc_struct_ids: &HashSet<majit_ir::descr::StructId>,
        raw_alloc_flavor_structs: &HashSet<String>,
    ) -> SemanticProgram {
        let module_filter = mir::normalize_module_filter(module_paths);
        let paint_tombstones = mir::prelink_crate(&llbc, cross_tombstoned_leaves);
        let mut state = CrateLoweringState::new(
            &llbc,
            &paint_tombstones,
            self.tables.func_hints.clone(),
            self.tables.skipped.clone(),
            gc_struct_ids,
            raw_alloc_flavor_structs,
            None,
            None,
        );
        for prev in &self.crates {
            state.absorb_struct_fields(prev.state.struct_fields());
            state.absorb_raw_alloc_flavor_structs(prev.state.raw_alloc_flavor_structs());
        }
        let krate = Rc::new(ProvidedCrate { llbc, state });
        let functions = self.declare_crate(&krate, module_filter.as_ref(), None);
        let mut program = krate.state.finish(functions);
        mir::harden_duplicate_leaf_metadata(
            &mut program.struct_fields,
            &mut program.struct_origins,
            &mut program.enum_variant_by_discriminant,
            Some(&program.struct_ids),
        );
        self.crates.push(krate);
        program
    }

    /// `lower_prelinked_crate` restricted to leaf names. The production
    /// pipeline declares every admitted funcobj; a test that only reads
    /// one body should not lower the rest.
    #[cfg(test)]
    fn lower_named_for_test(&mut self, llbc: Llbc, function_names: &[&str]) -> SemanticProgram {
        let function_filter: HashSet<String> = function_names
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        let paint_tombstones = mir::prelink_crate(&llbc, &HashSet::new());
        let gc_struct_ids = mir::harvest_declared_gc_facts(&llbc).gc_struct_ids();
        let mut state = CrateLoweringState::new(
            &llbc,
            &paint_tombstones,
            self.tables.func_hints.clone(),
            self.tables.skipped.clone(),
            &gc_struct_ids,
            &HashSet::new(),
            None,
            Some(&function_filter),
        );
        for prev in &self.crates {
            state.absorb_struct_fields(prev.state.struct_fields());
            state.absorb_raw_alloc_flavor_structs(prev.state.raw_alloc_flavor_structs());
        }
        let krate = Rc::new(ProvidedCrate { llbc, state });
        let functions = self.declare_crate(&krate, None, Some(&function_filter));
        let mut program = krate.state.finish(functions);
        mir::harden_duplicate_leaf_metadata(
            &mut program.struct_fields,
            &mut program.struct_origins,
            &mut program.enum_variant_by_discriminant,
            Some(&program.struct_ids),
        );
        self.crates.push(krate);
        program
    }

    /// A funcobj per declaration of `krate` the membership gates admit,
    /// each holding its body unbuilt.
    fn declare_crate(
        &self,
        krate: &Rc<ProvidedCrate>,
        module_filter: Option<&HashSet<String>>,
        function_filter: Option<&HashSet<String>>,
    ) -> Vec<SemanticFunction> {
        // `bookkeeper.py` `Bookkeeper.getdesc` keys by the function object.
        // Body-carrying FunDecls were recorded on this crate's
        // `StructFieldRegistry` in `CrateLoweringState::new`.
        krate.lowering(&self.tables, |lowering| {
            krate
                .llbc
                .iter_local_fns()
                .filter(|fd| lowering.admit_decl(fd, module_filter, function_filter))
                .filter_map(|fd| {
                    let header = lowering.decl_header(fd);
                    let stamp = header.graph_stamp();
                    // A declaration with no body (a required trait method,
                    // an extern) is no function object.
                    let Some(declared) = lowering.decl_header_graph(fd, &stamp) else {
                        lowering.record_decl_failure(fd, DeclBuildError::NoBody);
                        return None;
                    };
                    let declared = Rc::new(declared);
                    if lowering.declare_llexternal(fd) {
                        return Some(header.into_semantic(LazyGraph::external(declared)));
                    }
                    let (krate, tables, def_id) = (krate.clone(), self.tables.clone(), fd.def_id);
                    let graph = LazyGraph::deferred(declared.clone(), move || {
                        let graph = krate.build_decl_graph(&tables, def_id, &stamp, &declared);
                        krate.declare_queued_specs(&tables);
                        graph
                    });
                    Some(header.into_semantic(graph))
                })
                .collect()
        })
    }

    /// Locate the funcobj whose Charon `name_path()` is `name_path`, if
    /// exactly one carries it.
    ///
    /// `bookkeeper.py getdesc(pyobj)` keys a descriptor by the function
    /// object itself, so two distinct functions are never conflated even
    /// when they render the same name.  A name path carries no such
    /// identity, and nothing stops two extracted `FunDecl`s from sharing
    /// one, so an ambiguous name resolves to nothing rather than to an
    /// arbitrary one of the candidates: binding the wrong body is silent,
    /// while a miss is not.  Identity on the demand path is
    /// [`GraphBodySource`], recorded when the funcobj was registered.
    ///
    /// Linear over the corpus, so it is a registration-time helper (and
    /// the test seam), not a per-demand lookup.
    #[cfg(test)]
    pub(crate) fn source_for_name_path(&self, name_path: &str) -> Option<GraphBodySource> {
        let mut found = None;
        for (i, krate) in self.crates.iter().enumerate() {
            for fd in krate.llbc.iter_local_fns() {
                if fd.item_meta.name_path() != name_path {
                    continue;
                }
                if found.is_some() {
                    return None;
                }
                found = Some(GraphBodySource {
                    llbc_index: i as u32,
                    def_id: fd.def_id,
                });
            }
        }
        found
    }

    /// Build the funcobj `src` names with the state its crate was lowered
    /// with, reproducing what the whole-program loop produced for it. Which
    /// funcobjs exist is decided when they are registered, so no membership
    /// gate runs here, and a body that does not lower answers its own error.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "no funcobj records its body source yet")
    )]
    pub(crate) fn build(&self, src: GraphBodySource) -> Result<SemanticFunction, LowerError> {
        let idx = src.llbc_index as usize;
        let krate = self
            .crates
            .get(idx)
            .ok_or_else(|| LowerError::Unsupported(format!("llbc index {idx} out of range")))?;
        let fd = krate.llbc.fn_by_id(src.def_id).ok_or_else(|| {
            LowerError::Unsupported(format!("no FunDecl for def_id {}", src.def_id))
        })?;
        // Same order as the deferred closure in `declare_crate`: the header
        // is snapshotted before the body exists, the body is stamped
        // afterwards, and clause specializations the body queued are
        // declared before the funcobj is returned.
        let (header, stamp, declared) = krate.lowering(&self.tables, |lowering| {
            let header = lowering.decl_header(fd);
            let stamp = header.graph_stamp();
            let declared = lowering.decl_header_graph(fd, &stamp);
            (header, stamp, declared)
        });
        let Some(declared) = declared else {
            return Err(LowerError::Unsupported(format!(
                "{}: no Unstructured body",
                fd.item_meta.name_path()
            )));
        };
        let Some(graph) = krate.build_decl_graph(&self.tables, src.def_id, &stamp, &declared)
        else {
            return Err(LowerError::Unsupported(format!(
                "{}: body did not lower",
                fd.item_meta.name_path()
            )));
        };
        krate.declare_queued_specs(&self.tables);
        Ok(header.into_semantic(LazyGraph::built(graph)))
    }
}

impl ProvidedCrate {
    /// Declare the clause specializations the bodies built so far queued,
    /// in queue order.
    fn declare_queued_specs(self: &Rc<Self>, tables: &Rc<ProviderTables>) {
        while let Some(req) = self.lowering(tables, |lowering| lowering.pop_spec()) {
            if let Some(spec) = self.declare_spec(tables, req) {
                tables.declarations.push(spec);
            }
        }
    }

    /// The funcobj of the clause specialization `req`, its graph unbuilt.
    /// `None`, recorded, when its body does not substitute.
    fn declare_spec(
        self: &Rc<Self>,
        tables: &Rc<ProviderTables>,
        req: crate::front::clause_spec::SpecRequest,
    ) -> Option<DeclaredFuncObj> {
        self.lowering(tables, |lowering| {
            let spec = lowering.declare_spec(req)?;
            let stamp = spec.header.graph_stamp();
            let declared = Rc::new(lowering.spec_header_graph(&spec));
            let (krate, tables, body) = (self.clone(), tables.clone(), spec.body.clone());
            let graph = LazyGraph::deferred(declared.clone(), move || {
                let graph = krate.lowering(&tables, |lowering| lowering.build_spec_body(&body))?;
                krate.declare_queued_specs(&tables);
                let mut graph = stamp_declared(&stamp, graph, &declared);
                // A spec that folded the const has no sentinel and stays
                // look-inside. One that still passes it is the template.
                mir::stamp_unresolved_trait_const_residual(&mut graph);
                Some(graph)
            });
            Some(spec.into_declared(graph))
        })
    }

    /// Run `f` over this crate's lowering context.
    fn lowering<R>(&self, tables: &ProviderTables, f: impl FnOnce(&CrateLowering<'_>) -> R) -> R {
        tables.with_static_addrs(|static_addrs| {
            f(&CrateLowering::new(
                &self.llbc,
                static_addrs,
                &tables.jitdriver_receiver_roots,
                &self.state,
            ))
        })
    }

    /// Build the body of the declaration `def_id` and stamp its header on
    /// it. `None`, recorded as the declaration's failure, when it does not
    /// lower.
    fn build_decl_graph(
        &self,
        tables: &ProviderTables,
        def_id: u64,
        stamp: &GraphStamp,
        declared: &FunctionGraph,
    ) -> Option<FunctionGraph> {
        let fd = self.llbc.fn_by_id(def_id)?;
        self.lowering(tables, |lowering| match lowering.build_decl_body(fd) {
            Ok(graph) => {
                // The header stamp was taken before this body existed, so
                // it cannot see an unresolved `TraitConst`. Same residual
                // as `residualize_unresolved_trait_const`: `policy.py`
                // `look_inside_graph` reads the hint off the built graph.
                let mut graph = stamp_declared(stamp, graph, declared);
                mir::stamp_unresolved_trait_const_residual(&mut graph);
                Some(graph)
            }
            Err(error) => {
                lowering.record_decl_failure(fd, error);
                None
            }
        })
    }
}

/// Stamp the funcobj's header on its built body, which must keep what the
/// funcobj declared.
fn stamp_declared(
    stamp: &GraphStamp,
    graph: FunctionGraph,
    declared: &FunctionGraph,
) -> FunctionGraph {
    let graph = stamp.apply(graph);
    assert_eq!(
        declaration(&graph),
        declaration(declared),
        "the built body of {} departs from its declaration",
        graph.name
    );
    graph
}

/// The part of a graph its declaration fixes: identity, `FUNC.RESULT`,
/// and the startblock's parameters with their declared types.
fn declaration(
    graph: &FunctionGraph,
) -> (
    GraphKey,
    Option<String>,
    Vec<String>,
    usize,
    Vec<(String, crate::model::ValueType, Option<String>)>,
) {
    let startblock = graph.block(graph.startblock);
    let inputs = startblock
        .operations
        .iter()
        .filter_map(|op| match &op.kind {
            crate::model::OpKind::Input {
                name,
                ty,
                class_root,
            } => {
                let index = startblock
                    .inputargs
                    .iter()
                    .position(|arg| op.result.as_ref() == Some(arg))?;
                Some((format!("{index}:{name}"), ty.clone(), class_root.clone()))
            }
            _ => None,
        })
        .collect();
    (
        graph.graph_key(),
        graph.return_type.clone(),
        graph.hints.clone(),
        startblock.inputargs.len(),
        inputs,
    )
}

impl ProviderTables {
    /// Run `f` with the borrowed [`crate::HostStaticAddrs`] view of the
    /// owned tables.
    fn with_static_addrs<R>(&self, f: impl FnOnce(crate::HostStaticAddrs<'_>) -> R) -> R {
        let pytypes = borrowed(&self.pytypes);
        let pytypes_by_struct = borrowed(&self.pytypes_by_struct);
        let refs = borrowed(&self.refs);
        let int_values = borrowed(&self.int_values);
        let carrier = &self.error_carrier;
        let carrier_wrappers = borrowed_segments(&carrier.carrier_wrappers);
        let scalar_field_stores: Vec<crate::ScalarFieldStore<'_>> = self
            .scalar_field_stores
            .iter()
            .map(OwnedScalarFieldStore::borrowed)
            .collect();
        let to_exc_object = carrier.to_exc_object.as_deref().map(borrowed_segments);
        f(crate::HostStaticAddrs {
            pytypes: &pytypes,
            pytypes_by_struct: &pytypes_by_struct,
            refs: &refs,
            int_values: &int_values,
            error_carrier: crate::ErrorCarrierSpec {
                carrier_path: &carrier.carrier_path,
                carrier_class: &carrier.carrier_class,
                carrier_wrappers: &carrier_wrappers,
                to_exc_object: to_exc_object.as_deref(),
                from_exc_object: carrier
                    .from_exc_object
                    .as_ref()
                    .map(|(receiver, method)| (receiver.as_str(), method.as_str())),
            },
            scalar_field_stores: &scalar_field_stores,
        })
    }
}

fn borrowed(rows: &[(String, i64)]) -> Vec<(&str, i64)> {
    rows.iter().map(|(k, v)| (k.as_str(), *v)).collect()
}

fn borrowed_segments(segments: &[String]) -> Vec<&str> {
    segments.iter().map(String::as_str).collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::model::FunctionGraph;

    const CORPUS: &str = crate::runtime_names::artifacts::CHARON_CORPUS_ULLBC;

    /// Rewrite every `id: N` in `s` to the position at which that id was
    /// first seen across the graph, so two lowerings that differ only in
    /// which range of the process-global `NEXT_VAR_ID` they drew from
    /// render the same.  Aliasing is preserved: one id maps to one slot,
    /// so "these two operands are the same variable" still shows up.
    fn normalize_ids(s: &str, seen: &mut HashMap<u64, usize>) -> String {
        const KEY: &str = "id: ";
        let mut out = String::with_capacity(s.len());
        let mut rest = s;
        while let Some(at) = rest.find(KEY) {
            let (before, after) = rest.split_at(at + KEY.len());
            out.push_str(before);
            let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
            match digits.parse::<u64>() {
                Ok(id) => {
                    let next = seen.len();
                    let slot = *seen.entry(id).or_insert(next);
                    out.push_str(&format!("#{slot}"));
                    rest = &after[digits.len()..];
                }
                Err(_) => rest = after,
            }
        }
        out.push_str(rest);
        out
    }

    /// An id-independent shape of a lowered graph.  Variables carry ids
    /// from process-global counters, so two lowerings of one funcobj are
    /// never `==`; what must match is the structure the codewriter reads.
    fn shape(
        g: &FunctionGraph,
    ) -> (
        String,
        Option<String>,
        usize,
        Vec<(usize, usize)>,
        Vec<String>,
    ) {
        let mut seen: HashMap<u64, usize> = HashMap::new();
        let mut ops = Vec::new();
        let mut blocks = Vec::new();
        for b in &g.blocks {
            blocks.push((b.inputargs.len(), b.operations.len()));
            for arg in &b.inputargs {
                ops.push(normalize_ids(&format!("inputarg {arg:?}"), &mut seen));
            }
            for op in &b.operations {
                ops.push(normalize_ids(&format!("{:?}", op.kind), &mut seen));
            }
        }
        (
            g.name.clone(),
            g.return_type.clone(),
            g.blocks.len(),
            blocks,
            ops,
        )
    }

    /// A body built through the provider is the body the whole-program
    /// loop built: same graph shape, from the same `FunDecl`, for every
    /// funcobj in the corpus that lowers at all.
    ///
    /// Also asserts every lowered funcobj's name path is unique in the
    /// corpus: `source_for_name_path` resolves an ambiguous name to
    /// `None`, so a duplicate surfaces here as a lookup miss.
    #[test]
    fn provider_reproduces_the_eagerly_lowered_body() {
        let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");
        let gc_struct_ids = mir::harvest_declared_gc_facts(&llbc).gc_struct_ids();
        let mut provider = GraphBodyProvider::new(
            crate::HostStaticAddrs::default(),
            &[],
            HashMap::new(),
            FuncObjDeclarations::default(),
            Default::default(),
        );
        let program = provider.lower_prelinked_crate(
            llbc,
            &[],
            &HashSet::new(),
            &gc_struct_ids,
            &HashSet::new(),
        );
        let mut compared = 0;
        for f in &program.functions {
            let Some(fd) = f
                .fun_decl_id
                .and_then(|id| provider.crates[0].llbc.fn_by_id(id))
            else {
                continue;
            };
            let name_path = fd.item_meta.name_path();
            let src = provider
                .source_for_name_path(&name_path)
                .unwrap_or_else(|| panic!("no unique GraphBodySource for {name_path}"));
            let got = provider
                .build(src)
                .unwrap_or_else(|e| panic!("provider failed to build {name_path}: {e}"));
            // A clause specialization shares its generic's `FunDecl` under
            // its own name.
            if got.name != f.name {
                continue;
            }
            assert_eq!(shape(got.graph()), shape(f.graph()), "{name_path}");
            compared += 1;
        }
        assert!(compared > 0, "corpus fixture lowered no bodies at all");
    }

    /// A funcobj's harvested `_jit_*_` attributes land on it and on its
    /// graph when the header is built, with no pass over the program
    /// afterwards.
    #[test]
    fn a_harvested_hint_is_stamped_as_the_funcobj_is_built() {
        let fn_path = |f: &SemanticFunction| {
            if f.module_path.is_empty() {
                f.name.clone()
            } else {
                format!("{}::{}", f.module_path, f.name)
            }
        };
        let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");
        let gc_struct_ids = mir::harvest_declared_gc_facts(&llbc).gc_struct_ids();
        let unhinted = GraphBodyProvider::new(
            crate::HostStaticAddrs::default(),
            &[],
            HashMap::new(),
            FuncObjDeclarations::default(),
            Default::default(),
        )
        .lower_prelinked_crate(llbc, &[], &HashSet::new(), &gc_struct_ids, &HashSet::new());
        let target = unhinted
            .functions
            .iter()
            .find(|f| f.hints.is_empty())
            .map(fn_path)
            .expect("corpus has an unhinted funcobj");
        let hints = HashMap::from([(target.clone(), vec!["unroll_safe".to_string()])]);
        let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");
        let gc_struct_ids = mir::harvest_declared_gc_facts(&llbc).gc_struct_ids();
        let program = GraphBodyProvider::new(
            crate::HostStaticAddrs::default(),
            &[],
            hints,
            FuncObjDeclarations::default(),
            Default::default(),
        )
        .lower_prelinked_crate(llbc, &[], &HashSet::new(), &gc_struct_ids, &HashSet::new());
        let f = program
            .functions
            .iter()
            .find(|f| fn_path(f) == target)
            .expect("the hinted funcobj still lowers");
        assert_eq!(f.hints, vec!["unroll_safe".to_string()]);
        assert!(f.graph().hints.iter().any(|h| h == "unroll_safe"));
    }

    /// The dispatcher builds through `GraphBodyProvider`, which snapshots
    /// the header before the body exists. `malloc_typed_stable` still
    /// passes the unresolved `TraitConst` sentinel, so the built graph
    /// carries `dont_look_inside` and `policy.py` `look_inside_graph`
    /// declines it. `malloc_typed` mentions `T::SIZE` only inside
    /// `debug_assert`, so that operand does not survive and the graph
    /// stays look-inside. The header's `SemanticFunction.hints` is the
    /// snapshot, not the hint the codewriter reads.
    #[test]
    fn deferred_malloc_template_with_a_clause_size_is_not_look_inside() {
        let llbc = Llbc::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../target/llbc/pyre-object.ullbc"
        ))
        .expect("pyre-object.ullbc is already extracted");
        let mut provider = GraphBodyProvider::new(
            crate::HostStaticAddrs::default(),
            &[],
            HashMap::new(),
            FuncObjDeclarations::default(),
            Default::default(),
        );
        let program = provider.lower_named_for_test(
            llbc,
            &[
                "malloc_typed",
                "malloc_typed_managed",
                "malloc_typed_stable",
            ],
        );
        let function = |name: &str| {
            program
                .functions
                .iter()
                .find(|function| function.name == name)
                .unwrap_or_else(|| panic!("missing {name}"))
        };
        let calls_sentinel = |name: &str| {
            function(name).graph().blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        crate::model::OpKind::Call {
                            target: crate::model::CallTarget::FunctionPath { segments, .. },
                            ..
                        } if segments.len() == 2
                            && segments[0] == "__str_const"
                            && segments[1] == "__trait_const"
                    )
                })
            })
        };
        for name in ["malloc_typed_managed", "malloc_typed_stable"] {
            assert!(
                calls_sentinel(name),
                "{name} dropped the unresolved TraitConst"
            );
            assert!(
                function(name)
                    .graph()
                    .hints
                    .iter()
                    .any(|hint| hint == "dont_look_inside"),
                "{name} graph hints {:?}",
                function(name).graph().hints
            );
        }
        assert!(
            !calls_sentinel("malloc_typed"),
            "malloc_typed kept a TraitConst the debug_assert dropped"
        );
        assert!(
            !function("malloc_typed")
                .graph()
                .hints
                .iter()
                .any(|hint| hint == "dont_look_inside"),
            "malloc_typed graph hints {:?}",
            function("malloc_typed").graph().hints
        );
        // `build` is the demand entry. It must stamp the same header and
        // the same body the deferred closure did, including a template
        // whose `dont_look_inside` exists only on the built graph.
        for name in ["malloc_typed", "malloc_typed_stable"] {
            let deferred = function(name);
            let src = GraphBodySource {
                llbc_index: 0,
                def_id: deferred
                    .fun_decl_id
                    .unwrap_or_else(|| panic!("{name} has no FunDecl")),
            };
            let demanded = provider
                .build(src)
                .unwrap_or_else(|e| panic!("build {name}: {e}"));
            assert_eq!(demanded.hints, deferred.hints, "{name} header hints");
            assert_eq!(
                demanded.graph().hints,
                deferred.graph().hints,
                "{name} graph hints"
            );
        }
    }
}
