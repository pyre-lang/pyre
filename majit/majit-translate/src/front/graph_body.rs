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
    /// Crates already in the provider when this one was created:
    /// whole-program graph visibility the per-crate extraction splits
    /// apart.
    prior: Vec<Rc<ProvidedCrate>>,
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
        let prior = self.crates.clone();
        let krate = Rc::new(ProvidedCrate { llbc, state, prior });
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
        let prior = self.crates.clone();
        let krate = Rc::new(ProvidedCrate { llbc, state, prior });
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
            let prior_fat_dyn_callee =
                |name_path: &str| self.lower_prior_fat_dyn_callee(tables, name_path);
            f(&CrateLowering::new(
                &self.llbc,
                static_addrs,
                &tables.jitdriver_receiver_roots,
                &self.state,
                Some(&prior_fat_dyn_callee),
            ))
        })
    }

    /// Lower a fat-`&dyn` callee whose body lives in a crate already in
    /// `prior` — whole-program graph visibility the per-crate extraction
    /// splits apart. An ambiguous `name_path` in one crate answers
    /// `None`, as [`GraphBodyProvider::source_for_name_path`] does.
    fn lower_prior_fat_dyn_callee(
        &self,
        tables: &ProviderTables,
        name_path: &str,
    ) -> Option<FunctionGraph> {
        for prior in self.prior.iter().rev() {
            let mut found = None;
            for fd in prior.llbc.iter_local_fns() {
                if fd.item_meta.name_path() != name_path || !fd.has_unstructured_body() {
                    continue;
                }
                if found.is_some() {
                    return None;
                }
                found = Some(fd.def_id);
            }
            let Some(def_id) = found else {
                continue;
            };
            return prior.lowering(tables, |l| l.lower_fat_dyn_callee_local(def_id));
        }
        None
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
    use std::collections::{HashMap, HashSet};

    use super::*;
    use crate::flowspace::model::Variable;
    use crate::model::{
        BlockId, CallTarget, ConcreteType, FunctionGraph, OpKind, SpaceOperation, VecFieldPart,
    };

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

    const OBJECT_LLBC: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    );
    const INTERP_LLBC: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );

    fn load_llbc_if_present(path: &str) -> Option<Llbc> {
        if !std::path::Path::new(path).is_file() {
            eprintln!("skipping: {path} is missing; run `python3 scripts/extract-llbc.py`");
            return None;
        }
        Some(Llbc::load(path).unwrap_or_else(|err| panic!("load {path}: {err}")))
    }

    fn call_names_get_strategy(target: &CallTarget) -> bool {
        match target {
            CallTarget::FunctionPath { segments, .. } => segments
                .iter()
                .any(|segment| segment == "w_dict_get_strategy"),
            CallTarget::Method { name, .. } => name == "w_dict_get_strategy",
            _ => false,
        }
    }

    fn assert_strategy_getter_spliced(graph: &FunctionGraph) {
        let still_calls_strategy = graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                let OpKind::Call { target, .. } = &op.kind else {
                    return false;
                };
                call_names_get_strategy(target)
            })
        });
        assert!(
            !still_calls_strategy,
            "w_dict_get_strategy stayed a call\n{}",
            graph.dump()
        );
        let has_imp_fat = graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| match &op.kind {
                OpKind::FieldRead { field, .. } => {
                    field.name == "imp"
                        && matches!(
                            field.vec_part,
                            Some(
                                VecFieldPart::FatVtable
                                    | VecFieldPart::FatLen
                                    | VecFieldPart::FatData
                            )
                        )
                }
                _ => false,
            })
        });
        assert!(
            has_imp_fat,
            "no FieldRead of imp tagged FatVtable/FatLen/FatData\n{}",
            graph.dump()
        );
    }

    fn program_function<'a>(
        program: &'a SemanticProgram,
        module_path: &str,
        name: &str,
    ) -> &'a SemanticFunction {
        program
            .functions
            .iter()
            .find(|function| function.module_path == module_path && function.name == name)
            .unwrap_or_else(|| {
                let names: Vec<_> = program
                    .functions
                    .iter()
                    .map(|function| format!("{}::{}", function.module_path, function.name))
                    .collect();
                panic!("missing {module_path}::{name} in {names:?}")
            })
    }

    /// Interpreter callers of a pyre-object fat `&dyn` return splice the
    /// earlier crate's body so `DictStrategyRef.imp` is two field reads.
    #[test]
    fn finditem_str_named_splices_the_cross_crate_strategy_getter() {
        let Some(object_llbc) = load_llbc_if_present(OBJECT_LLBC) else {
            return;
        };
        let Some(interp_llbc) = load_llbc_if_present(INTERP_LLBC) else {
            return;
        };
        let mut provider = GraphBodyProvider::new(
            crate::HostStaticAddrs::default(),
            &[],
            HashMap::new(),
            FuncObjDeclarations::default(),
            Default::default(),
        );
        provider.lower_named_for_test(object_llbc, &["w_dict_get_strategy"]);
        let program =
            provider.lower_named_for_test(interp_llbc, &["finditem_str_named", "view_as_kwargs"]);
        for name in ["finditem_str_named", "view_as_kwargs"] {
            let function = program_function(&program, "baseobjspace", name);
            let graph = function.lazy_graph().get().unwrap_or_else(|| {
                panic!(
                    "{}::{} body did not lower",
                    function.module_path, function.name
                )
            });
            assert_strategy_getter_spliced(graph);
        }
    }

    fn producer(graph: &FunctionGraph, var_id: u64) -> Option<(BlockId, &SpaceOperation)> {
        graph.blocks.iter().find_map(|block| {
            block
                .operations
                .iter()
                .find(|op| {
                    op.result
                        .as_ref()
                        .is_some_and(|result| result.id() == var_id)
                })
                .map(|op| (block.id, op))
        })
    }

    fn is_cast_op(kind: &OpKind) -> Option<&Variable> {
        match kind {
            OpKind::UnaryOp { op, operand, .. }
                if op == "same_as" || op == "cast_int_to_ptr" || op == "cast_ptr_to_int" =>
            {
                Some(operand)
            }
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if segments.last().is_some_and(|s| {
                s == "cast_int_to_ptr" || s == "cast_ptr_to_int" || s == "same_as"
            }) =>
            {
                args.first().and_then(|arg| arg.as_variable())
            }
            _ => None,
        }
    }

    fn walk_imp_vec_part(
        graph: &FunctionGraph,
        block: BlockId,
        var: &Variable,
        want: &[VecFieldPart],
        seen: &mut HashSet<(usize, u64)>,
    ) -> bool {
        if !seen.insert((block.0, var.id())) {
            return false;
        }
        if let Some((def_block, op)) = producer(graph, var.id()) {
            return match &op.kind {
                OpKind::FieldRead { field, .. }
                    if field.name == "imp"
                        && field.vec_part.is_some_and(|part| want.contains(&part)) =>
                {
                    true
                }
                kind if is_cast_op(kind).is_some() => {
                    walk_imp_vec_part(graph, def_block, is_cast_op(kind).unwrap(), want, seen)
                }
                _ => false,
            };
        }
        let Some(block_ref) = graph.blocks.get(block.0) else {
            return false;
        };
        let preds = graph.predecessors(block);
        if preds.is_empty() {
            return false;
        }
        let index = block_ref.inputargs.iter().position(|arg| arg == var);
        preds.iter().all(|pred| {
            let Some(pred_block) = graph.blocks.get(pred.0) else {
                return false;
            };
            let incoming: Vec<_> = pred_block
                .exits
                .iter()
                .filter(|link| link.target == block)
                .collect();
            if incoming.is_empty() {
                return false;
            }
            incoming.iter().all(|link| {
                let src = if let Some(index) = index {
                    link.args.get(index).and_then(|arg| arg.as_variable())
                } else {
                    Some(var)
                };
                src.is_some_and(|src| walk_imp_vec_part(graph, *pred, src, want, seen))
            })
        })
    }

    fn assert_vtable_slot_indirect_call(
        graph: &FunctionGraph,
        trait_root: &str,
        method_name: &str,
    ) {
        let field_name = format!("method_{method_name}");
        let expected_key = Some((trait_root.to_string(), method_name.to_string()));
        let mut hits = 0usize;
        for block in &graph.blocks {
            for op in &block.operations {
                let OpKind::IndirectCall {
                    funcptr,
                    args,
                    family_key,
                    ..
                } = &op.kind
                else {
                    continue;
                };
                if family_key != &expected_key {
                    continue;
                }
                hits += 1;
                let Some((_, funcptr_op)) = producer(graph, funcptr.id()) else {
                    panic!(
                        "{trait_root}::{method_name} funcptr has no producer\n{}",
                        graph.dump()
                    );
                };
                let OpKind::FieldRead { base, field, .. } = &funcptr_op.kind else {
                    panic!(
                        "{trait_root}::{method_name} funcptr is not a FieldRead: {:?}\n{}",
                        funcptr_op.kind,
                        graph.dump()
                    );
                };
                assert_eq!(
                    field.name, field_name,
                    "{trait_root}::{method_name} funcptr field"
                );
                assert_eq!(
                    field.owner_root.as_deref(),
                    Some("dictmultiobject::DictStrategy::{vtable}"),
                    "{trait_root}::{method_name} vtable owner_root"
                );
                let mut seen = HashSet::new();
                assert!(
                    walk_imp_vec_part(
                        graph,
                        block.id,
                        base,
                        &[VecFieldPart::FatLen, VecFieldPart::FatMeta],
                        &mut seen
                    ),
                    "{field_name} base does not reach FatLen/FatMeta of imp\n{}",
                    graph.dump()
                );
                let receiver = args.first().unwrap_or_else(|| {
                    panic!("{trait_root}::{method_name} IndirectCall has no args[0]")
                });
                let mut seen = HashSet::new();
                assert!(
                    walk_imp_vec_part(
                        graph,
                        block.id,
                        receiver,
                        &[VecFieldPart::FatData],
                        &mut seen
                    ),
                    "{trait_root}::{method_name} args[0] does not reach FatData of imp\n{}",
                    graph.dump()
                );
                assert_eq!(
                    FunctionGraph::concretetype_of(base),
                    ConcreteType::Signed,
                    "vtable word must be an int so getfield_raw_i loads it"
                );
            }
        }
        assert!(
            hits > 0,
            "no IndirectCall family_key=({trait_root}, {method_name})\n{}",
            graph.dump()
        );
    }

    /// Path B: a bodyless `dyn DictStrategy` method lowers to the same
    /// `{vtable}.method_*` FieldRead plus `IndirectCall { family_key }` as
    /// Path A's inline vtable-slot dispatch.
    #[test]
    fn finditem_str_named_calls_getitem_str_through_the_vtable_slot() {
        let Some(object_llbc) = load_llbc_if_present(OBJECT_LLBC) else {
            return;
        };
        let Some(interp_llbc) = load_llbc_if_present(INTERP_LLBC) else {
            return;
        };
        let mut provider = GraphBodyProvider::new(
            crate::HostStaticAddrs::default(),
            &[],
            HashMap::new(),
            FuncObjDeclarations::default(),
            Default::default(),
        );
        provider.lower_named_for_test(object_llbc, &["w_dict_get_strategy"]);
        let program =
            provider.lower_named_for_test(interp_llbc, &["finditem_str_named", "view_as_kwargs"]);
        let finditem = program_function(&program, "baseobjspace", "finditem_str_named");
        let finditem_graph = finditem.lazy_graph().get().unwrap_or_else(|| {
            panic!(
                "{}::{} body did not lower",
                finditem.module_path, finditem.name
            )
        });
        assert_vtable_slot_indirect_call(finditem_graph, "DictStrategy", "getitem_str");
        let view = program_function(&program, "baseobjspace", "view_as_kwargs");
        let view_graph = view
            .lazy_graph()
            .get()
            .unwrap_or_else(|| panic!("{}::{} body did not lower", view.module_path, view.name));
        assert_vtable_slot_indirect_call(view_graph, "DictStrategy", "view_as_kwargs");
    }

    fn impls_with_owner_leaf<'a>(
        program: &'a SemanticProgram,
        method: &str,
        owner_leaf: &str,
    ) -> Vec<&'a SemanticFunction> {
        program
            .functions
            .iter()
            .filter(|function| {
                function.name == method
                    && function
                        .self_ty_root
                        .as_deref()
                        .is_some_and(|owner| owner.rsplit("::").next() == Some(owner_leaf))
            })
            .collect()
    }

    fn require_impl<'a>(
        program: &'a SemanticProgram,
        method: &str,
        owner_leaf: &str,
    ) -> &'a SemanticFunction {
        let impls = impls_with_owner_leaf(program, method, owner_leaf);
        let trait_impls: Vec<_> = impls
            .iter()
            .copied()
            .filter(|function| function.trait_root.is_some())
            .collect();
        match trait_impls.as_slice() {
            [one] => *one,
            [] => panic!(
                "no trait-impl {owner_leaf}::{method}; candidates: {:?}",
                impls
                    .iter()
                    .map(|function| (
                        function.self_ty_root.as_deref(),
                        function.trait_root.as_deref(),
                        function.trait_qualified.as_deref(),
                        function.trait_impl_identity.as_deref()
                    ))
                    .collect::<Vec<_>>()
            ),
            _ => panic!(
                "ambiguous trait-impl {owner_leaf}::{method}: {} matches",
                trait_impls.len()
            ),
        }
    }

    fn family_mentions(family: &[crate::parse::CallPath], needle: &str) -> bool {
        family
            .iter()
            .any(|path| path.segments.iter().any(|segment| segment.contains(needle)))
    }

    /// `lib.rs` family insert (`rpbc.py FunctionReprBase.call` /
    /// `row_of_graphs.values()`): every `(self_ty_root, trait_root)` pair.
    fn register_trait_families_like_production(
        program: &SemanticProgram,
        call_control: &mut crate::call::CallControl,
    ) {
        for func in &program.functions {
            let (Some(owner), Some(trait_leaf)) = (&func.self_ty_root, &func.trait_root) else {
                continue;
            };
            let path = match func.trait_impl_identity.as_deref() {
                Some(identity) => crate::parse::CallPath::from_segments(
                    crate::front::mir::trait_impl_identity_path(owner, identity, &func.name),
                ),
                None => crate::parse::CallPath::for_impl_method(owner, &func.name),
            };
            call_control.register_trait_family_path(&func.name, trait_leaf, path);
        }
        for func in &program.functions {
            let (None, Some(trait_leaf)) = (&func.self_ty_root, &func.trait_root) else {
                continue;
            };
            let impl_type = format!("<default methods of {trait_leaf}>");
            call_control.register_trait_family_member(&func.name, trait_leaf, &impl_type);
        }
    }

    /// Interpreter-crate overrides of a pyre-object trait join the same
    /// PBC family the object-crate impls and the vtable call sites use.
    /// Charon `item_meta.is_local` is the extraction crate, not the
    /// translated-program boundary (`local_crates::is_local_crate_root`).
    #[test]
    fn dict_strategy_indirect_families_include_cross_crate_overrides() {
        let Some(object_llbc) = load_llbc_if_present(OBJECT_LLBC) else {
            return;
        };
        let Some(interp_llbc) = load_llbc_if_present(INTERP_LLBC) else {
            return;
        };
        let object_crate = object_llbc.crate_name().to_string();
        let interp_crate = interp_llbc.crate_name().to_string();
        crate::local_crates::with_local_crate_root(&object_crate, || {
            crate::local_crates::with_local_crate_root(&interp_crate, || {
                let skipped: mir::LoweringSkips = Default::default();
                let mut provider = GraphBodyProvider::new(
                    crate::HostStaticAddrs::default(),
                    &[],
                    HashMap::new(),
                    FuncObjDeclarations::default(),
                    skipped.clone(),
                );
                let object_program =
                    provider.lower_named_for_test(object_llbc, &["getitem_str", "view_as_kwargs"]);
                let interp_program =
                    provider.lower_named_for_test(interp_llbc, &["getitem_str", "view_as_kwargs"]);

                let object_get = require_impl(&object_program, "getitem_str", "ObjectDictStrategy");
                let unicode_get =
                    require_impl(&object_program, "getitem_str", "UnicodeDictStrategy");
                let module_get = require_impl(&object_program, "getitem_str", "ModuleDictStrategy");
                let mapdict_get = require_impl(&interp_program, "getitem_str", "MapDictStrategy");
                let classdict_get =
                    require_impl(&interp_program, "getitem_str", "ClassDictStrategySlot");
                let unicode_kwargs =
                    require_impl(&object_program, "view_as_kwargs", "UnicodeDictStrategy");
                let default_get = object_program
                    .functions
                    .iter()
                    .find(|function| {
                        function.name == "getitem_str"
                            && function.self_ty_root.is_none()
                            && function.trait_root.as_deref() == Some("DictStrategy")
                    })
                    .expect("DictStrategy default getitem_str");

                let trait_root = object_get.trait_root.as_deref();
                let trait_qualified = object_get.trait_qualified.as_deref();
                assert_eq!(
                    trait_root,
                    Some("DictStrategy"),
                    "object-side trait_root={trait_root:?} trait_qualified={trait_qualified:?}"
                );
                for (label, function) in [
                    ("ObjectDictStrategy", object_get),
                    ("UnicodeDictStrategy", unicode_get),
                    ("ModuleDictStrategy", module_get),
                    ("MapDictStrategy", mapdict_get),
                    ("ClassDictStrategySlot", classdict_get),
                    ("UnicodeDictStrategy.view_as_kwargs", unicode_kwargs),
                ] {
                    assert_eq!(
                        (
                            function.trait_root.as_deref(),
                            function.trait_qualified.as_deref()
                        ),
                        (trait_root, trait_qualified),
                        "{label} trait_root={:?} trait_qualified={:?} vs object-side {trait_root:?} {trait_qualified:?}",
                        function.trait_root,
                        function.trait_qualified
                    );
                }
                // Default bodies leave `trait_qualified` empty; the family
                // key is still the leaf `trait_root`.
                assert_eq!(default_get.trait_root.as_deref(), Some("DictStrategy"));
                assert_eq!(default_get.trait_qualified, None);

                let mut call_control = crate::call::CallControl::new();
                register_trait_families_like_production(&object_program, &mut call_control);
                register_trait_families_like_production(&interp_program, &mut call_control);
                let family_get = call_control.all_impls_for_indirect("DictStrategy", "getitem_str");
                let family_kw =
                    call_control.all_impls_for_indirect("DictStrategy", "view_as_kwargs");
                for needle in [
                    "MapDictStrategy",
                    "ClassDictStrategySlot",
                    "ObjectDictStrategy",
                    "UnicodeDictStrategy",
                    "ModuleDictStrategy",
                    "<default methods of DictStrategy>",
                ] {
                    assert!(
                        family_mentions(&family_get, needle),
                        "getitem_str family missing {needle}: {family_get:?}"
                    );
                }
                assert!(
                    family_mentions(&family_kw, "UnicodeDictStrategy"),
                    "view_as_kwargs family missing UnicodeDictStrategy: {family_kw:?}"
                );

                let unicode_lowered = unicode_kwargs.lazy_graph().get();
                let skips = skipped.borrow();
                let unicode_skips: Vec<_> = skips
                    .iter()
                    .filter(|(name, _)| name.contains("view_as_kwargs"))
                    .cloned()
                    .collect();
                eprintln!(
                    "UnicodeDictStrategy::view_as_kwargs lowered={} DeclBuildError skips={unicode_skips:?}",
                    unicode_lowered.is_some()
                );
            })
        });
    }
}
