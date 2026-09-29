//! A build-time jitcode table and the one descr pool its bodies index.
//!
//! `CodeWriter.make_jitcodes()` (codewriter.py) produces two lists that
//! only mean anything together: `all_jitcodes`, and the single shared
//! `Assembler.descrs` (assembler.py) that every `d`/`j` argcode in every
//! body indexes. A host that runs the codewriter at build time and embeds the
//! two serialized lists in its binary has to join them back into the runtime
//! shapes — `Arc<JitCode>` shells and a `RuntimeBhDescr` pool. This type is
//! that join, and it is the only place the two lists meet.
//!
//! **What the join has to preserve.** A `BhDescr::JitCode` slot names its
//! callee by an `all_jitcodes` index, and the object that index resolves to
//! must be *the* object the table holds at it — `codewriter.py
//! all_jitcodes[jitcode.index] is jitcode`, an identity, not an equality.
//! Minting a second shell per pool slot satisfies every read that only wants a
//! body and breaks every read that asks whether two references name the same
//! jitcode: an `Arc::ptr_eq` dedup while flattening a registry sees one callee
//! twice, and an index stamped on one shell is read back off the other.
//!
//! **Why there is no cycle to break.** The shells carry an EMPTY per-jitcode
//! `exec.descrs` and resolve every operand through this pool as the
//! process-global fallback ([`JitCode::descr_at`],
//! [`init_global_build_descr_pool`]). So the pool holds jitcodes and the
//! jitcodes hold nothing — a `BC_INLINE_CALL` chain of any depth resolves
//! against one table. Copying the pool into each shell instead makes the depth
//! the copy was taken at the depth that resolves: the callees inside a copied
//! pool are shells of their own, and whatever pool they were given is the one
//! their own operands read.

use std::sync::Arc;

use super::{CanonicalBhDescr, CanonicalJitCode, JitCode, RuntimeBhDescr, RuntimeDescrTable};

/// A materialized build-time `all_jitcodes` + `Assembler.descrs` pair.
///
/// Process-lifetime by construction: [`Self::materialize`] leaks both lists,
/// because the pool is installed as the global `descr_at` fallback, whose
/// entries are handed out as `&'static`.
pub struct EmbeddedJitCodeTable {
    jitcodes: &'static [Arc<JitCode>],
    descrs: &'static [RuntimeBhDescr],
}

/// The half of the impls below that does not need arguing. Both payloads
/// `materialize` actually stores are thread-safe on their own terms —
/// `Box<CanonicalBhDescr>` structurally, `Arc<JitCode>` through `JitCode`'s own
/// manual impls — so a payload that stops being either fails the build instead
/// of quietly hollowing out a comment.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Box<CanonicalBhDescr>>();
    assert_send_sync::<Arc<JitCode>>();
};

// SAFETY: only the `descrs` field needs these; by the assertion above the
// `jitcodes` field is already `Send + Sync` on its own. `RuntimeBhDescr` is
// neither, because `AssemblerToken`'s `JitCallAssemblerTarget` holds a
// `*const ()`. `materialize` is the sole writer of both lists and builds only
// the `Descr` and `JitCode` variants, so neither raw-pointer variant — that one
// and `Call`'s `JitCallTarget` — is ever constructed here: there is no pointer
// in either list for this type to hand across a thread. A future arm that mints
// one invalidates this.
unsafe impl Send for EmbeddedJitCodeTable {}
unsafe impl Sync for EmbeddedJitCodeTable {}

impl EmbeddedJitCodeTable {
    /// Codewriter support functions shared by every embedded interpreter.
    /// `support.py _ll_2_int_floordiv/_ll_2_int_mod` use truncating word arithmetic.
    pub fn builtin_fnaddrs() -> [(&'static str, i64); 2] {
        let div = crate::blackhole::_ll_2_int_floordiv as *const () as usize as i64;
        let rem = crate::blackhole::_ll_2_int_mod as *const () as usize as i64;
        [("_ll_2_int_floordiv", div), ("_ll_2_int_mod", rem)]
    }

    /// Restore an embedded image's complete effect-descriptor population.
    ///
    /// `pyjitpl.py::finish_setup_descrs` uses all of GcCache, not merely the
    /// opcode descriptor table. Publish opcode-owned layouts first, then the
    /// analyzer-only members and their frozen `compute_bitstrings` stamps.
    pub fn materialize_with_frozen_effects(
        canonical: &[Arc<CanonicalJitCode>],
        mut descrs: Vec<CanonicalBhDescr>,
        symbolic_paths: &[(i64, String)],
        runtime_bindings: &[(&str, i64)],
        effect_mints: &[majit_ir::effectinfo::DescrMintEntry],
    ) -> &'static Self {
        for descr in &descrs {
            if matches!(descr, CanonicalBhDescr::Field { .. }) {
                crate::pyjitpl::dispatch::field_descr_ref_from_bh(descr);
            }
        }
        super::frozen_effects::publish_mints(effect_mints);
        for descr in &mut descrs {
            if let CanonicalBhDescr::Call { calldescr }
            | CanonicalBhDescr::JitCode { calldescr, .. } = descr
            {
                super::frozen_effects::prepare(&mut calldescr.extra_info);
            }
        }
        let canonical: Vec<_> = canonical
            .iter()
            .map(|code| {
                let mut code = (**code).clone();
                if let Some(calldescr) = code.calldescr.get_mut() {
                    super::frozen_effects::prepare(&mut calldescr.extra_info);
                }
                Arc::new(code)
            })
            .collect();
        Self::materialize_with_symbolic_fnaddrs(
            &canonical,
            descrs,
            symbolic_paths,
            runtime_bindings,
        )
    }

    /// Join the two serialized lists into runtime shells and their pool.
    ///
    /// `canonical` must be `all_jitcodes` in allocation order, which
    /// `codewriter.py jitcode.index = index` makes the same thing as
    /// "indexed by `jitcode.index`" — asserted here, since every `j` operand
    /// in the pool is an index into it and a list that drifted from its own
    /// indices resolves silently to the wrong callee.
    pub fn materialize(
        canonical: &[Arc<CanonicalJitCode>],
        descrs: Vec<CanonicalBhDescr>,
    ) -> &'static Self {
        Self::materialize_with_replacements(canonical, descrs, &[])
    }

    /// Join a table after resolving symbolic function addresses supplied by
    /// the embedding runtime.
    ///
    /// A build-script translator cannot take callable addresses from the
    /// final process. For an unbound callee it therefore stores a stable
    /// symbolic value and records its path in `symbolic_paths`. The host owns
    /// the ABI shims for those paths; this method replaces their symbolic
    /// values in JitCode shells, constant pools, and inline-call descriptors
    /// before any runtime object is published.
    pub fn materialize_with_symbolic_fnaddrs(
        canonical: &[Arc<CanonicalJitCode>],
        descrs: Vec<CanonicalBhDescr>,
        symbolic_paths: &[(i64, String)],
        runtime_bindings: &[(&str, i64)],
    ) -> &'static Self {
        let replacements: Vec<(i64, i64)> = symbolic_paths
            .iter()
            .filter_map(|(symbolic, _)| {
                let runtime =
                    Self::rebound_symbolic_fnaddr(*symbolic, symbolic_paths, runtime_bindings);
                (*symbolic != runtime).then_some((*symbolic, runtime))
            })
            .collect();
        Self::materialize_with_replacements(canonical, descrs, &replacements)
    }

    /// Replace one build-time function address the way
    /// [`Self::materialize_with_symbolic_fnaddrs`] replaces jitcode constants.
    ///
    /// An address that is not a recorded symbolic path hash is returned
    /// unchanged. A recorded hash is replaced when `runtime_bindings` (or the
    /// shared builtin table) names its path.
    pub fn rebound_symbolic_fnaddr(
        fnaddr: i64,
        symbolic_paths: &[(i64, String)],
        runtime_bindings: &[(&str, i64)],
    ) -> i64 {
        let Some((_, path)) = symbolic_paths
            .iter()
            .find(|(symbolic, _)| *symbolic == fnaddr)
        else {
            return fnaddr;
        };
        let builtins = Self::builtin_fnaddrs();
        runtime_bindings
            .iter()
            .chain(builtins.iter())
            .find(|(binding_path, _)| *binding_path == path)
            .map(|(_, runtime)| *runtime)
            .unwrap_or(fnaddr)
    }

    fn materialize_with_replacements(
        canonical: &[Arc<CanonicalJitCode>],
        descrs: Vec<CanonicalBhDescr>,
        replacements: &[(i64, i64)],
    ) -> &'static Self {
        let jitcodes: &'static [Arc<JitCode>] = Box::leak(
            canonical
                .iter()
                .enumerate()
                .map(|(index, core)| {
                    assert_eq!(
                        core.try_index(),
                        Some(index),
                        "jitcode {:?} sits at position {index} but names index {:?}; \
                         every `j` operand indexes this list",
                        core.name,
                        core.try_index(),
                    );
                    assert!(
                        core.try_body().is_some(),
                        "embedded all_jitcodes artefact contains bodyless jitcode {:?} at index \
                         {index}; CodeWriter::make_jitcodes publishes only completed entries",
                        core.name,
                    );
                    let mut core = (**core).clone();
                    if let Some((_, runtime)) = replacements
                        .iter()
                        .find(|(symbolic, _)| *symbolic == core.fnaddr)
                    {
                        core.fnaddr = *runtime;
                    }
                    for constant in &mut core.body_mut().constants_i {
                        if let Some((_, runtime)) = replacements
                            .iter()
                            .find(|(symbolic, _)| symbolic == constant)
                        {
                            *constant = *runtime;
                        }
                    }
                    Arc::new(JitCode::from_canonical(core))
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        let pool: &'static [RuntimeBhDescr] = Box::leak(
            descrs
                .into_iter()
                .map(|descr| match descr {
                    // The callee is the table's own entry, cloned — an
                    // `Arc::clone`, so `ptr_eq` against the table holds.
                    CanonicalBhDescr::JitCode { jitcode_index, .. } => {
                        RuntimeBhDescr::JitCode(Arc::clone(&jitcodes[jitcode_index]))
                    }
                    // Every other variant is an ordinary `d` descr and carries
                    // through unchanged: the runtime pool element and the
                    // build-time one are the same type.
                    other => RuntimeBhDescr::Descr(Box::new(other)).into_resolved(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        Box::leak(Box::new(Self {
            jitcodes,
            descrs: pool,
        }))
    }

    /// `metainterp_sd.jitcodes` (warmspot.py) — the flat registry
    /// `resume.py:1338-1340` indexes by a frame's `jitcode_pos`.
    pub fn jitcodes(&self) -> &'static [Arc<JitCode>] {
        self.jitcodes
    }

    /// The shared pool, in `Assembler.descrs` order.
    pub fn descrs(&self) -> &'static [RuntimeBhDescr] {
        self.descrs
    }

    /// The jitcode an `all_jitcodes` index names.
    pub fn by_index(&self, index: usize) -> Option<&'static Arc<JitCode>> {
        self.jitcodes.get(index)
    }

    /// The jitcode a graph leaf name names, or `None`.
    ///
    /// A name is not unique — the codewriter derives it from the graph leaf,
    /// and two distinct graphs can end on the same one — so this returns the
    /// first match and is only a lookup for callers that hold a name and
    /// nothing better. Anything that can carry an index should use
    /// [`Self::by_index`], which is what the operands themselves do.
    pub fn by_name(&self, name: &str) -> Option<&'static Arc<JitCode>> {
        self.jitcodes.iter().find(|jitcode| jitcode.name() == name)
    }

    /// Install this pool as the process-global `descr_at` fallback.
    ///
    /// Idempotent through [`crate::jitcode::init_global_build_descr_pool`]: the
    /// first table wins. Until this runs, a shell's operands resolve against
    /// its own empty `exec.descrs` and every lookup returns `None`.
    pub fn install_as_global_pool(&'static self) {
        super::init_global_build_descr_pool(self);
    }
}

impl RuntimeDescrTable for EmbeddedJitCodeTable {
    fn get(&self, index: usize) -> Option<&'static RuntimeBhDescr> {
        self.descrs.get(index)
    }

    fn len(&self) -> usize {
        self.descrs.len()
    }

    fn jitcodes(&self) -> &'static [Arc<JitCode>] {
        self.jitcodes
    }

    fn jitcode_at(&self, index: usize) -> Option<Arc<JitCode>> {
        self.jitcodes.get(index).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A native helper mutating a field must invalidate the very descriptor
    /// read by a getfield opcode, after serialization has erased raw Arcs.
    #[test]
    fn frozen_effects_rejoin_the_opcode_field_after_serialization() {
        use majit_ir::effectinfo::{DescrMintEntry, DescrMintSpec, DescrSetKeys, DescrSetMember};
        use majit_ir::{EffectInfo, ExtraEffect, Type};
        use majit_jitcode::jitcode::{BhCallDescr, BhFieldSpec, BhSizeSpec};
        let member = DescrSetMember::Field {
            struct_id: 0x475241494e454649,
            field_name: "depth".into(),
        };
        let mut writer =
            EffectInfo::const_new(ExtraEffect::CannotRaise, majit_ir::OopSpecIndex::None);
        writer.descr_set_keys = Some(
            DescrSetKeys {
                write_fields: vec![member.clone()],
                ..Default::default()
            }
            .into(),
        );
        let mut reader =
            EffectInfo::const_new(ExtraEffect::CannotRaise, majit_ir::OopSpecIndex::None);
        reader.descr_set_keys = Some(
            DescrSetKeys {
                readonly_fields: vec![member.clone()],
                ..Default::default()
            }
            .into(),
        );
        let layout =
            majit_ir::effectinfo::compute_frozen_bitstrings(&mut [&mut writer, &mut reader]);
        let stamp = layout.descr_indices[0][&member];
        let mint = DescrMintEntry {
            member,
            spec: DescrMintSpec::Field {
                struct_size: 8,
                offset: 0,
                field_size: 8,
                field_type: Type::Int,
                flag: majit_ir::descr::ArrayFlag::Unsigned,
                is_immutable: false,
                is_quasi_immutable: false,
                index_in_parent: 0,
            },
            ei_index: stamp,
        };
        let field = CanonicalBhDescr::Field {
            offset: 0,
            field_size: 8,
            field_type: Type::Int,
            field_flag: majit_ir::descr::ArrayFlag::Unsigned,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: Some(0),
            parent: Some(Arc::new(BhSizeSpec {
                size: 8,
                type_id: 0x475241494e454649,
                vtable: 0,
                owner: String::new(),
                is_gc_managed: false,
                headerless: true,
                all_fielddescrs: vec![BhFieldSpec {
                    index: 0,
                    field_key: "depth".into(),
                    name: "Frame.depth".into(),
                    offset: 0,
                    field_size: 8,
                    field_type: Type::Int,
                    field_flag: majit_ir::descr::ArrayFlag::Unsigned,
                    is_field_signed: false,
                    is_immutable: false,
                    is_quasi_immutable: false,
                    index_in_parent: 0,
                    is_class_word: None,
                    load_is_acquire: false,
                }],
            })),
            name: "depth".into(),
            owner: "Frame".into(),
            load_is_acquire: false,
        };
        let descrs = vec![
            field,
            CanonicalBhDescr::Call {
                calldescr: BhCallDescr {
                    extra_info: writer,
                    ..Default::default()
                },
            },
            CanonicalBhDescr::Call {
                calldescr: BhCallDescr {
                    extra_info: reader,
                    ..Default::default()
                },
            },
        ];
        let descrs = bincode::deserialize(&bincode::serialize(&descrs).unwrap()).unwrap();
        let mints: Vec<DescrMintEntry> =
            bincode::deserialize(&bincode::serialize(&vec![mint]).unwrap()).unwrap();
        let table =
            EmbeddedJitCodeTable::materialize_with_frozen_effects(&[], descrs, &[], &[], &mints);
        let field = table.descrs()[0].as_optimizer_descr().unwrap();
        assert_eq!(field.get_ei_index(), stamp);
        let effect = |i: usize| {
            let CanonicalBhDescr::Call { calldescr } = table.descrs()[i].as_bh_descr().unwrap()
            else {
                panic!()
            };
            &calldescr.extra_info
        };
        assert!(effect(1).check_write_descr_field(field.get_ei_index()));
        assert!(!effect(2).check_write_descr_field(field.get_ei_index()));
        assert!(Arc::ptr_eq(
            field,
            &effect(1)._write_descrs_fields.as_ref().unwrap()[0]
        ));
        assert_ne!(
            effect(1),
            effect(2),
            "typed-call re-interning must not merge different effects"
        );
    }

    /// Two jitcodes, the second reachable from the first through a `j` slot.
    fn fixture() -> (Vec<Arc<CanonicalJitCode>>, Vec<CanonicalBhDescr>) {
        let caller = Arc::new(CanonicalJitCode::new("caller"));
        caller.set_index(0);
        caller.set_body(Default::default());
        let callee = Arc::new(CanonicalJitCode::new("callee"));
        callee.set_index(1);
        callee.set_body(Default::default());
        (
            vec![caller, callee],
            vec![CanonicalBhDescr::JitCode {
                jitcode_index: 1,
                fnaddr: 0,
                calldescr: Default::default(),
            }],
        )
    }

    /// The subject: a `j` slot resolves to the table's own entry, not to a
    /// second shell that merely carries the same body.
    #[test]
    fn a_jitcode_slot_resolves_to_the_table_entry_itself() {
        let (canonical, descrs) = fixture();
        let table = EmbeddedJitCodeTable::materialize(&canonical, descrs);
        let from_slot = table.descrs()[0]
            .as_jitcode()
            .expect("the `j` slot must resolve to a jitcode");
        let from_table = table.by_index(1).expect("index 1 is in the table");
        assert!(
            Arc::ptr_eq(from_slot, from_table),
            "`all_jitcodes[jitcode.index] is jitcode` (codewriter.py:80) is an \
             identity: a second shell with the same body passes every read of \
             the body and fails every `Arc::ptr_eq`",
        );
    }

    /// The shells hold no pool of their own, so nothing is stored twice and
    /// depth cannot decide what resolves.
    #[test]
    fn a_shell_carries_no_pool_of_its_own() {
        let (canonical, descrs) = fixture();
        let table = EmbeddedJitCodeTable::materialize(&canonical, descrs);
        for jitcode in table.jitcodes() {
            assert!(
                jitcode.exec.descrs.is_empty(),
                "a build-time shell resolves through the global pool; a \
                 per-shell copy makes its callees' operands read whichever \
                 pool those callees were handed",
            );
        }
    }

    /// A list whose positions disagree with its own `jitcode.index` stamps
    /// cannot be indexed by a `j` operand, and says so at materialization
    /// rather than resolving to the wrong callee.
    #[test]
    #[should_panic(expected = "names index")]
    fn a_table_out_of_order_with_its_indices_is_refused() {
        let (mut canonical, descrs) = fixture();
        canonical.swap(0, 1);
        let _ = EmbeddedJitCodeTable::materialize(&canonical, descrs);
    }

    /// `all_jitcodes` is the codewriter's completed list; a numbered shell
    /// with no body is a malformed embedded artefact, not a runtime bodyless
    /// state for the materializer to preserve.
    #[test]
    #[should_panic(expected = "embedded all_jitcodes artefact contains bodyless jitcode")]
    fn a_bodyless_all_jitcodes_entry_is_rejected_at_materialization() {
        let core = Arc::new(CanonicalJitCode::new("bodyless"));
        core.set_index(0);

        let _ = EmbeddedJitCodeTable::materialize(&[core], Vec::new());
    }

    #[test]
    fn symbolic_fnaddrs_are_resolved_before_the_table_is_published() {
        let symbolic = majit_jitcode::codewriter::call::SYMBOLIC_FNADDR_BASE as i64 | 7;
        let runtime = 0x1234_i64;
        let callee = CanonicalJitCode::new("callee");
        callee.set_index(0);
        callee.set_body(majit_jitcode::jitcode::JitCodeBody {
            constants_i: vec![symbolic, 11],
            ..Default::default()
        });
        let mut callee = Arc::new(callee);
        Arc::get_mut(&mut callee).unwrap().fnaddr = symbolic;

        let table = EmbeddedJitCodeTable::materialize_with_symbolic_fnaddrs(
            &[callee],
            vec![CanonicalBhDescr::JitCode {
                jitcode_index: 0,
                fnaddr: symbolic,
                calldescr: Default::default(),
            }],
            &[(symbolic, "runtime::callee".to_string())],
            &[("runtime::callee", runtime)],
        );

        let loaded = table.by_index(0).unwrap();
        assert_eq!(loaded.fnaddr, runtime);
        assert_eq!(loaded.body().constants_i, vec![runtime, 11]);
        assert!(Arc::ptr_eq(loaded, table.descrs()[0].as_jitcode().unwrap()));
    }

    #[test]
    fn support_bindings_are_automatic_but_allow_runtime_override() {
        for (path, address) in EmbeddedJitCodeTable::builtin_fnaddrs() {
            for override_address in [None, Some(0x1234_i64)] {
                let symbolic = -97;
                let mut code = CanonicalJitCode::new("support");
                code.fnaddr = symbolic;
                code.set_index(0);
                code.set_body(majit_jitcode::jitcode::JitCodeBody {
                    constants_i: vec![symbolic],
                    ..Default::default()
                });
                let bindings: Vec<_> = override_address.map(|a| (path, a)).into_iter().collect();
                let table = EmbeddedJitCodeTable::materialize_with_symbolic_fnaddrs(
                    &[Arc::new(code)],
                    Vec::new(),
                    &[(symbolic, path.into())],
                    &bindings,
                );
                let loaded = table.by_index(0).unwrap();
                assert_eq!(loaded.fnaddr, override_address.unwrap_or(address));
                assert_eq!(
                    loaded.body().constants_i,
                    [override_address.unwrap_or(address)]
                );
            }
        }
    }
}
