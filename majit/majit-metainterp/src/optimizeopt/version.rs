//! Loop versioning.
//!
//! Mirrors RPython's `version.py`: LoopVersionInfo and LoopVersion.
//! When a guard is known to fail frequently, a specialized version of
//! the loop can be compiled and stitched directly to that guard.

use majit_ir::{Op, OpRef};

use indexmap::IndexMap;

/// version.py: LoopVersionInfo(BasicLoopInfo)
///
/// Tracks multiple compiled versions of a loop, keyed by guard descr.
/// `descrs` is the ordered list of guard descriptors that have loop versions.
/// `leads_to` maps each descr's fail_index to the LoopVersion.
pub struct LoopVersionInfo {
    /// version.py:19 — ordered list of fail indices for tracked guards.
    pub descrs: Vec<u32>,
    /// version.py — maps fail_index → LoopVersion.
    pub leads_to: IndexMap<u32, LoopVersion>,
    /// version.py — insertion index for track(). -1 means append.
    insert_index: i32,
    /// version.py:22 — compiled loop versions.
    pub versions: Vec<LoopVersion>,
}

impl LoopVersionInfo {
    /// version.py:10-22
    pub fn new() -> Self {
        LoopVersionInfo {
            descrs: Vec::new(),
            leads_to: IndexMap::new(),
            insert_index: -1,
            versions: Vec::new(),
        }
    }

    /// version.py: mark()
    pub fn mark(&mut self) {
        self.insert_index = self.descrs.len() as i32;
    }

    /// version.py: clear()
    pub fn clear(&mut self) {
        self.insert_index = -1;
    }

    /// version.py: track(op, descr, version)
    ///
    /// `fail_index` is the descr's unique identifier.
    pub fn track(&mut self, fail_index: u32, version: LoopVersion) {
        let i = self.insert_index;
        if i >= 0 {
            self.descrs.insert(i as usize, fail_index);
        } else {
            self.descrs.push(fail_index);
        }
        assert!(
            !self.leads_to.contains_key(&fail_index),
            "version.py:35 assert descr not in self.leads_to"
        );
        self.leads_to.insert(fail_index, version);
    }

    /// version.py: remove(descr)
    pub fn remove(&mut self, fail_index: u32) {
        self.leads_to
            .swap_remove(&fail_index)
            .unwrap_or_else(|| panic!("version.py:42 could not remove fail_index={}", fail_index));
        self.descrs.retain(|d| *d != fail_index);
    }

    /// version.py: get(descr)
    pub fn get(&self, fail_index: u32) -> Option<&LoopVersion> {
        self.leads_to.get(&fail_index)
    }

    /// version.py: snapshot(loop)
    ///
    /// Clone the loop operations and create a new LoopVersion.
    /// The version is registered in `self.versions`.
    pub fn snapshot(&mut self, ops: &[Op], label_args: &[OpRef]) -> LoopVersion {
        let newloop = ops.to_vec();
        let mut version = LoopVersion::new(newloop, label_args.to_vec());
        version.setup_once(self);
        self.versions.push(version.clone());
        version
    }

    /// version.py: post_loop_compilation(...)
    ///
    /// Compile each version and stitch to guard descriptors.
    /// This is called after the root trace is already compiled.
    ///
    /// In majit, the actual compilation is deferred to the backend.
    /// This method returns the list of (fail_index, LoopVersion) pairs
    /// that need to be compiled as bridges.
    pub fn pending_compilations(&self) -> Vec<(u32, LoopVersion)> {
        if self.versions.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::new();
        for &fail_index in &self.descrs {
            if let Some(version) = self.leads_to.get(&fail_index) {
                result.push((fail_index, version.clone()));
            }
        }
        result
    }

    /// Number of tracked versions.
    pub fn num_versions(&self) -> usize {
        self.leads_to.len()
    }
}

impl Default for LoopVersionInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// version.py: LoopVersion
///
/// A specialized version of a trace loop, attached to a guard descriptor.
#[derive(Clone, Debug)]
pub struct LoopVersion {
    /// The specialized loop body operations.
    pub ops: Vec<Op>,
    /// Input arguments for this version (label arglist copy).
    pub inputargs: Vec<OpRef>,
}

impl LoopVersion {
    pub fn new(ops: Vec<Op>, inputargs: Vec<OpRef>) -> Self {
        LoopVersion { ops, inputargs }
    }

    /// version.py: setup_once(info)
    ///
    /// Clone guard descriptors in the version's ops so each version
    /// has its own descr identity. Re-track cloned loop-version guards.
    /// version.py: setup_once(info)
    ///
    /// Clone guard descriptors in the version's ops so each version
    /// has its own descr identity. Uses `olddescr.clone()` which
    /// preserves the concrete type (CompileLoopVersionDescr stays
    /// CompileLoopVersionDescr, ResumeGuardDescr stays ResumeGuardDescr).
    pub fn setup_once(&mut self, info: &mut LoopVersionInfo) {
        for op in &mut self.ops {
            if !op.opcode.is_guard() {
                continue;
            }
            // version.py:104-105
            let old_descr = match op.getdescr() {
                Some(d) => d,
                None => continue,
            };
            // version.py:107: descr = olddescr.clone()
            // Subtype-preserving clone: CompileLoopVersionDescr.clone()
            // returns CompileLoopVersionDescr, preserving loop_version()==true.
            let Some(new_descr) = old_descr.clone_descr() else {
                continue;
            };
            // version.py:108: op.setdescr(descr)
            op.setdescr(new_descr.clone());
            // version.py:109-114: if descr.loop_version()
            let Some(new_fd) = new_descr.as_fail_descr() else {
                continue;
            };
            if new_fd.loop_version() {
                // `version.py:32-36 leads_to[descr]` keys by descr object
                // identity (`compute_unique_id(faildescr)` /
                // `descr_identity`).  Pyre uses `Descr::index()` — the
                // globally unique `alloc_fail_index()` value baked at descr
                // construction.
                let old_fi = old_descr.index();
                let toversion = info.leads_to.get(&old_fi).cloned();
                if let Some(tv) = toversion {
                    info.track(new_descr.index(), tv);
                } else {
                    panic!("version.py:114 olddescr must be found in leads_to");
                }
            }
        }
    }
}
