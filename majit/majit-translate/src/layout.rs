//! Layout provider — RPython `symbolic.get_field_token` / `symbolic.get_size`.
//!
//! RPython uses `llmemory.offsetof()` and `llmemory.sizeof()` (backed by C-level
//! layout via ll2ctypes) to compute exact field offsets and struct sizes. The JIT
//! codewriter calls `cpu.fielddescrof(STRUCT, fieldname)` which delegates to
//! `symbolic.get_field_token(STRUCT, fieldname, translate_support_code)`.
//!
//! In majit, the `LayoutProvider` trait serves the same purpose: it supplies
//! real struct layouts from the host runtime. The default `HeuristicLayoutProvider`
//! approximates `#[repr(C)]` layout from parsed type strings; production runtimes
//! (e.g. pyre) should override with layouts from `std::mem::offset_of!()` /
//! `std::mem::size_of::<T>()`.

use std::collections::{HashMap, HashSet};

use crate::call::StructLayout;
use crate::model::ImmutableRank;

/// Byte width of a pointer on the target these layouts describe — RPython's
/// `WORD`.
///
/// The layouts computed here are baked into build-time artefacts (the descr
/// pool `pyre-jit-trace/build.rs` serialises) that the *target* binary reads
/// back, so they must describe the target's structs, not the build host's.
/// Cargo exports the target's pointer width to build scripts as
/// `CARGO_CFG_TARGET_POINTER_WIDTH`; the layout model also runs in-process
/// (where the host *is* the target), and there the host width applies.
///
/// Read once: neither the environment nor the target changes within a run.
pub fn target_word_size() -> usize {
    static WORD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *WORD.get_or_init(|| {
        std::env::var("CARGO_CFG_TARGET_POINTER_WIDTH")
            .ok()
            .and_then(|bits| bits.parse::<usize>().ok())
            .map(|bits| bits / 8)
            .unwrap_or(std::mem::size_of::<usize>())
    })
}

/// Whether `target` is a real cross target — non-empty and distinct from the
/// build `host` — so its layout sidecars apply. Native builds share the host's
/// struct layouts and need no sidecar.
pub fn is_cross_target(target: &str, host: &str) -> bool {
    !target.is_empty() && target != host
}

/// Filename of the cross-target layout sidecar for `crate_stem` (an `.ullbc`
/// stem such as `pyre-object`) and `target`, e.g.
/// `pyre-object.wasm32-unknown-unknown.layouts.ullbc`.
///
/// The single home for this naming convention: `pyre-jit-trace/build.rs`
/// (preflight, rerun tracking, the codegen cache key) and
/// [`crate::auto_discover_workspace_llbc_paths`] both derive sidecar names
/// through here, so the convention cannot drift between them.
pub fn layout_sidecar_filename(crate_stem: &str, target: &str) -> String {
    format!("{crate_stem}.{target}.layouts.ullbc")
}

/// RPython: `symbolic.get_field_token` + `symbolic.get_size` provider.
///
/// Supplies struct layouts for the codewriter pipeline. Each struct is
/// identified by name (RPython uses LLTYPE identity).
pub trait LayoutProvider {
    /// Return the layout for a struct, or None to fall back to heuristic.
    fn get_struct_layout(&self, struct_name: &str) -> Option<StructLayout>;

    /// Return the layout corrected with exact rtyper-resolved per-field byte
    /// offsets and total size — the offsets `symbolic.get_field_token` reads
    /// from the C compiler in RPython (`symbolic.py:7`), which the heuristic
    /// only approximates.  No default: a provider must decide explicitly how
    /// it honours the exact layout (the heuristic applies the offsets over
    /// its `#[repr(C)]` approximation; an `offset_of!()`-backed provider
    /// returns its already-exact layout).  There is no "optionally ignore
    /// the symbolic layout" path — that would silently emit approximate
    /// offsets where the exact ones were known.
    fn get_struct_layout_exact(
        &self,
        struct_name: &str,
        exact_offsets: &HashMap<String, u64>,
        exact_size: Option<u64>,
    ) -> Option<StructLayout>;
}

/// Default provider using type-string heuristic.
///
/// Approximates `#[repr(C)]` layout by parsing field type strings and
/// applying alignment rules. Equivalent to the pre-LayoutProvider code path.
///
/// Production runtimes should provide a concrete `LayoutProvider` with
/// real offsets from `std::mem::offset_of!()` instead.
pub struct HeuristicLayoutProvider {
    known_struct_names: HashSet<String>,
    known_struct_sizes: HashMap<String, usize>,
    known_struct_aligns: HashMap<String, usize>,
    fields_by_struct: HashMap<String, Vec<(String, String)>>,
    /// RPython: per-class `_immutable_fields_` declarations paired with
    /// `ImmutableRank` (see `rpython/rtyper/rclass.py _parse_field_list`).  Empty
    /// when the source did not declare any immutable fields for that
    /// struct.
    immutable_fields_by_struct: HashMap<String, HashMap<String, ImmutableRank>>,
}

impl HeuristicLayoutProvider {
    /// Build the heuristic provider from struct field definitions.
    ///
    /// Runs fixed-point iteration to resolve nested struct sizes, matching
    /// the convergence loop that `symbolic.get_size()` doesn't need (because
    /// RPython queries the C compiler directly).
    ///
    /// `immutable_fields_by_struct`: parsed `#[jit_immutable_fields(...)]`
    /// per struct (RPython `_immutable_fields_`). Used to set
    /// `StructFieldLayout.is_immutable` and `StructFieldLayout.rank`.
    pub fn from_struct_fields(
        fields_by_struct: &HashMap<String, Vec<(String, String)>>,
        known_struct_names: &HashSet<String>,
        immutable_fields_by_struct: &HashMap<String, Vec<(String, ImmutableRank)>>,
    ) -> Self {
        let immutable: HashMap<String, HashMap<String, ImmutableRank>> = immutable_fields_by_struct
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().map(|(n, r)| (n.clone(), *r)).collect()))
            .collect();
        let mut known_sizes: HashMap<String, usize> = HashMap::new();
        let mut known_aligns: HashMap<String, usize> = fields_by_struct
            .keys()
            .map(|name| (name.clone(), 1))
            .collect();
        loop {
            let mut changed = false;
            for (struct_name, fields) in fields_by_struct {
                let imm_ranks = immutable.get(struct_name).cloned().unwrap_or_default();
                let layout = StructLayout::from_type_strings(
                    fields,
                    &HashSet::new(),
                    &known_sizes,
                    &known_aligns,
                    &imm_ranks,
                );
                if known_sizes.get(struct_name) != Some(&layout.size)
                    || known_aligns.get(struct_name) != Some(&layout.align)
                {
                    known_sizes.insert(struct_name.clone(), layout.size);
                    known_aligns.insert(struct_name.clone(), layout.align);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Self {
            known_struct_names: known_struct_names.clone(),
            known_struct_sizes: known_sizes,
            known_struct_aligns: known_aligns,
            fields_by_struct: fields_by_struct.clone(),
            immutable_fields_by_struct: immutable,
        }
    }
}

impl LayoutProvider for HeuristicLayoutProvider {
    fn get_struct_layout(&self, struct_name: &str) -> Option<StructLayout> {
        let fields = self.fields_by_struct.get(struct_name)?;
        let imm_ranks = self
            .immutable_fields_by_struct
            .get(struct_name)
            .cloned()
            .unwrap_or_default();
        Some(StructLayout::from_type_strings(
            fields,
            &self.known_struct_names,
            &self.known_struct_sizes,
            &self.known_struct_aligns,
            &imm_ranks,
        ))
    }

    fn get_struct_layout_exact(
        &self,
        struct_name: &str,
        exact_offsets: &HashMap<String, u64>,
        exact_size: Option<u64>,
    ) -> Option<StructLayout> {
        let mut layout = self.get_struct_layout(struct_name)?;
        let rows = self.fields_by_struct.get(struct_name)?;
        let imm_ranks = self
            .immutable_fields_by_struct
            .get(struct_name)
            .cloned()
            .unwrap_or_default();
        layout.apply_exact_layout(
            rows,
            exact_offsets,
            exact_size,
            &self.known_struct_names,
            &self.known_struct_sizes,
            &imm_ranks,
        );
        Some(layout)
    }
}

/// One STRUCT's layout ready to install on `CallControl`.
///
/// Exact Charon layouts are emitted first, keyed by STRUCT identity
/// (`descr.py` `get_field_descr(STRUCT, fieldname)`), even when the name
/// table withdrew the spelling. Heuristic layouts fill remaining names.
pub(crate) struct ProgramStructLayout {
    pub spelling: String,
    pub sid: majit_ir::descr::StructId,
    pub exact: Option<crate::front::semantic::ExactLayout>,
    pub layout: StructLayout,
}

fn field_rows_cover_exact(
    rows: &[crate::front::semantic::FieldRow],
    exact: &crate::front::semantic::ExactLayout,
) -> bool {
    exact
        .field_offsets
        .keys()
        .all(|name| rows.iter().any(|row| row.name == *name))
}

fn spelling_covering_exact<'a>(
    sid: majit_ir::descr::StructId,
    exact: &crate::front::semantic::ExactLayout,
    struct_fields: &'a crate::front::semantic::StructFieldRegistry,
) -> Option<&'a str> {
    let mut fallback = None;
    for (name, rows) in struct_fields.fields.iter() {
        if rows.is_empty() || !field_rows_cover_exact(rows, exact) {
            continue;
        }
        if majit_ir::descr::struct_id_for_name(name) == Some(sid) {
            return Some(name.as_str());
        }
        if fallback.is_none() {
            fallback = Some(name.as_str());
        }
    }
    fallback
}

/// Exact layouts by STRUCT identity, then heuristic layouts for names
/// that still have no identity-keyed row.
///
/// An instantiated STRUCT does not inherit the template's offsets, but a
/// missing Charon row must still get a layout: otherwise
/// `compute_struct_size` / `fielddescrof_concrete` fall through to the
/// C-order accumulator on every use.
pub(crate) fn program_struct_layouts(
    struct_fields: &crate::front::semantic::StructFieldRegistry,
    exact_layouts: &HashMap<majit_ir::descr::StructId, crate::front::semantic::ExactLayout>,
    provider: &dyn LayoutProvider,
) -> Vec<ProgramStructLayout> {
    let mut installed = HashSet::new();
    let mut out = Vec::new();
    for (sid, exact) in exact_layouts {
        if exact.field_offsets.is_empty() {
            continue;
        }
        let Some(spelling) = spelling_covering_exact(*sid, exact, struct_fields) else {
            continue;
        };
        let Some(layout) =
            provider.get_struct_layout_exact(spelling, &exact.field_offsets, exact.size)
        else {
            continue;
        };
        installed.insert(*sid);
        out.push(ProgramStructLayout {
            spelling: spelling.to_string(),
            sid: *sid,
            exact: Some(exact.clone()),
            layout,
        });
    }
    for (struct_name, rows) in struct_fields.fields.iter() {
        let Some(sid) = majit_ir::descr::struct_id_for_name(struct_name) else {
            continue;
        };
        if !installed.insert(sid) {
            continue;
        }
        let exact = exact_layouts.get(&sid);
        if exact.is_some_and(|layout| !layout.field_offsets.is_empty()) && rows.is_empty() {
            continue;
        }
        let layout = match exact {
            Some(exact) if !exact.field_offsets.is_empty() => {
                provider.get_struct_layout_exact(struct_name, &exact.field_offsets, exact.size)
            }
            _ => provider.get_struct_layout(struct_name),
        };
        let Some(layout) = layout else {
            continue;
        };
        out.push(ProgramStructLayout {
            spelling: struct_name.clone(),
            sid,
            exact: exact.cloned(),
            layout,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_provider_basic_struct() {
        let mut fields = HashMap::new();
        fields.insert(
            "MyStruct".to_string(),
            vec![
                ("x".to_string(), "i64".to_string()),
                ("y".to_string(), "i64".to_string()),
            ],
        );
        let provider =
            HeuristicLayoutProvider::from_struct_fields(&fields, &HashSet::new(), &HashMap::new());
        let layout = provider.get_struct_layout("MyStruct").unwrap();
        assert_eq!(layout.size, 16);
        assert_eq!(layout.fields.len(), 2);
        assert_eq!(layout.fields[0].name, "x");
        assert_eq!(layout.fields[0].offset, 0);
        assert!(!layout.fields[0].is_immutable());
        assert_eq!(layout.fields[1].name, "y");
        assert_eq!(layout.fields[1].offset, 8);
        assert!(!layout.fields[1].is_immutable());
    }

    #[test]
    fn test_heuristic_provider_unknown_struct() {
        let fields = HashMap::new();
        let provider =
            HeuristicLayoutProvider::from_struct_fields(&fields, &HashSet::new(), &HashMap::new());
        assert!(provider.get_struct_layout("NoSuchStruct").is_none());
    }

    #[test]
    fn test_heuristic_provider_immutable_field() {
        // RPython parity: STRUCT._immutable_field("pools") == True
        // when "pools" is in _immutable_fields_.
        let mut fields = HashMap::new();
        fields.insert(
            "Storage".to_string(),
            vec![
                ("pools".to_string(), "i64".to_string()),
                ("scratch".to_string(), "i64".to_string()),
            ],
        );
        let mut immutable = HashMap::new();
        immutable.insert(
            "Storage".to_string(),
            vec![("pools".to_string(), ImmutableRank::Immutable)],
        );
        let provider =
            HeuristicLayoutProvider::from_struct_fields(&fields, &HashSet::new(), &immutable);
        let layout = provider.get_struct_layout("Storage").unwrap();
        assert_eq!(layout.fields[0].name, "pools");
        assert!(layout.fields[0].is_immutable());
        assert_eq!(layout.fields[1].name, "scratch");
        assert!(!layout.fields[1].is_immutable());
    }

    #[test]
    fn test_heuristic_provider_quasi_immutable_field_rank() {
        // RPython parity: `?` suffix → `IR_QUASIIMMUTABLE` rank.
        let mut fields = HashMap::new();
        fields.insert(
            "Cell".to_string(),
            vec![
                ("value".to_string(), "i64".to_string()),
                ("flag".to_string(), "i64".to_string()),
            ],
        );
        let mut immutable = HashMap::new();
        immutable.insert(
            "Cell".to_string(),
            vec![("value".to_string(), ImmutableRank::QuasiImmutable)],
        );
        let provider =
            HeuristicLayoutProvider::from_struct_fields(&fields, &HashSet::new(), &immutable);
        let layout = provider.get_struct_layout("Cell").unwrap();
        assert_eq!(layout.fields[0].name, "value");
        assert_eq!(layout.fields[0].rank, Some(ImmutableRank::QuasiImmutable));
        assert!(layout.fields[0].is_immutable());
        assert!(layout.fields[0].is_quasi_immutable());
        assert_eq!(layout.fields[1].name, "flag");
        // RPython `STRUCT._immutable_field` returns False (not a default
        // rank) for fields outside `_immutable_fields_`.
        assert_eq!(layout.fields[1].rank, None);
        assert!(!layout.fields[1].is_immutable());
    }

    /// finding 3a: the enum base's `__discriminant` field resolves at the
    /// tag's real byte position (`discriminant_offset`), not the heuristic
    /// 0.  Mirrors the enum arm of `derive_program_metadata` (base
    /// `ExactLayout` carries `{"__discriminant": discriminant_offset}`)
    /// feeding the bridge's `get_struct_layout_exact`.
    #[test]
    fn enum_base_discriminant_resolves_at_tag_offset() {
        use majit_charon_reader::ullbc::TypeLayout;

        // A niche/tag enum whose tag sits at byte 8, not 0.
        let niche: TypeLayout = serde_json::from_str(
            r#"{ "size": 16, "align": 8,
                 "variant_layouts": [{"field_offsets": [0]}, {"field_offsets": [0]}],
                 "discriminator": {"Branch": {"offset": 8}} }"#,
        )
        .unwrap();
        assert_eq!(niche.discriminant_offset(), Some(8));

        // The base row registry: the discriminant-only sentinel the enum
        // arm registers under the base spelling.
        let mut fields = HashMap::new();
        fields.insert(
            "module::Enum".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        let provider =
            HeuristicLayoutProvider::from_struct_fields(&fields, &HashSet::new(), &HashMap::new());

        // Heuristic alone would place the tag at offset 0.
        assert_eq!(
            provider.get_struct_layout("module::Enum").unwrap().fields[0].offset,
            0
        );

        // The base ExactLayout the enum arm builds from the niche layout.
        let mut base_offsets = HashMap::new();
        base_offsets.insert(
            "__discriminant".to_string(),
            niche.discriminant_offset().unwrap_or(0),
        );
        let exact = provider
            .get_struct_layout_exact("module::Enum", &base_offsets, niche.size)
            .unwrap();
        let disc = exact
            .fields
            .iter()
            .find(|f| f.name == "__discriminant")
            .expect("base layout carries the discriminant field");
        assert_eq!(disc.offset, 8, "niche tag resolves at its real offset");
        assert_eq!(exact.size, 16);

        // A tag at offset 0 (the common case) registers 0, matching the
        // heuristic exactly.
        let mut zero_offsets = HashMap::new();
        zero_offsets.insert("__discriminant".to_string(), 0u64);
        let exact0 = provider
            .get_struct_layout_exact("module::Enum", &zero_offsets, Some(8))
            .unwrap();
        assert_eq!(
            exact0
                .fields
                .iter()
                .find(|f| f.name == "__discriminant")
                .unwrap()
                .offset,
            0
        );
    }
}
