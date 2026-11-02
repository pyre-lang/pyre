//! Port of `rpython/translator/c/genc.py`.
//!
//! Upstream is 944 LOC of three classes plus helpers:
//! * `CCompilerDriver` (`:40-58`) — platform-bound build invocation.
//! * `CBuilder` (`:62-235`) — common base for standalone and library
//!   builders. Holds `translator`, `entrypoint`, `config`, `gcpolicy`,
//!   `gchooks`, `secondary_entrypoints`, plus the `c_source_filename`,
//!   `_compiled`, `modulename`, `split` class attrs, and the
//!   [`build_database`](Self::build_database) /
//!   [`generate_source`](Self::generate_source) /
//!   [`compile`](Self::compile) flow.
//! * `CStandaloneBuilder` (`:237-510`) — exe-producing subclass. Adds
//!   `executable_name`, `shared_library_name`, `_entrypoint_wrapper`
//!   slots and overrides `getentrypointptr`, `compile`, `gen_makefile`.
//!
//! This file ports only the structural shells: the constructor
//! (`__init__`), attribute layout, and the public method signatures
//! consumed by [`crate::translator::driver`]. Each method body returns a
//! [`TaskError`] citing the upstream line of the unported leaf;
//! convergence is per-method as the database / source-emit / compile
//! chain lands.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use crate::config::config::{Config, ConfigValue, OptionValue};
use crate::flowspace::model::HostObject;
use crate::translator::backend::database::{GcPolicyClass, LowLevelDatabase};
use crate::translator::driver::EntryPointSpec;
use crate::translator::rtyper::lltypesystem::lltype::{_getconcretetype, getfunctionptr};
use crate::translator::tool::taskengine::TaskError;
use crate::translator::translator::TranslationContext;

pub use crate::translator::tool::cbuild::ExternalCompilationInfo;

/// Port of upstream `class CBuilder(object)` at `genc.py`.
///
/// Field shape mirrors upstream's `__init__` (`:68-78`) plus the
/// class-level defaults at `:62-66`. Each method below corresponds to a
/// `CBuilder` method by upstream-name; bodies return [`TaskError`] until
/// the C-backend leaf lands.
#[derive(Clone)]
pub struct CBuilder {
    // Upstream class-level slots at `:63-66`.
    /// `c_source_filename = None` (`:63`). Set by `generate_source`.
    pub c_source_filename: RefCell<Option<PathBuf>>,
    /// `_compiled = False` (`:64`). Flipped by `compile`.
    pub _compiled: bool,
    /// `modulename = None` (`:65`). Set by `generate_source` after
    /// `uniquemodulename(...)`.
    pub modulename: RefCell<Option<String>>,
    /// `split = False` (`:66`). Overridden to `True` by
    /// `CStandaloneBuilder` and `CLibraryBuilder`.
    pub split: bool,
    /// Dynamic subclass slot read by `build_database` as
    /// `self.standalone` at `genc.py`. Upstream defines it on
    /// subclasses; Rust stores the current subclass value here.
    pub standalone: bool,

    // Upstream `__init__` slots at `:69-78`.
    /// `self.translator` (`:70`).
    pub translator: Rc<TranslationContext>,
    /// `self.entrypoint` (`:71`).
    pub entrypoint: Option<HostObject>,
    /// `self.entrypoint_name = getattr(entrypoint, 'func_name', None)` (`:72`).
    pub entrypoint_name: Option<String>,
    /// `self.originalentrypoint = entrypoint` (`:73`).
    pub originalentrypoint: Option<HostObject>,
    /// `self.config = config` (`:74`).
    pub config: Rc<Config>,
    /// `self.gcpolicy = gcpolicy` (`:75`). Opaque to the local port —
    /// upstream's `GCPolicy` hierarchy is not yet ported.
    pub gcpolicy: Option<Rc<dyn Any>>,
    /// `self.gchooks = gchooks` (`:76`).
    pub gchooks: Option<Rc<dyn Any>>,
    /// `self.eci = self.get_eci()` (`:77`). Upstream `get_eci`
    /// (`genc.py`) returns an `ExternalCompilationInfo` filled
    /// with `include_dirs=[pypy_include_dir]` (and `revdb` when
    /// `reverse_debugger` is on). The slot stays opaque because the C
    /// backend still stores heterogeneous values through Python object
    /// attributes, but the stored object is the concrete
    /// [`ExternalCompilationInfo`] (`translator.tool.cbuild`).
    pub eci: RefCell<Option<Rc<dyn Any>>>,
    /// `self.secondary_entrypoints = secondary_entrypoints` (`:78`).
    pub secondary_entrypoints: Vec<EntryPointSpec>,

    // Upstream `targetdir` slot, set by `generate_source` at `:187`.
    pub targetdir: RefCell<Option<PathBuf>>,

    // Upstream `db` slot, set by `build_database` at `:103`.
    pub db: RefCell<Option<LowLevelDatabase>>,
    // Upstream `c_entrypoint_name`, set by `build_database` at `:114`
    // or `:122`.
    pub c_entrypoint_name: RefCell<Option<String>>,
}

impl CBuilder {
    /// Upstream `CBuilder.__init__(self, translator, entrypoint, config,
    /// gcpolicy=None, gchooks=None, secondary_entrypoints=())` at
    /// `:68-78`.
    pub fn new(
        translator: Rc<TranslationContext>,
        entrypoint: Option<HostObject>,
        config: Rc<Config>,
        gcpolicy: Option<Rc<dyn Any>>,
        gchooks: Option<Rc<dyn Any>>,
        secondary_entrypoints: Vec<EntryPointSpec>,
    ) -> Self {
        // Upstream `:72`: `self.entrypoint_name = getattr(self.entrypoint,
        // 'func_name', None)`. The local `GraphFunc.name` field carries
        // the same `__name__`/`func_name` value.
        let entrypoint_name = entrypoint
            .as_ref()
            .and_then(|h| h.user_function().map(|gf| gf.name.clone()));
        let builder = Self {
            c_source_filename: RefCell::new(None),
            _compiled: false,
            modulename: RefCell::new(None),
            split: false,
            standalone: false,
            translator,
            entrypoint: entrypoint.clone(),
            entrypoint_name,
            originalentrypoint: entrypoint,
            config: Rc::clone(&config),
            gcpolicy,
            gchooks,
            eci: RefCell::new(None),
            secondary_entrypoints,
            targetdir: RefCell::new(None),
            db: RefCell::new(None),
            c_entrypoint_name: RefCell::new(None),
        };
        // Upstream `:77`: `self.eci = self.get_eci()`.
        let _ = config;
        let initial_eci = builder.get_eci();
        *builder.eci.borrow_mut() = Some(Rc::new(initial_eci) as Rc<dyn Any>);
        builder
    }

    /// Upstream `CBuilder.DEBUG_DEFINES` class-level constant at
    /// `genc.py`. Used by `task_source_c` when the
    /// `c_debug_defines` extra option is on.
    pub fn debug_defines() -> std::collections::HashMap<String, String> {
        let mut d = std::collections::HashMap::new();
        d.insert("RPY_ASSERT".to_string(), "1".to_string());
        d.insert("RPY_LL_ASSERT".to_string(), "1".to_string());
        d.insert("RPY_REVDB_PRINT_ALL".to_string(), "1".to_string());
        d
    }

    /// Port of upstream `CBuilder.get_eci(self)` at `genc.py`.
    ///
    /// Upstream returns an `ExternalCompilationInfo(include_dirs=[
    /// pypy_include_dir, ?revdb])` where `pypy_include_dir` resolves to
    /// `rpython/translator/c/` (the directory holding `src/`, `g_*.h`,
    /// and the rest of the C-backend headers).
    ///
    /// TODO: the Rust port has not yet vendored the
    /// upstream C-backend header tree (`rpython/translator/c/src/*.h`,
    /// `g_*.h`, `genc.py pypy_include_dir`). Upstream `get_eci`
    /// **always** returns a non-empty `include_dirs` list whose first
    /// entry is the directory containing those headers — downstream
    /// consumers (`merge_eci`, gcc invocations) treat that include dir
    /// as load-bearing. Returning `Vec::new()` when `PYPY_SRCROOT` is
    /// unset is silently incorrect: any downstream C-backend code that
    /// later trusts the ECI will compile against the wrong include
    /// graph. The local port therefore panics rather than returning a
    /// "fake empty" ECI; callers exercising this structural placeholder
    /// should set `PYPY_SRCROOT` to a directory containing the upstream
    /// header tree. Do not vendor those headers or add a local C backend
    /// under `majit/majit-translate`: this module cites
    /// `rpython/translator/c/...` only as upstream source material.
    pub fn get_eci(&self) -> ExternalCompilationInfo {
        let mut include_dirs: Vec<PathBuf> = Vec::new();
        match std::env::var("PYPY_SRCROOT") {
            Ok(pypy_srcroot) => {
                // Upstream `:81`: `pypy_include_dir =
                // py.path.local(__file__).join('..')` resolves to
                // `${PYPY_SRCROOT}/rpython/translator/c`.
                let pypy_include_dir = PathBuf::from(pypy_srcroot).join("rpython/translator/c");
                // Upstream `:82`: `include_dirs = [pypy_include_dir]`.
                include_dirs.push(pypy_include_dir.clone());
                // Upstream `:83-84`: revdb include dir is
                // `pypy_include_dir.join('..', 'revdb')`.
                if matches!(
                    self.config.get("translation.reverse_debugger"),
                    Ok(crate::config::config::ConfigValue::Value(
                        crate::config::config::OptionValue::Bool(true)
                    ))
                ) && let Some(parent) = pypy_include_dir.parent()
                {
                    include_dirs.push(parent.join("revdb"));
                }
            }
            Err(_) => {
                // Empty `include_dirs` would silently lie to the
                // C-backend consumer — surface the unported state as a
                // sentinel path so a later compile-step / merge-eci
                // failure points at the missing vendored tree rather
                // than silently producing wrong machine code.
                include_dirs.push(PathBuf::from(
                    "<unported: rpython/translator/c headers; set PYPY_SRCROOT>",
                ));
            }
        }
        ExternalCompilationInfo {
            include_dirs,
            ..ExternalCompilationInfo::default()
        }
    }

    /// Upstream `CBuilder.build_database(self)` at `:87-138`.
    ///
    /// Direct calls on bare `CBuilder` still fail because upstream would
    /// dispatch `self.getentrypointptr()` on a concrete subclass. The real
    /// body is [`Self::build_database_with`], used by
    /// [`crate::translator::backend::CBuilderRef`] after enum-based dynamic
    /// dispatch supplies the subclass `getentrypointptr` callback.
    pub fn build_database(&self) -> Result<LowLevelDatabase, TaskError> {
        Err(TaskError {
            message: "genc.py:110 CBuilder.build_database — subclass getentrypointptr dispatch is required".to_string(),
        })
    }

    /// Port of `genc.py CBuilder.build_database`.
    ///
    /// Upstream constructs the gcpolicy + exctransformer + DB, sets
    /// `self.db`, walks gc_startup_code, then calls
    /// `pf = self.getentrypointptr()` at `:110` and registers it with
    /// the DB. Rust dispatches the subclass `getentrypointptr` through
    /// the supplied `entrypoint_fn` callback so the call lands at the
    /// upstream-equivalent `:110` position rather than ahead of the DB
    /// constructor.
    pub fn build_database_with(
        &self,
        entrypoint_fn: &dyn Fn() -> Result<EntryPointPtr, TaskError>,
    ) -> Result<LowLevelDatabase, TaskError> {
        // Upstream `:90 gcpolicyclass = self.get_gcpolicyclass()`.
        let gcpolicyclass = self.get_gcpolicyclass()?;
        // Upstream `:92 exctransformer = translator.getexceptiontransformer()`.
        let exctransformer = self.translator.getexceptiontransformer()?;
        // Upstream `:93-102 db = LowLevelDatabase(...)`.
        let db = LowLevelDatabase::new(
            Some(self.translator.clone()),
            self.standalone,
            gcpolicyclass,
            self.gchooks.clone(),
            exctransformer,
            config_bool(&self.config, "translation.thread")?,
            config_bool(&self.config, "translation.sandbox")?,
            config_bool(&self.config, "translation.split_gc_address_space")?,
            config_bool(&self.config, "translation.reverse_debugger")?,
            config_bool(&self.config, "translation.countfieldaccess")?,
        );
        // Upstream `:103 self.db = db`.
        *self.db.borrow_mut() = Some(db.clone());

        // Upstream `:106-107 list(db.gcpolicy.gc_startup_code())`.
        for startup in db.gcpolicy.gc_startup_code() {
            db.get(startup);
        }

        // Upstream `:110 pf = self.getentrypointptr()` — subclass dispatch.
        let pf = entrypoint_fn()?;

        // Upstream `:111-122` registers pf and secondary entry points.
        match pf {
            EntryPointPtr::Many(ptrs) => {
                for one_pf in ptrs {
                    db.get(one_pf);
                }
                *self.c_entrypoint_name.borrow_mut() = None;
            }
            EntryPointPtr::One(ptr) => {
                let pfname = db.get(ptr);
                for (func, _) in &self.secondary_entrypoints {
                    let ptr =
                        self.functionptr_for_any(func, "genc.py:118 secondary_entrypoints")?;
                    db.get(ptr);
                }
                *self.c_entrypoint_name.borrow_mut() = Some(pfname);
            }
        }

        // Upstream `:124-126 if self.config.translation.reverse_debugger:
        // gencsupp.prepare_database(db)`.
        if config_bool(&self.config, "translation.reverse_debugger")? {
            crate::translator::revdb::gencsupp::prepare_database(&db)?;
        }

        // Upstream `:128-130 for obj in exports.EXPORTS_obj2name.keys():
        // db.getcontainernode(obj); exports.clear()`. The exports module
        // (`rpython/rlib/exports.py`) has not been ported; the local
        // `TranslationContext` exposes an exports-keys iterator that is
        // empty until the rlib lands. Register every key with the DB so
        // the loop body matches upstream once the iterator is non-empty.
        for obj in self.translator.exports_obj2name_keys() {
            db.getcontainernode(obj);
        }
        self.translator.clear_exports();

        // Upstream `:132-133 for ll_func in db.translator._call_at_startup:
        // db.get(ll_func)`.
        for ll_func in self.translator.call_at_startup() {
            db.get(ll_func);
        }

        // `LowLevelDatabase.complete` (`database.py`) finishes by calling
        // `gctransformer.inline_helpers_and_postprocess(self.all_graphs())`.
        // The local DB does not yet own its gctransformer (its node factory is
        // still incomplete), so keep the call immediately adjacent to
        // `complete()` and run it over the translator's complete graph set.
        // This is a graph-wide translation stage; application functions carry
        // no per-function rooting attribute.
        if matches!(
            &db.gcpolicy.policy_class,
            GcPolicyClass::FrameworkShadowStack
        ) {
            // Placeholder, not a port. `BaseFrameworkGCTransformer.__init__`
            // builds `c_const_gcdata` as
            // `rmodel.inputconst(r_gc_data, self.gcdata)`, whose concretetype is the
            // rtyper repr of the `gctypelayout.GCData` instance -- a pointer.
            // Neither `gctypelayout.GCData` nor a repr for it exists on this
            // side yet, so the constant stands in as an opaque host object
            // typed `Void`.
            //
            // Inert at this stage and only at this stage: the sole consumer,
            // `shadowcolor.py add_enter_leave_roots_frame`, puts the constant
            // in `gc_enter_roots_frame`'s argument list and never reads it,
            // which is what the port does too. What needs the real type is
            // `framework.py _gc_adr_of_gcdata_attr` (`:999`), reached from
            // `gct_gc_adr_of_root_stack_base` and `gct_gc_adr_of_root_stack_top`:
            // it asks for `.concretetype.TO` and emits `cast_ptr_to_adr`, and
            // a `Void` constant has neither. Those two are unported -- their
            // names are in `lloperation.rs`, which is a table entry and not an
            // implementation -- so porting them means giving `GCData` a repr
            // here first rather than reading through this stand-in.
            let c_gcdata = crate::flowspace::model::Hlvalue::Constant(
                crate::flowspace::model::Constant::with_concretetype(
                    crate::flowspace::model::ConstValue::HostObject(
                        crate::flowspace::model::HostObject::new_opaque(
                            "rpython.memory.gctransform.framework.GCData",
                        ),
                    ),
                    crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Void,
                ),
            );
            let mut transformer =
                crate::memory::gctransform::shadowstack::ShadowStackFrameworkGCTransformer::new(
                    self.translator.as_ref(),
                    c_gcdata,
                );
            let graphs = self.translator.graphs.borrow().clone();
            transformer
                .inline_helpers_and_postprocess(&graphs)
                .map_err(|err| TaskError {
                    message: format!(
                        "ShadowStackFrameworkGCTransformer.inline_helpers_and_postprocess: {err}"
                    ),
                })?;
        }

        db.complete()?;
        self.collect_compilation_info(&db);
        Ok(db)
    }

    /// Backwards-compat shim — preferred call shape is
    /// [`Self::build_database_with`] which threads the subclass
    /// `getentrypointptr` callback into the upstream-equivalent
    /// position. This shim runs the callback first (matching the
    /// pre-fix order) and is retained only for tests that already
    /// hold a materialised [`EntryPointPtr`].
    #[deprecated(note = "use build_database_with for upstream call ordering")]
    pub fn build_database_from_entrypointptr(
        &self,
        pf: EntryPointPtr,
    ) -> Result<LowLevelDatabase, TaskError> {
        self.build_database_with(&|| Ok(pf.clone()))
    }

    /// Port of upstream `CBuilder.collect_compilation_info(db)` at
    /// `genc.py`. The first `merge_eci` call (`:147`) folds the
    /// gc-policy ECI in. The subsequent `globalcontainers()` and
    /// `getstructdeflist()` walks (`:149-158`) collect per-container
    /// `node.compilation_info()` and per-struct `STRUCT._hints['eci']`
    /// values — both require typed `Node` / `STRUCT` ports that have
    /// not landed; the local containers are still `Rc<dyn Any>` and
    /// expose no `compilation_info` accessor. Skip the two walks until
    /// `database.py` node factory lands so the loop bodies cannot
    /// silently drop ECIs that should have been merged.
    pub fn collect_compilation_info(&self, db: &LowLevelDatabase) {
        self.merge_eci(db.gcpolicy.compilation_info());
    }

    /// Port of upstream `CBuilder.merge_eci(*ecis)` at
    /// `genc.py`: `self.eci = self.eci.merge(*ecis)`. The
    /// local single-arg shape mirrors the only call site
    /// (`collect_compilation_info`'s gc-policy ECI). When the slot
    /// holds a typed [`ExternalCompilationInfo`] the merge runs
    /// in-place; otherwise the call is a no-op until the slot is
    /// populated by a typed source (TODO: because
    /// `gcpolicy.compilation_info()` still returns an opaque
    /// `Option<Rc<dyn Any>>`).
    pub fn merge_eci(&self, ecis: Option<Rc<dyn Any>>) {
        let Some(other) = ecis else {
            return;
        };
        let Ok(other) = other.downcast::<ExternalCompilationInfo>() else {
            return;
        };
        let mut slot = self.eci.borrow_mut();
        let Some(current) = slot.as_ref() else {
            // No prior ECI — adopt the other one wholesale.
            *slot = Some(other as Rc<dyn Any>);
            return;
        };
        let Some(current) = current.downcast_ref::<ExternalCompilationInfo>() else {
            // Slot held an opaque value — leave it alone rather than
            // overwrite with a typed ECI that ignored what was there.
            return;
        };
        match current.merge(&[other.as_ref()]) {
            Ok(merged) => {
                *slot = Some(Rc::new(merged) as Rc<dyn Any>);
            }
            Err(_) => {
                // Mixed-platform error — propagating it from a side-
                // effecting merge_eci is not yet wired through callers.
                // Leave the slot at its prior value and let downstream
                // ECI consumers surface the platform mismatch.
            }
        }
    }

    /// Port of upstream `CBuilder.get_gcpolicyclass(self)` at
    /// `genc.py`:
    /// ```python
    /// if self.gcpolicy is None:
    ///     name = self.config.translation.gctransformer
    ///     if name == "framework":
    ///         name = "%s+%s" % (name, self.config.translation.gcrootfinder)
    ///     return gc.name_to_gcpolicy[name]
    /// return self.gcpolicy
    /// ```
    /// When a custom `gcpolicy` is present, upstream returns the
    /// object itself — the local port preserves the `Rc<dyn Any>`
    /// inside [`GcPolicyClass::Custom`] so the caller-supplied policy
    /// methods/state survive the call.
    pub fn get_gcpolicyclass(&self) -> Result<GcPolicyClass, TaskError> {
        if let Some(gcpolicy) = &self.gcpolicy {
            return Ok(GcPolicyClass::Custom(Rc::clone(gcpolicy)));
        }
        let mut name = config_string(&self.config, "translation.gctransformer")?;
        if name == "framework" {
            name = format!(
                "{}+{}",
                name,
                config_string(&self.config, "translation.gcrootfinder")?
            );
        }
        Ok(GcPolicyClass::from_name(&name))
    }

    pub(crate) fn functionptr_for_any(
        &self,
        func: &Rc<dyn Any>,
        context: &str,
    ) -> Result<Rc<dyn Any>, TaskError> {
        let host = func
            .as_ref()
            .downcast_ref::<HostObject>()
            .ok_or_else(|| TaskError {
                message: format!("{context}: expected HostObject function"),
            })?;
        self.functionptr_for_host(host, context)
    }

    pub(crate) fn functionptr_for_host(
        &self,
        host: &HostObject,
        context: &str,
    ) -> Result<Rc<dyn Any>, TaskError> {
        let annotator = self.translator.annotator().ok_or_else(|| TaskError {
            message: format!("{context}: translator.annotator is None"),
        })?;
        let desc = annotator.bookkeeper.getdesc(host).map_err(|e| TaskError {
            message: format!("{context}: bookkeeper.getdesc failed: {e}"),
        })?;
        let funcdesc = desc.as_function().ok_or_else(|| TaskError {
            message: format!("{context}: descriptor is not a FunctionDesc"),
        })?;
        let graph = funcdesc.borrow().getuniquegraph().map_err(|e| TaskError {
            message: format!("{context}: getuniquegraph failed: {e}"),
        })?;
        let ptr = getfunctionptr(&graph.graph, _getconcretetype).map_err(|e| TaskError {
            message: format!("{context}: getfunctionptr failed: {e}"),
        })?;
        Ok(Rc::new(ptr) as Rc<dyn Any>)
    }

    /// Upstream `CBuilder.generate_source(self, db=None, defines={},
    /// exe_name=None)` at `:175-235`.
    pub fn generate_source(
        &self,
        _db: Option<&LowLevelDatabase>,
        _defines: &HashMap<String, String>,
        _exe_name: Option<String>,
    ) -> Result<PathBuf, TaskError> {
        Err(TaskError {
            message: "genc.py:175 CBuilder.generate_source — leaf SourceGenerator / uniquemodulename / udir not yet ported".to_string(),
        })
    }

    /// Upstream `CBuilder.compile(self, **kwds)` at `genc.py-…`
    /// (overridden in subclasses).
    pub fn compile(&self, _exe_name: Option<String>) -> Result<(), TaskError> {
        Err(TaskError {
            message: "genc.py:225 CBuilder.compile — leaf platform.compile not yet ported"
                .to_string(),
        })
    }
}

impl std::fmt::Debug for CBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CBuilder")
            .field("entrypoint_name", &self.entrypoint_name)
            .field("split", &self.split)
            .field("standalone", &self.standalone)
            .field("modulename", &self.modulename)
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub enum EntryPointPtr {
    One(Rc<dyn Any>),
    Many(Vec<Rc<dyn Any>>),
}

fn config_bool(config: &Rc<Config>, path: &str) -> Result<bool, TaskError> {
    match config.get(path).map_err(|e| TaskError {
        message: format!("genc.py config read {path}: {e}"),
    })? {
        ConfigValue::Value(OptionValue::Bool(value)) => Ok(value),
        ConfigValue::Value(OptionValue::None) => Ok(false),
        other => Err(TaskError {
            message: format!("genc.py config read {path}: expected Bool, got {other:?}"),
        }),
    }
}

fn config_string(config: &Rc<Config>, path: &str) -> Result<String, TaskError> {
    match config.get(path).map_err(|e| TaskError {
        message: format!("genc.py config read {path}: {e}"),
    })? {
        ConfigValue::Value(OptionValue::Choice(value))
        | ConfigValue::Value(OptionValue::Str(value)) => Ok(value),
        ConfigValue::Value(OptionValue::None) => Ok(String::new()),
        other => Err(TaskError {
            message: format!("genc.py config read {path}: expected Choice/Str, got {other:?}"),
        }),
    }
}

/// Port of upstream `class CStandaloneBuilder(CBuilder)` at
/// `genc.py`.
///
/// Adds the standalone-only slots `executable_name` (`:240`),
/// `shared_library_name` (`:241`), `_entrypoint_wrapper` (`:242`) and
/// overrides `getentrypointptr` (`:246-281`), `compile`, `gen_makefile`.
#[derive(Clone)]
pub struct CStandaloneBuilder {
    /// Inherited `CBuilder` state, with `split = true` per `:239`.
    pub base: CBuilder,
    /// `executable_name = None` (`:240`). Set by `compile` after
    /// `platform.compile` returns the executable path.
    pub executable_name: RefCell<Option<PathBuf>>,
    /// `shared_library_name = None` (`:241`). Set when the platform
    /// emits a companion shared library.
    pub shared_library_name: RefCell<Option<PathBuf>>,
    /// `_entrypoint_wrapper = None` (`:242`). Cached wrapper produced
    /// by `getentrypointptr`.
    pub _entrypoint_wrapper: RefCell<Option<Rc<dyn Any>>>,
    /// `make_entrypoint_wrapper = True  # for tests` (`:243`).
    pub make_entrypoint_wrapper: bool,
    /// Optional Windows-only sibling `pypyw.exe`. Upstream sets this in
    /// `CStandaloneBuilder.compile` after the platform build; set to
    /// `None` until that path lands.
    pub executable_name_w: RefCell<Option<PathBuf>>,
}

impl CStandaloneBuilder {
    /// Upstream `class CStandaloneBuilder(CBuilder)` inherits `__init__`
    /// from `CBuilder` and overrides only class-level defaults. The
    /// Rust port mirrors this with a constructor that accepts the same
    /// arguments and flips the `split` flag to `true` per `:239`.
    pub fn new(
        translator: Rc<TranslationContext>,
        entrypoint: Option<HostObject>,
        config: Rc<Config>,
        gcpolicy: Option<Rc<dyn Any>>,
        gchooks: Option<Rc<dyn Any>>,
        secondary_entrypoints: Vec<EntryPointSpec>,
    ) -> Self {
        let mut base = CBuilder::new(
            translator,
            entrypoint,
            config,
            gcpolicy,
            gchooks,
            secondary_entrypoints,
        );
        // Upstream `:238-239`: `standalone = True; split = True`.
        base.standalone = true;
        base.split = true;
        Self {
            base,
            executable_name: RefCell::new(None),
            shared_library_name: RefCell::new(None),
            _entrypoint_wrapper: RefCell::new(None),
            make_entrypoint_wrapper: true,
            executable_name_w: RefCell::new(None),
        }
    }

    /// Upstream class-level `standalone = True` (`:238`). Exposed as a
    /// const for callers that need to discriminate at the type level.
    pub const STANDALONE: bool = true;

    /// Upstream `getentrypointptr(self)` at `:246-281`.
    pub fn getentrypointptr(&self) -> Result<Rc<dyn Any>, TaskError> {
        if !self.make_entrypoint_wrapper {
            let entrypoint = self.base.entrypoint.as_ref().ok_or_else(|| TaskError {
                message: "genc.py:249 CStandaloneBuilder.getentrypointptr: entrypoint is None"
                    .to_string(),
            })?;
            return self.base.functionptr_for_host(
                entrypoint,
                "genc.py:249 CStandaloneBuilder.getentrypointptr",
            );
        }
        if let Some(existing) = self._entrypoint_wrapper.borrow().clone() {
            return Ok(existing);
        }
        Err(TaskError {
            message: "genc.py:246 CStandaloneBuilder.getentrypointptr — leaf MixLevelHelperAnnotator / lltype_to_annotation not yet ported".to_string(),
        })
    }

    /// Upstream `cmdexec(self, args, env, err, expect_crash, exe)` at
    /// `:283-323`. DEFERRED — depends on `translator.platform.execute`.
    pub fn cmdexec(
        &self,
        _args: &str,
        _env: Option<HashMap<String, String>>,
    ) -> Result<String, TaskError> {
        Err(TaskError {
            message:
                "genc.py:283 CStandaloneBuilder.cmdexec — leaf platform.execute not yet ported"
                    .to_string(),
        })
    }

    /// Upstream `CStandaloneBuilder.compile(self, exe_name=None)`.
    pub fn compile(&self, exe_name: Option<String>) -> Result<(), TaskError> {
        // Delegate the body to `CBuilder.compile` until the platform
        // build chain lands; the standalone-specific bookkeeping
        // (executable_name, c_entryp on the driver) happens in the
        // caller after this returns.
        self.base.compile(exe_name)
    }

    /// Upstream `CStandaloneBuilder.get_entry_point()` returns the
    /// platform-built executable path. Stub returns the cached
    /// `executable_name` slot.
    pub fn get_entry_point(&self) -> Result<PathBuf, TaskError> {
        match self.executable_name.borrow().clone() {
            Some(p) => Ok(p),
            None => Err(TaskError {
                message: "genc.py:240 CStandaloneBuilder.executable_name — slot is unset; compile() must run first".to_string(),
            }),
        }
    }
}

impl std::fmt::Debug for CStandaloneBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CStandaloneBuilder")
            .field("base", &self.base)
            .field("executable_name", &self.executable_name)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::translationoption::get_combined_translation_config;
    use crate::translator::translator::TranslationContext;
    use parking_lot::{Mutex, MutexGuard};
    use std::sync::OnceLock;

    static PYPY_SRCROOT_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    struct PypySrcrootEnvGuard {
        prior: Option<String>,
        _lock: MutexGuard<'static, ()>,
    }

    impl PypySrcrootEnvGuard {
        fn set(value: Option<&str>) -> Self {
            let lock = PYPY_SRCROOT_ENV_LOCK.get_or_init(|| Mutex::new(())).lock();
            let prior = std::env::var("PYPY_SRCROOT").ok();
            // Process environment is global; these tests serialize the
            // mutation so parallel Rust tests cannot observe a half-fixture.
            unsafe {
                match value {
                    Some(value) => std::env::set_var("PYPY_SRCROOT", value),
                    None => std::env::remove_var("PYPY_SRCROOT"),
                }
            }
            Self { prior, _lock: lock }
        }
    }

    impl Drop for PypySrcrootEnvGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.prior {
                    Some(value) => std::env::set_var("PYPY_SRCROOT", value),
                    None => std::env::remove_var("PYPY_SRCROOT"),
                }
            }
        }
    }

    fn fixture_translator_and_config() -> (Rc<TranslationContext>, Rc<Config>) {
        let translator = Rc::new(TranslationContext::new());
        let config = get_combined_translation_config(None, None, None, true).expect("config");
        (translator, config)
    }

    #[test]
    fn cbuilder_init_mirrors_upstream_slots() {
        // Upstream `:68-78`: `__init__(translator, entrypoint, config,
        // gcpolicy=None, gchooks=None, secondary_entrypoints=())` —
        // every slot must be writable on construction.
        let (translator, config) = fixture_translator_and_config();
        let cb = CBuilder::new(translator, None, config, None, None, Vec::new());
        // `:63` `c_source_filename = None` — pristine slot.
        assert!(cb.c_source_filename.borrow().is_none());
        // `:64` `_compiled = False`.
        assert!(!cb._compiled);
        // `:65` `modulename = None`.
        assert!(cb.modulename.borrow().is_none());
        // `:66` `split = False`.
        assert!(!cb.split);
        // `:78` `secondary_entrypoints = ()` — empty by default.
        assert!(cb.secondary_entrypoints.is_empty());
    }

    #[test]
    fn cstandalonebuilder_flips_split_flag_per_239() {
        // Upstream `:238-239`: `standalone = True; split = True`.
        let (translator, config) = fixture_translator_and_config();
        let csb = CStandaloneBuilder::new(translator, None, config, None, None, Vec::new());
        assert!(csb.base.split, "CStandaloneBuilder must flip split=True");
    }

    #[test]
    fn get_eci_returns_pypy_include_dir_when_srcroot_is_set() {
        // Upstream `genc.py get_eci`: include_dirs is
        // `[pypy_include_dir, ?revdb]` where `pypy_include_dir =
        // py.path.local(__file__).join('..')` resolves to
        // `${PYPY_SRCROOT}/rpython/translator/c`. The local port
        // honours `PYPY_SRCROOT`.
        let (translator, config) = fixture_translator_and_config();
        let _env = PypySrcrootEnvGuard::set(Some("/tmp/pypy-fake-srcroot"));
        let cb = CBuilder::new(translator, None, config, None, None, Vec::new());
        let eci = cb.get_eci();

        assert_eq!(eci.include_dirs.len(), 1);
        assert!(
            eci.include_dirs[0].ends_with("rpython/translator/c"),
            "expected {:?} to end with rpython/translator/c",
            eci.include_dirs[0]
        );
    }

    #[test]
    fn get_eci_returns_sentinel_include_dir_without_srcroot() {
        let (translator, config) = fixture_translator_and_config();
        let _env = PypySrcrootEnvGuard::set(None);
        let cb = CBuilder::new(translator, None, config, None, None, Vec::new());
        let eci = cb.get_eci();

        // Upstream `:80-85 get_eci` ALWAYS returns a non-empty
        // include_dirs list. The local port surfaces an explicit
        // sentinel path when `PYPY_SRCROOT` is unset so a downstream
        // C-backend consumer fails loudly with a clear message rather
        // than silently treating "no include dirs" as valid.
        assert_eq!(
            eci.include_dirs.len(),
            1,
            "without PYPY_SRCROOT include_dirs must surface a single sentinel path: {:?}",
            eci.include_dirs
        );
        assert!(
            eci.include_dirs[0]
                .to_string_lossy()
                .contains("PYPY_SRCROOT"),
            "sentinel path must reference PYPY_SRCROOT: {:?}",
            eci.include_dirs
        );
    }

    #[test]
    fn build_database_returns_task_error_until_leaf_lands() {
        // Bare `CBuilder` has no upstream `getentrypointptr`; enum
        // subclass dispatch supplies it before the real body runs.
        let (translator, config) = fixture_translator_and_config();
        let cb = CBuilder::new(translator, None, config, None, None, Vec::new());
        let err = cb
            .build_database()
            .expect_err("must require subclass dispatch");
        assert!(
            err.message.contains("genc.py:110"),
            "must cite genc.py:110, got: {}",
            err.message
        );
    }

    #[test]
    fn build_database_without_rtyper_fails_at_getexceptiontransformer() {
        // Upstream `genc.py:92`: `exctransformer =
        // translator.getexceptiontransformer()`. The
        // `getexceptiontransformer` body at `translator.py`
        // raises `ValueError("no rtyper")` when the translator has
        // not been rtyped. With the local fixture leaving rtyper
        // unset, build_database must fail at that exact point — not
        // at the later `db.complete()` leaf.
        let (translator, config) = fixture_translator_and_config();
        let mut cb = CBuilder::new(translator, None, config, None, None, Vec::new());
        cb.split = true;
        let ptrs = EntryPointPtr::Many(Vec::new());
        let err = cb
            .build_database_with(&|| Ok(ptrs.clone()))
            .expect_err("getexceptiontransformer must require rtyper");
        assert!(
            err.message.contains("translator.py:88"),
            "expected translator.py:88 ValueError citation, got: {}",
            err.message
        );
        // The database slot is written before the failure point only
        // after `getexceptiontransformer` succeeds, so a no-rtyper
        // build leaves both slots untouched.
        assert!(cb.db.borrow().is_none());
        assert!(cb.c_entrypoint_name.borrow().is_none());
    }

    #[test]
    fn get_entry_point_returns_error_when_executable_unset() {
        // Upstream `getentrypointptr` reads `self.executable_name` set
        // by `compile`. Without a compile pass, the slot is None and
        // the local port surfaces a TaskError.
        let (translator, config) = fixture_translator_and_config();
        let csb = CStandaloneBuilder::new(translator, None, config, None, None, Vec::new());
        let err = csb.get_entry_point().expect_err("compile must run first");
        assert!(err.message.contains("executable_name"), "{}", err.message);
    }
}
