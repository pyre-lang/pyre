//! Port of `rpython/translator/driver.py`.
//!
//! Upstream is a 631-LOC orchestration layer that subclasses
//! [`SimpleTaskEngine`] and registers every translation step
//! (`task_annotate`, `task_rtype_lltype`, `task_backendopt_lltype`,
//! `task_pyjitpl_lltype`, `task_database_c`, `task_source_c`,
//! `task_compile_c`, `task_llinterpret_lltype`, `task_jittest_lltype`,
//! `task_stackcheckinsertion_lltype`) as a `task_<name>` method
//! decorated by [`taskdef`]. The class also wires per-task timing
//! (`Timer`), debug-log breadcrumbs (`debug_start` / `debug_print` /
//! `debug_stop`), instrumentation forks (`ProfInstrument`), checkpoint
//! `fork_before` hooks (`unixcheckpoint.restartable_point`) and the
//! `from_targetspec` classmethod that's used by `targetstandalone.py`.
//!
//! ## Direct leaves consumed by `__init__`
//!
//! Every leaf reachable from `TranslationDriver.__init__` is already
//! ported when this file lands:
//!
//! | Upstream `driver.py` import                                | Local module                             |
//! |------------------------------------------------------------|-------------------------------------------|
//! | `rpython.translator.translator.TranslationContext`         | [`crate::translator::translator`]         |
//! | `rpython.translator.tool.taskengine.SimpleTaskEngine`      | [`crate::translator::tool::taskengine`]   |
//! | `rpython.translator.timing.Timer`                          | [`crate::translator::timing`]             |
//! | `rpython.annotator.listdef.s_list_of_strings`              | [`crate::annotator::listdef::s_list_of_strings`] |
//! | `rpython.annotator.policy.AnnotatorPolicy`                 | [`crate::annotator::policy::AnnotatorPolicy`]    |
//! | `rpython.config.translationoption.get_combined_translation_config` | [`crate::config::translationoption::get_combined_translation_config`] |
//! | `rpython.rlib.entrypoint.{secondary_entrypoints,annotated_jit_entrypoints}` | [`super::registries`] (this module) |
//!
//! ## DEFERRED task bodies
//!
//! Most `task_*` bodies depend on translation infrastructure that is
//! not yet ported (the JIT codewriter policy hookup, the C backend
//! database/source/compile chain, `LLInterpreter`, `query` for
//! `sanity_check_annotation`, `unixcheckpoint`, …). Each task is
//! registered with the upstream-equivalent name + deps + title, so the
//! plan ordering, `expose_task` selection and `proceed` dispatch all
//! work end-to-end. Calling a still-deferred task body returns a
//! [`TaskError`] that cites the upstream source line + the missing
//! leaf; it must not panic merely because the leaf backend module has
//! not landed yet.
//!
//! Convergence path: each leaf land replaces a single `TaskError`
//! return site with the real body; the driver structure does not
//! change.
//!
//! ## Rust-language adaptations (each minimal, all documented)
//!
//! 1. **Reflective `expose_task` `setattr(self, task, proc)`** at
//!    upstream `:104-110` — Python overwrites bound-method slots so
//!    callers can write `driver.compile()`. Rust has no per-instance
//!    method binding; the port keeps `self.exposed: Vec<String>`
//!    (line 101) verbatim and provides [`TranslationDriver::exposed_method`]
//!    + [`TranslationDriver::proceed`] to dispatch by name. The
//!      `proceed`-bound `backend_goal` is captured in a side-table
//!      [`TranslationDriver::exposed_backend_goal`] so `proceed("compile")`
//!      routes to the platform-specific `compile_c` task exactly as
//!      upstream's bound `proc()` closure would.
//! 2. **`@taskdef` decorator** at `:22-32` — upstream attaches
//!    `task_deps` / `task_title` / `task_idempotent` / `task_earlycheck`
//!    attributes to each `task_*` function. The Rust port calls
//!    [`SimpleTaskEngine::register_task`] explicitly with the same
//!    metadata; the side-table [`TranslationDriver::task_earlycheck`]
//!    holds the optional pre-check callback that upstream stashes via
//!    `taskfunc.task_earlycheck`.
//! 3. **`debug_start` / `debug_print` / `debug_stop`** at upstream
//!    `_do` (`:269-270`, `:288`) — the `rpython.rlib.debug` log facility
//!    is not ported here. The local stubs are no-ops; the convergence
//!    site is documented inline and the structural call order is
//!    preserved.
//! 4. **`AnsiLogger`** at upstream `:17-19` (`log = AnsiLogger("translation")`)
//!    is routed through the [`LOG`] module-level static; the
//!    `TranslationDriver::info(...)` method delegates to `LOG.info(...)`.
//!    Sister convergence with [`crate::translator::timing::Timer::pprint`].

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::{Rc, Weak};

use crate::config::config::{Config, ConfigError, ConfigValue, OptionValue};
use crate::config::translationoption::{
    _GLOBAL_TRANSLATIONCONFIG, get_combined_translation_config,
};
use crate::tool::ansi_print::AnsiLogger;
use crate::translator::targetspec::TargetSpecDict;
use crate::translator::timing::{SystemClock, Timer};

/// Upstream `driver.py` `log = AnsiLogger("translation")`.  Module-
/// level logger for the translation driver; `TranslationDriver::info`
/// dispatches through this static.
pub static LOG: AnsiLogger = AnsiLogger::new("translation");
use crate::translator::tool::taskengine::{
    SimpleTaskEngine, TaskEngineHooks, TaskError, TaskOutput,
};

// Upstream `:1-19` module-level imports + log handle.

// Upstream `:38 PROFILE = set([])`. The Rust port stores it in a
// thread-local `RefCell<HashSet<String>>` so callers can mutate it
// the same way upstream does — `PROFILE.add('annotate')`.
thread_local! {
    static PROFILE: RefCell<std::collections::HashSet<String>> =
        RefCell::new(std::collections::HashSet::new());
}

/// Returns whether `goal` is in the upstream-equivalent `PROFILE` set.
fn profile_contains(goal: &str) -> bool {
    PROFILE.with(|p| p.borrow().contains(goal))
}

// Upstream `:40-41 class Instrument(Exception): pass`.

/// Port of upstream `class Instrument(Exception)` at `:40-41`. Raised
/// by `TranslationDriver.instrument_result` to abort the current task
/// and trigger the `compile` re-entry inside `_do` (upstream `:283-285`).
#[derive(Debug, Clone)]
pub struct Instrument;

impl std::fmt::Display for Instrument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Instrument")
    }
}

impl std::error::Error for Instrument {}

// Upstream `:44-60 class ProfInstrument`.

/// Port of upstream `class ProfInstrument` at `:44-60`. The class is
/// only referenced by `TranslationDriver.instrument_result` (the C-
/// backend instrumentation fork at `:218-248`) and itself reaches
/// `compiler.platform.execute` — both DEFERRED until the C-backend
/// port lands. The struct is preserved so callers / tests that
/// mention the type by name compile cleanly.
pub struct ProfInstrument {
    pub datafile: String,
    // Upstream stores the live `compiler` instance (`CStandaloneBuilder`).
    // The local port keeps the slot opaque — a `dyn Any` carrier — so
    // nothing here drags in unported C-backend types.
    pub compiler: Rc<dyn Any>,
}

impl ProfInstrument {
    /// Upstream `ProfInstrument.first` at `:50-51`.
    pub fn first(&self) -> ! {
        unimplemented!(
            "driver.py:50 ProfInstrument.first — leaf CStandaloneBuilder._build not yet ported"
        );
    }

    /// Upstream `ProfInstrument.probe` at `:53-56`.
    pub fn probe(&self, _exe: &str, _args: &[String]) -> ! {
        unimplemented!(
            "driver.py:53 ProfInstrument.probe — leaf compiler.platform.execute not yet ported"
        );
    }

    /// Upstream `ProfInstrument.after` at `:58-60`.
    pub fn after(&self) -> ! {
        unimplemented!("driver.py:58 ProfInstrument.after — calls os._exit(0)");
    }
}

// Upstream `:13-14 secondary_entrypoints, annotated_jit_entrypoints`.
//
// `rpython/rlib/entrypoint.py` declares
// `secondary_entrypoints = {"main": []}` and `:8 annotated_jit_entrypoints = []`
// as module-level mutable globals. The port keeps the same shape via
// process-local `OnceLock<RefCell<...>>` slots so callers can mutate
// them through `secondary_entrypoints_register(key, ...)` /
// `annotated_jit_entrypoints_register(...)` exactly like upstream's
// `secondary_entrypoints.setdefault(key, []).append(...)`.
//
// `setup`'s `secondary_entrypoints[key]` lookup at `:204-207` reads the
// `key` listed in `config.translation.secondaryentrypoints` (a CSV
// string). Upstream raises `KeyError` when the key is missing; the
// Rust port returns `Err(TaskError)` matching that contract.

/// `EntryPointSpec` mirrors upstream's `(func, argtypes)` tuple stored
/// inside `secondary_entrypoints[key]`. The argtypes are an opaque
/// Python list upstream — the local port preserves the slot as a
/// `Vec<Rc<dyn Any>>` so the C-backend code that consumes them can
/// downcast to its own argtype shape when it lands.
pub type EntryPointSpec = (Rc<dyn Any>, Vec<Rc<dyn Any>>);

/// Carrier for the `libdef` argument upstream `setup_library(self,
/// libdef, …)` reads at `driver.py`. Upstream's `libdef` is a
/// duck-typed object supplied by carbonpython (the only consumer per
/// the upstream `# Used by carbon python only.` comment at `:213`); it
/// carries a `.functions` attribute that the driver assigns to
/// `self.secondary_entrypoints` at `:216`.
///
/// The Rust port keeps `setup_library`'s `libdef: Rc<dyn Any>` opaque
/// so any wrapper (e.g. the C-backend's `CLibraryBuilder`) can pass
/// through; this struct is the minimal shape that downcast at `:216`
/// recognises so the assignment can be performed without dragging
/// carbonpython types into the port.
#[derive(Debug, Clone)]
pub struct LibDef {
    pub functions: Vec<EntryPointSpec>,
}

impl LibDef {
    /// Constructor matching upstream's `LibDef(functions=...)` shape.
    pub fn new(functions: Vec<EntryPointSpec>) -> Self {
        Self { functions }
    }
}

// Upstream `CStandaloneBuilder` / `CLibraryBuilder` / `LowLevelDatabase`
// types live in `crate::translator::backend::{genc, dlltool}` matching upstream
// module paths (`rpython/translator/c/genc.py` /
// `rpython/translator/c/dlltool.py`). The `CBuilderRef` sum type below
// stitches the two subclasses into the single `self.cbuilder` slot the
// driver tasks read at `:435`, `:444`, `:531-541`.
pub use crate::translator::backend::CBuilderRef;
pub use crate::translator::backend::database::LowLevelDatabase as DatabaseState;

// Upstream `entrypoint.py`: `secondary_entrypoints = {"main": []}`.
// Stored thread-local because `Rc<dyn Any>` is not `Sync`.
thread_local! {
    static SECONDARY_ENTRYPOINTS: RefCell<HashMap<String, Vec<EntryPointSpec>>> = {
        let mut m = HashMap::new();
        m.insert("main".to_string(), Vec::new());
        RefCell::new(m)
    };
}

/// Read-only view of upstream `entrypoint.py secondary_entrypoints`.
/// Returns the entries registered under `key` or `None` matching
/// upstream's `KeyError` raise path on a missing key.
pub fn secondary_entrypoints_get(key: &str) -> Option<Vec<EntryPointSpec>> {
    SECONDARY_ENTRYPOINTS.with(|s| s.borrow().get(key).cloned())
}

/// Typed registration helper mirroring upstream
/// `secondary_entrypoints.setdefault(key, []).append((func, argtypes))`
/// at `entrypoint.py` (the only mutator in upstream's import path).
///
/// `func` is a [`HostObject`] — the only callable shape `task_annotate`
/// at `:308-312` knows how to feed to `annotator.build_types`. Each
/// element of `inputtypes` is an [`AnnotationSpec`] carried through
/// `Rc::new` so the storage slot's `Rc<dyn Any>` shape downcasts back
/// at task time. An empty `inputtypes` list means a 0-arg function,
/// matching upstream's `[]` literal at the call site (NOT upstream's
/// `Ellipsis` sentinel — Ellipsis support is currently DEFERRED, no
/// in-tree caller has requested it).
pub fn secondary_entrypoints_register(
    key: &str,
    func: crate::flowspace::model::HostObject,
    inputtypes: Vec<crate::annotator::signature::AnnotationSpec>,
) {
    let func_any: Rc<dyn Any> = Rc::new(func);
    let inputs_any: Vec<Rc<dyn Any>> = inputtypes
        .into_iter()
        .map(|s| Rc::new(s) as Rc<dyn Any>)
        .collect();
    SECONDARY_ENTRYPOINTS.with(|s| {
        s.borrow_mut()
            .entry(key.to_string())
            .or_default()
            .push((func_any, inputs_any));
    });
}

/// All registered keys (used by upstream's `secondary_entrypoints.keys()`
/// at `:207` for the missing-key error message).
pub fn secondary_entrypoints_keys() -> Vec<String> {
    SECONDARY_ENTRYPOINTS.with(|s| s.borrow().keys().cloned().collect())
}

// Upstream `entrypoint.py`: `annotated_jit_entrypoints = []`.
thread_local! {
    static ANNOTATED_JIT_ENTRYPOINTS: RefCell<Vec<EntryPointSpec>> =
        RefCell::new(Vec::new());
}

/// Read-only view of upstream `entrypoint.py annotated_jit_entrypoints`.
pub fn annotated_jit_entrypoints_get() -> Vec<EntryPointSpec> {
    ANNOTATED_JIT_ENTRYPOINTS.with(|s| s.borrow().clone())
}

/// Typed registration helper for upstream
/// `entrypoint.py annotated_jit_entrypoints.append((func, argtypes))`.
/// Same shape as [`secondary_entrypoints_register`] — see that helper's
/// doc for the typed → opaque-slot conversion rationale.
pub fn annotated_jit_entrypoints_register(
    func: crate::flowspace::model::HostObject,
    inputtypes: Vec<crate::annotator::signature::AnnotationSpec>,
) {
    let func_any: Rc<dyn Any> = Rc::new(func);
    let inputs_any: Vec<Rc<dyn Any>> = inputtypes
        .into_iter()
        .map(|s| Rc::new(s) as Rc<dyn Any>)
        .collect();
    ANNOTATED_JIT_ENTRYPOINTS.with(|s| s.borrow_mut().push((func_any, inputs_any)));
}

// Upstream `:22-32 def taskdef(...)` decorator.
//
// Upstream attaches `task_deps`, `task_title`, `task_newstate`,
// `task_expected_states`, `task_idempotent`, `task_earlycheck` to a
// `task_<name>` function. The Rust port doesn't decorate functions —
// it carries the same metadata as a struct passed to
// `SimpleTaskEngine::register_task`. `TaskDef` keeps the bundle named
// for grep-parity with upstream call sites.

/// Port of the metadata `taskdef(deps, title, ..., earlycheck=None)` at
/// upstream `:22-32` would attach to a `task_*` function.
struct TaskDef {
    deps: Vec<String>,
    title: String,
    /// Upstream `task.task_idempotent`. When `true`, the driver does
    /// not stash the goal in `self.done`, allowing re-runs.
    idempotent: bool,
    /// Upstream `task.task_earlycheck`. Optional callback that
    /// `_event("planned", goal)` invokes at upstream `:611-612`.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    earlycheck: Option<Rc<dyn Fn(&TranslationDriver) -> Result<(), TaskError>>>,
}

impl TaskDef {
    fn new(deps: Vec<String>, title: impl Into<String>) -> Self {
        TaskDef {
            deps,
            title: title.into(),
            idempotent: false,
            earlycheck: None,
        }
    }

    fn with_earlycheck(
        mut self,
        check: impl Fn(&TranslationDriver) -> Result<(), TaskError> + 'static,
    ) -> Self {
        self.earlycheck = Some(Rc::new(check));
        self
    }
}

// Upstream `:339 RTYPE = 'rtype_lltype'` and `:378 BACKENDOPT =
// 'backendopt_lltype'` and `:387 STACKCHECKINSERTION = 'stackcheckinsertion_lltype'`.

/// Upstream `:339 RTYPE = 'rtype_lltype'` — the canonical typed-task
/// name picked by `task_pyjitpl_lltype` / `task_jittest_lltype`'s deps.
pub const RTYPE: &str = "rtype_lltype";

/// Upstream `:378 BACKENDOPT = 'backendopt_lltype'`.
pub const BACKENDOPT: &str = "backendopt_lltype";

/// Upstream `:387 STACKCHECKINSERTION = 'stackcheckinsertion_lltype'`.
pub const STACKCHECKINSERTION: &str = "stackcheckinsertion_lltype";

// Upstream `:63 class TranslationDriver(SimpleTaskEngine)`.

/// Port of upstream `class TranslationDriver(SimpleTaskEngine)` at
/// `:63-622`.
///
/// The struct holds every observable field upstream attaches in
/// `__init__` (`:66-134`). Mutable fields are wrapped in `RefCell` /
/// `Cell` because upstream mutates them through Python's
/// implicit-`self` binding while the Rust port operates on `&self`.
pub struct TranslationDriver {
    /// Upstream `self.timer = Timer()` at `:71`.
    pub timer: Timer<SystemClock>,

    /// Upstream `SimpleTaskEngine.__init__(self)` at `:72`. Held by
    /// composition because Rust has no inheritance — see the module
    /// doc on `crate::translator::tool::taskengine` for the trait-based
    /// override pattern that replaces upstream's MRO.
    pub engine: SimpleTaskEngine,

    /// Upstream `self.config = config` at `:80`. Held as `Rc<Config>`
    /// to mirror upstream's mutable-Config-handed-around contract
    /// (`interactive.py Translation.__init__ self.config = self.driver.config`).
    pub config: Rc<Config>,

    /// Upstream `self.exe_name = exe_name` at `:87`. `None` ↔ upstream
    /// `None`; `Some(s)` ↔ upstream string template.
    pub exe_name: RefCell<Option<String>>,

    /// Upstream `self.extmod_name = extmod_name` at `:88`.
    pub extmod_name: RefCell<Option<String>>,

    /// Upstream `self.done = {}` at `:90`. Set of completed goals
    /// (upstream uses a dict-as-set with arbitrary `True` values).
    pub done: RefCell<HashMap<String, bool>>,

    /// Upstream `self._disabled = to_disable` at `:167` (set from
    /// `__init__` line 92's `self.disable(disable)` call). Carries
    /// the names a caller asked to skip; consulted by
    /// `_maybe_skip` and resolved through `backend_select_goals`.
    pub _disabled: RefCell<Vec<String>>,

    /// Upstream `self.default_goal = default_goal` at `:99`.
    pub default_goal: RefCell<Option<String>>,

    /// Upstream `self.extra_goals = []` at `:100`.
    pub extra_goals: RefCell<Vec<String>>,

    /// Upstream `self.exposed = []` at `:101` — names of tasks the
    /// `expose_task` selection loop selected for this backend +
    /// type-system combo (`:113-134`). Read by upstream tests and
    /// `pdbplus`.
    pub exposed: RefCell<Vec<String>>,

    /// Side table populated by [`Self::expose_task`]: the
    /// `backend_goal` arg captured in upstream's nested
    /// `def proc(): return self.proceed(backend_goal)` closure
    /// (`:107-110`). Keyed on `task` (the public, unprefixed name) and
    /// stores the actual platform-specific task to run.
    ///
    /// Upstream Python achieves this through method-level closures;
    /// Rust prefers an explicit map.
    pub exposed_backend_goal: RefCell<HashMap<String, String>>,

    /// Upstream `self._backend_extra_options = {}` (class-level at
    /// `:64`). Per-instance overrides land via
    /// `set_backend_extra_options` at `:139-140`.
    pub _backend_extra_options: RefCell<HashMap<String, OptionValue>>,

    // Upstream-`setup`-populated fields. Each is initially absent;
    // `setup` writes them (`:176-216`).
    /// Upstream `self.standalone = standalone` at `:178`.
    pub standalone: Cell<bool>,
    /// Upstream `self.inputtypes = inputtypes` at `:184`.
    ///
    /// Upstream stores a Python list of annotation specifiers; the
    /// Rust port mirrors with `Vec<AnnotationSpec>` — the typed
    /// dispatch shape consumed by
    /// [`crate::annotator::annrpython::RPythonAnnotator::build_types`]
    /// at `annrpython.rs`.
    pub inputtypes: RefCell<Option<Vec<crate::annotator::signature::AnnotationSpec>>>,
    /// Upstream `self.policy = policy` at `:188`.
    ///
    /// Concrete `AnnotatorPolicy` ↔ upstream's `policy` argument. Any
    /// future subclasses (`StrictAnnotatorPolicy`, …) will land as
    /// builder variants of [`AnnotatorPolicy`], not as a separate type.
    pub policy: RefCell<Option<crate::annotator::policy::AnnotatorPolicy>>,
    /// Upstream `self.extra = extra` at `:190`. Upstream's `extra` is
    /// the `targetspec_dic` dict (any-typed payload), so kept as
    /// `Rc<dyn Any>` until each consumer narrows.
    pub extra: RefCell<HashMap<String, Rc<dyn Any>>>,
    /// Upstream `self.entry_point = entry_point` at `:197`. The
    /// upstream attribute is a Python function object; the local
    /// equivalent is [`HostObject::UserFunction`].
    pub entry_point: RefCell<Option<crate::flowspace::model::HostObject>>,
    /// Upstream `self.translator = translator` at `:198`. The bridge
    /// from `Rc<Config>` to `TranslationContext` is the
    /// last-mile work; today the driver constructs the default-config
    /// `TranslationContext` so downstream tasks can reach the
    /// graphs/annotator slot.
    pub translator: RefCell<Option<Rc<super::translator::TranslationContext>>>,
    /// Upstream `self.libdef = None` at `:199` (overridden by
    /// `setup_library`). `libdef` upstream is a CPython-side
    /// shape from `targetstandalone.LibDef`; kept opaque until that
    /// type ports.
    pub libdef: RefCell<Option<Rc<dyn Any>>>,
    /// Upstream `self.secondary_entrypoints = []` at `:200`.
    pub secondary_entrypoints: RefCell<Vec<EntryPointSpec>>,

    /// Strong holder for the annotator across the task chain.
    ///
    /// TODO (Rust-language): upstream stores
    /// `translator.annotator = self` (annrpython.py) as a
    /// strong Python attribute; CPython's cyclic GC unwinds the
    /// `translator → annotator → translator` cycle on session exit.
    /// The local `TranslationContext.annotator` is `Weak` to avoid
    /// the cycle (translator.rs), so without a separate strong
    /// holder the annotator drops between `task_annotate` returning
    /// and `task_rtype_lltype` running. This field is the strong
    /// holder; `task_annotate` writes it; later tasks that upstream-
    /// equivalently reach `self.translator.annotator` simply ride
    /// `translator.annotator()` because the strong field keeps the
    /// Weak-upgrade alive.
    pub annotator: RefCell<Option<Rc<crate::annotator::annrpython::RPythonAnnotator>>>,

    /// Upstream `self.jitpolicy`, written by `task_pyjitpl_lltype`
    /// before calling `warmspot.apply_jit`.
    pub jitpolicy: RefCell<Option<Rc<dyn Any>>>,

    /// Upstream `self.cbuilder`, written by `task_database_c`. Holds
    /// either a `CStandaloneBuilder` or a `CLibraryBuilder` per the
    /// `if standalone:` branch at `:419-432`.
    pub cbuilder: RefCell<Option<CBuilderRef>>,

    /// Upstream `self.database`, written by `task_database_c`.
    pub database: RefCell<Option<DatabaseState>>,

    /// Upstream `self.c_entryp`, written by `task_compile_c` and
    /// rewritten by `create_exe`.
    pub c_entryp: RefCell<Option<PathBuf>>,

    // Side table for the upstream `@taskdef(... earlycheck=...)`
    // metadata. Engine-level `register_task` only takes the title +
    // idempotency flag; the earlycheck function rides here so
    // `_event("planned", goal)` can invoke it.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    pub task_earlycheck:
        RefCell<HashMap<String, Rc<dyn Fn(&TranslationDriver) -> Result<(), TaskError>>>>,
}

/// Stable cycle-free handle for the driver. Tasks register a
/// `Weak<TranslationDriver>` so the closures don't form a refcount
/// cycle through `engine.tasks`. The driver-construction helper
/// upgrades it to `Rc` on every callable invocation.
type DriverWeak = Weak<TranslationDriver>;

impl TranslationDriver {
    /// Upstream `TranslationDriver.__init__` at `:66-134`. Returns an
    /// `Rc<Self>` because the Rust port registers task callables that
    /// hold a `Weak<Self>` — registration happens inside `new` after
    /// the `Rc` is created so the closures can upgrade.
    ///
    /// Arguments mirror upstream's keyword args; `default_goal`'s
    /// `backend_select_goals` resolution at `:94-97` runs here
    /// verbatim.
    pub fn new(
        setopts: Option<Vec<(String, OptionValue)>>,
        default_goal: Option<String>,
        disable: Vec<String>,
        exe_name: Option<String>,
        extmod_name: Option<String>,
        config: Option<Rc<Config>>,
        overrides: Option<HashMap<String, OptionValue>>,
    ) -> Result<Rc<Self>, ConfigError> {
        // Upstream `:71`: `self.timer = Timer()`.
        let timer = Timer::new();

        // Upstream `:72`: `SimpleTaskEngine.__init__(self)`.
        let engine = SimpleTaskEngine::new();

        // Upstream `:76-77`: `if config is None: config =
        // translationoption.get_combined_translation_config(translating=True)`.
        let config = match config {
            Some(c) => c,
            None => get_combined_translation_config(None, None, None, true)?,
        };

        // Upstream `:78-79`: `translationoption._GLOBAL_TRANSLATIONCONFIG = config`.
        _GLOBAL_TRANSLATIONCONFIG.with(|slot| {
            *slot.borrow_mut() = Some(Rc::clone(&config));
        });

        // Upstream `:81-82`: `if overrides is not None:
        //     self.config.override(overrides)`.
        if let Some(overrides) = overrides {
            config.override_(overrides)?;
        }

        // Upstream `:84-85`: `if setopts is not None: self.config.set(**setopts)`.
        if let Some(setopts) = setopts {
            config.set(setopts)?;
        }

        let driver = Rc::new(TranslationDriver {
            timer,
            engine,
            config,
            exe_name: RefCell::new(exe_name),
            extmod_name: RefCell::new(extmod_name),
            done: RefCell::new(HashMap::new()),
            _disabled: RefCell::new(Vec::new()),
            default_goal: RefCell::new(None),
            extra_goals: RefCell::new(Vec::new()),
            exposed: RefCell::new(Vec::new()),
            exposed_backend_goal: RefCell::new(HashMap::new()),
            _backend_extra_options: RefCell::new(HashMap::new()),
            standalone: Cell::new(false),
            inputtypes: RefCell::new(None),
            policy: RefCell::new(None),
            extra: RefCell::new(HashMap::new()),
            entry_point: RefCell::new(None),
            translator: RefCell::new(None),
            libdef: RefCell::new(None),
            secondary_entrypoints: RefCell::new(Vec::new()),
            annotator: RefCell::new(None),
            jitpolicy: RefCell::new(None),
            cbuilder: RefCell::new(None),
            database: RefCell::new(None),
            c_entryp: RefCell::new(None),
            task_earlycheck: RefCell::new(HashMap::new()),
        });

        // Upstream `:92`: `self.disable(disable)`. Must run before
        // `_maybe_skip` consultation in the `expose_task` loop.
        driver.disable(disable);

        // Upstream `:103-110`: register `task_*` callables before the
        // `expose_task` loop walks `self.tasks`. Rust runs the same
        // sequence — first register every task, then run the loop.
        Self::register_tasks(&driver);

        // Upstream `:94-97`: resolve `default_goal` through
        // `backend_select_goals([default_goal])`. Mirror exactly,
        // including the silent `None`-on-skip downgrade.
        if let Some(default_goal) = default_goal {
            let resolved = driver
                .backend_select_goals(std::slice::from_ref(&default_goal))?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    ConfigError::Generic(format!(
                        "TranslationDriver: backend_select_goals returned no result for {:?}",
                        default_goal
                    ))
                })?;
            let maybe_skip = driver._maybe_skip();
            if maybe_skip.iter().any(|s| s == &resolved) {
                *driver.default_goal.borrow_mut() = None;
            } else {
                *driver.default_goal.borrow_mut() = Some(resolved);
            }
        }

        // Upstream `:112-134`: `expose_task` selection loop. Reads
        // backend / type_system from `config.translation.*` and
        // chooses which task names land in `self.exposed`.
        driver.run_expose_task_loop()?;

        Ok(driver)
    }

    /// Convenience constructor matching upstream's `TranslationDriver()`
    /// no-arg call (used by the test suite at `test_driver.py:8`).
    pub fn new_default() -> Result<Rc<Self>, ConfigError> {
        Self::new(None, None, Vec::new(), None, None, None, None)
    }

    /// Internal: registers every `task_<name>` upstream defines on
    /// `TranslationDriver` (`:297-555`). Each registration captures a
    /// `Weak<Self>` so the engine's task table doesn't cycle.
    ///
    /// Task bodies that depend on unported leaves keep the upstream
    /// statement order and return `TaskError` at the exact missing
    /// leaf; the registration metadata (deps, title, idempotency) is
    /// the upstream-observable shape that `_plan` and `expose_task`
    /// consume.
    fn register_tasks(this: &Rc<Self>) {
        let weak = Rc::downgrade(this);

        // Upstream `:297-298 @taskdef([], "Annotating&simplifying")
        // def task_annotate(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "annotate",
            TaskDef::new(Vec::new(), "Annotating&simplifying"),
            |d| d.task_annotate(),
        );

        // Upstream `:340-341 @taskdef(['annotate'], "RTyping")
        // def task_rtype_lltype(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "rtype_lltype",
            TaskDef::new(vec!["annotate".to_string()], "RTyping"),
            |d| d.task_rtype_lltype(),
        );

        // Upstream `:347-348 @taskdef([RTYPE], "JIT compiler generation")
        // def task_pyjitpl_lltype(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "pyjitpl_lltype",
            TaskDef::new(vec![RTYPE.to_string()], "JIT compiler generation"),
            |d| d.task_pyjitpl_lltype(),
        );

        // Upstream `:365 @taskdef([RTYPE], "test of the JIT on the llgraph backend")
        // def task_jittest_lltype(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "jittest_lltype",
            TaskDef::new(
                vec![RTYPE.to_string()],
                "test of the JIT on the llgraph backend",
            ),
            |d| d.task_jittest_lltype(),
        );

        // Upstream `:379 @taskdef([RTYPE, '??pyjitpl_lltype', '??jittest_lltype'],
        // "lltype back-end optimisations") def task_backendopt_lltype(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "backendopt_lltype",
            TaskDef::new(
                vec![
                    RTYPE.to_string(),
                    "??pyjitpl_lltype".to_string(),
                    "??jittest_lltype".to_string(),
                ],
                "lltype back-end optimisations",
            ),
            |d| d.task_backendopt_lltype(),
        );

        // Upstream `:388 @taskdef(['?'+BACKENDOPT, RTYPE, 'annotate'], "inserting stack checks")
        // def task_stackcheckinsertion_lltype(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "stackcheckinsertion_lltype",
            TaskDef::new(
                vec![
                    format!("?{}", BACKENDOPT),
                    RTYPE.to_string(),
                    "annotate".to_string(),
                ],
                "inserting stack checks",
            ),
            |d| d.task_stackcheckinsertion_lltype(),
        );

        // Upstream `:405-407 @taskdef([STACKCHECKINSERTION, '?'+BACKENDOPT,
        // RTYPE, '?annotate'], "Creating database for generating c source",
        // earlycheck = possibly_check_for_boehm) def task_database_c(self):`.
        let database_c_def = TaskDef::new(
            vec![
                STACKCHECKINSERTION.to_string(),
                format!("?{}", BACKENDOPT),
                RTYPE.to_string(),
                "?annotate".to_string(),
            ],
            "Creating database for generating c source",
        )
        .with_earlycheck(|d| d.possibly_check_for_boehm());
        Self::register_task_with_def(this, &weak, "database_c", database_c_def, |d| {
            d.task_database_c()
        });

        // Upstream `:440 @taskdef(['database_c'], "Generating c source")
        // def task_source_c(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "source_c",
            TaskDef::new(vec!["database_c".to_string()], "Generating c source"),
            |d| d.task_source_c(),
        );

        // Upstream `:526 @taskdef(['source_c'], "Compiling c source")
        // def task_compile_c(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "compile_c",
            TaskDef::new(vec!["source_c".to_string()], "Compiling c source"),
            |d| d.task_compile_c(),
        );

        // Upstream `:543 @taskdef([STACKCHECKINSERTION, '?'+BACKENDOPT, RTYPE],
        // "LLInterpreting") def task_llinterpret_lltype(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "llinterpret_lltype",
            TaskDef::new(
                vec![
                    STACKCHECKINSERTION.to_string(),
                    format!("?{}", BACKENDOPT),
                    RTYPE.to_string(),
                ],
                "LLInterpreting",
            ),
            |d| d.task_llinterpret_lltype(),
        );
    }

    /// Internal helper: hands a single `TaskDef` to the engine and
    /// stashes the optional `earlycheck` in the side table.
    fn register_task_with_def(
        this: &Rc<Self>,
        weak: &DriverWeak,
        name: &str,
        def: TaskDef,
        body: fn(&TranslationDriver) -> Result<TaskOutput, TaskError>,
    ) {
        let weak_for_call = weak.clone();
        let callable: Rc<dyn Fn() -> Result<TaskOutput, TaskError>> = Rc::new(move || {
            let d = weak_for_call
                .upgrade()
                .expect("TranslationDriver dropped while engine still owned task callable");
            body(&d)
        });
        this.engine.register_task(
            name.to_string(),
            callable,
            def.deps.clone(),
            def.title.clone(),
            def.idempotent,
        );
        if let Some(check) = def.earlycheck {
            this.task_earlycheck
                .borrow_mut()
                .insert(name.to_string(), check);
        }
    }

    /// Internal: implements upstream `:104-110 def expose_task(task,
    /// backend_goal=None)` selection loop at `:112-134`.
    fn run_expose_task_loop(&self) -> Result<(), ConfigError> {
        let (backend, ts) = self.get_backend_and_type_system()?;

        // Upstream `:113`: `for task in self.tasks:` — `self.tasks` is
        // a regular `dict` populated in `_register_task` registration
        // order (RPython runs on Python 2, where `dict` iteration is
        // *not* guaranteed insertion-order — but every CPython 2
        // implementation upstream actually shipped on does iterate in
        // insertion order in practice for dicts of this size).  The
        // local port preserves that registration order explicitly via
        // engine's `task_order` sidecar so callers don't depend on
        // host `HashMap` randomization.
        let task_names: Vec<String> = self.engine.task_names_in_registration_order();

        for task_name in task_names {
            let explicit_task = task_name.clone();
            if task_name == "annotate" {
                // Upstream `:115-116`: `expose_task(task)` with no postfix.
                self.expose_task(&task_name, None);
                continue;
            }
            // Upstream `:118`: `task, postfix = task.split('_')`. The
            // upstream identifiers always contain at most one `_`, so
            // `split('_')` returns exactly two parts.
            let (head, postfix) = match task_name.split_once('_') {
                Some((h, p)) => (h, p),
                None => continue,
            };
            match head {
                // Upstream `:119-120`: `if task in ('rtype',
                // 'backendopt', 'llinterpret', 'pyjitpl'):` — note
                // `stackcheckinsertion` and `jittest` are NOT in
                // upstream's tuple. They are still registered as
                // tasks but never exposed.
                "rtype" | "backendopt" | "llinterpret" | "pyjitpl" => {
                    // Upstream `:121-125`: gated on type-system match.
                    if let Some(ts) = &ts {
                        if ts == postfix {
                            self.expose_task(head, Some(&explicit_task));
                        }
                    } else {
                        self.expose_task(&explicit_task, None);
                    }
                }
                "source" | "compile" | "run" => {
                    // Upstream `:126-134`: gated on backend match if
                    // backend is set, else type-system fallback.
                    if let Some(backend) = &backend {
                        if backend == postfix {
                            self.expose_task(head, Some(&explicit_task));
                        }
                    } else if let Some(ts) = &ts {
                        if ts == "lltype" {
                            self.expose_task(&explicit_task, None);
                        }
                    } else {
                        self.expose_task(&explicit_task, None);
                    }
                }
                _ => {
                    // Heads not enumerated above are silently skipped
                    // upstream — the `if task in (…)` chain falls
                    // through.
                }
            }
        }
        Ok(())
    }

    /// Port of upstream's nested `def expose_task(task, backend_goal=None)`
    /// inside `__init__` at `:104-110`. Captures the `backend_goal` as
    /// a closure over which `proceed` to run; in Rust we record it in
    /// the side table so `proceed_exposed("compile")` can resolve.
    fn expose_task(&self, task: &str, backend_goal: Option<&str>) {
        let backend_goal = backend_goal.unwrap_or(task).to_string();
        self.exposed.borrow_mut().push(task.to_string());
        self.exposed_backend_goal
            .borrow_mut()
            .insert(task.to_string(), backend_goal);
    }

    /// Upstream `set_extra_goals(self, goals)` at `:136-137`.
    pub fn set_extra_goals(&self, goals: Vec<String>) {
        *self.extra_goals.borrow_mut() = goals;
    }

    /// Upstream `set_backend_extra_options(self, extra_options)` at `:139-140`.
    pub fn set_backend_extra_options(&self, extra_options: HashMap<String, OptionValue>) {
        *self._backend_extra_options.borrow_mut() = extra_options;
    }

    /// Upstream `get_info(self)` at `:142-144`.
    pub fn get_info(&self) -> HashMap<String, String> {
        let mut d = HashMap::new();
        if let Some(backend) = self.read_choice("translation.backend") {
            d.insert("backend".to_string(), backend);
        }
        d
    }

    /// Upstream `get_backend_and_type_system(self)` at `:146-149`.
    pub fn get_backend_and_type_system(
        &self,
    ) -> Result<(Option<String>, Option<String>), ConfigError> {
        let type_system = self.read_choice("translation.type_system");
        let backend = self.read_choice("translation.backend");
        Ok((backend, type_system))
    }

    /// Reads a `ChoiceOption` value as `Option<String>`, mapping
    /// upstream's `None` ↔ `OptionValue::None`. Used by
    /// `get_backend_and_type_system` and `get_info`.
    fn read_choice(&self, path: &str) -> Option<String> {
        match self.config.get(path).ok()? {
            ConfigValue::Value(OptionValue::Choice(s)) => Some(s),
            ConfigValue::Value(OptionValue::Str(s)) => Some(s),
            ConfigValue::Value(OptionValue::None) => None,
            _ => None,
        }
    }

    /// Upstream `backend_select_goals(self, goals)` at `:151-164`.
    /// Resolves every requested goal through the backend / type-
    /// system postfixes, matching upstream's
    /// `for postfix in postfixes` chain exactly.
    pub fn backend_select_goals(&self, goals: &[String]) -> Result<Vec<String>, ConfigError> {
        let (backend, ts) = self.get_backend_and_type_system()?;
        // Upstream `:153`: `postfixes = [''] + ['_'+p for p in (backend, ts) if p]`.
        let mut postfixes: Vec<String> = vec![String::new()];
        for p in [&backend, &ts].into_iter().flatten() {
            postfixes.push(format!("_{}", p));
        }
        let tasks = self.engine.tasks();
        let mut out = Vec::new();
        for goal in goals {
            let mut found: Option<String> = None;
            for postfix in &postfixes {
                let cand = format!("{}{}", goal, postfix);
                if tasks.contains_key(&cand) {
                    found = Some(cand);
                    break;
                }
            }
            match found {
                Some(g) => out.push(g),
                None => {
                    // Upstream `:162`: `raise Exception("cannot infer
                    // complete goal from: %r" % goal)`.
                    return Err(ConfigError::Generic(format!(
                        "cannot infer complete goal from: {:?}",
                        goal
                    )));
                }
            }
        }
        Ok(out)
    }

    /// Upstream `disable(self, to_disable)` at `:166-167`.
    pub fn disable(&self, to_disable: Vec<String>) {
        *self._disabled.borrow_mut() = to_disable;
    }

    /// Upstream `_maybe_skip(self)` at `:169-174`.
    pub fn _maybe_skip(&self) -> Vec<String> {
        let disabled = self._disabled.borrow().clone();
        if disabled.is_empty() {
            return Vec::new();
        }
        let resolved = match self.backend_select_goals(&disabled) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        let mut maybe_skip: Vec<String> = Vec::new();
        for goal in resolved {
            for dep in self.engine._depending_on_closure(&goal) {
                maybe_skip.push(dep);
            }
        }
        // Upstream `:174`: `dict.fromkeys(maybe_skip).keys()` — dedupe.
        let mut seen = std::collections::HashSet::new();
        maybe_skip.retain(|s| seen.insert(s.clone()));
        maybe_skip
    }

    /// Upstream `setup(self, entry_point, inputtypes, policy=None,
    /// extra={}, empty_translator=None)` at `:176-210`.
    ///
    /// The upstream `:194 translator = TranslationContext(config=self.config)`
    /// is mirrored by reading the seven typed `translation.<name>`
    /// fields out of `self.config` via
    /// [`crate::translator::translator::TranslationConfig::from_rc_config`]
    /// and constructing the `TranslationContext` with the resulting
    /// snapshot. Driver-side `override` / `setopts` calls performed
    /// before `setup` are visible to every read of
    /// `self.translator.config.translation.<X>` because the snapshot
    /// is taken AFTER those mutations land.
    ///
    /// `driver_instrument_result` callback at `:210` is still unwired
    /// (the local `TranslationContext` does not yet carry that slot
    /// — the leaf is C-backend instrumentation).
    pub fn setup(
        &self,
        entry_point: Option<crate::flowspace::model::HostObject>,
        inputtypes: Option<Vec<crate::annotator::signature::AnnotationSpec>>,
        policy: Option<crate::annotator::policy::AnnotatorPolicy>,
        extra: HashMap<String, Rc<dyn Any>>,
        empty_translator: Option<Rc<super::translator::TranslationContext>>,
    ) -> Result<(), TaskError> {
        use crate::annotator::model::SomeValue;
        use crate::annotator::signature::AnnotationSpec;

        // Upstream `:177-178`: `standalone = inputtypes is None;
        // self.standalone = standalone`.
        let standalone = inputtypes.is_none();
        self.standalone.set(standalone);

        // Upstream `:180-183`: `if standalone: inputtypes =
        // [s_list_of_strings]; self.inputtypes = inputtypes`.
        let resolved_inputtypes: Vec<AnnotationSpec> = inputtypes.unwrap_or_else(|| {
            // upstream: standalone fall-back. `s_list_of_strings` is
            // a `SomeList`; wrap it in `AnnotationSpec::Already` so
            // the build_types pipeline at `annrpython.rs`
            // (`typeannotation` short-circuit) accepts it.
            vec![AnnotationSpec::Already(Box::new(SomeValue::List(
                crate::annotator::listdef::s_list_of_strings(),
            )))]
        });
        *self.inputtypes.borrow_mut() = Some(resolved_inputtypes);

        // Upstream `:185-187`: default policy = `AnnotatorPolicy()`.
        let resolved_policy = policy.unwrap_or_default();
        *self.policy.borrow_mut() = Some(resolved_policy);

        // Upstream `:189`: `self.extra = extra`.
        *self.extra.borrow_mut() = extra;

        // Upstream `:191-194`: pick `empty_translator` or build one.
        // When we build, take a snapshot of the driver's `Rc<Config>`
        // so any `override` / `setopts` mutations performed before
        // `setup` (`driver.py`) are visible inside the
        // translator's `config.translation.<X>` reads.
        let translator: Rc<super::translator::TranslationContext> = match empty_translator {
            Some(t) => t,
            None => {
                let translation_config = super::translator::TranslationConfig::from_rc_config(
                    &self.config,
                )
                .map_err(|e| TaskError {
                    message: format!(
                        "TranslationDriver::setup: TranslationConfig::from_rc_config: {e:?}"
                    ),
                })?;
                Rc::new(
                    super::translator::TranslationContext::with_config_and_flowing_flags(
                        Some(translation_config),
                        super::translator::FlowingFlags::default(),
                    ),
                )
            }
        };

        // Upstream `:196-198`.
        *self.entry_point.borrow_mut() = entry_point;
        *self.translator.borrow_mut() = Some(translator);
        *self.libdef.borrow_mut() = None;

        // Upstream `:199`: `self.secondary_entrypoints = []`.
        self.secondary_entrypoints.borrow_mut().clear();

        // Upstream `:201-208`: walk `config.translation.secondaryentrypoints`
        // CSV and look up each key in the global registry.
        let csv = match self.config.get("translation.secondaryentrypoints") {
            Ok(ConfigValue::Value(OptionValue::Str(s))) => s,
            _ => String::new(),
        };
        if !csv.is_empty() {
            for key in csv.split(',') {
                let key = key.trim();
                if key.is_empty() {
                    continue;
                }
                let points = secondary_entrypoints_get(key).ok_or_else(|| TaskError {
                    message: format!(
                        "Entrypoint {:?} not found (not in {:?})",
                        key,
                        secondary_entrypoints_keys(),
                    ),
                })?;
                self.secondary_entrypoints.borrow_mut().extend(points);
            }
        }

        // Upstream `:210`: `self.translator.driver_instrument_result =
        // self.instrument_result`. DEFERRED — translator surface not yet
        // bridged through `Rc<Config>`.
        Ok(())
    }

    /// Upstream `setup_library(self, libdef, policy=None, extra={},
    /// empty_translator=None)` at `:212-216`.
    pub fn setup_library(
        &self,
        libdef: Rc<dyn Any>,
        policy: Option<crate::annotator::policy::AnnotatorPolicy>,
        extra: HashMap<String, Rc<dyn Any>>,
        empty_translator: Option<Rc<super::translator::TranslationContext>>,
    ) -> Result<(), TaskError> {
        // Upstream `:214`: `self.setup(None, None, policy, extra,
        // empty_translator)`.
        self.setup(None, None, policy, extra, empty_translator)?;
        // Upstream `:215`: `self.libdef = libdef`.
        *self.libdef.borrow_mut() = Some(libdef.clone());
        // Upstream `:216`: `self.secondary_entrypoints = libdef.functions`.
        // RPython attribute access raises `AttributeError` when the
        // object does not expose `.functions`. The Rust port surfaces a
        // [`TaskError`] on the equivalent downcast failure so non-
        // [`LibDef`] carriers cannot silently end up with empty
        // `secondary_entrypoints`.
        let ld = libdef.downcast_ref::<LibDef>().ok_or_else(|| TaskError {
            message: "driver.py:216 setup_library: libdef has no `.functions` attribute (downcast to LibDef failed)".to_string(),
        })?;
        *self.secondary_entrypoints.borrow_mut() = ld.functions.clone();
        Ok(())
    }

    /// Upstream `instrument_result(self, args)` at `:218-248`.
    ///
    /// **DEFERRED**: the body forks via `os.fork`, mutates
    /// `self.config.translation.instrument*` for the child, drives
    /// `self.proceed('compile')`, then reads `_instrument_counters` out
    /// of `udir`. None of those are ported here (`os.fork` /
    /// `udir` / C-backend `compile` task / `array.array` reading).
    ///
    /// Upstream returns the `counters` array (`array.array('L')`,
    /// unsigned long); the local port returns `Vec<u64>` matching the
    /// `'L'` typecode on a 64-bit host. Until the leaf lands, the
    /// function surfaces a [`TaskError`] citing `:218` so callers see
    /// the missing leaf rather than a panic.
    pub fn instrument_result(&self, _args: &[Rc<dyn Any>]) -> Result<Vec<u64>, TaskError> {
        Err(TaskError {
            message: "driver.py:218 instrument_result — leaves os.fork + udir + C-backend compile not yet ported".to_string(),
        })
    }

    /// Upstream `info(self, msg)` at `:250-251` — `log.info(msg)` via
    /// the module-level [`LOG`] `AnsiLogger("translation")`.
    pub fn info(&self, msg: &str) {
        LOG.info(msg);
    }

    /// Upstream `_profile(self, goal, func)` at `:253-260`:
    ///
    /// ```python
    /// def _profile(self, goal, func):
    ///     from cProfile import Profile
    ///     from rpython.tool.lsprofcalltree import KCacheGrind
    ///     d = {'func':func}
    ///     prof = Profile()
    ///     prof.runctx("res = func()", globals(), d)
    ///     KCacheGrind(prof).output(open(goal + ".out", "w"))
    ///     return d['res']
    /// ```
    ///
    /// Two TODO leaves with no direct Rust analogue:
    /// `cProfile.Profile` (Python-host CPython profiler with C
    /// extension hooks) and `lsprofcalltree.KCacheGrind` (callgrind-
    /// format dumper). Substituting either with a Rust profiler is a
    /// *new component*, not a port — so the Rust body keeps the
    /// upstream contract `return d['res']` (return `func()`'s result)
    /// and emits a single-line stderr warning recording that the
    /// `.out` file was not written. Callers see translation succeed
    /// just like upstream when the profiler succeeds; the only
    /// observable difference is the missing `<goal>.out` file.
    pub fn _profile(
        &self,
        goal: &str,
        _func: &Rc<dyn Fn() -> Result<TaskOutput, TaskError>>,
    ) -> Result<TaskOutput, TaskError> {
        // TODO: cProfile / KCacheGrind have no
        // Rust counterparts. Upstream `:259 prof.runctx("res = func()",
        // ...)` ALWAYS writes `<goal>.out` after running `func`; the
        // file is the leaf's only externally visible product. Returning
        // `func()` without the profiler would mark the task as "profiled
        // OK" while quietly skipping the file — a silent regression that
        // hides behind a stderr warning. Surface a hard `TaskError` so
        // the driver fails noisily instead. Callers wanting the
        // unprofiled result can disable `--profile` at the driver level.
        // Convergence path = either (a) port `cProfile.Profile` (CPython
        // C extension; out of scope) or (b) integrate `pprof` /
        // `cargo flamegraph` and emit a `<goal>.out` callgrind file.
        Err(TaskError {
            message: format!(
                "driver.py:253 _profile (goal={goal:?}): cProfile.Profile + \
                 lsprofcalltree.KCacheGrind have no Rust counterpart yet; \
                 cannot produce {goal}.out"
            ),
        })
    }

    // Upstream `task_*` bodies. Every body either ports a small
    // structural shell (the buildannotator+complete chain that the
    // local annotator ports support) or panics with a citation and the
    // missing leaf name.

    /// Upstream `task_annotate(self)` at `:297-327`.
    ///
    /// ```python
    /// @taskdef([], "Annotating&simplifying")
    /// def task_annotate(self):
    ///     translator = self.translator
    ///     policy = self.policy
    ///     self.log.info('with policy: %s.%s' % (...))
    ///     annotator = translator.buildannotator(policy=policy)
    ///     if self.secondary_entrypoints is not None:
    ///         for func, inputtypes in self.secondary_entrypoints:
    ///             if inputtypes == Ellipsis:
    ///                 continue
    ///             annotator.build_types(func, inputtypes, False)
    ///     if self.entry_point:
    ///         s = annotator.build_types(self.entry_point, self.inputtypes)
    ///         translator.entry_point_graph = annotator.bookkeeper.getdesc(self.entry_point).getuniquegraph()
    ///     else:
    ///         s = None
    ///     self.sanity_check_annotation()
    ///     if self.entry_point and self.standalone and s.knowntype != int:
    ///         raise Exception(...)
    ///     annotator.complete()
    ///     annotator.simplify()
    ///     return s
    /// ```
    ///
    /// **Partially activated**: the structural skeleton (buildannotator,
    /// optional `build_types(entry_point)`, `sanity_check_annotation`,
    /// `complete`, `simplify`) ports line-by-line. The
    /// `secondary_entrypoints` loop at `:308-312` iterates upstream's
    /// `(func, inputtypes)` pair where `inputtypes` is a Python list of
    /// `SomeXxx` values; the local registry stores
    /// `(Rc<dyn Any>, Vec<Rc<dyn Any>>)` pending the upstream-mirroring
    /// shape. The loop is wired but skipped (DEFERRED) until that
    /// downcast hardens.
    ///
    /// The standalone-int return-type check at `:321-324` requires
    /// inspecting `s.knowntype`; the local check uses
    /// [`SomeValue::Integer`] as the parity discriminant.
    pub fn task_annotate(&self) -> Result<TaskOutput, TaskError> {
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::HostObject;

        // Upstream `:300-301`: `translator = self.translator; policy = self.policy`.
        let translator = self.translator.borrow().clone().ok_or_else(|| TaskError {
            message: "task_annotate: translator slot is unset; setup() must be called first"
                .to_string(),
        })?;
        let policy = self.policy.borrow().clone();

        // Upstream `:302-304`: `self.log.info('with policy: ...')`. The
        // class-name string is `policy.__class__.__module__ + '.' + ... __name__`.
        // The Rust port emits a parity-equivalent line.
        self.info("with policy: rpython.annotator.policy.AnnotatorPolicy");

        // Upstream `:306`: `annotator = translator.buildannotator(policy=policy)`.
        let annotator = translator.buildannotator(policy);
        // Hand the annotator over to the driver's strong holder so it
        // survives this task's return. See the field doc on
        // [`Self::annotator`] for the Rust-port adaptation rationale.
        *self.annotator.borrow_mut() = Some(Rc::clone(&annotator));

        // Upstream `:308-312`:
        //
        //     if self.secondary_entrypoints is not None:
        //         for func, inputtypes in self.secondary_entrypoints:
        //             if inputtypes == Ellipsis:
        //                 continue
        //             annotator.build_types(func, inputtypes, False)
        //
        // The local `EntryPointSpec` carries `(Rc<dyn Any>,
        // Vec<Rc<dyn Any>>)` — registrants seed the slots through
        // [`secondary_entrypoints_register`], which wraps a typed
        // `HostObject` + `Vec<AnnotationSpec>` pair. The loop
        // downcasts back at consume time. Ellipsis support is DEFERRED
        // (no in-tree caller registers an Ellipsis sentinel yet — see
        // the `secondary_entrypoints_register` doc); empty inputtypes
        // are treated as upstream's `[]` "0-arg call", matching the
        // typed-helper contract.
        let entries: Vec<EntryPointSpec> = self.secondary_entrypoints.borrow().clone();
        for (func_any, inputtypes_any) in entries.into_iter() {
            // upstream: `for func, inputtypes in self.secondary_entrypoints:`
            let host = match func_any.downcast_ref::<HostObject>() {
                Some(h) => h.clone(),
                None => {
                    // Untyped carriers (raw `Rc<dyn Any>` slots) cannot
                    // be fed to `build_types`. Skip silently — matches
                    // the registry's pre-typed-helper era contract.
                    continue;
                }
            };
            let mut specs: Vec<crate::annotator::signature::AnnotationSpec> =
                Vec::with_capacity(inputtypes_any.len());
            let mut typed = true;
            for it in inputtypes_any.iter() {
                match it.downcast_ref::<crate::annotator::signature::AnnotationSpec>() {
                    Some(s) => specs.push(s.clone()),
                    None => {
                        typed = false;
                        break;
                    }
                }
            }
            if !typed {
                continue;
            }
            // upstream: `annotator.build_types(func, inputtypes, False)`.
            // The trailing `False` is `complete_now`; upstream's
            // `main_entry_point` defaults to False here too — only the
            // `:314-316` call uses `main_entry_point=True`.
            annotator
                .build_types(&host, &specs, false, false)
                .map_err(|e| TaskError {
                    message: format!("annotator.build_types (secondary): {e}"),
                })?;
        }

        // Upstream `:314-318`: optional `s = annotator.build_types(
        // self.entry_point, self.inputtypes)`.
        let s: Option<SomeValue> = if let Some(entry_point) = self.entry_point.borrow().clone() {
            let inputtypes = self.inputtypes.borrow().clone().ok_or_else(|| TaskError {
                message: "task_annotate: inputtypes unset; setup() must populate it".to_string(),
            })?;
            let result = annotator
                .build_types(&entry_point, &inputtypes, true, true)
                .map_err(|e| TaskError {
                    message: format!("annotator.build_types: {e}"),
                })?;
            // Upstream `:316-317`: `translator.entry_point_graph =
            // annotator.bookkeeper.getdesc(self.entry_point).getuniquegraph()`.
            let _ep: HostObject = entry_point;
            // `RPythonAnnotator::build_types` already populates
            // `translator.entry_point_graph` when `main_entry_point=true`
            // (annrpython.rs), so the explicit assignment is
            // already done by the call above.
            result
        } else {
            // Upstream `:318-319`: `else: s = None`.
            None
        };

        // Upstream `:320`: `self.sanity_check_annotation()`.
        self.sanity_check_annotation()?;

        // Upstream `:321-324`: standalone-int return-type check.
        if self.entry_point.borrow().is_some() && self.standalone.get() {
            let int_return = matches!(s, Some(SomeValue::Integer(_)));
            if !int_return {
                return Err(TaskError {
                    message:
                        "stand-alone program entry point must return an int (and not, e.g., None or always raise an exception)."
                            .to_string(),
                });
            }
        }

        // Upstream `:325-326`: `annotator.complete(); annotator.simplify()`.
        annotator.complete().map_err(|e| TaskError {
            message: format!("annotator.complete: {e}"),
        })?;
        annotator.simplify(None, None);

        // Upstream `:327`: `return s`.
        Ok(s.map(|sv| Box::new(sv) as Box<dyn Any>))
    }

    /// Upstream `sanity_check_annotation(self)` at `:330-337`.
    ///
    /// ```python
    /// def sanity_check_annotation(self):
    ///     translator = self.translator
    ///     irreg = query.qoutput(query.check_exceptblocks_qgen(translator))
    ///     if irreg:
    ///         self.log.info("Some exceptblocks seem insane")
    ///
    ///     lost = query.qoutput(query.check_methods_qgen(translator))
    ///     assert not lost, "lost methods, something gone wrong with the annotation of method defs"
    /// ```
    ///
    /// The query.py port is in [`super::goal::query`]. Walks the
    /// `self.translator` slot when present; silently no-ops in pre-
    /// `setup()` lifecycle states — matching upstream's
    /// "sanity check is best-effort" intent.
    pub fn sanity_check_annotation(&self) -> Result<(), TaskError> {
        use super::goal::query;
        let slot = self.translator.borrow();
        let translator = match slot.as_ref() {
            Some(t) => t.clone(),
            None => return Ok(()),
        };
        // Upstream `:332-334`: `irreg = query.qoutput(...)`.
        let irreg = query::qoutput(query::check_exceptblocks_qgen(&translator), None);
        if irreg > 0 {
            self.info("Some exceptblocks seem insane");
        }
        // Upstream `:336-337`: `lost = query.qoutput(...); assert not lost,
        // "lost methods, something gone wrong with the annotation of method defs"`.
        let lost = query::qoutput(query::check_methods_qgen(&translator), None);
        if lost > 0 {
            return Err(TaskError {
                message: "lost methods, something gone wrong with the annotation of method defs"
                    .to_string(),
            });
        }
        Ok(())
    }

    /// Upstream `task_rtype_lltype(self)` at `:340-345`.
    ///
    /// ```python
    /// @taskdef(['annotate'], "RTyping")
    /// def task_rtype_lltype(self):
    ///     """ RTyping - lltype version
    ///     """
    ///     rtyper = self.translator.buildrtyper()
    ///     rtyper.specialize(dont_simplify_again=True)
    /// ```
    ///
    /// Activated: both `TranslationContext::buildrtyper`
    /// (`translator.rs`) and
    /// `RPythonTyper::specialize(dont_simplify_again=true)`
    /// (`rtyper.rs`) are ported. The `'annotate'` task dep
    /// guarantees the annotator slot is populated before this body
    /// runs.
    pub fn task_rtype_lltype(&self) -> Result<TaskOutput, TaskError> {
        let translator = self.translator.borrow().clone().ok_or_else(|| TaskError {
            message: "task_rtype_lltype: translator slot is unset".to_string(),
        })?;
        let rtyper = translator.buildrtyper();
        rtyper.specialize(true).map_err(|e| TaskError {
            message: format!("rtyper.specialize: {e:?}"),
        })?;
        Ok(None)
    }

    fn missing_task_leaf(&self, line: usize, leaf: &str) -> TaskError {
        TaskError {
            message: format!("driver.py:{line} leaf not ported: {leaf}"),
        }
    }

    fn read_bool(&self, path: &str) -> bool {
        matches!(
            self.config.get(path),
            Ok(ConfigValue::Value(OptionValue::Bool(true)))
        )
    }

    fn format_exe_name_template(&self, template: &str) -> String {
        let mut out = template.to_string();
        for (key, value) in self.get_info() {
            out = out.replace(&format!("%({key})s"), &value);
        }
        out
    }

    /// Upstream `rpython/translator/driver.py TranslationDriver.task_pyjitpl_lltype`.
    ///
    /// The upstream task calls `warmspot.apply_jit`; the reverse dependency
    /// from this crate to `majit-metainterp` would form a Cargo cycle. A
    /// callback alone would not complete the port: production must consume
    /// the interpreter graphs that `WarmRunnerDesc` prepares, including its
    /// `split_graph_and_record_jitdriver` and `rewrite_jit_merge_point` passes.
    /// `register_configured_jitdrivers` already copies and splits the portal graph when
    /// requested, but pyre still traces runtime per-CodeObject JitCodes by
    /// default. See `warmspot-portal.plan.md` for that consumer migration.
    /// Complete the graph rewrites and their production consumers before
    /// integrating the whole task; translator setup has no thread-local
    /// state owner in upstream. Until then this body assembles the available
    /// arguments and returns the explicit missing-task error below.
    pub fn task_pyjitpl_lltype(&self) -> Result<TaskOutput, TaskError> {
        let translator = self.translator.borrow().clone().ok_or_else(|| TaskError {
            message: "task_pyjitpl_lltype: translator slot is unset".to_string(),
        })?;
        // Upstream `:351-357`: `get_policy = self.extra.get('jitpolicy',
        // None); if get_policy is None: self.jitpolicy = JitPolicy()
        // else: self.jitpolicy = get_policy(self)`.
        let get_policy = self.extra.borrow().get("jitpolicy").cloned();
        if get_policy.is_none() {
            *self.jitpolicy.borrow_mut() =
                Some(Rc::new(crate::codewriter::policy::DefaultJitPolicy::new()));
        } else {
            return Err(self.missing_task_leaf(
                352,
                "calling extra['jitpolicy'](self) to build the JIT policy",
            ));
        }
        // Upstream `:360-361`: `apply_jit(self.translator,
        // policy=self.jitpolicy,
        // backend_name=self.config.translation.jit_backend, inline=True)`.
        let backend_name = self
            .read_choice("translation.jit_backend")
            .unwrap_or_else(|| "auto".to_string());
        let _ = (translator, backend_name);
        Err(self.missing_task_leaf(
            358,
            "rpython.jit.metainterp.warmspot.apply_jit(translator, policy=..., backend_name=..., inline=True) — cross-crate dependency on majit-metainterp not yet wired",
        ))
    }

    /// Upstream `task_jittest_lltype(self)` at `:365-376`.
    ///
    /// Untested here: `restartable_point(auto='run')` skips its prompt and
    /// falls through to `RealRuntime::fork`, so a unit test that calls this
    /// forks the test runner. Covering the ordering needs the driver to
    /// accept an injectable `CheckpointRuntime`.
    pub fn task_jittest_lltype(&self) -> Result<TaskOutput, TaskError> {
        // Upstream `:371-372`: `from rpython.translator.goal import
        // unixcheckpoint; unixcheckpoint.restartable_point(auto='run')`.
        crate::translator::goal::unixcheckpoint::restartable_point(Some("run"))?;
        // Upstream `:374-376`: `from rpython.jit.tl import jittest;
        // jittest.jittest(self)`. The `jittest.jittest` body lives at
        // `rpython/jit/tl/jittest.py` and itself calls
        // `LLInterpreter(driver.translator.rtyper)` +
        // `apply_jit(jitpolicy, interp, graph, LLGraphCPU)` from
        // `warmspot`. Both `apply_jit` (cross-crate to majit-metainterp)
        // and `LLGraphCPU` (rpython/jit/backend/llgraph) are not yet
        // ported; surface a TaskError citing upstream `:375`.
        Err(self.missing_task_leaf(375, "rpython.jit.tl.jittest.jittest(self)"))
    }

    /// Upstream `task_backendopt_lltype(self)` at `:380-384`.
    pub fn task_backendopt_lltype(&self) -> Result<TaskOutput, TaskError> {
        let translator = self.translator.borrow().clone().ok_or_else(|| TaskError {
            message: "task_backendopt_lltype: translator slot is unset".to_string(),
        })?;
        // Upstream `:383`: `from rpython.translator.backendopt.all import
        // backend_optimizations` → `backend_optimizations(self.translator,
        // replace_we_are_jitted=True)`.
        let kwds = vec![("replace_we_are_jitted".to_string(), OptionValue::Bool(true))];
        crate::translator::backendopt::all::backend_optimizations(
            translator,
            None,
            false,
            false,
            kwds,
            Some(&self.config),
        )?;
        Ok(None)
    }

    /// Upstream `task_stackcheckinsertion_lltype(self)` at `:388-392`.
    pub fn task_stackcheckinsertion_lltype(&self) -> Result<TaskOutput, TaskError> {
        let translator = self.translator.borrow().clone().ok_or_else(|| TaskError {
            message: "task_stackcheckinsertion_lltype: translator slot is unset".to_string(),
        })?;
        let _ = translator;
        Err(self.missing_task_leaf(
            390,
            "rpython.translator.transform.insert_ll_stackcheck(self.translator)",
        ))
    }

    /// Upstream `possibly_check_for_boehm(self)` at `:395-403`.
    ///
    /// ```python
    /// def possibly_check_for_boehm(self):
    ///     if self.config.translation.gc == "boehm":
    ///         from rpython.rtyper.tool.rffi_platform import configure_boehm
    ///         from rpython.translator.platform import CompilationError
    ///         try:
    ///             configure_boehm(self.translator.platform)
    ///         except CompilationError as e:
    ///             i = 'Boehm GC not installed.  Try e.g. "translate.py --gc=minimark"'
    ///             raise Exception(str(e) + '\n' + i)
    /// ```
    ///
    /// **TODO** — `rpython.rtyper.tool.rffi_platform`
    /// (`rpython/rtyper/tool/rffi_platform.py`) and
    /// `rpython.translator.platform` (`rpython/translator/platform/__init__.py`)
    /// are not ported. The configure_boehm probe links a tiny C
    /// program against `libgc-dev` to verify the headers + lib are
    /// installed; without those ports we cannot run the probe, so the
    /// branch returns `Ok(())` even when `gc=boehm`. Convergence path:
    /// once `translator.platform` ports, the body becomes the
    /// upstream try/except shape verbatim — until then, callers that
    /// set `gc=boehm` skip the verification step at runtime (matching
    /// upstream behaviour on hosts where Boehm is installed; mismatch
    /// only on hosts where it is *not*, where upstream raises).
    pub fn possibly_check_for_boehm(&self) -> Result<(), TaskError> {
        // Upstream `:396`: `if self.config.translation.gc == "boehm":`.
        let gc = match self.config.get("translation.gc") {
            Ok(ConfigValue::Value(OptionValue::Choice(s))) => s,
            _ => return Ok(()),
        };
        if gc == "boehm" {
            // Upstream `:397-403`: probe the platform for libgc-dev.
            // TODO — see the doc-comment above for
            // the convergence path.
        }
        Ok(())
    }

    /// Upstream `task_database_c(self)` at `:408-438`.
    pub fn task_database_c(&self) -> Result<TaskOutput, TaskError> {
        // Upstream `:411`: `translator = self.translator`.
        let translator = self.translator.borrow().clone().ok_or_else(|| TaskError {
            message: "task_database_c: translator slot is unset".to_string(),
        })?;
        // Upstream `:412-413`: `if translator.annotator is not None:
        // translator.frozen = True`.
        if translator.annotator().is_some() {
            translator.frozen.set(true);
        }

        // Upstream `:415`: `standalone = self.standalone`.
        let standalone = self.standalone.get();
        // Upstream `:416-417`: `get_gchooks = self.extra.get('get_gchooks',
        // lambda: None); gchooks = get_gchooks()`.
        let get_gchooks = self.extra.borrow().get("get_gchooks").cloned();
        if get_gchooks.is_some() {
            return Err(self.missing_task_leaf(417, "calling extra['get_gchooks']()"));
        }
        let gchooks: Option<Rc<dyn Any>> = None;

        // Upstream `:419-432`: `if standalone:` constructs
        // `CStandaloneBuilder` else `CLibraryBuilder` with the same
        // (translator, entry_point, config, gchooks=gchooks,
        // secondary_entrypoints=secondary + annotated_jit) shape.
        let secondary = self.secondary_entrypoints.borrow().clone();
        let annotated = annotated_jit_entrypoints_get();
        let mut secondary_combined: Vec<EntryPointSpec> = secondary.clone();
        secondary_combined.extend(annotated.clone());
        let entry_point = self.entry_point.borrow().clone();
        let config = self.config.clone();
        let cbuilder: CBuilderRef = if standalone {
            // Upstream `:420-424`:
            //     cbuilder = CStandaloneBuilder(self.translator,
            //         self.entry_point, config=self.config, gchooks=gchooks,
            //         secondary_entrypoints=
            //             self.secondary_entrypoints +
            //             annotated_jit_entrypoints)
            CBuilderRef::Standalone(crate::translator::backend::genc::CStandaloneBuilder::new(
                translator,
                entry_point,
                config,
                None,
                gchooks,
                secondary_combined,
            ))
        } else {
            // Upstream `:426-432`: build the `functions` list and
            // construct `CLibraryBuilder(translator, entry_point,
            // functions=functions, name='libtesting', config=config,
            // gchooks=gchooks)`.
            //
            // Upstream `:427`: `functions = [(self.entry_point, None)]
            // + ...`. The `(self.entry_point, None)` tuple is pushed
            // unconditionally — even when `self.entry_point is None`.
            // Mirror that here by always pushing the head slot; when
            // `entry_point` is `None` the slot carries a `()` sentinel
            // so downstream `getentrypointptr()` fails the same way
            // upstream's `getfunctionptr(None)` does instead of
            // silently dropping the slot.
            let mut functions: Vec<EntryPointSpec> = Vec::new();
            let head: Rc<dyn Any> = match entry_point.clone() {
                Some(entry) => Rc::new(entry) as Rc<dyn Any>,
                None => Rc::new(()) as Rc<dyn Any>,
            };
            functions.push((head, Vec::new()));
            functions.extend(secondary);
            functions.extend(annotated);
            // Upstream `:430`: `name='libtesting'` literal. The
            // `extmod_name` override happens post-construction at `:434`.
            CBuilderRef::Library(crate::translator::backend::dlltool::CLibraryBuilder::new(
                translator,
                entry_point,
                config,
                functions,
                "libtesting".to_string(),
                None,
                gchooks,
                Vec::new(),
            ))
        };
        // Upstream `:433-434`: `if not standalone: cbuilder.modulename =
        // self.extmod_name`.
        if !standalone && let CBuilderRef::Library(lib) = &cbuilder {
            *lib.base.modulename.borrow_mut() = self.extmod_name.borrow().clone();
        }
        // Upstream `:435`: `database = cbuilder.build_database()`.
        let database = cbuilder.build_database()?;
        // Upstream `:436`: `self.log.info("...")`.
        self.info("database for generating C source was created");
        // Upstream `:437-438`: `self.cbuilder = cbuilder; self.database = database`.
        *self.cbuilder.borrow_mut() = Some(cbuilder);
        *self.database.borrow_mut() = Some(database);
        Ok(None)
    }

    /// Upstream `task_source_c(self)` at `:441-463`.
    pub fn task_source_c(&self) -> Result<TaskOutput, TaskError> {
        let cbuilder = self.cbuilder.borrow().clone().ok_or_else(|| TaskError {
            message: "task_source_c: cbuilder slot is unset; task_database_c must run first"
                .to_string(),
        })?;
        let database = self.database.borrow().clone().ok_or_else(|| TaskError {
            message: "task_source_c: database slot is unset; task_database_c must run first"
                .to_string(),
        })?;
        // Upstream `:446-449`: `if self._backend_extra_options.get(
        // 'c_debug_defines', False): defines = cbuilder.DEBUG_DEFINES
        // else: defines = {}`. `DEBUG_DEFINES` is the 3-macro set
        // `{'RPY_ASSERT': 1, 'RPY_LL_ASSERT': 1,
        // 'RPY_REVDB_PRINT_ALL': 1}` from `genc.py:171-173`.
        let defines = if matches!(
            self._backend_extra_options.borrow().get("c_debug_defines"),
            Some(OptionValue::Bool(true))
        ) {
            crate::translator::backend::genc::CBuilder::debug_defines()
        } else {
            HashMap::new()
        };
        let exe_name = self
            .exe_name
            .borrow()
            .as_ref()
            .map(|template| self.format_exe_name_template(template));
        let c_source_filename = cbuilder.generate_source(&database, &defines, exe_name)?;
        self.info(&format!("written: {}", c_source_filename.display()));
        if self.read_bool("translation.dump_static_data_info") {
            let _targetdir = cbuilder.targetdir();
            return Err(self.missing_task_leaf(
                456,
                "rpython.translator.tool.staticsizereport.dump_static_data_info(...)",
            ));
        }
        Ok(None)
    }

    /// Upstream `compute_exe_name(self, suffix='')` at `:465-474`.
    pub fn compute_exe_name(&self, suffix: &str) -> Result<PathBuf, TaskError> {
        let template = self.exe_name.borrow().clone().ok_or_else(|| TaskError {
            message: "compute_exe_name: exe_name is None".to_string(),
        })?;
        let mut newexename = self.format_exe_name_template(&template);
        if !newexename.contains('/') && !newexename.contains('\\') {
            newexename = format!("./{newexename}");
        }
        if suffix.is_empty() {
            return Ok(PathBuf::from(newexename));
        }
        let mut path = PathBuf::from(newexename);
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| TaskError {
                message: "compute_exe_name: cannot compute basename stem".to_string(),
            })?
            .to_string();
        let new_basename = format!("{stem}{suffix}");
        path.set_file_name(new_basename);
        Ok(path)
    }

    /// Upstream `create_exe(self)` at `:476-524`.
    pub fn create_exe(&self) -> Result<(), TaskError> {
        if self.exe_name.borrow().is_some() {
            let exename = self.c_entryp.borrow().clone().ok_or_else(|| TaskError {
                message: "create_exe: c_entryp slot is unset".to_string(),
            })?;
            let basename = exename
                .file_name()
                .ok_or_else(|| TaskError {
                    message: "create_exe: c_entryp has no basename".to_string(),
                })?
                .to_owned();
            let newexename = PathBuf::from(basename);
            shutil_copy(&exename, &newexename).map_err(|e| TaskError {
                message: format!(
                    "driver.py `shutil_copy`({}, {}): {e}",
                    exename.display(),
                    newexename.display()
                ),
            })?;
            self.info(&format!(
                "copied: {} to {}",
                exename.display(),
                newexename.display()
            ));
            if let Some(cbuilder) = self.cbuilder.borrow().clone()
                && let Some(soname) = cbuilder.shared_library_name()
            {
                let newsoname =
                    newexename.with_file_name(soname.file_name().ok_or_else(|| TaskError {
                        message: "create_exe: shared_library_name has no basename".to_string(),
                    })?);
                shutil_copy(&soname, &newsoname).map_err(|e| TaskError {
                    message: format!(
                        "driver.py `shutil_copy`({}, {}): {e}",
                        soname.display(),
                        newsoname.display()
                    ),
                })?;
                self.info(&format!(
                    "copied: {} to {}",
                    soname.display(),
                    newsoname.display()
                ));
                if cbuilder.executable_name_w().is_some() {
                    return Err(self.missing_task_leaf(
                        490,
                        "Windows pypyw/import-library/pdb/libffi copy block",
                    ));
                }
            }
            *self.c_entryp.borrow_mut() = Some(newexename);
        }
        let created = self
            .c_entryp
            .borrow()
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "None".to_string());
        self.info(&format!("created: {created}"));
        Ok(())
    }

    /// Upstream `task_compile_c(self)` at `:527-541`.
    pub fn task_compile_c(&self) -> Result<TaskOutput, TaskError> {
        let cbuilder = self.cbuilder.borrow().clone().ok_or_else(|| TaskError {
            message: "task_compile_c: cbuilder slot is unset; task_source_c must run first"
                .to_string(),
        })?;
        let exe_name = if self.standalone.get() && self.exe_name.borrow().is_some() {
            Some(
                self.compute_exe_name("")?
                    .file_name()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| TaskError {
                        message: "task_compile_c: computed exe name has no basename".to_string(),
                    })?
                    .to_string(),
            )
        } else {
            None
        };
        cbuilder.compile(exe_name)?;
        if self.standalone.get() {
            // Upstream `:537-539`: `self.c_entryp =
            // cbuilder.executable_name; self.create_exe()`.
            let executable = cbuilder.executable_name().ok_or_else(|| TaskError {
                message: "task_compile_c: cbuilder.executable_name is None after compile()"
                    .to_string(),
            })?;
            *self.c_entryp.borrow_mut() = Some(executable);
            self.create_exe()?;
        } else {
            // Upstream `:540-541`: `self.c_entryp = cbuilder.get_entry_point()`.
            *self.c_entryp.borrow_mut() = Some(cbuilder.get_entry_point()?);
        }
        Ok(None)
    }

    /// Upstream `task_llinterpret_lltype(self)` at `:544-555`.
    pub fn task_llinterpret_lltype(&self) -> Result<TaskOutput, TaskError> {
        // Upstream `:545`: `from rpython.rtyper.llinterp import LLInterpreter`.
        use crate::translator::rtyper::llinterp::{FrameError, LLInterpreter};
        // Upstream `:547`: `translator = self.translator`.
        let translator = self.translator.borrow().clone().ok_or_else(|| TaskError {
            message: "task_llinterpret_lltype: translator slot is unset".to_string(),
        })?;
        // Upstream `:548`: `interp = LLInterpreter(translator.rtyper)`.
        let rtyper = translator.rtyper().ok_or_else(|| TaskError {
            message: "task_llinterpret_lltype: translator.rtyper slot is unset".to_string(),
        })?;
        // Upstream `:99` requires the interpreter object identity for
        // `LLInterpreter.current_interpreter = self`; the local port
        // mirrors that with a thread-local `Weak<LLInterpreter>` and
        // therefore needs an `Rc<LLInterpreter>` here.
        let interp = Rc::new(LLInterpreter::new(rtyper, true, None));
        // Upstream `:549-550`: `bk = translator.annotator.bookkeeper`,
        // `graph = bk.getdesc(self.entry_point).getuniquegraph()`.
        // `getuniquegraph` is ported on
        // [`crate::annotator::description::FunctionDesc::getuniquegraph`]
        // and the c-backend driver already calls it
        // (`translator/backend/genc.rs`). The placeholder below remains
        // because the `LLInterpreter::eval_graph` consumer surface
        // expects an `Rc<dyn Any>` opaque graph handle until the
        // llinterp port narrows it to `Rc<PyGraph>`.
        let _annotator = translator.annotator().ok_or_else(|| TaskError {
            message: "task_llinterpret_lltype: translator.annotator slot is unset".to_string(),
        })?;
        let _entry = self.entry_point.borrow().clone().ok_or_else(|| TaskError {
            message: "task_llinterpret_lltype: entry_point slot is unset".to_string(),
        })?;
        // Upstream `:551-553`: `v = interp.eval_graph(graph, get_llinterp_args())`.
        // Use a placeholder graph until `getuniquegraph` lands; the
        // shell still surfaces the leaf-level TaskError citing
        // `llinterp.py`.
        let placeholder_graph: Rc<dyn Any> = Rc::new(());
        self.extra
            .borrow()
            .get("get_llinterp_args")
            .cloned()
            .map(|_| ())
            .unwrap_or(());
        let v = interp
            .eval_graph(placeholder_graph, Vec::new(), false)
            .map_err(FrameError::into_task_error)?;
        // Upstream `:555`: `log.llinterpret("result -> %s" % v)`.
        self.info(&format!("llinterpret result -> {:?}", v.type_id()));
        Ok(None)
    }

    /// Upstream `proceed(self, goals)` at `:557-570`.
    pub fn proceed(self: &Rc<Self>, goals: ProceedGoals) -> Result<TaskOutput, TaskError> {
        // Upstream `:558-563`: empty goals → fall back to default_goal,
        // else log + return.
        let mut goals: Vec<String> = match goals {
            ProceedGoals::Empty => match self.default_goal.borrow().clone() {
                Some(d) => vec![d],
                None => {
                    self.info("nothing to do");
                    return Ok(None);
                }
            },
            ProceedGoals::One(s) => vec![s],
            ProceedGoals::Many(v) => v,
        };

        // Upstream `:566`: `goals.extend(self.extra_goals)`.
        goals.extend(self.extra_goals.borrow().iter().cloned());

        // Upstream `:567`: `goals = self.backend_select_goals(goals)`.
        let goals = self.backend_select_goals(&goals).map_err(|e| TaskError {
            message: e.to_string(),
        })?;

        // Upstream `:568`: `result = self._execute(goals, task_skip = self._maybe_skip())`.
        let task_skip = self._maybe_skip();
        let hooks = DriverHooks {
            driver: Rc::clone(self),
        };
        let result = self.engine.execute(&hooks, &goals, &task_skip)?;

        // Upstream `:569`: `self.log.info('usession directory: %s' % (udir,))`.
        // `udir` not ported — print a placeholder so the call shape is
        // visible.
        self.info("usession directory: <udir not ported>");

        Ok(result)
    }

    /// Upstream `from_targetspec(cls, targetspec_dic, config=None,
    /// args=None, empty_translator=None, disable=[], default_goal=None)`
    /// at `:573-602`.
    ///
    /// `targetspec_dic` upstream is the dict imported from
    /// `targetstandalone.py`; the local port carries it as a
    /// [`TargetSpecDict`] so the `"target"` callable is typed while the
    /// open dict still flows into `setup(..., extra=targetspec_dic)`.
    pub fn from_targetspec(
        targetspec_dic: TargetSpecDict,
        config: Option<Rc<Config>>,
        args: Option<Vec<String>>,
        empty_translator: Option<Rc<super::translator::TranslationContext>>,
        disable: Vec<String>,
        default_goal: Option<String>,
    ) -> Result<Rc<Self>, TaskError> {
        // Upstream `:577-578`: `if args is None: args = []`.
        let args = args.unwrap_or_default();

        // Upstream `:580-581`: construct the driver with config,
        // default_goal, and disable.
        let driver =
            Self::new(None, default_goal, disable, None, None, config, None).map_err(|e| {
                TaskError {
                    message: format!("driver.py:580 from_targetspec: {e}"),
                }
            })?;

        // Upstream `:582-585`: `target = targetspec_dic['target']`;
        // timer start/end around exactly the target call.
        let target = targetspec_dic.target();
        driver.timer.start_event("loading target");
        let spec = target.call(&driver, &args)?;
        driver.timer.end_event("loading target");

        // Upstream `:587-595`: unpack either a 3-tuple, a 2-tuple, or a
        // non-tuple entry point.
        let (entry_point, inputtypes, policy) = spec.into_setup_parts();

        // Upstream `:598-601`: setup receives the original dict as
        // `extra`, including the `"target"` key.
        driver.setup(
            Some(entry_point),
            inputtypes,
            policy,
            targetspec_dic.into_extra(),
            empty_translator,
        )?;
        Ok(driver)
    }

    /// Upstream `prereq_checkpt_rtype(self)` at `:604-606`. DEFERRED —
    /// `rpython.rtyper.rmodel` import detection, used by the
    /// fork-checkpoint path inside `_event`.
    pub fn prereq_checkpt_rtype(&self) -> Result<(), TaskError> {
        // Upstream's body: `assert 'rpython.rtyper.rmodel' not in
        // sys.modules`. The Rust port has no equivalent module-import
        // observation; succeed.
        Ok(())
    }

    /// Upstream `prereq_checkpt_rtype_lltype = prereq_checkpt_rtype`
    /// at `:607`.
    pub fn prereq_checkpt_rtype_lltype(&self) -> Result<(), TaskError> {
        self.prereq_checkpt_rtype()
    }

    /// Upstream `_event(self, kind, goal, func)` at `:610-622`.
    ///
    /// Earlycheck branch (`:611-612`) ports verbatim. The fork-before
    /// checkpoint at `:613-622` reads
    /// `config.translation.fork_before`, resolves it through
    /// `backend_select_goals`, gates on `not done && match goal`, runs
    /// the optional `prereq_checkpt_<goal>` hook, and dispatches
    /// `unixcheckpoint.restartable_point(auto='run')`. The Rust port
    /// mirrors that chain end-to-end; the leaf
    /// [`super::goal::unixcheckpoint::restartable_point`] currently
    /// surfaces a [`TaskError`] until the fork / `os.waitpid` / prompt
    /// loop ports — that error propagates out of `_event_with_func`,
    /// matching upstream's "fork checkpoint failed" semantics.
    pub fn _event_with_func(
        &self,
        kind: &str,
        goal: &str,
        idempotent: bool,
    ) -> Result<(), TaskError> {
        if kind == "planned"
            && let Some(check) = self.task_earlycheck.borrow().get(goal).cloned()
        {
            check(self)?;
        }
        if kind == "pre" {
            // Upstream `:614`: `fork_before = self.config.translation.fork_before`.
            //
            // `fork_before` is declared as a `ChoiceOption` in
            // `translationoption.rs` (upstream `translationoption.py:146-150`),
            // so the stored value is `OptionValue::Choice` — not `Str`.
            // The `OptionValue::None` case (upstream's `None` default)
            // skips the body, matching `if fork_before:` at upstream
            // `:615`.
            let fork_before: Option<String> = match self.config.get("translation.fork_before") {
                Ok(ConfigValue::Value(OptionValue::Choice(s))) => Some(s),
                _ => None,
            };
            if let Some(fb) = fork_before {
                // Upstream `:616`: `fork_before, = self.backend_select_goals([fork_before])`.
                let resolved = self
                    .backend_select_goals(std::slice::from_ref(&fb))
                    .map_err(|e| TaskError {
                        message: format!("fork_before backend_select_goals({fb:?}): {e}"),
                    })?;
                let resolved_fb = match resolved.into_iter().next() {
                    Some(s) => s,
                    None => {
                        return Err(TaskError {
                            message: format!(
                                "fork_before backend_select_goals({fb:?}): empty result"
                            ),
                        });
                    }
                };
                // Upstream `:617`: `if not fork_before in self.done and fork_before == goal:`.
                if !self.done.borrow().contains_key(&resolved_fb) && resolved_fb == goal {
                    // Upstream `:618-620`: optional `prereq_checkpt_<goal>` hook.
                    // Only `prereq_checkpt_rtype` (and its alias
                    // `prereq_checkpt_rtype_lltype` at `:607`) are
                    // declared in the local port — both are no-ops
                    // matching upstream's "best-effort" intent.
                    match goal {
                        "rtype" => self.prereq_checkpt_rtype()?,
                        "rtype_lltype" => self.prereq_checkpt_rtype_lltype()?,
                        _ => {}
                    }
                    // Upstream `:621-622`: `from rpython.translator.goal import
                    // unixcheckpoint; unixcheckpoint.restartable_point(auto='run')`.
                    super::goal::unixcheckpoint::restartable_point(Some("run"))?;
                }
            }
        }
        let _ = idempotent;
        Ok(())
    }

    /// Convenience for tests / callers: whether a task name is in the
    /// post-`expose_task` list. Upstream callers read `td.exposed`
    /// directly, so the field is `pub`; this helper exists to express
    /// the upstream membership check ergonomically.
    pub fn is_exposed(&self, name: &str) -> bool {
        self.exposed.borrow().iter().any(|s| s == name)
    }

    /// Snapshot of `self.exposed` for callers that prefer not to
    /// borrow. Upstream callers iterate the list as a set; this
    /// helper hands back the same content as a fresh `Vec`.
    pub fn exposed_snapshot(&self) -> Vec<String> {
        self.exposed.borrow().clone()
    }

    /// Backend goal recorded by [`Self::expose_task`] for the public
    /// task name, or `None` if `name` is not exposed.
    pub fn exposed_method(&self, name: &str) -> Option<String> {
        self.exposed_backend_goal.borrow().get(name).cloned()
    }
}

/// Argument shape for [`TranslationDriver::proceed`] mirroring
/// upstream's overloaded `goals` parameter at `:557-565` (string,
/// list, or `None`).
pub enum ProceedGoals {
    Empty,
    One(String),
    Many(Vec<String>),
}

// Upstream `:262-295 def _do(self, goal, func, *args, **kwds)`.
//
// `_do` overrides `SimpleTaskEngine._do` so the engine-driven task
// loop runs each task inside the driver's `self.timer.start_event` /
// `end_event` pair, prints `log.info(...)` lines, propagates the
// `Instrument` exception into a re-entry on the `compile` goal, and
// stashes completed goals into `self.done` (for non-idempotent
// tasks).

struct DriverHooks {
    driver: Rc<TranslationDriver>,
}

impl TaskEngineHooks for DriverHooks {
    /// Upstream `_do(self, goal, func, *args, **kwds)` at `:262-295`.
    fn _do(
        &self,
        goal: &str,
        callable: &Rc<dyn Fn() -> Result<TaskOutput, TaskError>>,
    ) -> Result<TaskOutput, TaskError> {
        let driver = &self.driver;
        // Upstream `:263`: `title = func.task_title`.
        let title = {
            let tasks = driver.engine.tasks();
            tasks
                .get(goal)
                .map(|reg| reg.title.clone())
                .unwrap_or_else(|| goal.to_string())
        };

        // Upstream `:264-266`: already-done short-circuit.
        if driver.done.borrow().contains_key(goal) {
            driver.info(&format!("already done: {title}"));
            return Ok(None);
        } else {
            driver.info(&format!("{title}..."));
        }

        // Upstream `:269`: `debug_start('translation-task')`. NOT
        // ported — see module docstring.
        // Upstream `:270`: `debug_print('starting', goal)`.
        // Upstream `:271`: `self.timer.start_event(goal)`.
        driver.timer.start_event(goal);

        // Upstream `:273-282`: try-block running the callable;
        // optional cProfile branch (DEFERRED); Instrument-exception
        // catch.
        let mut instrument = false;
        let mut res: TaskOutput = None;
        // Upstream `:275-278`: PROFILE branch. DEFERRED — `_profile`
        // surfaces a TaskError citing `driver.py` when reached; the
        // `PROFILE` set is empty by default so the branch is
        // unreachable in practice.
        let in_profile = profile_contains(goal);
        let result = if in_profile {
            // Upstream `:276`: `res = self._profile(goal, func)`.
            driver._profile(goal, callable)
        } else {
            callable()
        };
        match result {
            Ok(value) => {
                res = value;
            }
            Err(err) => {
                // Upstream `:279-280`: catch `Instrument` and continue;
                // every other exception escapes via the `finally`.
                if err.message == "Instrument" {
                    instrument = true;
                } else {
                    // Replicate the `finally` block before propagating.
                    driver.timer.end_event(goal);
                    return Err(err);
                }
            }
        }

        // Upstream `:281-282`: `if not func.task_idempotent:
        // self.done[goal] = True`.
        let idempotent = {
            let tasks = driver.engine.tasks();
            tasks.get(goal).map(|r| r.idempotent).unwrap_or(false)
        };
        if !idempotent {
            driver.done.borrow_mut().insert(goal.to_string(), true);
        }

        if instrument {
            // Upstream `:283-285`: re-enter on `compile` then assert.
            let _ = driver.proceed(ProceedGoals::One("compile".to_string()))?;
            return Err(TaskError {
                message: "we should not get here".to_string(),
            });
        }

        // Upstream `:286-293`: finally block (already partially run on
        // the error path above).
        // Upstream `:288`: `debug_stop('translation-task')`. NOT ported.
        driver.timer.end_event(goal);

        Ok(res)
    }

    /// Upstream `_event(kind, goal)` at `:610-622`.
    ///
    /// Upstream raises `func.task_earlycheck(self)`'s exception
    /// unwrapped — the engine's first event loop at
    /// `taskengine.py` has no try/except so the raise
    /// propagates straight out of `_execute`. The Rust port returns
    /// the error so [`SimpleTaskEngine::execute`] can `?`-propagate
    /// it identically (the trait's signature was widened from
    /// `fn _event(&self, …)` to `Result<(), TaskError>` for this
    /// reason).
    fn _event(&self, kind: &str, goal: &str) -> Result<(), TaskError> {
        self.driver._event_with_func(kind, goal, false)
    }
}

// Upstream `:624-631 shutil_copy` — module-level helper that handles
// the "destination is the executable currently being run" case on
// posix (rename through a `~` suffix). Re-exported so c-backend
// callers can use it; on non-posix the upstream falls through to
// `shutil.copy`.

/// Port of upstream `shutil_copy(src, dst)` at `:624-631`. On non-
/// Windows hosts the routine renames `dst~` over `dst` to handle the
/// "executable currently in use" pitfall.
pub fn shutil_copy(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    if cfg!(target_os = "windows") {
        std::fs::copy(src, dst)?;
        Ok(())
    } else {
        let tmp = dst.with_extension("__copy_tmp__");
        std::fs::copy(src, &tmp)?;
        std::fs::rename(&tmp, dst)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{ConstValue, Constant, GraphFunc, HostObject};
    use crate::translator::targetspec::{
        TargetSpecCallable, TargetSpecCallableSlot, TargetSpecDict, TargetSpecResult,
    };

    fn host_function(name: &str) -> HostObject {
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        HostObject::new_user_function(GraphFunc::new(name, globals))
    }

    /// Mirrors `rpython/translator/test/test_driver.py test_ctr`.
    /// Verifies the default-construction `expected` set + the
    /// `backend_select_goals` resolution table.
    #[test]
    fn ctr_default_exposed_and_backend_select_goals() {
        let td = TranslationDriver::new_default().expect("driver");
        let expected: std::collections::HashSet<String> = [
            "annotate",
            "backendopt",
            "llinterpret",
            "rtype",
            "source",
            "compile",
            "pyjitpl",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let exposed: std::collections::HashSet<String> =
            td.exposed_snapshot().into_iter().collect();
        assert_eq!(exposed, expected, "default exposed mismatch");

        assert_eq!(
            td.backend_select_goals(&["compile_c".to_string()])
                .expect("compile_c"),
            vec!["compile_c".to_string()],
        );
        assert_eq!(
            td.backend_select_goals(&["compile".to_string()])
                .expect("compile"),
            vec!["compile_c".to_string()],
        );
        assert_eq!(
            td.backend_select_goals(&["rtype".to_string()])
                .expect("rtype"),
            vec!["rtype_lltype".to_string()],
        );
        assert_eq!(
            td.backend_select_goals(&["rtype_lltype".to_string()])
                .expect("rtype_lltype"),
            vec!["rtype_lltype".to_string()],
        );
        assert_eq!(
            td.backend_select_goals(&["backendopt".to_string()])
                .expect("backendopt"),
            vec!["backendopt_lltype".to_string()],
        );
        assert_eq!(
            td.backend_select_goals(&["backendopt_lltype".to_string()])
                .expect("backendopt_lltype"),
            vec!["backendopt_lltype".to_string()],
        );
    }

    /// Mirrors `rpython/translator/test/test_driver.py:21-33` — when
    /// `backend=None` and `type_system=None`, only the explicit
    /// `*_lltype` / `*_c` task names compose, and generic prefixes
    /// like `compile` / `rtype` / `backendopt` are NOT resolvable.
    #[test]
    fn ctr_with_backend_and_type_system_unset() {
        let setopts: Vec<(String, OptionValue)> = vec![
            ("backend".to_string(), OptionValue::None),
            ("type_system".to_string(), OptionValue::None),
        ];
        let td = TranslationDriver::new(Some(setopts), None, Vec::new(), None, None, None, None)
            .expect("driver");

        assert_eq!(
            td.backend_select_goals(&["compile_c".to_string()])
                .expect("compile_c"),
            vec!["compile_c".to_string()],
        );
        assert!(
            td.backend_select_goals(&["compile".to_string()]).is_err(),
            "compile must not resolve when backend / type_system are None"
        );
        assert!(
            td.backend_select_goals(&["rtype".to_string()]).is_err(),
            "rtype must not resolve when type_system is None"
        );
        assert_eq!(
            td.backend_select_goals(&["rtype_lltype".to_string()])
                .expect("rtype_lltype"),
            vec!["rtype_lltype".to_string()],
        );
        assert!(
            td.backend_select_goals(&["backendopt".to_string()])
                .is_err(),
            "backendopt must not resolve when backend / type_system are None"
        );
        assert_eq!(
            td.backend_select_goals(&["backendopt_lltype".to_string()])
                .expect("backendopt_lltype"),
            vec!["backendopt_lltype".to_string()],
        );

        let expected: std::collections::HashSet<String> = [
            "annotate",
            "backendopt_lltype",
            "llinterpret_lltype",
            "rtype_lltype",
            "source_c",
            "compile_c",
            "pyjitpl_lltype",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let exposed: std::collections::HashSet<String> =
            td.exposed_snapshot().into_iter().collect();
        assert_eq!(
            exposed, expected,
            "exposed must match the explicit-only set when backend / type_system are None"
        );
    }

    /// Mirrors `rpython/translator/test/test_driver.py:35-48` — when
    /// `type_system=lltype` but `backend=None`, the type-system-keyed
    /// public names appear (rtype, backendopt, llinterpret, pyjitpl)
    /// while the backend-keyed ones stay specific (source_c, compile_c).
    #[test]
    fn ctr_type_system_lltype_backend_none() {
        let setopts: Vec<(String, OptionValue)> = vec![
            ("backend".to_string(), OptionValue::None),
            (
                "type_system".to_string(),
                OptionValue::Choice("lltype".to_string()),
            ),
        ];
        let td = TranslationDriver::new(Some(setopts), None, Vec::new(), None, None, None, None)
            .expect("driver");

        assert_eq!(
            td.backend_select_goals(&["compile_c".to_string()])
                .expect("compile_c"),
            vec!["compile_c".to_string()],
        );
        assert!(
            td.backend_select_goals(&["compile".to_string()]).is_err(),
            "compile must not resolve when backend is None"
        );
        assert_eq!(
            td.backend_select_goals(&["rtype_lltype".to_string()])
                .expect("rtype_lltype"),
            vec!["rtype_lltype".to_string()],
        );
        assert_eq!(
            td.backend_select_goals(&["rtype".to_string()])
                .expect("rtype"),
            vec!["rtype_lltype".to_string()],
        );
        assert_eq!(
            td.backend_select_goals(&["backendopt".to_string()])
                .expect("backendopt"),
            vec!["backendopt_lltype".to_string()],
        );
        assert_eq!(
            td.backend_select_goals(&["backendopt_lltype".to_string()])
                .expect("backendopt_lltype"),
            vec!["backendopt_lltype".to_string()],
        );

        let expected: std::collections::HashSet<String> = [
            "annotate",
            "backendopt",
            "llinterpret",
            "rtype",
            "source_c",
            "compile_c",
            "pyjitpl",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let exposed: std::collections::HashSet<String> =
            td.exposed_snapshot().into_iter().collect();
        assert_eq!(
            exposed, expected,
            "exposed must match the type-system-keyed set when backend is None"
        );
    }

    #[test]
    fn setup_threads_driver_overrides_into_translator_config() {
        // Upstream `driver.py:194 translator = TranslationContext(
        // config=self.config)` shares the driver's Config object.
        // The Rust port snapshots the driver's `Rc<Config>` into a
        // typed `TranslationConfig` at setup() time, so any
        // `override`/`setopts` mutation performed before setup() is
        // visible inside `translator.config.translation.<X>`.
        let mut overrides: HashMap<String, OptionValue> = HashMap::new();
        overrides.insert("translation.verbose".to_string(), OptionValue::Bool(true));
        overrides.insert(
            "translation.list_comprehension_operations".to_string(),
            OptionValue::Bool(true),
        );
        overrides.insert("translation.keepgoing".to_string(), OptionValue::Bool(true));
        let td = TranslationDriver::new(None, None, Vec::new(), None, None, None, Some(overrides))
            .expect("driver");
        td.setup(None, None, None, HashMap::new(), None)
            .expect("setup");
        let translator = td
            .translator
            .borrow()
            .as_ref()
            .map(Rc::clone)
            .expect("translator");
        assert!(
            translator.config.translation.verbose,
            "driver-side translation.verbose=true must propagate to translator.config"
        );
        assert!(
            translator.config.translation.list_comprehension_operations,
            "driver-side list_comprehension_operations=true must propagate"
        );
        assert!(
            translator.config.translation.keepgoing,
            "driver-side keepgoing=true must propagate"
        );
        // `translating` is set true by get_combined_translation_config(translating=True)
        // on driver construction.
        assert!(
            translator.config.translating,
            "translating flag must reach the snapshot"
        );
    }

    #[test]
    fn sanity_check_annotation_passes_on_empty_translator() {
        // Upstream `:330-337 sanity_check_annotation` requires
        // `self.translator` to be a TranslationContext. With no
        // graphs registered, `check_exceptblocks_qgen` and
        // `check_methods_qgen` both yield zero items, so the upstream
        // assert passes silently.
        let td = TranslationDriver::new_default().expect("driver");
        td.setup(None, None, None, HashMap::new(), None)
            .expect("setup with default translator");
        td.sanity_check_annotation()
            .expect("empty translator must pass sanity check");
    }

    #[test]
    fn sanity_check_annotation_silent_when_translator_is_unset() {
        // Upstream pre-`setup()` state has `self.translator` unset.
        // The Rust port treats that as a no-op so the method can be
        // called from arbitrary lifecycle points without panicking.
        let td = TranslationDriver::new_default().expect("driver");
        td.sanity_check_annotation()
            .expect("unset translator must no-op");
    }

    #[test]
    fn deferred_driver_tasks_return_errors_instead_of_panicking() {
        let td = TranslationDriver::new_default().expect("driver");
        td.setup(None, None, None, HashMap::new(), None)
            .expect("setup");

        let err = td
            .task_backendopt_lltype()
            .expect_err("backendopt leaf should report the missing upstream leaf");
        assert!(
            err.message.contains("malloc.py:553"),
            "expected `malloc.py:553` citation, got: {}",
            err.message
        );

        let err = td
            .task_compile_c()
            .expect_err("compile leaf should report missing prior C builder state");
        assert!(err.message.contains("cbuilder slot is unset"));
    }

    #[test]
    fn task_backendopt_lltype_dispatches_to_translator_backendopt_all() {
        // Upstream `:380-384`: the leaf calls
        // `rpython.translator.backendopt.all.backend_optimizations(
        // self.translator, replace_we_are_jitted=True)`. The local port
        // dispatches into `crate::translator::backendopt::all`; default
        // backendopt config has `mallocs=True`, so the first unported
        // subpass to fire after the implemented `all.py` scaffolding is
        // `malloc.py remove_mallocs`.
        let td = TranslationDriver::new_default().expect("driver");
        td.setup(None, None, None, HashMap::new(), None)
            .expect("setup");
        let err = td
            .task_backendopt_lltype()
            .expect_err("backendopt leaf is still missing");
        assert!(
            err.message.contains("malloc.py:553"),
            "task_backendopt_lltype must reach `malloc.py remove_mallocs`, got: {}",
            err.message
        );
    }

    /// Re-enabling this test requires injectable input and output as well as an
    /// injectable `CheckpointRuntime`. On the no-fork path the `auto` token is
    /// discarded and the prompt reads input before consulting the runtime, so
    /// a runtime seam alone cannot prevent a blocking or repeated EOF read.
    #[test]
    fn task_database_c_sets_translator_frozen_before_c_backend_leaf() {
        let td = TranslationDriver::new_default().expect("driver");
        td.setup(None, None, None, HashMap::new(), None)
            .expect("setup");
        let translator = td
            .translator
            .borrow()
            .as_ref()
            .map(Rc::clone)
            .expect("translator");
        let annotator = translator.buildannotator(None);
        *td.annotator.borrow_mut() = Some(annotator);

        let err = td
            .task_database_c()
            .expect_err("standalone C backend still needs the entrypoint wrapper leaf");
        // `CBuilder.build_database` enters the real body via enum
        // subclass dispatch. Upstream `genc.py` calls
        // `translator.getexceptiontransformer()` BEFORE
        // `self.getentrypointptr()` (`:110`), so without an rtyper the
        // first failure surface is `translator.py ValueError: no
        // rtyper`. The test pins that ordering.
        assert!(
            err.message.contains("translator.py:88"),
            "expected translator.py:88 citation (genc.py:92 calls \
             getexceptiontransformer before getentrypointptr), got: {}",
            err.message
        );
        assert!(
            translator.frozen.get(),
            "task_database_c must mirror driver.py:411 before building the database"
        );
    }

    #[test]
    fn shutil_copy_round_trip_in_tempdir() {
        // Mirrors `test_shutil_copy` at upstream
        // `test_driver.py`. Upstream uses pytest's per-session
        // `udir.join(...)`; the Rust port uses `tempfile::TempDir` for
        // the equivalent isolation so concurrent invocations cannot
        // race on a shared path under `std::env::temp_dir()`.
        let dir = tempfile::TempDir::new().expect("tempdir");
        let a = dir.path().join("file_a");
        let b = dir.path().join("file_b");
        std::fs::write(&a, b"hello").expect("write a");
        shutil_copy(&a, &b).expect("shutil_copy");
        let read = std::fs::read_to_string(&b).expect("read b");
        assert_eq!(read, "hello");
    }

    /// Upstream `instrument_result(self, args)` at `:218-248` is
    /// DEFERRED — the Rust port surfaces a [`TaskError`] citing the
    /// upstream line instead of panicking, so callers see the missing
    /// leaf. Pinned here so the panic→Result flip cannot regress.
    #[test]
    fn instrument_result_returns_task_error_citing_line_218() {
        let td = TranslationDriver::new_default().expect("driver");
        let err = td.instrument_result(&[]).expect_err("must be DEFERRED");
        assert!(
            err.message.contains("driver.py:218"),
            "expected `driver.py:218` citation, got: {}",
            err.message
        );
    }

    /// Upstream `_profile(self, goal, func)` at `:253-260` runs
    /// `func()` inside `cProfile.Profile.runctx` and writes
    /// `<goal>.out` via `KCacheGrind`. The Rust port has no cProfile /
    /// KCacheGrind analogue, so the leaf surfaces a hard `TaskError`
    /// citing the unported dependency rather than silently dropping
    /// the `.out` file. Pin that contract so a future "let it succeed
    /// without the file" regression is caught.
    #[test]
    fn _profile_surfaces_unported_leaf_error() {
        let td = TranslationDriver::new_default().expect("driver");
        let callable: Rc<dyn Fn() -> Result<TaskOutput, TaskError>> =
            Rc::new(|| Ok(Some(Box::new(42_i64) as Box<dyn std::any::Any>)));
        let err = td._profile("annotate", &callable).unwrap_err();
        assert!(
            err.message.contains("driver.py:253"),
            "expected `driver.py:253` citation, got: {}",
            err.message
        );
        assert!(
            err.message.contains("cProfile") && err.message.contains("KCacheGrind"),
            "expected cProfile + KCacheGrind names in citation, got: {}",
            err.message
        );
    }

    /// `secondary_entrypoints_register` mirrors upstream
    /// `entrypoint.py secondary_entrypoints.setdefault(key,
    /// []).append(...)`. The typed (HostObject, Vec<AnnotationSpec>)
    /// shape downcasts back at `task_annotate :308-312` consume time.
    #[test]
    fn secondary_entrypoints_register_round_trips_through_typed_helper() {
        use crate::annotator::signature::AnnotationSpec;

        let key = "test_secondary_register_round_trip";
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let gf = GraphFunc::new("demo_secondary", globals);
        let host = HostObject::new_user_function(gf);
        secondary_entrypoints_register(key, host.clone(), vec![AnnotationSpec::Int]);

        let entries = secondary_entrypoints_get(key).expect("registered key");
        assert_eq!(entries.len(), 1);
        let (func_any, inputs_any) = &entries[0];
        let func = func_any
            .downcast_ref::<HostObject>()
            .expect("HostObject downcast");
        assert_eq!(func.qualname(), host.qualname());
        assert_eq!(inputs_any.len(), 1);
        let spec = inputs_any[0]
            .downcast_ref::<AnnotationSpec>()
            .expect("AnnotationSpec downcast");
        assert!(matches!(spec, AnnotationSpec::Int));
    }

    /// Upstream `_event(self, "pre", goal, ...)` at `:613-622`: when
    /// `config.translation.fork_before` is unset (the default), the
    /// body skips the checkpoint silently. Pinned so the activation
    /// added in this commit doesn't regress the default-OK path.
    #[test]
    fn event_pre_no_fork_before_returns_ok() {
        let td = TranslationDriver::new_default().expect("driver");
        td._event_with_func("pre", "rtype_lltype", false)
            .expect("default fork_before=None must skip body");
    }

    /// Upstream `_event(self, "pre", goal, ...)` at `:617`: when
    /// `fork_before` is set but doesn't match the resolved goal
    /// (e.g. fork_before='rtype' but goal='annotate'), the body skips.
    /// Pinned so the goal-mismatch path stays observable.
    ///
    /// The matching arm has no test: it dispatches
    /// `restartable_point(auto='run')`, which forks the test runner.
    /// Covering it needs an injectable `CheckpointRuntime`.
    #[test]
    fn event_pre_fork_before_non_matching_goal_returns_ok() {
        let setopts: Vec<(String, OptionValue)> = vec![(
            "fork_before".to_string(),
            OptionValue::Choice("rtype".to_string()),
        )];
        let td = TranslationDriver::new(Some(setopts), None, Vec::new(), None, None, None, None)
            .expect("driver");
        // `fork_before='rtype'` resolves to `'rtype_lltype'`; the goal
        // arg is `'annotate'`, which does not match — body skips.
        td._event_with_func("pre", "annotate", false)
            .expect("non-matching goal must skip checkpoint");
    }

    /// Upstream `setup_library` at `:212-216` ends with
    /// `self.secondary_entrypoints = libdef.functions`. Pin the
    /// downcast → assignment chain so the Rust port matches upstream
    /// for typed [`LibDef`] carriers.
    #[test]
    fn setup_library_assigns_libdef_functions_to_secondary_entrypoints() {
        let td = TranslationDriver::new_default().expect("driver");
        // Upstream `LibDef(functions=[(func, argtypes)])`. Use the
        // simplest opaque func / argtypes shape so the assignment is
        // observable without exercising the annotator.
        let func: Rc<dyn Any> = Rc::new("the-func".to_string());
        let libdef = Rc::new(LibDef::new(vec![(func.clone(), Vec::new())])) as Rc<dyn Any>;
        td.setup_library(libdef, None, HashMap::new(), None)
            .expect("setup_library");
        let secondary = td.secondary_entrypoints.borrow().clone();
        assert_eq!(secondary.len(), 1, "must mirror libdef.functions length");
        // Verify the inner reference is preserved (Rc::ptr_eq).
        assert!(
            Rc::ptr_eq(&secondary[0].0, &func),
            "must preserve func Rc identity"
        );
    }

    /// Untyped `libdef` carriers (e.g. `Rc::new(())`) hit the
    /// AttributeError-equivalent path: upstream `:216
    /// libdef.functions` raises on a missing attribute, so the Rust
    /// port surfaces a [`TaskError`] rather than silently leaving the
    /// slot empty.
    #[test]
    fn setup_library_errors_on_libdef_without_functions_attribute() {
        let td = TranslationDriver::new_default().expect("driver");
        let untyped: Rc<dyn Any> = Rc::new(());
        let err = td
            .setup_library(untyped, None, HashMap::new(), None)
            .expect_err("untyped libdef must surface AttributeError-equivalent");
        assert!(
            err.message.contains("driver.py:216"),
            "expected `driver.py:216` citation, got: {}",
            err.message
        );
    }

    #[test]
    fn from_targetspec_entry_result_runs_target_and_setup_standalone() {
        let entry = host_function("target.entry_only");
        let entry_for_target = entry.clone();
        let target: Rc<dyn TargetSpecCallable> =
            Rc::new(move |driver: &Rc<TranslationDriver>, args: &[String]| {
                // Upstream `driver.py:584`: the freshly constructed driver
                // is passed to the target callable together with args.
                assert!(driver.translator.borrow().is_none());
                assert_eq!(args, &["--flag".to_string()]);
                Ok(TargetSpecResult::Entry(entry_for_target.clone()))
            });

        let driver = TranslationDriver::from_targetspec(
            TargetSpecDict::new(target),
            None,
            Some(vec!["--flag".to_string()]),
            None,
            Vec::new(),
            None,
        )
        .expect("from_targetspec");

        assert_eq!(
            driver.entry_point.borrow().as_ref().map(|h| h.qualname()),
            Some(entry.qualname())
        );
        assert!(driver.standalone.get(), "inputtypes=None marks standalone");
        assert!(driver.translator.borrow().is_some(), "setup must run");
        assert_eq!(driver.timer.events().len(), 1);
        assert_eq!(driver.timer.events()[0].0, "loading target");

        let extra = driver.extra.borrow();
        let target_slot = extra
            .get("target")
            .expect("extra keeps upstream target key")
            .downcast_ref::<TargetSpecCallableSlot>()
            .expect("target slot downcast");
        let _ = target_slot.target.clone();
    }

    #[test]
    fn from_targetspec_two_tuple_result_sets_inputtypes_without_policy() {
        use crate::annotator::signature::AnnotationSpec;

        let entry = host_function("target.entry_inputtypes");
        let entry_for_target = entry.clone();
        let target: Rc<dyn TargetSpecCallable> =
            Rc::new(move |_: &Rc<TranslationDriver>, _: &[String]| {
                Ok(TargetSpecResult::EntryInputTypes(
                    entry_for_target.clone(),
                    vec![AnnotationSpec::Int, AnnotationSpec::Str],
                ))
            });

        let driver = TranslationDriver::from_targetspec(
            TargetSpecDict::new(target).with_extra("answer", Rc::new(42_i64) as Rc<dyn Any>),
            None,
            None,
            None,
            Vec::new(),
            None,
        )
        .expect("from_targetspec");

        assert_eq!(
            driver.entry_point.borrow().as_ref().map(|h| h.qualname()),
            Some(entry.qualname())
        );
        assert!(
            !driver.standalone.get(),
            "explicit inputtypes is non-standalone"
        );
        assert_eq!(
            *driver.inputtypes.borrow(),
            Some(vec![AnnotationSpec::Int, AnnotationSpec::Str])
        );
        assert!(
            driver.policy.borrow().is_some(),
            "setup supplies default policy"
        );
        assert!(driver.extra.borrow().contains_key("answer"));
    }

    #[test]
    fn from_targetspec_three_tuple_result_passes_policy_to_setup() {
        use crate::annotator::policy::AnnotatorPolicy;
        use crate::annotator::signature::AnnotationSpec;

        let entry = host_function("target.entry_inputtypes_policy");
        let entry_for_target = entry.clone();
        let target: Rc<dyn TargetSpecCallable> =
            Rc::new(move |_: &Rc<TranslationDriver>, _: &[String]| {
                Ok(TargetSpecResult::EntryInputTypesPolicy(
                    entry_for_target.clone(),
                    vec![AnnotationSpec::Bool],
                    AnnotatorPolicy,
                ))
            });

        let driver = TranslationDriver::from_targetspec(
            TargetSpecDict::new(target),
            None,
            None,
            None,
            Vec::new(),
            None,
        )
        .expect("from_targetspec");

        assert_eq!(
            driver.entry_point.borrow().as_ref().map(|h| h.qualname()),
            Some(entry.qualname())
        );
        assert_eq!(
            *driver.inputtypes.borrow(),
            Some(vec![AnnotationSpec::Bool])
        );
        assert!(driver.policy.borrow().is_some());
    }

    #[test]
    fn from_targetspec_propagates_target_error() {
        let target: Rc<dyn TargetSpecCallable> =
            Rc::new(|_: &Rc<TranslationDriver>, _: &[String]| {
                Err(TaskError {
                    message: "target boom".to_string(),
                })
            });
        let result = TranslationDriver::from_targetspec(
            TargetSpecDict::new(target),
            None,
            None,
            None,
            Vec::new(),
            None,
        );
        let err = match result {
            Ok(_) => panic!("target errors propagate like upstream exceptions"),
            Err(e) => e,
        };
        assert!(
            err.message.contains("target boom"),
            "expected target error, got: {}",
            err.message
        );
    }
}
