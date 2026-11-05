//! Port of `rpython/translator/tool/taskengine.py`.
//!
//! Upstream is a 130-LOC self-contained task-dependency graph
//! executor used as the base class of
//! `translator.driver.TranslationDriver`. Subclasses define
//! `task_<name>` methods; `SimpleTaskEngine.__init__` discovers them
//! via `dir(self)` and populates `self.tasks` with
//! `{name: (callable, deps)}` entries. `_plan(goals, skip)` topo-
//! sorts the subset needed to reach `goals`; `_execute(goals)`
//! iterates the plan calling `self._do(goal, callable, ...)`, which
//! subclasses override for timing / logging.
//!
//! **Rust adaptation** (documented):
//!
//! Python's `dir(self)` reflection has no direct Rust equivalent —
//! there is no runtime way to enumerate the `task_<name>` methods of
//! a type without reflection machinery that Rust intentionally lacks.
//! The port replaces the implicit reflective scan with explicit
//! [`SimpleTaskEngine::register_task`] calls that subclasses perform
//! inside their constructors (i.e. from `TranslationDriver::new`
//! once that lands). The observable behaviour matches upstream
//! line-by-line: after the constructor returns, `self.tasks` holds
//! the same `(name, callable, deps)` triples upstream produces.
//!
//! The override-hook pattern (`_do` / `_event` / `_error`) uses a
//! Rust trait, [`TaskEngineHooks`], with no-op defaults. Subclasses
//! implement it and the default [`SimpleTaskEngine::execute`] method
//! dispatches through the trait, preserving upstream's Python-MRO
//! override semantics.

use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// Return type of a task callable. Upstream task bodies return
/// whatever the user wants (annotate returns the annotator, rtype
/// returns None, compile returns the entry-point pointer, …);
/// `Box<dyn Any>` carries that shape across the generic engine API.
pub type TaskOutput = Option<Box<dyn Any>>;

pub use majit_eci::taskengine::TaskError;

/// Port of a task callable + its registered metadata.
///
/// Upstream at `taskengine.py:10-14` stores tasks as
/// `{name: (callable, deps)}` pairs. The callable is a Python-bound
/// method with optional `task_title` and `task_idempotent`
/// attributes set by a decorator (upstream `@taskdef(...)` produces
/// these). The Rust port bundles the attributes directly on the
/// registration struct — reflection-less — matching the observable
/// data.
pub struct TaskRegistration {
    /// Upstream `taskcallable`. Shared via `Rc` because `_plan_cache`
    /// preserves the registration between plan resolutions.
    pub callable: Rc<dyn Fn() -> Result<TaskOutput, TaskError>>,
    /// Upstream `getattr(task, 'task_deps', [])` at
    /// `taskengine.py:12`. Names of prerequisite goals the planner
    /// must execute before this task.
    pub deps: Vec<String>,
    /// Upstream `task.task_title` consulted by `TranslationDriver._do`
    /// (`driver.py`). Human-readable progress label; the engine
    /// itself only propagates it.
    pub title: String,
    /// Upstream `task.task_idempotent` consulted by
    /// `TranslationDriver._do` (`driver.py`). When `true`, the
    /// driver does *not* stash the goal in `self.done`, so a
    /// re-request re-runs the task. The engine does not consult this
    /// itself; it rides along for subclasses.
    pub idempotent: bool,
}

/// Plan cache key: `(goals, skip)` tuples stored in
/// `self._plan_cache` at `taskengine.py:19`.
type PlanKey = (Vec<String>, Vec<String>);

/// Port of upstream's nested `consider(subgoal)` inside `_plan`
/// (`taskengine.py`):
///
/// ```python
/// def consider(subgoal):
///     if subgoal in seen:
///         return
///     else:
///         seen[subgoal] = True
///     constraints.append([subgoal])
///     deps = subgoals(subgoal)
///     for dep in deps:
///         constraints.append([subgoal, dep])
///         consider(dep)
/// ```
///
/// Free-function form because Rust closures cannot recurse without
/// extra plumbing, and `consider` shares mutable state
/// (`seen` / `constraints`) across calls. `tasks` / `goal_set` /
/// `skip` are `&` borrows — they don't change inside `_plan`.
///
/// The matching upstream `subgoals(task_name)` (`taskengine.py`)
/// — the `??` / `?` sigil resolver — is inlined here as the
/// `resolved_deps` filter, matching the same per-dep emission order.
fn consider(
    subgoal: &str,
    tasks: &HashMap<String, TaskRegistration>,
    goal_set: &HashSet<&String>,
    skip: &[String],
    seen: &mut HashSet<String>,
    constraints: &mut Vec<Vec<String>>,
) -> Result<(), TaskError> {
    // Upstream `:42-43`: `if subgoal in seen: return`.
    if seen.contains(subgoal) {
        return Ok(());
    }
    // Upstream `:44-45`: `seen[subgoal] = True`.
    seen.insert(subgoal.to_string());
    // Upstream `:46`: `constraints.append([subgoal])`.
    constraints.push(vec![subgoal.to_string()]);

    let reg = tasks.get(subgoal).ok_or_else(|| TaskError {
        message: format!("SimpleTaskEngine._plan: unknown goal {:?}", subgoal),
    })?;

    // Upstream nested `subgoals(task_name)` at `:26-37` — apply the
    // `??` (optional) / `?` (suggested) sigils.
    let resolved_deps: Vec<String> = reg
        .deps
        .iter()
        .filter_map(|dep| {
            if let Some(rest) = dep.strip_prefix("??") {
                // Upstream `:29-32`: optional, only if already in goals.
                if goal_set.contains(&rest.to_string()) {
                    Some(rest.to_string())
                } else {
                    None
                }
            } else if let Some(rest) = dep.strip_prefix('?') {
                // Upstream `:33-36`: suggested, included unless skipped.
                if skip.iter().any(|s| s == rest) {
                    None
                } else {
                    Some(rest.to_string())
                }
            } else {
                // Upstream `:37 yield dep` — bare requirement.
                Some(dep.clone())
            }
        })
        .collect();

    // Upstream `:48-50`: `for dep in deps: constraints.append([subgoal,
    // dep]); consider(dep)`. The append-then-recurse interleave is the
    // ordering PyPy's topo picker breaks ties on; preserve it
    // line-for-line.
    for dep in &resolved_deps {
        constraints.push(vec![subgoal.to_string(), dep.clone()]);
        consider(dep, tasks, goal_set, skip, seen, constraints)?;
    }
    Ok(())
}

/// Port of `rpython/translator/tool/taskengine.py SimpleTaskEngine
/// SimpleTaskEngine` state.
///
/// Upstream is a class whose subclasses inherit its methods. Rust
/// has no inheritance; the port exposes this struct as a composable
/// field on subclasses (e.g. `TranslationDriver.engine:
/// SimpleTaskEngine`), with the override-hook methods routed through
/// the [`TaskEngineHooks`] trait.
pub struct SimpleTaskEngine {
    /// Upstream `self._plan_cache = {}` at `taskengine.py`. The
    /// cache is keyed on `(tuple(goals), tuple(skip))` per
    /// `taskengine.py:19`, storing the topologically-sorted plan.
    _plan_cache: RefCell<HashMap<PlanKey, Vec<String>>>,
    /// Upstream `self.tasks = tasks = {}` at `taskengine.py`. Keyed
    /// on task name, value is the `(callable, deps)` pair upstream
    /// stores; the Rust port carries the full [`TaskRegistration`] to
    /// preserve `task_title` / `task_idempotent` for subclass hooks.
    tasks: RefCell<HashMap<String, TaskRegistration>>,
    /// Sidecar to [`tasks`] tracking registration order. Upstream's
    /// `self.tasks` is a Python dict whose iteration order is the
    /// insertion order (Python 3.7+); callers like
    /// `driver.py for task in self.tasks:` observe that order.
    /// Rust's `HashMap` has no ordering guarantee, so the port mirrors
    /// the dict-insertion-order trail in this auxiliary `Vec`.
    /// Re-registering an existing task name preserves the position
    /// upstream's dict assignment would (`d['k'] = v` keeps `'k'`'s
    /// position).
    task_order: RefCell<Vec<String>>,
}

impl Default for SimpleTaskEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleTaskEngine {
    /// Upstream `SimpleTaskEngine.__init__` at `taskengine.py`.
    /// Initializes the plan cache + task map; the reflective scan over
    /// `task_<name>` methods at lines 7-14 is deferred to explicit
    /// [`register_task`](Self::register_task) calls by subclasses
    /// (Rust adaptation; see module docstring).
    pub fn new() -> Self {
        SimpleTaskEngine {
            _plan_cache: RefCell::new(HashMap::new()),
            tasks: RefCell::new(HashMap::new()),
            task_order: RefCell::new(Vec::new()),
        }
    }

    /// Replacement for upstream's `dir(self)` reflective scan
    /// (`taskengine.py:7-14`). Subclasses call this once per `task_*`
    /// method during their construction, passing the stripped task
    /// name (`"annotate"`, not `"task_annotate"`), the callable, its
    /// declared deps, and the title / idempotency attributes that
    /// upstream reads via `getattr(task, 'task_deps', [])`.
    ///
    /// Re-registering under an existing name overwrites the entry,
    /// matching upstream `taskengine.py:14 tasks[task_name] = task,
    /// task_deps` dict-assignment semantics. The order recorded in
    /// `task_order` is preserved on overwrite (Python dict assignment
    /// keeps the existing key's position).
    pub fn register_task(
        &self,
        name: impl Into<String>,
        callable: Rc<dyn Fn() -> Result<TaskOutput, TaskError>>,
        deps: Vec<String>,
        title: impl Into<String>,
        idempotent: bool,
    ) {
        let name = name.into();
        let title = title.into();
        let mut tasks = self.tasks.borrow_mut();
        let mut order = self.task_order.borrow_mut();
        if !tasks.contains_key(&name) {
            order.push(name.clone());
        }
        tasks.insert(
            name,
            TaskRegistration {
                callable,
                deps,
                title,
                idempotent,
            },
        );
    }

    /// Upstream `SimpleTaskEngine._plan(goals, skip)` at
    /// `taskengine.py`. Topologically sorts the subset of tasks
    /// needed to reach `goals`, respecting declared `task_deps` and
    /// applying the `?` / `??` suffix sigils on optional / suggested
    /// dependencies:
    ///
    /// - `??<dep>` — optional, only included if `<dep>` is already in
    ///   `goals`. Line 29-32.
    /// - `?<dep>` — suggested, included unless `<dep>` is in `skip`.
    ///   Line 33-36.
    /// - bare `<dep>` — required.
    ///
    /// The returned plan is in execution order (oldest dep first,
    /// innermost last).
    ///
    /// `goals` that are not in `skip` filter the `skip` list (line 17).
    pub fn _plan(&self, goals: &[String], skip: &[String]) -> Result<Vec<String>, TaskError> {
        // Upstream `skip = [toskip for toskip in skip if toskip not in goals]`.
        let goal_set: HashSet<&String> = goals.iter().collect();
        let skip: Vec<String> = skip
            .iter()
            .filter(|s| !goal_set.contains(*s))
            .cloned()
            .collect();

        let key: PlanKey = (goals.to_vec(), skip.clone());
        if let Some(cached) = self._plan_cache.borrow().get(&key) {
            return Ok(cached.clone());
        }

        let tasks = self.tasks.borrow();

        // Upstream `constraints = []` + nested `consider(subgoal)`
        // that pushes `[subgoal]` then for every dep pushes
        // `[subgoal, dep]` and recurses. The flat list of mini-
        // constraint-lists is then repeatedly drained in topo order.
        let mut constraints: Vec<Vec<String>> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        // Upstream `for goal in goals: consider(goal)` at
        // `taskengine.py`. Drives `consider` in goal order so
        // independent goals' constraints are emitted in the same order
        // PyPy's recursion produces — the topo picker at `:64-69`
        // breaks ties on this ordering, so a LIFO worklist would
        // shuffle independent goals' constraints and yield a
        // *different* plan after the final `plan.reverse()`. The
        // explicit-recursion `consider` helper below mirrors PyPy's
        // call-stack recursion verbatim.
        for goal in goals {
            consider(goal, &tasks, &goal_set, &skip, &mut seen, &mut constraints)?;
        }

        // Upstream `while True: …` at lines 59-76. Repeatedly pick a
        // candidate whose leading slot appears in no other
        // constraint's tail, append to plan, then remove it from every
        // constraint's head. Detects a circular dependency when no
        // candidate survives.
        let mut plan: Vec<String> = Vec::new();
        loop {
            // Upstream `cands = dict.fromkeys([constr[0] for constr in
            // constraints if constr])` — ordered unique leading slots.
            let mut cands: Vec<String> = Vec::new();
            let mut cand_seen: HashSet<String> = HashSet::new();
            for constr in &constraints {
                if let Some(head) = constr.first()
                    && cand_seen.insert(head.clone())
                {
                    cands.push(head.clone());
                }
            }
            if cands.is_empty() {
                break;
            }

            // Upstream nested loop: pick the first `cand` that is not
            // named in any constraint's tail (`constr[1:]`).
            let mut picked: Option<String> = None;
            for cand in &cands {
                let appears_in_tail = constraints
                    .iter()
                    .any(|c| c.iter().skip(1).any(|s| s == cand));
                if !appears_in_tail {
                    picked = Some(cand.clone());
                    break;
                }
            }
            let cand = picked.ok_or_else(|| TaskError {
                message: "SimpleTaskEngine._plan: circular dependecy".to_string(),
            })?;

            plan.push(cand.clone());
            // Upstream: drop every constraint whose head equals the
            // picked candidate (the `del constr[0]` drops the head and
            // an empty constraint disappears from subsequent `cands`).
            for constr in &mut constraints {
                if constr.first() == Some(&cand) {
                    constr.remove(0);
                }
            }
        }

        plan.reverse();

        self._plan_cache.borrow_mut().insert(key, plan.clone());
        Ok(plan)
    }

    /// Upstream `SimpleTaskEngine._depending_on(goal)` at
    /// `taskengine.py`. Returns every task whose `task_deps`
    /// contains `goal`. Used by `_depending_on_closure` to propagate
    /// `disable()` effects.
    pub fn _depending_on(&self, goal: &str) -> Vec<String> {
        let tasks = self.tasks.borrow();
        let mut out = Vec::new();
        for (name, reg) in tasks.iter() {
            if reg.deps.iter().any(|d| {
                // Upstream compares against the raw `task_deps` entry
                // (including `?` / `??` prefix). Mirror that so a
                // `disable("annotate")` reaches tasks that only
                // suggest it.
                d == goal
            }) {
                out.push(name.clone());
            }
        }
        out
    }

    /// Upstream `SimpleTaskEngine._depending_on_closure(goal)` at
    /// `taskengine.py`. Transitive closure of `_depending_on`.
    pub fn _depending_on_closure(&self, goal: &str) -> Vec<String> {
        let mut d: HashSet<String> = HashSet::new();
        let mut stack: Vec<String> = vec![goal.to_string()];
        while let Some(cur) = stack.pop() {
            if !d.insert(cur.clone()) {
                continue;
            }
            for dep in self._depending_on(&cur) {
                stack.push(dep);
            }
        }
        d.into_iter().collect()
    }

    /// Access the registered tasks. Mirrors upstream's direct
    /// `self.tasks` attribute access at
    /// `driver.py for task in self.tasks:`.
    pub fn tasks(&self) -> std::cell::Ref<'_, HashMap<String, TaskRegistration>> {
        self.tasks.borrow()
    }

    /// Task names in registration (== upstream dict-insertion) order.
    /// Use this when iterating `self.tasks` matters, e.g. in
    /// `driver.py for task in self.tasks:` whose loop order is
    /// observable through `self.exposed`.
    pub fn task_names_in_registration_order(&self) -> Vec<String> {
        self.task_order.borrow().clone()
    }
}

/// Override hooks for [`SimpleTaskEngine::execute`]. Port of upstream
/// `_do` / `_event` / `_error` methods on `SimpleTaskEngine` at
/// `taskengine.py`.
///
/// Upstream relies on Python's MRO so that when
/// `SimpleTaskEngine._execute` calls `self._do(...)`, the call
/// dispatches to `TranslationDriver._do` (the subclass override).
/// Rust has no inheritance; subclasses implement this trait and pass
/// `self` to [`SimpleTaskEngine::execute`], which routes the calls
/// through the trait for the same observable dispatch.
///
/// Defaults match upstream:
/// - `_do(goal, func)` just invokes `func()`.
/// - `_event(kind, goal)` / `_error(goal)` are no-ops.
///
/// `_event` returns `Result<(), TaskError>` so the upstream
/// `taskengine.py` / `:112` plain calls — which raise unwrapped on
/// `task_earlycheck` failure inside `driver.py` — propagate
/// instead of being silently swallowed. The default impl returns
/// `Ok(())`, matching upstream's `def _event(self, kind, goal, taskcallable)
/// : pass` body.
pub trait TaskEngineHooks {
    fn _do(
        &self,
        _goal: &str,
        callable: &Rc<dyn Fn() -> Result<TaskOutput, TaskError>>,
    ) -> Result<TaskOutput, TaskError> {
        callable()
    }

    fn _event(&self, _kind: &str, _goal: &str) -> Result<(), TaskError> {
        Ok(())
    }

    fn _error(&self, _goal: &str) {}
}

/// Default `TaskEngineHooks` impl — every method is the upstream
/// default body. Used when a caller invokes [`SimpleTaskEngine::execute`]
/// without a subclass wrapper.
pub struct DefaultHooks;

impl TaskEngineHooks for DefaultHooks {}

impl SimpleTaskEngine {
    /// Upstream `SimpleTaskEngine._execute(goals, *args, **kwds)` at
    /// `taskengine.py`. Runs the plan for `goals` in order,
    /// calling `hooks._event("planned"/"pre"/"post", goal)` around each
    /// `hooks._do(goal, callable)`. Returns the result of the LAST
    /// `_do` call (upstream behaviour: `res = None` then repeatedly
    /// overwritten, final value returned).
    ///
    /// `task_skip` is the `_plan` `skip` argument, passed via
    /// upstream's `**kwds` dict (line 104). Port makes it a distinct
    /// parameter since Rust has no `**kwds` analogue.
    ///
    /// Errors raised inside `hooks._do` surface unchanged; the
    /// engine routes through `hooks._error(goal)` before propagating,
    /// matching upstream's `except:` block at lines 117-119.
    pub fn execute<H: TaskEngineHooks>(
        &self,
        hooks: &H,
        goals: &[String],
        task_skip: &[String],
    ) -> Result<TaskOutput, TaskError> {
        let plan = self._plan(goals, task_skip)?;
        // Upstream: first loop emits "planned" events for every task.
        // Upstream `taskengine.py` lets a `_event` raise
        // bubble straight out of `_execute` (no try/except wraps the
        // first loop), so any error returned by the planned-phase
        // earlycheck (`driver.py`) must propagate here.
        for goal in &plan {
            hooks._event("planned", goal)?;
        }

        // Upstream: second loop actually runs the tasks.
        let mut res: TaskOutput = None;
        for goal in &plan {
            // Upstream `:112` — `_event('pre', ...)` is also outside
            // the try/except block, so a raise here propagates
            // unwrapped just like the planned phase.
            hooks._event("pre", goal)?;
            // Clone the callable Rc out so we don't hold the `tasks`
            // borrow across the _do call (which may re-enter the
            // engine for recursive tasks).
            let callable = {
                let tasks = self.tasks.borrow();
                let reg = tasks.get(goal).ok_or_else(|| TaskError {
                    message: format!("SimpleTaskEngine.execute: unknown goal {:?}", goal),
                })?;
                Rc::clone(&reg.callable)
            };
            match hooks._do(goal, &callable) {
                Ok(value) => {
                    res = value;
                }
                Err(err) => {
                    hooks._error(goal);
                    return Err(err);
                }
            }
            // Upstream `:120` — same outside-try shape applies.
            hooks._event("post", goal)?;
        }
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn noop_callable() -> Rc<dyn Fn() -> Result<TaskOutput, TaskError>> {
        Rc::new(|| Ok(None))
    }

    #[test]
    fn register_task_populates_tasks_map() {
        // Upstream `taskengine.py:5-14` — `self.tasks[task_name] =
        // task, task_deps`. The Rust port's explicit registration
        // must land the same triple.
        let e = SimpleTaskEngine::new();
        e.register_task(
            "annotate",
            noop_callable(),
            vec![],
            "Annotating&simplifying",
            false,
        );
        let tasks = e.tasks.borrow();
        let reg = tasks.get("annotate").expect("registered");
        assert!(reg.deps.is_empty());
        assert_eq!(reg.title, "Annotating&simplifying");
        assert!(!reg.idempotent);
    }

    #[test]
    fn register_task_overwrites_on_duplicate_preserving_order_position() {
        // Upstream `taskengine.py SimpleTaskEngine.__init__ tasks[task_name] = task, task_deps`
        // is dict assignment. Re-registering must overwrite the
        // callable/deps/title/idempotent and preserve the existing key's
        // position in the iteration order (Python dict-assignment
        // semantics).
        let e = SimpleTaskEngine::new();
        e.register_task("a", noop_callable(), vec![], "alpha", false);
        e.register_task("b", noop_callable(), vec![], "bravo", true);
        e.register_task("a", noop_callable(), vec!["b".to_string()], "alpha2", true);

        let tasks = e.tasks.borrow();
        let reg_a = tasks.get("a").expect("a is registered");
        assert_eq!(reg_a.title, "alpha2");
        assert_eq!(reg_a.deps, vec!["b".to_string()]);
        assert!(reg_a.idempotent);
        drop(tasks);

        // Position of "a" stays at index 0 — Python dict assignment
        // does not re-order existing keys.
        assert_eq!(
            e.task_names_in_registration_order(),
            vec!["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn task_names_follow_registration_order() {
        // Upstream `driver.py for task in self.tasks:` iterates dict
        // insertion order. Pin the trail.
        let e = SimpleTaskEngine::new();
        e.register_task("annotate", noop_callable(), vec![], "a", false);
        e.register_task("rtype_lltype", noop_callable(), vec![], "r", false);
        e.register_task("backendopt_lltype", noop_callable(), vec![], "b", false);
        assert_eq!(
            e.task_names_in_registration_order(),
            vec![
                "annotate".to_string(),
                "rtype_lltype".to_string(),
                "backendopt_lltype".to_string(),
            ]
        );
    }

    #[test]
    fn plan_empty_goals_is_empty_list() {
        let e = SimpleTaskEngine::new();
        let plan = e._plan(&[], &[]).expect("empty plan");
        assert!(plan.is_empty());
    }

    #[test]
    fn plan_single_task_no_deps_returns_task_only() {
        // Upstream: `goals=["annotate"]` with `tasks={"annotate":
        // (fn, [])}` → plan `["annotate"]`. Smallest feasible case.
        let e = SimpleTaskEngine::new();
        e.register_task("annotate", noop_callable(), vec![], "a", false);
        let plan = e._plan(&["annotate".to_string()], &[]).expect("plan");
        assert_eq!(plan, vec!["annotate".to_string()]);
    }

    #[test]
    fn plan_iterates_goals_in_forward_order_after_reverse() {
        // Upstream `taskengine.py:52-53` — `for goal in goals:
        // consider(goal)` drives `consider` in goal order.
        // Constraints land in forward order, then `plan.reverse()` at
        // `:78` flips the final plan so the LATEST goal lands first.
        // Fixture: two independent leaves with no deps.
        //
        // The pre-fix port used a LIFO worklist
        // (`worklist.pop()`), which iterated goals in reverse —
        // producing constraints `[[b], [a]]` instead of `[[a], [b]]`,
        // and a final plan `[a, b]` instead of `[b, a]`. This test
        // pins the fix.
        let e = SimpleTaskEngine::new();
        e.register_task("a", noop_callable(), vec![], "a", false);
        e.register_task("b", noop_callable(), vec![], "b", false);
        let plan = e
            ._plan(&["a".to_string(), "b".to_string()], &[])
            .expect("plan");
        // PyPy trace: consider("a") → [[a]]; consider("b") → [[a],[b]];
        // topo picks a then b → plan [a,b]; reverse → [b,a].
        assert_eq!(plan, vec!["b".to_string(), "a".to_string()]);
    }

    #[test]
    fn plan_with_required_dep_orders_dep_first() {
        // Upstream `taskengine.py:43-52` + `:59-78` — topo sort puts
        // `annotate` before `rtype_lltype` because the latter
        // declares the former as a dep.
        let e = SimpleTaskEngine::new();
        e.register_task("annotate", noop_callable(), vec![], "a", false);
        e.register_task(
            "rtype_lltype",
            noop_callable(),
            vec!["annotate".to_string()],
            "r",
            false,
        );
        let plan = e._plan(&["rtype_lltype".to_string()], &[]).expect("plan");
        assert_eq!(
            plan,
            vec!["annotate".to_string(), "rtype_lltype".to_string()]
        );
    }

    #[test]
    fn plan_optional_double_question_included_only_when_already_in_goals() {
        // Upstream `taskengine.py:29-32`: `??<dep>` is only pulled in
        // when `<dep>` is already in the requested goals.
        let e = SimpleTaskEngine::new();
        e.register_task("extra", noop_callable(), vec![], "e", false);
        e.register_task(
            "driver",
            noop_callable(),
            vec!["??extra".to_string()],
            "d",
            false,
        );
        // goals = ["driver"]: extra is NOT in goals, so `??extra` is
        // skipped — plan has only driver.
        let plan_without = e._plan(&["driver".to_string()], &[]).expect("plan");
        assert_eq!(plan_without, vec!["driver".to_string()]);

        // goals = ["driver", "extra"]: extra IS in goals, so the
        // optional dep fires and orders extra before driver.
        let plan_with = e
            ._plan(&["driver".to_string(), "extra".to_string()], &[])
            .expect("plan");
        assert_eq!(plan_with, vec!["extra".to_string(), "driver".to_string()]);
    }

    #[test]
    fn plan_suggested_question_included_unless_in_skip() {
        // Upstream `taskengine.py:33-36`: `?<dep>` is included by
        // default; `skip=["dep"]` removes it.
        let e = SimpleTaskEngine::new();
        e.register_task("optional", noop_callable(), vec![], "o", false);
        e.register_task(
            "driver",
            noop_callable(),
            vec!["?optional".to_string()],
            "d",
            false,
        );
        let plan_default = e._plan(&["driver".to_string()], &[]).expect("plan");
        assert_eq!(
            plan_default,
            vec!["optional".to_string(), "driver".to_string()]
        );
        let plan_skipped = e
            ._plan(&["driver".to_string()], &["optional".to_string()])
            .expect("plan");
        assert_eq!(plan_skipped, vec!["driver".to_string()]);
    }

    #[test]
    fn plan_skip_is_ignored_for_entries_in_goals() {
        // Upstream `taskengine.py:17`: `skip = [toskip for toskip in
        // skip if toskip not in goals]`. Pass a `skip` entry that is
        // also in `goals` — it must be filtered out, so the task
        // still runs.
        let e = SimpleTaskEngine::new();
        e.register_task("optional", noop_callable(), vec![], "o", false);
        e.register_task(
            "driver",
            noop_callable(),
            vec!["?optional".to_string()],
            "d",
            false,
        );
        let plan = e
            ._plan(
                &["driver".to_string(), "optional".to_string()],
                &["optional".to_string()],
            )
            .expect("plan");
        assert_eq!(plan, vec!["optional".to_string(), "driver".to_string()]);
    }

    #[test]
    fn plan_circular_dep_reports_error() {
        // Upstream `taskengine.py:71`: `raise RuntimeError("circular
        // dependecy")` — literal typo preserved in the port's message
        // text so parity grepping matches.
        let e = SimpleTaskEngine::new();
        e.register_task("a", noop_callable(), vec!["b".to_string()], "a", false);
        e.register_task("b", noop_callable(), vec!["a".to_string()], "b", false);
        let err = e._plan(&["a".to_string()], &[]).unwrap_err();
        assert!(
            err.message.contains("circular"),
            "message: {:?}",
            err.message
        );
    }

    #[test]
    fn plan_cache_hits_on_second_call() {
        // Upstream `taskengine.py`: `self._plan_cache[key] = plan`
        // — verify the cache actually fires.
        let e = SimpleTaskEngine::new();
        e.register_task("a", noop_callable(), vec![], "a", false);
        let _ = e._plan(&["a".to_string()], &[]).expect("first");
        assert_eq!(e._plan_cache.borrow().len(), 1);
        let _ = e._plan(&["a".to_string()], &[]).expect("second");
        // Still size 1 — same key re-hit the cache.
        assert_eq!(e._plan_cache.borrow().len(), 1);
    }

    #[test]
    fn depending_on_lists_direct_dependents() {
        // Upstream `taskengine.py _depending_on`.
        let e = SimpleTaskEngine::new();
        e.register_task("a", noop_callable(), vec![], "a", false);
        e.register_task("b", noop_callable(), vec!["a".to_string()], "b", false);
        e.register_task("c", noop_callable(), vec!["b".to_string()], "c", false);
        let deps_a = e._depending_on("a");
        assert_eq!(deps_a, vec!["b".to_string()]);
        let deps_b = e._depending_on("b");
        assert_eq!(deps_b, vec!["c".to_string()]);
        assert!(e._depending_on("c").is_empty());
    }

    #[test]
    fn depending_on_closure_is_transitive() {
        // Upstream `taskengine.py _depending_on_closure` — closes over
        // `_depending_on` repeatedly.
        let e = SimpleTaskEngine::new();
        e.register_task("a", noop_callable(), vec![], "a", false);
        e.register_task("b", noop_callable(), vec!["a".to_string()], "b", false);
        e.register_task("c", noop_callable(), vec!["b".to_string()], "c", false);
        let closure: HashSet<String> = e._depending_on_closure("a").into_iter().collect();
        let expected: HashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        assert_eq!(closure, expected);
    }

    #[test]
    fn execute_runs_plan_and_returns_last_task_result() {
        // Upstream `taskengine.py` — `res = None` then each
        // `_do` overwrites. Final `res` is returned.
        let observed: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        let obs_a = Rc::clone(&observed);
        let obs_b = Rc::clone(&observed);
        let e = SimpleTaskEngine::new();
        e.register_task(
            "a",
            Rc::new(move || {
                obs_a.borrow_mut().push("a".to_string());
                Ok(Some(Box::new(1_i64) as Box<dyn Any>))
            }),
            vec![],
            "a",
            false,
        );
        e.register_task(
            "b",
            Rc::new(move || {
                obs_b.borrow_mut().push("b".to_string());
                Ok(Some(Box::new(42_i64) as Box<dyn Any>))
            }),
            vec!["a".to_string()],
            "b",
            false,
        );
        let res = e
            .execute(&DefaultHooks, &["b".to_string()], &[])
            .expect("execute");
        // Observed order: a first, then b.
        assert_eq!(
            observed.borrow().clone(),
            vec!["a".to_string(), "b".to_string()],
        );
        // Returned `res` is b's return value.
        let value = res.expect("b returned Some");
        let downcast = value.downcast::<i64>().expect("i64 payload");
        assert_eq!(*downcast, 42);
    }

    #[test]
    fn execute_calls_event_hooks_in_planned_pre_post_order() {
        // Upstream `taskengine.py:109-120` — "planned" fires for every
        // goal before any "pre"/"post", then "pre" / task / "post" per
        // goal. Exercise via a custom Hooks impl.
        struct Record {
            events: RefCell<Vec<(String, String)>>,
        }
        impl TaskEngineHooks for Record {
            fn _event(&self, kind: &str, goal: &str) -> Result<(), TaskError> {
                self.events
                    .borrow_mut()
                    .push((kind.to_string(), goal.to_string()));
                Ok(())
            }
        }
        let e = SimpleTaskEngine::new();
        e.register_task("a", noop_callable(), vec![], "a", false);
        e.register_task("b", noop_callable(), vec!["a".to_string()], "b", false);
        let hooks = Record {
            events: RefCell::new(Vec::new()),
        };
        e.execute(&hooks, &["b".to_string()], &[]).expect("execute");

        let evs = hooks.events.borrow();
        // First 2: planned for each goal, in plan order.
        assert_eq!(evs[0], ("planned".to_string(), "a".to_string()),);
        assert_eq!(evs[1], ("planned".to_string(), "b".to_string()),);
        // Then per-goal pre/post.
        assert_eq!(evs[2], ("pre".to_string(), "a".to_string()));
        assert_eq!(evs[3], ("post".to_string(), "a".to_string()));
        assert_eq!(evs[4], ("pre".to_string(), "b".to_string()));
        assert_eq!(evs[5], ("post".to_string(), "b".to_string()));
    }

    #[test]
    fn execute_error_hook_fires_on_task_error() {
        // Upstream `taskengine.py`: `except: self._error(goal);
        // raise`. In Rust, we route the error AFTER calling `_error`.
        struct Track {
            err_goal: RefCell<Option<String>>,
        }
        impl TaskEngineHooks for Track {
            fn _error(&self, goal: &str) {
                *self.err_goal.borrow_mut() = Some(goal.to_string());
            }
        }
        let e = SimpleTaskEngine::new();
        e.register_task(
            "boom",
            Rc::new(|| {
                Err(TaskError {
                    message: "nope".to_string(),
                })
            }),
            vec![],
            "boom",
            false,
        );
        let hooks = Track {
            err_goal: RefCell::new(None),
        };
        let res = e.execute(&hooks, &["boom".to_string()], &[]);
        assert!(res.is_err());
        assert_eq!(hooks.err_goal.borrow().clone(), Some("boom".to_string()),);
    }

    #[test]
    fn execute_custom_do_hook_wraps_callable() {
        // Upstream `driver.py-..` overrides `_do` to run the
        // callable inside a timer. Exercise the hook surface by
        // counting how many times `_do` runs and what it returns.
        struct Counter {
            invocations: Cell<u32>,
        }
        impl TaskEngineHooks for Counter {
            fn _do(
                &self,
                _goal: &str,
                callable: &Rc<dyn Fn() -> Result<TaskOutput, TaskError>>,
            ) -> Result<TaskOutput, TaskError> {
                self.invocations.set(self.invocations.get() + 1);
                callable()
            }
        }
        let e = SimpleTaskEngine::new();
        e.register_task("a", noop_callable(), vec![], "a", false);
        e.register_task("b", noop_callable(), vec!["a".to_string()], "b", false);
        let hooks = Counter {
            invocations: Cell::new(0),
        };
        e.execute(&hooks, &["b".to_string()], &[]).expect("execute");
        assert_eq!(hooks.invocations.get(), 2);
    }
}
