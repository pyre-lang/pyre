//! `pypy/module/pypyjit/policy.py` — `PyPyJitPolicy`.
//!
//! The JIT policy the interpreter hands the translator: which interpreter
//! functions the codewriter may look inside.  It extends
//! `rpython/jit/codewriter/policy.py` `JitPolicy`
//! (`majit_translate::codewriter::policy`) and overrides
//! `look_inside_function` only.
//!
//! This file is not part of `pyre-interpreter`'s module tree.  The policy
//! is translation-time code: upstream `targetpypystandalone.py jitpolicy`
//! imports it only while translating, and pyre's translation runs in the
//! `pyre-jit-trace` build script, which includes this file by `#[path]`
//! (`build/prepass.rs`).  `pyre-interpreter` itself does not link the
//! translator the trait lives in.
//!
//! Upstream reads `func.__module__`, a dotted Python module path; pyre reads
//! the same attribute off `graph.func.module`, spelled as `module_path!()`
//! spells it.  Each upstream root is matched against pyre's counterpart
//! root:
//!
//! | upstream | pyre |
//! |---|---|
//! | `pypy.module.` | `pyre_interpreter::module::`, `pyre_module::module::` |
//! | `pypy.interpreter.astcompiler.` | `pyre_interpreter::astcompiler::` |
//! | `pypy.interpreter.pyparser.` | `pyre_interpreter::pyparser::` |
//! | `rpython.rlib.rlocale` | `majit_rlib::rlocale` |
//! | `rpython.rlib.rsocket` | `majit_rlib::rsocket` |
//!
//! `pypy.module` has two counterparts because pyre splits the builtin
//! modules across two crates: `pyre-interpreter/src/module/` and the
//! optional `pyre-module/src/module/` mirror the same `pypy/module/` tree.

use majit_translate::codewriter::policy::{JitPolicy, JitPolicyState};
use majit_translate::model::FunctionGraph;

/// `pypy.module.` — the roots `look_inside_function` strips before calling
/// `look_inside_pypy_module`.
const PYPY_MODULE_ROOTS: [&str; 2] = ["pyre_interpreter::module::", "pyre_module::module::"];

/// policy.py `class PyPyJitPolicy(JitPolicy)`.
#[derive(Debug, Clone, Default)]
pub struct PyPyJitPolicy {
    pub state: JitPolicyState,
}

impl PyPyJitPolicy {
    /// `PyPyJitPolicy(jithookiface)`.  `targetpypystandalone.py jitpolicy`
    /// passes `pypy_hooks` (`pypy/module/pypyjit/hooks.py`); pyre has no
    /// JIT hook interface, so `state.jithookiface` stays unset.
    pub fn new() -> Self {
        Self {
            state: JitPolicyState::new(),
        }
    }

    /// policy.py `look_inside_pypy_module(self, modname)`.
    ///
    /// `modname` is the module path below `pypy.module.`, `::`-separated.
    /// `unicodedata` / `gc` / `_minimal_curses` are rejected only below the
    /// package: their bodies live in `interp_*` submodules, and the package
    /// itself holds the `moduledef` table.
    pub fn look_inside_pypy_module(&self, modname: &str) -> bool {
        if modname == "__builtin__::operation"
            || modname == "__builtin__::abstractinst"
            || modname == "__builtin__::interp_classobj"
            || modname == "__builtin__::functional"
            || modname == "__builtin__::descriptor"
            || modname == "thread::os_local"
            || modname == "thread::os_thread"
            || modname.starts_with("_rawffi::alt")
        {
            return true;
        }
        let (modname, rest) = match modname.split_once("::") {
            Some((modname, rest)) => {
                if ["unicodedata", "gc", "_minimal_curses"].contains(&modname) {
                    return false;
                }
                (modname, rest)
            }
            None => (modname, ""),
        };
        if modname == "pypyjit" && rest.contains("interp_resop") {
            return false;
        }
        true
    }
}

impl JitPolicy for PyPyJitPolicy {
    fn state(&self) -> &JitPolicyState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut JitPolicyState {
        &mut self.state
    }

    /// policy.py `look_inside_function(self, func)`.
    fn look_inside_function(&self, func: &FunctionGraph) -> bool {
        let module = func.func.module.as_deref().unwrap_or("?");

        if module == "majit_rlib::rlocale" || module == "majit_rlib::rsocket" {
            return false;
        }
        if module.starts_with("pyre_interpreter::astcompiler::") {
            return false;
        }
        if module.starts_with("pyre_interpreter::pyparser::") {
            return false;
        }
        if let Some(modname) = PYPY_MODULE_ROOTS
            .iter()
            .find_map(|root| module.strip_prefix(root))
            && !self.look_inside_pypy_module(modname)
        {
            return false;
        }

        true
    }

    fn snapshot_look_inside_function(&self) -> Box<dyn Fn(&FunctionGraph) -> bool> {
        let this = self.clone();
        Box::new(move |func| JitPolicy::look_inside_function(&this, func))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn func_in(module: Option<&str>) -> FunctionGraph {
        let mut graph = FunctionGraph::new("f");
        graph.func.module = module.map(str::to_string);
        graph
    }

    fn looks_inside(module: &str) -> bool {
        PyPyJitPolicy::new().look_inside_function(&func_in(Some(module)))
    }

    fn pypypolicy() -> PyPyJitPolicy {
        PyPyJitPolicy::new()
    }

    // `pypy/module/pypyjit/test/test_policy.py`, one test per upstream test.

    #[test]
    fn test_id_any() {
        assert!(looks_inside("pyre_object::intobject"));
    }

    #[test]
    fn test_rlocale() {
        assert!(!looks_inside("majit_rlib::rlocale"));
    }

    #[test]
    fn test_astcompiler() {
        assert!(!looks_inside("pyre_interpreter::astcompiler::ast"));
    }

    #[test]
    fn test_pyparser() {
        assert!(!looks_inside("pyre_interpreter::pyparser::parser"));
    }

    #[test]
    fn test_property() {
        assert!(looks_inside(
            "pyre_interpreter::module::__builtin__::descriptor"
        ));
    }

    #[test]
    fn test_thread_local() {
        assert!(looks_inside("pyre_interpreter::module::thread::os_local"));
        assert!(looks_inside("pyre_interpreter::module::thread::os_thread"));
    }

    #[test]
    fn test_time() {
        assert!(looks_inside("pyre_interpreter::module::time::interp_time"));
    }

    #[test]
    fn test_io() {
        assert!(looks_inside(
            "pyre_interpreter::module::_io::interp_bytesio"
        ));
    }

    #[test]
    fn test_thread() {
        assert!(looks_inside("pyre_interpreter::module::thread::os_lock"));
    }

    #[test]
    fn test_select() {
        assert!(looks_inside("pyre_module::module::select::interp_select"));
    }

    #[test]
    fn test_pypy_module() {
        assert!(looks_inside("pyre_module::module::_random::interp_random"));
        assert!(looks_inside(
            "pyre_interpreter::module::_collections::interp_deque"
        ));
        let policy = pypypolicy();
        assert!(policy.look_inside_pypy_module("__builtin__::operation"));
        assert!(policy.look_inside_pypy_module("__builtin__::abstractinst"));
        assert!(policy.look_inside_pypy_module("__builtin__::functional"));
        assert!(policy.look_inside_pypy_module("__builtin__::descriptor"));
        assert!(policy.look_inside_pypy_module("exceptions::interp_exceptions"));
        for modname in ["pypyjit", "signal", "micronumpy", "math", "imp"] {
            assert!(policy.look_inside_pypy_module(modname));
            assert!(policy.look_inside_pypy_module(&format!("{modname}::foo")));
        }
        assert!(!policy.look_inside_pypy_module("pypyjit::interp_resop"));
    }

    #[test]
    fn test_see_jit_module() {
        assert!(pypypolicy().look_inside_pypy_module("pypyjit::interp_jit"));
    }

    // The pyre-side mappings the upstream tests do not reach.

    /// `unicodedata` / `gc` / `_minimal_curses` are rejected below the
    /// package in both module crates; the package path itself is not.
    #[test]
    fn excluded_modules_are_rejected_below_the_package_in_both_crates() {
        for root in PYPY_MODULE_ROOTS {
            for modname in ["unicodedata", "gc", "_minimal_curses"] {
                assert!(looks_inside(&format!("{root}{modname}")));
                assert!(!looks_inside(&format!("{root}{modname}::interp")));
                // `moduledef.py` is `pypy.module.<name>.moduledef`.
                assert!(!looks_inside(&format!("{root}{modname}::moduledef")));
            }
        }
    }

    /// `rpython.rlib.rsocket` is matched exactly, like `rlocale`; its rffi
    /// layer is not in the list.
    #[test]
    fn rsocket_is_rejected_and_its_rffi_layer_is_not() {
        assert!(!looks_inside("majit_rlib::rsocket"));
        assert!(looks_inside("majit_rlib::_rsocket_rffi"));
    }

    /// `func.__module__ or '?'`: a function with no module is looked into.
    #[test]
    fn a_function_without_a_module_is_looked_into() {
        assert!(pypypolicy().look_inside_function(&func_in(None)));
    }

    /// `policy.py look_inside_graph` reads `_jit_look_inside_` before it
    /// asks `look_inside_function`, so the hint overrides the module.
    #[test]
    fn look_inside_hint_overrides_the_module_rejection() {
        let mut func = func_in(Some("pyre_module::module::unicodedata::interp_ucd"));
        let mut policy = pypypolicy();
        assert!(!policy.look_inside_graph(&func));
        func.push_hint("jit_look_inside");
        assert!(policy.look_inside_graph(&func));
    }
}
