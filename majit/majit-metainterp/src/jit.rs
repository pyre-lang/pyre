//! User-facing runtime half of `rpython/rlib/jit.py`.
//!
//! Line-by-line structural port of the no-op hint API that interpreter
//! code and the `#[jit_interp]` proc macro import as
//! `majit_metainterp::jit::*`.
//!
//! This is intentionally not a same-stem sibling of any
//! `rpython/jit/metainterp/*.py` file. The translator-only
//! marker/extregistry half of `rlib/jit.py` lives in
//! `majit_translate::rlib::jit`, because it depends on annotator,
//! bookkeeper, and extregistry structures that belong to the
//! translation crate.
//!
//! Item ordering in this file follows jit.py exactly. Each section is
//! annotated with the corresponding jit.py line number.
//!
//! Functions in this module are no-ops at interpreter runtime. When the
//! `#[jit_interp]` proc macro processes user code, calls to these functions
//! are recognized and lowered to JitCode operations:
//!
//! - `promote(x)` → `int_guard_value` / `ref_guard_value` / `float_guard_value`
//! - `record_exact_class(v, cls)` → RECORD_EXACT_CLASS IR op

// ── DEBUG_ELIDABLE_FUNCTIONS ──
// rlib/jit.py

/// rlib/jit.py — enables runtime consistency checks on elidable functions.
pub const DEBUG_ELIDABLE_FUNCTIONS: bool = false;

// ── elidable, purefunction ──
// rlib/jit.py — `@elidable` decorator → proc macro `#[elidable]`
// rlib/jit.py — `purefunction` is a deprecated alias for `elidable`.
//                   Rust users should write `#[elidable]` directly.

// ── hint ──
// rlib/jit.py

/// Hint for the JIT.
///
/// Possible hint kinds (RPython keyword arguments):
/// - promote — promote the argument from a variable into a constant
/// - promote_string — same, but promote string by value
/// - promote_unicode — same, but promote unicode string by value
/// - access_directly — directly access a virtualizable as a structure
/// - fresh_virtualizable — virtualizable was just allocated
/// - force_virtualizable — force the virtualizable early
///
/// In Rust, these are separate functions: `promote()`, `promote_string()`, etc.
/// Virtualizable hints are handled by the `virtualizable!` macro.
///
/// rlib/jit.py — `hint(x, **kwds)` returns x unchanged.
#[inline(always)]
pub fn hint<T>(x: T) -> T {
    x
}

// ── access_directly / fresh_virtualizable ──
// rlib/jit.py:88-93

/// Access a virtualizable directly as a structure, without treating it
/// as a virtualizable.
///
/// rlib/jit.py — `hint(x, access_directly=True)`
#[inline(always)]
pub fn hint_access_directly<T>(x: T) -> T {
    hint(x)
}

/// Stop treating a virtualizable as one: from here on the value is an
/// ordinary object.
///
/// rlib/jit.py — `hint(x, access_directly=False)`
///
/// Upstream's single `hint(x, **kwds)` takes the flag's value as a keyword,
/// and the `False` spelling is not a no-op: the entry DELETES
/// `access_directly` and `fresh_virtualizable` from the annotation's flags.
/// `executioncontext.py` uses it at both points where a frame is handed to
/// arbitrary Python ("from here on, frame is just a normal w_object"), so
/// the callee set below is not specialised on the virtualizable protocol.
///
/// Pyre dispatches one helper per kwarg, so the `False` value becomes this
/// separate name rather than an argument.
#[inline(always)]
pub fn hint_no_access_directly<T>(x: T) -> T {
    hint(x)
}

/// Declare that the virtualizable was just allocated, so storing
/// directly on it is what is wanted — `Frame.__init__` is the motivating
/// case (`pypy/interpreter/pyframe.py`).  Its redirected fields are
/// not under the virtualizable protocol yet, so the codewriter leaves
/// them as ordinary struct fields instead of lowering them to
/// `getfield_vable_*` / `setfield_vable_*` and instead of admitting an
/// array field to `vable_array_vars`
/// (`rpython/jit/codewriter/jtransform.py`).
///
/// Upstream this hint "has to come with access_directly=True"
/// (`rlib/jit.py`, asserted at `rlib/jit.py`), because there
/// the two flags ride one `SomeInstance` and `access_directly` is what
/// suppresses the `jit_force_virtualizable` call
/// (`rpython/rtyper/rvirtualizable.py`).  Pyre dispatches one
/// helper per kwarg and has no `hook_access_field`, so this helper alone
/// carries the whole suppression; pair it with [`hint_access_directly`]
/// to keep the call sites readable as the upstream pair.
///
/// Because of that, placement matters here in a way it does not upstream.
/// The suppression is recorded per basic block, and upstream re-establishes
/// it in front of every redirected access, so its one `hint()` covers the
/// whole function.  This helper covers only the block it sits in.  Accesses
/// on the far side of a branch — an early return, a loop body, an assertion
/// behind a null test — are not covered, and each needs its own call.
///
/// rlib/jit.py — `hint(x, fresh_virtualizable=True)`
#[inline(always)]
pub fn hint_fresh_virtualizable<T>(x: T) -> T {
    hint(x)
}

/// Force a virtualizable before it escapes the portal while remaining live.
///
/// This is the Rust spelling of `hint(x, force_virtualizable=True)`.  The
/// interpreter observes the identity function; `jtransform.rewrite_op_hint`
/// turns it into `hint_force_virtualizable/r` in looked-inside JIT graphs.
#[inline(always)]
pub fn hint_force_virtualizable<T>(x: T) -> T {
    hint(x)
}

// ── promote ──
// rlib/jit.py

/// Promotes a variable in a trace to a constant.
///
/// When a variable is promoted, a guard is inserted that assumes the value
/// of the variable is constant. In other words, the value of the variable
/// is checked to be the same as it was at trace collection time.  Once the
/// variable is assumed constant, more aggressive constant folding may be
/// possible.
///
/// If however, the guard fails frequently, a bridge will be generated
/// this time assuming the constancy of the variable under its new value.
/// This optimisation should be used carefully, as in extreme cases, where
/// the promoted variable is not very constant at all, code explosion can
/// occur. In turn this leads to poor performance.
///
/// Overpromotion is characterised by a cascade of bridges branching from
/// very similar guard_value opcodes, each guarding the same variable under
/// a different value.
///
/// Note that promoting a string with `promote` will promote by pointer.
/// To promote a string by value, see `promote_string`.
///
/// rlib/jit.py — `promote(x) = hint(x, promote=True)`
#[inline(always)]
pub fn promote<T: Copy>(x: T) -> T {
    hint(x)
}

// ── promote_string ──
// rlib/jit.py

/// Promote a string by value (not by pointer).
///
/// rlib/jit.py — `promote_string(x) = hint(x, promote_string=True)`
#[inline(always)]
pub fn promote_string<T: Copy>(x: T) -> T {
    hint(x)
}

// ── promote_unicode ──
// rlib/jit.py

/// Promote a unicode string by value (not by pointer).
///
/// rlib/jit.py — `promote_unicode(x) = hint(x, promote_unicode=True)`
#[inline(always)]
pub fn promote_unicode<T: Copy>(x: T) -> T {
    hint(x)
}

// ── dont_look_inside, look_inside, unroll_safe, loop_invariant ──
// rlib/jit.py, 142, 151, 162
//
// These are decorators in RPython, implemented as proc macro attributes
// in majit-macros: #[dont_look_inside], #[unroll_safe], #[loop_invariant].
// `look_inside` is deprecated in RPython; no majit equivalent needed.

// ── _get_args ──
// rlib/jit.py
//
// In RPython `_get_args(func)` extracts parameter names for elidable_promote
// and look_inside_iff code generation. In majit, both proc macros collect
// parameters from `ItemFn::sig.inputs` inline; no separate helper needed.

// ── elidable_promote, purefunction_promote, look_inside_iff, oopspec, not_in_trace ──
// rlib/jit.py, 203, 208, 250, 260 — all proc macros in majit-macros.

// ── isconstant ──
// rlib/jit.py

/// While tracing, returns whether or not the value is currently known to be
/// constant. This is not perfect, values can become constant later. Mostly for
/// use with look_inside_iff.
///
/// This is for advanced usage only.
///
/// rlib/jit.py
#[inline(always)]
pub fn isconstant<T: ?Sized>(_value: &T) -> bool {
    false
}

// ── isvirtual ──
// rlib/jit.py

/// Returns if this value is virtual, while tracing. Can be wrong in both
/// directions. It tries to be conservative by default, but can also sometimes
/// return True for something that does not end up completely virtual.
///
/// This is for advanced usage only.
///
/// rlib/jit.py
#[inline(always)]
pub fn isvirtual<T: ?Sized>(_value: &T) -> bool {
    false
}

// ── loop_unrolling_heuristic ──
// rlib/jit.py

/// In which cases iterating over items of lst can be unrolled.
///
/// rlib/jit.py loop_unrolling_heuristic
#[inline(always)]
pub fn loop_unrolling_heuristic<T>(lst: &[T], size: usize, cutoff: usize) -> bool {
    // rlib/jit.py — `size == 0 or (isconstant(size) and (isvirtual(lst) or size <= cutoff))`
    // `isvirtual(lst)` is often lying; also require size to be constant.
    size == 0 || (isconstant(&size) && (isvirtual(lst) || size <= cutoff))
}

// ── we_are_jitted ──
// rlib/jit.py

/// Considered as true during tracing and blackholing,
/// so its consequences are reflected into jitted code.
///
/// rlib/jit.py:355. `inline(never)` + `dont_look_inside` so the
/// call survives to `fold_we_are_jitted_calls` and the hook body
/// cannot fold to const false.
#[inline(never)]
#[majit_macros::dont_look_inside]
pub fn we_are_jitted() -> bool {
    majit_backend::we_are_jitted()
}

// ── _we_are_jitted ──
// rlib/jit.py

/// rlib/jit.py — `_we_are_jitted = CDefinedIntSymbolic('0 /* we are not jitted here */', default=0)`
///
/// A C-defined symbolic integer representing the JIT runtime flag at translation
/// time. In majit this is a runtime thread-local flag exposed via `we_are_jitted()`.
#[allow(non_upper_case_globals)]
pub const _we_are_jitted: i32 = 0;

// ── _get_virtualizable_token ──
// rlib/jit.py

/// An obscure API to get vable token. Used by _vmprof.
///
/// rlib/jit.py _get_virtualizable_token — returns null GCREF pointer for untranslated/non-virtualizable frames.
#[inline(always)]
pub fn _get_virtualizable_token<T>(_frame: &T) -> *mut () {
    std::ptr::null_mut()
}

// ── current_trace_length ──
// rlib/jit.py

/// During JIT tracing, returns the current trace length (as a constant).
/// If not tracing, returns -1.
///
/// rlib/jit.py current_trace_length
#[inline(always)]
pub fn current_trace_length() -> i64 {
    -1
}

// ── jit_debug ──
// rlib/jit.py

/// When JITted, cause an extra operation JIT_DEBUG to appear in
/// the graphs.  Should not be left after debugging.
///
/// rlib/jit.py jit_debug
#[inline(always)]
pub fn jit_debug(_string: &str, _arg1: i64, _arg2: i64, _arg3: i64, _arg4: i64) {}

// ── assert_green ──
// rlib/jit.py

/// Very strong assert: checks that 'value' is a green
/// (a JIT compile-time constant).
///
/// rlib/jit.py assert_green
#[inline(always)]
pub fn assert_green<T>(_value: &T) {}

// ── AssertGreenFailed ──
// rlib/jit.py

/// rlib/jit.py AssertGreenFailed
#[derive(Debug)]
pub struct AssertGreenFailed;

impl std::fmt::Display for AssertGreenFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AssertGreenFailed")
    }
}

impl std::error::Error for AssertGreenFailed {}

// ── jit_callback ──
// rlib/jit.py

/// Use as a decorator for C callback functions, to insert a
/// jitdriver.jit_merge_point() at the start.  Only for callbacks
/// that typically invoke more app-level Python code.
///
/// rlib/jit.py
///
/// In majit, this concept maps to `#[jit_interp]` on the callback wrapper.
/// No standalone runtime helper is required; left as documentation for parity.
pub const fn jit_callback(_name: &'static str) {}

// ── virtual_ref ──
// rlib/jit.py

/// Creates a 'vref' object that contains a reference to 'x'.  Calls
/// to virtual_ref/virtual_ref_finish must be properly nested.  The idea
/// is that the object 'x' is supposed to be JITted as a virtual between
/// the calls to virtual_ref and virtual_ref_finish, but the 'vref'
/// object can escape at any point in time.  If at runtime it is
/// dereferenced, it returns 'x', which is then forced.
///
/// rlib/jit.py
#[inline(always)]
pub fn virtual_ref<T>(x: &T) -> DirectJitVRef<T> {
    // rlib/jit.py — DirectJitVRef.__init__: assert x is not None
    // In Rust a `&T` is never null, so this assertion is statically satisfied.
    DirectJitVRef {
        inner: DirectVRef {
            _x: x as *const T,
            _state: VRefState::NonForced,
        },
    }
}

// ── virtual_ref_finish ──
// rlib/jit.py

/// See docstring in virtual_ref(x).
///
/// rlib/jit.py
#[inline(always)]
pub fn virtual_ref_finish<T>(vref: &mut DirectJitVRef<T>, x: &T) {
    // rlib/jit.py:479 — keepalive_until_here(x); otherwise the whole function
    // call is removed. In Rust this is implicit via lifetime analysis.
    _virtual_ref_finish(vref, x);
}

// ── non_virtual_ref ──
// rlib/jit.py

/// Creates a 'vref' that just returns x when called; nothing more special.
/// Used for None or for frames outside JIT scope.
///
/// rlib/jit.py non_virtual_ref
#[inline(always)]
pub fn non_virtual_ref<T>(x: &T) -> DirectVRef<T> {
    DirectVRef {
        _x: x as *const T,
        _state: VRefState::NonForced,
    }
}

// ── InvalidVirtualRef ──
// rlib/jit.py

/// Raised if we try to call a non-forced virtualref after the call to
/// virtual_ref_finish.
///
/// rlib/jit.py
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidVirtualRef;

impl std::fmt::Display for InvalidVirtualRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InvalidVirtualRef")
    }
}

impl std::error::Error for InvalidVirtualRef {}

// rlib/jit.py:493

// ── DirectVRef ──
// rlib/jit.py

/// rlib/jit.py:498 — _state field
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VRefState {
    /// rlib/jit.py:498 — 'non-forced'
    NonForced,
    /// rlib/jit.py:502 — 'forced'
    Forced,
    /// rlib/jit.py:515 — 'invalid'
    Invalid,
}

/// rlib/jit.py — `class DirectVRef(object)`
#[derive(Debug)]
pub struct DirectVRef<T> {
    pub(crate) _x: *const T,
    pub(crate) _state: VRefState,
}

impl<T> DirectVRef<T> {
    /// rlib/jit.py — `def __call__(self):`
    pub fn force(&mut self) -> Result<&T, InvalidVirtualRef> {
        match self._state {
            // rlib/jit.py:501-502
            VRefState::NonForced => {
                self._state = VRefState::Forced;
                Ok(unsafe { &*self._x })
            }
            // rlib/jit.py:503-504
            VRefState::Invalid => Err(InvalidVirtualRef),
            VRefState::Forced => Ok(unsafe { &*self._x }),
        }
    }

    /// rlib/jit.py — `@property def virtual(self):`
    pub fn virtual_(&self) -> bool {
        self._state == VRefState::NonForced
    }

    /// rlib/jit.py — `def _finish(self):`
    pub(crate) fn _finish(&mut self) {
        if self._state == VRefState::NonForced {
            self._state = VRefState::Invalid;
        }
    }
}

// ── DirectJitVRef ──
// rlib/jit.py

/// rlib/jit.py — `class DirectJitVRef(DirectVRef)`
///
/// Rust has no inheritance; modelled as a newtype wrapping `DirectVRef<T>`.
/// Access to the base class fields goes through `self.inner`.
#[derive(Debug)]
pub struct DirectJitVRef<T> {
    pub(crate) inner: DirectVRef<T>,
}

impl<T> DirectJitVRef<T> {
    /// rlib/jit.py — delegate to DirectVRef.__call__
    pub fn force(&mut self) -> Result<&T, InvalidVirtualRef> {
        self.inner.force()
    }

    /// rlib/jit.py — delegate to DirectVRef.virtual
    pub fn virtual_(&self) -> bool {
        self.inner.virtual_()
    }
}

// ── _virtual_ref_finish ──
// rlib/jit.py

/// rlib/jit.py — `def _virtual_ref_finish(vref, x):`
///
/// Private helper that asserts identity and calls `_finish()`.
#[inline(always)]
fn _virtual_ref_finish<T>(vref: &mut DirectJitVRef<T>, x: &T) {
    // rlib/jit.py:523 — assert vref._x is x
    debug_assert!(
        std::ptr::eq(vref.inner._x, x),
        "Invalid call to virtual_ref_finish"
    );
    // rlib/jit.py — vref._finish()
    vref.inner._finish();
}

// ── vref_None ──
// rlib/jit.py

/// Pre-made vref for null/absent frames.
///
/// rlib/jit.py — `vref_None = non_virtual_ref(None)`
pub const VREF_NONE: DirectVRef<()> = DirectVRef {
    _x: std::ptr::null(),
    _state: VRefState::NonForced,
};

// ── JitHintError ──
// rlib/jit.py

/// Inconsistency in the JIT hints.
///
/// rlib/jit.py JitHintError
#[derive(Debug)]
pub struct JitHintError(pub String);

impl std::fmt::Display for JitHintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JitHintError: {}", self.0)
    }
}

impl std::error::Error for JitHintError {}

// ── ENABLE_ALL_OPTS ──
// rlib/jit.py

/// rlib/jit.py:562
pub const ENABLE_ALL_OPTS: &str = "intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll";

// ── PARAMETER_DOCS ──
// rlib/jit.py

/// Documentation strings for each JIT parameter.
///
/// rlib/jit.py — `PARAMETER_DOCS` dict
pub fn parameter_doc(name: &str) -> &'static str {
    match name {
        "threshold" => "number of times a loop has to run for it to become hot",
        "function_threshold" => {
            "number of times a function must run for it to become traced from start"
        }
        "trace_eagerness" => {
            "number of times a guard has to fail before we start compiling a bridge"
        }
        "decay" => "amount to regularly decay counters by (0=none, 1000=max)",
        "trace_limit" => {
            "number of recorded operations before we abort tracing with ABORT_TOO_LONG"
        }
        "inlining" => "inline functions or not (1/0)",
        "loop_longevity" => {
            "a parameter controlling how long loops will be kept before being freed, an estimate"
        }
        "retrace_limit" => "how many times we can try retracing before giving up",
        "pureop_historylength" => {
            "how many pure operations the optimizer should remember for CSE (internal)"
        }
        "max_retrace_guards" => "number of extra guards a retrace can cause",
        "max_unroll_loops" => "number of extra unrollings a loop can cause",
        "disable_unrolling" => "after how many operations we should not unroll",
        "enable_opts" => "optimizations to enable, or all",
        "max_unroll_recursion" => "how many levels deep to unroll a recursive function",
        "vec" => "turn on the vectorization optimization (vecopt)",
        "vec_cost" => {
            "threshold for which traces to bail. Unpacking increases the counter, \
             vector operation decrease the cost"
        }
        "vec_all" => "try to vectorize trace loops that occur outside of the numpypy library",
        _ => "(unknown parameter)",
    }
}

// ── PARAMETERS ──
// rlib/jit.py

/// Default JIT parameters.
///
/// rlib/jit.py:588
pub const PARAMETERS: JitParameters = JitParameters {
    threshold: 1039,          // just above 1024, prime
    function_threshold: 1619, // slightly more than one above, also prime
    trace_eagerness: 200,
    decay: 40,
    trace_limit: 6000,
    inlining: true,
    loop_longevity: 1000,
    retrace_limit: 0,
    pureop_historylength: 16,
    max_retrace_guards: 15,
    max_unroll_loops: 0,
    disable_unrolling: 200,
    enable_opts: EnableOpts::All,
    max_unroll_recursion: 7,
    vec: false,
    vec_all: false,
    vec_cost: 0,
};

/// rlib/jit.py — PARAMETERS dict
#[derive(Debug, Clone)]
pub struct JitParameters {
    pub threshold: u32,
    pub function_threshold: u32,
    pub trace_eagerness: u32,
    pub decay: u32,
    pub trace_limit: u32,
    pub inlining: bool,
    pub loop_longevity: u32,
    pub retrace_limit: u32,
    pub pureop_historylength: u32,
    pub max_retrace_guards: u32,
    pub max_unroll_loops: u32,
    pub disable_unrolling: u32,
    pub enable_opts: EnableOpts,
    pub max_unroll_recursion: u32,
    pub vec: bool,
    pub vec_all: bool,
    pub vec_cost: u32,
}

/// rlib/jit.py:562,600
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnableOpts {
    All,
    Custom(String),
}

// ── unroll_parameters ──
// rlib/jit.py

/// rlib/jit.py — `unroll_parameters = unrolling_iterable(PARAMETERS.items())`
///
/// The canonical list of (name, default_value_as_i64) pairs used by
/// `set_user_param` to iterate over all parameters in a fixed order.
pub const UNROLL_PARAMETERS: &[(&str, i64)] = &[
    ("threshold", 1039),
    ("function_threshold", 1619),
    ("trace_eagerness", 200),
    ("decay", 40),
    ("trace_limit", 6000),
    ("inlining", 1),
    ("loop_longevity", 1000),
    ("retrace_limit", 0),
    ("pureop_historylength", 16),
    ("max_retrace_guards", 15),
    ("max_unroll_loops", 0),
    ("disable_unrolling", 200),
    ("max_unroll_recursion", 7),
    ("vec", 0),
    ("vec_all", 0),
    ("vec_cost", 0),
];

// ── JitDriver ──
// rlib/jit.py+
//
// Implemented in `majit-metainterp/src/jitdriver.rs` as `JitDriver<State>`.

// ── _set_param ──
// rlib/jit.py

/// rlib/jit.py — `def _set_param(driver, name, value)`
///
/// Private dispatch to the driver-level parameter setter.
fn _set_param(params: &mut JitParameters, name: &str, value: i64) -> Result<(), JitHintError> {
    match name {
        "threshold" => params.threshold = value as u32,
        "function_threshold" => params.function_threshold = value as u32,
        "trace_eagerness" => params.trace_eagerness = value as u32,
        "decay" => params.decay = value as u32,
        "trace_limit" => params.trace_limit = value as u32,
        "inlining" => params.inlining = value != 0,
        "loop_longevity" => params.loop_longevity = value as u32,
        "retrace_limit" => params.retrace_limit = value as u32,
        "pureop_historylength" => params.pureop_historylength = value as u32,
        "max_retrace_guards" => params.max_retrace_guards = value as u32,
        "max_unroll_loops" => params.max_unroll_loops = value as u32,
        "disable_unrolling" => params.disable_unrolling = value as u32,
        "max_unroll_recursion" => params.max_unroll_recursion = value as u32,
        "vec" => params.vec = value != 0,
        "vec_all" => params.vec_all = value != 0,
        "vec_cost" => params.vec_cost = value as u32,
        _ => return Err(JitHintError(format!("unknown JIT parameter: {name}"))),
    }
    Ok(())
}

// ── set_param ──
// rlib/jit.py

/// Set a JIT parameter by name.
///
/// rlib/jit.py set_param — raises ValueError on unknown name.
pub fn set_param(params: &mut JitParameters, name: &str, value: i64) -> Result<(), JitHintError> {
    _set_param(params, name, value)
}

// ── set_param_to_default ──
// rlib/jit.py

/// Set a JIT parameter to its default value.
///
/// rlib/jit.py set_param_to_default
pub fn set_param_to_default(params: &mut JitParameters, name: &str) -> Result<(), JitHintError> {
    for (n, default) in UNROLL_PARAMETERS {
        if *n == name {
            return _set_param(params, name, *default);
        }
    }
    Err(JitHintError(format!("unknown JIT parameter: {name}")))
}

// ── TraceLimitTooHigh ──
// rlib/jit.py

/// This is raised when the trace limit is too high for the chosen
/// opencoder model, recompile your interpreter with 'big' as
/// jit_opencoder_model.
///
/// rlib/jit.py TraceLimitTooHigh
#[derive(Debug)]
pub struct TraceLimitTooHigh;

impl std::fmt::Display for TraceLimitTooHigh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TraceLimitTooHigh")
    }
}

impl std::error::Error for TraceLimitTooHigh {}

// ── set_user_param ──
// rlib/jit.py

/// The driver argument of `rlib/jit.py set_user_param`, specialized to the
/// parameter storage or live driver receiving the calls. Rust represents the
/// upstream integer/string parameter overload with two setters.
pub trait JitParameterTarget {
    fn set_param(&mut self, name: &str, value: i64) -> Result<(), JitHintError>;
    fn set_param_enable_opts(&mut self, value: &str);
}

impl JitParameterTarget for JitParameters {
    fn set_param(&mut self, name: &str, value: i64) -> Result<(), JitHintError> {
        set_param(self, name, value)
    }

    fn set_param_enable_opts(&mut self, value: &str) {
        self.enable_opts = if value == "all" {
            EnableOpts::All
        } else {
            EnableOpts::Custom(value.to_string())
        };
    }
}

/// Set the tunable JIT parameters from a user-supplied string
/// following the format 'param=value,param=value', or 'off' to
/// disable the JIT.  For programmatic setting of parameters, use
/// directly `set_param`.
///
/// rlib/jit.py
pub fn set_user_param(
    driver: &mut impl JitParameterTarget,
    text: &str,
) -> Result<(), JitHintError> {
    // rlib/jit.py:842-845
    if text == "off" {
        driver.set_param("threshold", -1)?;
        driver.set_param("function_threshold", -1)?;
        return Ok(());
    }
    // rlib/jit.py:846-849
    if text == "default" {
        for (name1, default) in UNROLL_PARAMETERS {
            driver.set_param(name1, *default)?;
            // PARAMETERS places the string-valued entry after this integer.
            if *name1 == "disable_unrolling" {
                driver.set_param_enable_opts("all");
            }
        }
        return Ok(());
    }
    // rlib/jit.py:850-874
    for s in text.split(',') {
        let s = s.trim_matches(' ');
        let parts: Vec<&str> = s.split('=').collect();
        if parts.len() != 2 {
            return Err(JitHintError(format!("malformed token: {s}")));
        }
        let name = parts[0];
        let value = parts[1];
        if name == "enable_opts" {
            driver.set_param_enable_opts(value);
            continue;
        }
        // rlib/jit.py — for name1, _ in unroll_parameters:
        let mut found = false;
        for (name1, _) in UNROLL_PARAMETERS {
            if *name1 == name && *name1 != "enable_opts" {
                let ivalue: i64 = value
                    .trim()
                    .parse()
                    .map_err(|_| JitHintError(format!("invalid value for {name}: {value}")))?;
                driver.set_param(name1, ivalue)?;
                found = true;
                break;
            }
        }
        if !found {
            return Err(JitHintError(format!("unknown JIT parameter: {name}")));
        }
    }
    Ok(())
}

// ── AsmInfo, JitDebugInfo, JitHookInterface ──
// rlib/jit.py, 1077, 1113
//
// `AsmInfo` is implemented in `majit-backend/src/lib.rs`. `JitDebugInfo` and
// `JitHookInterface` are debug-introspection interfaces that are not yet
// wired to user code in majit (hooks fire through compile/bridge paths).

// ── record_exact_class ──
// rlib/jit.py

/// Assure the JIT that value is an instance of cls. This is a precise
/// class check, like a guard_class.
///
/// See also `assert_not_none(x)`, which asserts that x is not None
/// and also assures the JIT that it is the case.
///
/// rlib/jit.py — `assert type(value) is cls`. RPython
/// `ll_record_exact_class(ll_value, ll_cls)` (rlib/jit.py) asserts
/// `ll_type(ll_value) is ll_cls`.  The blackhole bytecode signature is
/// `record_exact_class/ri` (`blackhole.py:616`): the instance is a ref
/// and the class pointer travels through the int register bank.
///
/// In Rust, `cls` is the raw class/vtable sentinel address as an integer.
/// Object references are checked by reading the first pointer-sized word,
/// matching RPython's low-level object layout where `ll_type(ll_value)`
/// reads `OBJECT.typeptr`.  Raw vtable callers can pass the vtable word
/// directly; that mirrors the pre-lowered `type(value) is cls` check.
pub trait RecordExactClassValue {
    fn record_exact_class_typeptr(self) -> usize;
}

impl RecordExactClassValue for usize {
    #[inline(always)]
    fn record_exact_class_typeptr(self) -> usize {
        self
    }
}

impl RecordExactClassValue for i64 {
    #[inline(always)]
    fn record_exact_class_typeptr(self) -> usize {
        self as usize
    }
}

impl<T: ?Sized> RecordExactClassValue for &T {
    #[inline(always)]
    fn record_exact_class_typeptr(self) -> usize {
        assert!(
            std::mem::size_of_val(self) >= std::mem::size_of::<usize>(),
            "record_exact_class called with value too small to carry typeptr"
        );
        let data = self as *const T as *const () as *const usize;
        // RPython rclass.ll_type(obj) reads the typeptr field from the
        // object header.  Pyre object layouts used with this hint put the
        // class/vtable word at offset zero.
        unsafe { *data }
    }
}

impl<T> RecordExactClassValue for *const T {
    #[inline(always)]
    fn record_exact_class_typeptr(self) -> usize {
        assert!(
            !self.is_null(),
            "record_exact_class called with None argument"
        );
        unsafe { *(self as *const usize) }
    }
}

impl<T> RecordExactClassValue for *mut T {
    #[inline(always)]
    fn record_exact_class_typeptr(self) -> usize {
        (self as *const T).record_exact_class_typeptr()
    }
}

#[inline(always)]
pub fn record_exact_class<V: RecordExactClassValue>(value: V, cls: usize) {
    assert_eq!(
        value.record_exact_class_typeptr(),
        cls,
        "record_exact_class called with invalid arguments"
    );
}

// ── assert_not_none ──
// rtyper/debug.py ll_assert_not_none

/// Assure the JIT that value is not None.
///
/// rtyper/debug.py ll_assert_not_none — `assert x is not None; return x`
///
/// In Rust there is no Python `None` per se; the `Option<&T>` / `Option<Box<T>>`
/// shape is the closest analog, and `Option::expect("ll_assert_not_none")`
/// preserves the upstream assertion semantics. The standalone identity
/// form (`x.assert_not_none()` on already-non-None Rust values) is a
/// no-op at runtime; the JIT consumer recognises calls and emits an
/// ASSERT_NOT_NONE op via [`crate::TraceCtx::trace_assert_not_none`].
#[inline(always)]
pub fn assert_not_none<T>(x: Option<T>) -> T {
    x.expect("ll_assert_not_none")
}

// ── _jit_record_known_result ──
// rlib/jit.py

/// Marker, special-cased by jtransform during JIT compilation.
///
/// rlib/jit.py — `def _jit_record_known_result(result, function, *args): pass`
///
/// At runtime this is a no-op. During JIT tracing, the codewriter
/// converts calls to this function into a `RecordKnownResult` operation
/// that pairs an elidable call with its known result for constant folding.
///
/// Called from the `record_known_result!` macro.
#[inline(always)]
pub fn _jit_record_known_result<T>(_known_result: T, _func_result: T) {
    // No-op at runtime; the codewriter emits RecordKnownResult.
}

// Implemented as a variadic macro in `majit-metainterp/src/lib.rs` since
// Rust lacks `*args`; see `record_known_result!`.

// ── record_known_result ──
// rlib/jit.py
//
// RPython: `record_known_result(result, func, *args)`
// Rust doesn't have *args; use the `record_known_result!` macro in lib.rs
// which accepts any arity via variadic macro pattern.

// ── conditional_call ──
// rlib/jit.py
//
// RPython: `conditional_call(condition, function, *args)`
//   → rtyper `jit_conditional_call(cond, funcptr, arg1, arg2, ...)` llop
//   → jtransform `conditional_call_ir_v` JitCode bytecode
//
// Rust: `conditional_call!(condition, func_path, arg1, arg2, ...)`
//   → jitcode_lower intercepts macro → `__builder.conditional_call_ir_v_typed_args`
//   → `conditional_call_ir_v/iiIRd` JitCode bytecode
//
// Use the `conditional_call!` macro in `#[jit_interp]` functions for the
// JIT path. The plain function below is the non-JIT runtime helper —
// rlib/jit.py `conditional_call` falls back to `if condition:
// function(*args)` outside JIT.

/// Does the same as:
///
/// ```text
/// if condition:
///     function(*args)
/// ```
///
/// rlib/jit.py conditional_call — non-JIT runtime helper.
///
/// For JIT-compiled `#[jit_interp]` functions, use `conditional_call!` macro
/// which takes `(condition, func_path, args...)` matching RPython's
/// `(condition, function, *args)`.
#[inline(always)]
pub fn conditional_call<F: FnOnce()>(condition: bool, function: F) {
    if condition {
        function()
    }
}

// ── conditional_call_elidable ──
// rlib/jit.py
//
// Same as conditional_call: use `conditional_call_elidable!` macro for JIT.
// The closure-based function below is a non-JIT runtime helper only.

// ── JitCondFalsy ──
// rlib/jit.py:1350-1357 — isinstance(value, int) and value == 0, else not value.

/// Trait expressing "falsy for `conditional_call_elidable`".
///
/// rlib/jit.py:
/// - If `value` is int: falsy iff `value == 0`
/// - Else (pointer/list/object): falsy iff `not value`
pub trait JitCondFalsy {
    fn jit_cond_falsy(&self) -> bool;
}

macro_rules! impl_jit_cond_falsy_int {
    ($($t:ty),*) => {
        $(
            impl JitCondFalsy for $t {
                #[inline(always)]
                fn jit_cond_falsy(&self) -> bool { *self == 0 }
            }
        )*
    };
}

impl_jit_cond_falsy_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl<T: ?Sized> JitCondFalsy for *const T {
    #[inline(always)]
    fn jit_cond_falsy(&self) -> bool {
        self.is_null()
    }
}

impl<T: ?Sized> JitCondFalsy for *mut T {
    #[inline(always)]
    fn jit_cond_falsy(&self) -> bool {
        self.is_null()
    }
}

impl<T> JitCondFalsy for Option<T> {
    #[inline(always)]
    fn jit_cond_falsy(&self) -> bool {
        self.is_none()
    }
}

// ── conditional_call_elidable ──
// rlib/jit.py

/// Does the same as:
///
/// ```text
/// if value == <0 or None or NULL>:
///     value = function(*args)
/// return value
/// ```
///
/// For the JIT.  Allows one branch which doesn't create a bridge,
/// typically used for caching.  The value and the function's return
/// type must match and cannot be a float: they must be either regular
/// 'int', or something that turns into a pointer.
///
/// Even if the function is not marked @elidable, it is still treated
/// mostly like one.  The only difference is that (in heapcache.py)
/// we don't assume this function won't change anything observable.
/// This is useful for caches.
///
/// rlib/jit.py — non-JIT runtime helper.
///
/// For JIT-compiled `#[jit_interp]` functions, use `conditional_call_elidable!` macro
/// which takes `(value, func_path, args...)` matching RPython's `(value, function, *args)`.
#[inline(always)]
pub fn conditional_call_elidable<T: JitCondFalsy, F: FnOnce() -> T>(value: T, function: F) -> T {
    if value.jit_cond_falsy() {
        function()
    } else {
        value
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promote_int() {
        assert_eq!(promote(42), 42);
    }

    #[test]
    fn test_promote_ref_copy() {
        let x: usize = 0xdead;
        assert_eq!(promote(x), 0xdead);
    }

    #[test]
    fn test_we_are_jitted_default() {
        assert!(!we_are_jitted());
    }

    #[test]
    fn test_isconstant_default() {
        assert!(!isconstant(&42));
    }

    #[test]
    fn test_isvirtual_default() {
        let v = vec![1, 2, 3];
        assert!(!isvirtual(&v));
    }

    #[test]
    fn test_current_trace_length_default() {
        assert_eq!(current_trace_length(), -1);
    }

    #[test]
    fn test_virtual_ref_lifecycle() {
        let val = 42i64;
        let mut vref = virtual_ref(&val);
        assert!(vref.virtual_());
        assert_eq!(*vref.force().unwrap(), 42);
        assert!(!vref.virtual_());
    }

    #[test]
    fn test_virtual_ref_finish_invalidates() {
        let val = 42i64;
        let mut vref = virtual_ref(&val);
        virtual_ref_finish(&mut vref, &val);
        assert!(vref.force().is_err());
    }

    #[test]
    fn record_exact_class_accepts_matching_typeptr_header() {
        #[repr(C)]
        struct Obj {
            typeptr: usize,
            payload: i64,
        }

        let obj = Obj {
            typeptr: 0xCAFE_BABE,
            payload: 7,
        };
        record_exact_class(&obj, 0xCAFE_BABE);
    }

    #[test]
    #[should_panic(expected = "record_exact_class called with invalid arguments")]
    fn record_exact_class_rejects_mismatched_typeptr_header() {
        #[repr(C)]
        struct Obj {
            typeptr: usize,
        }

        let obj = Obj { typeptr: 1 };
        record_exact_class(&obj, 2);
    }

    #[test]
    #[should_panic(expected = "record_exact_class called with None argument")]
    fn record_exact_class_rejects_null_pointer() {
        let ptr: *const usize = std::ptr::null();
        record_exact_class(ptr, 1);
    }

    #[test]
    fn test_parameters_default() {
        assert_eq!(PARAMETERS.threshold, 1039);
        assert_eq!(PARAMETERS.function_threshold, 1619);
        assert_eq!(PARAMETERS.trace_eagerness, 200);
        assert_eq!(PARAMETERS.trace_limit, 6000);
    }

    #[test]
    fn test_set_user_param() {
        let mut p = PARAMETERS;
        set_user_param(&mut p, "threshold=100,trace_limit=3000").unwrap();
        assert_eq!(p.threshold, 100);
        assert_eq!(p.trace_limit, 3000);
        assert_eq!(p.function_threshold, 1619);
    }

    #[test]
    fn test_set_user_param_off() {
        let mut p = PARAMETERS;
        set_user_param(&mut p, "off").unwrap();
        assert_eq!(p.threshold, u32::MAX);
        assert_eq!(p.function_threshold, u32::MAX);
    }

    #[test]
    fn test_set_user_param_default() {
        let mut p = PARAMETERS;
        p.threshold = 999;
        p.enable_opts = EnableOpts::Custom(String::new());
        p.vec_all = true;
        set_user_param(&mut p, "default").unwrap();
        assert_eq!(p.threshold, 1039);
        assert_eq!(p.enable_opts, EnableOpts::All);
        assert!(!p.vec_all);
    }

    #[test]
    fn test_set_user_param_oracle_grammar_and_partial_application() {
        // Measured with pypy3 pypyjit.set_param; rlib/jit.py set_user_param.
        for text in [
            "",
            "threshold=1,",
            "threshold =1",
            "enable_opts=a=b",
            "vectorize=1",
            "max_inline_depth=7",
        ] {
            let mut params = PARAMETERS;
            assert!(set_user_param(&mut params, text).is_err(), "{text:?}");
        }
        let mut p = PARAMETERS;
        set_user_param(&mut p, " threshold= 17 ,enable_opts=,vec=1").unwrap();
        assert_eq!(p.threshold, 17);
        assert_eq!(p.enable_opts, EnableOpts::Custom(String::new()));
        assert!(p.vec);
        assert!(set_user_param(&mut p, "threshold=23,bad=1").is_err());
        assert_eq!(p.threshold, 23);
    }

    #[test]
    fn test_set_user_param_live_warmstate() {
        let mut ws = crate::warmstate::WarmEnterState::new(7);
        set_user_param(&mut ws, "enable_opts=,vec_all=1").unwrap();
        assert!(ws.get_enable_opts().is_empty());
        set_user_param(&mut ws, "off").unwrap();
        assert_eq!(ws.threshold(), 0);
        set_user_param(&mut ws, "default").unwrap();
        assert_eq!(ws.threshold(), PARAMETERS.threshold);
        assert!(!ws.get_enable_opts().is_empty());
        assert!(!ws.vec_all());
    }

    #[test]
    fn test_set_param_unknown_errors() {
        let mut p = PARAMETERS;
        assert!(set_param(&mut p, "nonexistent", 42).is_err());
    }

    #[test]
    fn test_set_user_param_unknown_errors() {
        let mut p = PARAMETERS;
        assert!(set_user_param(&mut p, "nonexistent=42").is_err());
    }

    #[test]
    fn test_conditional_call_elidable_int() {
        let result = conditional_call_elidable(42_i64, || 99_i64);
        assert_eq!(result, 42);
        let result = conditional_call_elidable(0_i64, || 99_i64);
        assert_eq!(result, 99);
    }

    #[test]
    fn test_conditional_call_elidable_ptr() {
        let x: i64 = 42;
        let non_null: *const i64 = &x;
        let result = conditional_call_elidable(non_null, std::ptr::null::<i64>);
        assert_eq!(result, non_null);

        let null: *const i64 = std::ptr::null();
        let fallback: i64 = 99;
        let result = conditional_call_elidable(null, || (&fallback) as *const i64);
        assert_eq!(result, (&fallback) as *const i64);
    }

    #[test]
    fn test_conditional_call_elidable_option() {
        let some_val: Option<i64> = Some(42);
        let result = conditional_call_elidable(some_val, || Some(99_i64));
        assert_eq!(result, Some(42));
        let none_val: Option<i64> = None;
        let result = conditional_call_elidable(none_val, || Some(99_i64));
        assert_eq!(result, Some(99));
    }

    #[test]
    fn test_loop_unrolling_heuristic() {
        let lst: &[i32] = &[];
        assert!(loop_unrolling_heuristic(lst, 0, 2));
        let lst = &[1, 2];
        // rlib/jit.py — requires isconstant(size); runtime false → returns false.
        assert!(!loop_unrolling_heuristic(lst, 2, 2));
    }

    #[test]
    fn test_enable_opts_all() {
        assert_eq!(PARAMETERS.enable_opts, EnableOpts::All);
        assert_eq!(
            ENABLE_ALL_OPTS,
            "intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll"
        );
    }

    #[test]
    fn test_vref_none() {
        assert_eq!(VREF_NONE._x, std::ptr::null());
        assert_eq!(VREF_NONE._state, VRefState::NonForced);
    }

    #[test]
    fn test_trace_limit_too_high() {
        let e = TraceLimitTooHigh;
        assert_eq!(format!("{e}"), "TraceLimitTooHigh");
    }

    #[test]
    fn test_assert_green_failed() {
        let e = AssertGreenFailed;
        assert_eq!(format!("{e}"), "AssertGreenFailed");
    }
}
