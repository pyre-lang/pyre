//! _abc module — PyPy: `pypy/module/_abc/`.
//!
//! ABCMeta backing for `abc.py`.  `_abc_instancecheck` /
//! `_abc_subclasscheck` walk `__mro__` for direct inheritance and the
//! per-class `_abc_registry` list populated by `_abc_register` for
//! virtual subclasses.  Mirrors `pypy/module/_abc/app_abc.py`'s
//! `_abc_register` / `_abc_subclasscheck` flow, including its
//! positive/negative caches: without them every check that is not a direct
//! `__mro__` hit re-runs the subclass hook, the registry walk and the
//! `__subclasses__` walk, all recursively, on every single call.

use pyre_object::*;
use std::sync::atomic::{AtomicU64, Ordering};

// `abc_invalidation_counter` (`app_abc.py`): bumped by every successful
// `_abc_register` — and by nothing else — and read by `get_cache_token`.  A
// negative cache recorded before a bump no longer describes the registry, so
// `_abc_negative_cache_version` is compared against this on every check.
static INVALIDATION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Boxes holding the root slots [`root_forever`] registers, kept for the life
/// of the process.  A `Vec` reallocation would move a word the collector still
/// writes through, so the indirection is what makes the registration outlive
/// the next push.
static ROOTED_SLOTS: parking_lot::Mutex<Vec<Box<usize>>> = parking_lot::Mutex::new(Vec::new());

/// Keep an object this module will go on naming by address.
///
/// [`simple_weak_set_type`] hands out an address the caller dereferences, and
/// old-gen is mark-sweep: not moving is not staying alive.  Rebinding the
/// module attribute, or dropping the module, lets the sweep reclaim the type
/// while the stash still names it — the collector traces neither the stash nor
/// anything that would keep the object on its account.  Registering the address
/// in a stable slot is what buys survival, the same `try_gc_add_root`
/// registration `_structseq.rs` uses for each registry `cls_slot`.
///
/// Used only where the object must genuinely outlive its last Python
/// reference: the class itself, and the weak references in
/// [`SIMPLE_WEAK_SET_CONTAINS`], which retain nothing of what they name.  The
/// bodies those references identify are deliberately not rooted.
///
/// The stashes stay plain addresses rather than reads of these slots: a heap
/// type is born old-gen and does not move.
fn root_forever(obj: PyObjectRef) {
    let mut slot = Box::new(obj as usize);
    let root_slot = (&raw mut *slot) as *mut pyre_object::gc_hook::GCREF;
    unsafe { pyre_object::gc_hook::try_gc_add_root(root_slot) };
    ROOTED_SLOTS.lock().push(slot);
}

/// The app-level `SimpleWeakSet` (`app_abc.py`), stashed at module init
/// the way `weakref_type` stashes its own.  The registry and both caches are
/// instances of it, so the collection this module installs is the one
/// `_get_dump` describes and a collected member drops itself.
static SIMPLE_WEAK_SET_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

fn simple_weak_set_type() -> PyObjectRef {
    SIMPLE_WEAK_SET_TYPE
        .get()
        .expect("_abc.SimpleWeakSet must be installed at module init")
}

/// The `__contains__` `app_abc.py` defines, stashed beside its type as the four
/// words a caller can move independently: the method itself, the code object it
/// holds, and the two globals its body resolves at run time.  The class is an
/// ordinary heap type, so the method is rebindable and the function it holds is
/// mutable in place; answering natively is only sound while the body a
/// membership test would reach is still the one written in `app_abc.py`.
///
/// The globals are `ref` and `TypeError` — the constructor the probe calls and
/// the class its handler catches.  That pair is the body's complete global
/// surface, read off the compiled code object rather than off the source: its
/// `LOAD_GLOBAL` operands are exactly those two names, and it declares no free
/// variables, no cell variables and no defaults.
///
/// `ref` is imported by the module, so it is a real entry.  `TypeError` is not:
/// it resolves through the builtins fallback and the module dict holds no such
/// key, so what is stashed for it is the module entry that would *shadow*
/// builtins — `0` while there is none.  A caller who assigns
/// `__globals__["TypeError"]` moves it off `0` and the comparison declines,
/// which is the mutation this guards.  The builtins binding the name falls
/// through to is checked where it is consulted, on the handler's own arm in
/// [`simple_weak_set_contains`].
///
/// What is stored is not those four addresses but a weak reference to each,
/// [`weak_lifeline`], read back through [`installed_contains_identity`].  A
/// strong stash would answer the address question by making the object
/// immortal: rebinding `__contains__` and dropping the last reference to the
/// old function would still leave it, its code and the module dict its
/// `__globals__` names alive for the life of the process.  Weakly, a dead
/// referent reads as "the body is gone", which is the same answer the
/// comparison wanted, and a live one cannot have had its address reused.
///
/// This buys hygiene rather than a behaviour, and the difference is worth
/// stating: a `__contains__` that is replaced and dropped stays reachable here
/// and on the reference build alike — measured, still alive after seven full
/// collections — because a method-cache slot goes on naming it.  What the weak
/// stash avoids is the one retention no eviction can undo, a root that is
/// permanent by construction.
static SIMPLE_WEAK_SET_CONTAINS: std::sync::OnceLock<[usize; 4]> = std::sync::OnceLock::new();

/// A weak reference to `obj`, kept by a root of its own.
///
/// Rooting the reference does not root its referent — that is the whole point —
/// so the address stays readable while the object lives without the stash being
/// the reason it lives.
fn weak_lifeline(obj: PyObjectRef) -> Option<usize> {
    let lifeline = pyre_interpreter::module::_weakref::interp__weakref::descr__new__weakref(
        pyre_interpreter::module::_weakref::interp__weakref::weakref_type(),
        &[obj],
    )
    .ok()?;
    root_forever(lifeline);
    Some(lifeline as usize)
}

/// The identity [`SIMPLE_WEAK_SET_CONTAINS`] recorded, as addresses again.
///
/// `None` once any referent has died, because an object the class no longer
/// holds cannot be the body a membership test would reach.  A `0` entry is a
/// word that was absent at install time — the `TypeError` shadow — and stays
/// absent here rather than becoming a dead reference.
fn installed_contains_identity() -> Option<(usize, usize, usize, usize)> {
    let lifelines = SIMPLE_WEAK_SET_CONTAINS.get()?;
    let mut words = [0usize; 4];
    for (word, &lifeline) in words.iter_mut().zip(lifelines.iter()) {
        if lifeline == 0 {
            continue;
        }
        let referent = pyre_interpreter::module::_weakref::interp__weakref::dereference(
            lifeline as PyObjectRef,
        );
        if referent.is_null() {
            return None;
        }
        *word = referent as usize;
    }
    Some((words[0], words[1], words[2], words[3]))
}

/// The weak reference naming the class `TypeError` resolved to at module init,
/// as [`weak_lifeline`] records it.
static BUILTIN_TYPE_ERROR: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// What `TypeError` names in the builtins the body falls through to.
///
/// `None` while there is no reachable builtins module or it holds no such key.
/// The lookup is the fallback leg of `LOAD_GLOBAL` itself — `space.builtin`
/// under the default `honor__builtins__=False`, not the `__builtins__` entry of
/// the module dict.
fn builtin_type_error() -> Option<PyObjectRef> {
    let ec = pyre_interpreter::call::getexecutioncontext();
    if ec.is_null() {
        return None;
    }
    let w_builtin = unsafe { (*ec).get_builtin() };
    if w_builtin.is_null() || !unsafe { pyre_object::is_module(w_builtin) } {
        return None;
    }
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
    if w_dict.is_null() {
        return None;
    }
    unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(w_dict, "TypeError") }
}

fn record_builtin_type_error() {
    if let Some(installed) = builtin_type_error()
        && let Some(lifeline) = weak_lifeline(installed)
    {
        let _ = BUILTIN_TYPE_ERROR.set(lifeline);
    }
}

/// Whether `TypeError` still names the class it named when this module loaded.
///
/// `false` also when nothing was recorded, which costs a decline rather than an
/// answer taken on a binding never established.
fn type_error_names_the_installed_class() -> bool {
    let Some(&lifeline) = BUILTIN_TYPE_ERROR.get() else {
        return false;
    };
    let mut installed =
        pyre_interpreter::module::_weakref::interp__weakref::dereference(lifeline as PyObjectRef);
    if installed.is_null() {
        return false;
    }
    pyre_object::with_roots!(installed => builtin_type_error())
        .is_some_and(|now| std::ptr::eq(now, installed))
}

/// The installed `__contains__`, the code object it currently holds and its two
/// globals, as the tuple [`SIMPLE_WEAK_SET_CONTAINS`] stores.  Anything that is
/// not a function, or a body missing either global, reads as `(0, 0, 0, 0)`,
/// which no stash can equal: [`pyre_interpreter::function_get_code`] reads the `code` field
/// without checking the type, so a `__contains__` rebound to a non-function must
/// be rejected before the read rather than by comparing what the read returned.
fn simple_weak_set_contains_identity() -> (usize, usize, usize, usize) {
    const NONE: (usize, usize, usize, usize) = (0, 0, 0, 0);
    let Some(method) = (unsafe {
        pyre_interpreter::baseobjspace::lookup_in_type(simple_weak_set_type(), "__contains__")
    }) else {
        return NONE;
    };
    if !unsafe { pyre_interpreter::is_function(method) } {
        return NONE;
    }
    let globals = unsafe { pyre_interpreter::function_get_globals_obj(method) };
    if globals.is_null() {
        return NONE;
    }
    let global = |name| unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(globals, name) };
    // `ref` must be there; `TypeError` normally is not, and its absence reads
    // as `0` rather than as a failure to identify the body.
    let Some(ref_global) = global("ref") else {
        return NONE;
    };
    let type_error_shadow = global("TypeError").map_or(0, |obj| obj as usize);
    (
        method as usize,
        unsafe { pyre_interpreter::function_get_code(method) as usize },
        ref_global as usize,
        type_error_shadow,
    )
}

/// `SimpleWeakSet()` — the empty collection `_abc_init` installs and the
/// invalidation in `subclass_of` rebinds to.
fn new_simple_weak_set() -> Result<PyObjectRef, pyre_interpreter::PyError> {
    pyre_interpreter::call::call_function_impl_result(simple_weak_set_type(), &[])
}

/// `app_abc.py SimpleWeakSet.__contains__` through the membership
/// protocol, which is where the weakref probe and the `TypeError` fallback
/// live.
///
fn weak_cache_contains(
    cls: PyObjectRef,
    name: &str,
    item: PyObjectRef,
) -> Result<bool, pyre_interpreter::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[cls]);
    let item_slot = roots.publish(&[item]);
    let cache = cache_attr(roots.get(cls_slot), name)?;
    let cache_slot = roots.publish(&[cache]);
    if let Some(found) = simple_weak_set_contains(roots.get(cache_slot), roots.get(item_slot))? {
        return Ok(found);
    }
    pyre_interpreter::baseobjspace::contains(roots.get(cache_slot), roots.get(item_slot))
}

/// `app_abc.py SimpleWeakSet.__contains__` without the app-level frame,
/// for the collection `_abc_init` installed and nothing has replaced.
///
/// `None` means the receiver is not that collection — a rebound `_abc_cache`, or
/// one whose `data` is no longer a set — so the caller takes the membership
/// protocol and whatever the replacement spells.
///
/// `_abc_instancecheck` runs this once on a hit and twice on a miss, and the
/// frame it replaces is most of what the check costs: the body is three
/// operations, and entering Python to run them is the rest.
fn simple_weak_set_contains(
    mut cache: PyObjectRef,
    mut item: PyObjectRef,
) -> Result<Option<bool>, pyre_interpreter::PyError> {
    if !pyre_interpreter::typedef::r#type(cache)
        .is_some_and(|actual| std::ptr::eq(actual.as_ptr(), simple_weak_set_type()))
    {
        return Ok(None);
    }
    // The receiver being that class is not enough: rebinding the class's own
    // `__contains__`, assigning that function a different `__code__`, or
    // rebinding either global its body resolves — the `ref` it calls, the
    // `TypeError` it catches — each leaves every instance an exact
    // `SimpleWeakSet` while changing what a membership test answers.
    let Some(installed) = pyre_object::with_roots!(cache, item => installed_contains_identity())
    else {
        return Ok(None);
    };
    if simple_weak_set_contains_identity() != installed {
        return Ok(None);
    }
    let roots = pyre_object::gc_roots::push_roots();
    let cache_slot = roots.publish(&[cache]);
    let item_slot = roots.publish(&[item]);
    let data = pyre_interpreter::baseobjspace::getattr_str(roots.get(cache_slot), "data")?;
    // Exactly a `set`. A subclass carries `SET_USER_TYPE` (`typedef.py`
    // `_getusercls`) and `contains` dispatches its `__contains__` override
    // through `subclass_special_override`. Reading the table directly would
    // answer past the override.
    if !unsafe { pyre_object::is_exact_type(data, &pyre_object::setobject::SET_TYPE) } {
        return Ok(None);
    }
    let data_slot = roots.publish(&[data]);
    // `wr = ref(item)`.  The app-level spelling answers False for an item that
    // cannot carry a weakref and lets every other error out, so match both.
    let probe = match pyre_interpreter::module::_weakref::interp__weakref::descr__new__weakref(
        pyre_interpreter::module::_weakref::interp__weakref::weakref_type(),
        &[roots.get(item_slot)],
    ) {
        Ok(probe) => probe,
        Err(err) if matches!(err.kind, pyre_interpreter::error::PyErrorKind::TypeError) => {
            // `except TypeError` resolves the name where the handler runs, so
            // the binding matters on this arm and nowhere else.  A shadowing
            // module entry already moved an identity word and declined above,
            // which leaves the builtins entry the name falls through to:
            // rebound, the handler `app_abc.py` spells need not match what the
            // constructor raised, and answering `False` here would catch what
            // the body would have let out.  Hand the test back instead of
            // deciding it.  `ref` on an item that refuses one raises without
            // running anything observable, so re-running it costs the raise.
            if !type_error_names_the_installed_class() {
                return Ok(None);
            }
            return Ok(Some(false));
        }
        Err(err) => return Err(err),
    };
    // An empty collection holds no entry to match, so the membership answer is
    // `False` once the probe exists.  The order is what `__contains__` spells:
    // `ref(item)` runs before `wr in self.data` is reached, so an item that
    // cannot carry one raises whatever `ref` raises whether the collection is
    // empty or not.  Answering `False` before the probe would decide the case
    // the arm below hands back.
    if unsafe { pyre_object::w_set_len(roots.get(data_slot)) } == 0 {
        return Ok(Some(false));
    }
    let probe_slot = roots.publish(&[probe]);
    // `wr in self.data`.  The probe carries no callback, and a weakref hashes
    // and compares by its referent, so it finds the callback-carrying entry
    // `add` stored.
    match unsafe {
        pyre_object::w_set_contains_checked(roots.get(data_slot), roots.get(probe_slot))
    } {
        Ok(found) => Ok(Some(found)),
        // The probe hashes by its referent, so a metaclass `__hash__` that
        // raises arrives here.  Recover that exception rather than declining:
        // the protocol would build a second probe and run the same observable
        // `__hash__` again before raising, and the expression it stands in for
        // runs it once.  `wr in self.data` is a set membership test, so the
        // recovered error names a set element.
        Err(_) => Err(
            pyre_interpreter::baseobjspace::take_pending_set_element_error(roots.get(probe_slot)),
        ),
    }
}

/// `app_abc.py SimpleWeakSet.add` — `self.data.add(ref(item, self._remove))`.
/// Called rather than open-coded so the entry carries the callback that
/// discards it once the referent dies; a bare `ref` would leave a spent one
/// behind for every class the check ever saw.
///
/// Every item reaching here is a
/// class — `register` and `subclass_of` each reject a non-type argument — and a
/// class is weak-referenceable, so `add` needs no test of its own, which is
/// also why `app_abc.py add` has none.
fn weak_cache_add(
    cls: PyObjectRef,
    name: &str,
    item: PyObjectRef,
) -> Result<(), pyre_interpreter::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[cls]);
    let item_slot = roots.publish(&[item]);
    let cache = cache_attr(roots.get(cls_slot), name)?;
    let cache_slot = roots.publish(&[cache]);
    let add = pyre_interpreter::baseobjspace::getattr_str(roots.get(cache_slot), "add")?;
    let add_slot = roots.publish(&[add]);
    pyre_interpreter::call::call_function_impl_result(
        roots.get(add_slot),
        &[roots.get(item_slot)],
    )?;
    Ok(())
}

/// `SimpleWeakSet.clear` (`app_abc.py`) on the named collection, in
/// place, so anything already holding it sees the clear.
fn weak_cache_clear(cls: PyObjectRef, name: &str) -> Result<(), pyre_interpreter::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[cls]);
    let cache = cache_attr(roots.get(cls_slot), name)?;
    let cache_slot = roots.publish(&[cache]);
    let clear = pyre_interpreter::baseobjspace::getattr_str(roots.get(cache_slot), "clear")?;
    let clear_slot = roots.publish(&[clear]);
    pyre_interpreter::call::call_function_impl_result(roots.get(clear_slot), &[])?;
    Ok(())
}

/// The named collection attribute of `cls`, or null when it has none.  Read
/// fresh at every use: the walks between two reads run arbitrary Python, which
/// can rebind the attribute and can move the object.
///
/// `app_abc.py:110` reads the slot as a plain attribute, so a class that never
/// ran `_abc_init` raises `AttributeError` out of the check rather than being
/// answered as a class with an empty cache.  A metaclass hook that raises
/// something else propagates unchanged.
fn cache_attr(cls: PyObjectRef, name: &str) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    pyre_interpreter::baseobjspace::getattr_str(cls, name)
}

/// Compare `cls._abc_negative_cache_version` against the invalidation counter
/// the way `_abc_subclasscheck` (`<`) and `_abc_instancecheck` (`==`) do.
///
/// The stored value is passed to the comparison as it stands rather than
/// normalised to an integer first: a rebound non-int raises from the compare,
/// and a value with its own `__lt__` / `__eq__` stays observable.
fn negative_cache_version_compare(
    cls: PyObjectRef,
    op: pyre_interpreter::objspace::descroperation::CompareOp,
) -> Result<bool, pyre_interpreter::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[cls]);
    let version = cache_attr(roots.get(cls_slot), "_abc_negative_cache_version")?;
    let version_slot = roots.publish(&[version]);
    let counter_slot = roots.publish(&[w_int_new(
        INVALIDATION_COUNTER.load(Ordering::Relaxed) as i64
    )]);
    pyre_interpreter::baseobjspace::is_true(pyre_interpreter::objspace::descroperation::compare(
        roots.get(version_slot),
        roots.get(counter_slot),
        op,
    )?)
}

/// `app_abc.py _abc_init` — install the three collections and the
/// negative-cache generation, then compute `__abstractmethods__`.
fn abc_init(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if let Some(cls) = args.first().copied() {
        // A heap type is born in the nursery (`w_type_new`). Every
        // allocation below can collect, so `cls` lives on the shadow stack
        // for the whole body and is re-read from its slot after each
        // collecting call — the same livevar set `gct_fv_gc_malloc` would
        // emit around those constructors (`framework.py`).
        let roots = pyre_object::gc_roots::push_roots();
        let cls_slot = roots.pin_roots(&[cls]);
        // Slot indices of abstract-method names pinned on `roots`.  A plain
        // `Vec<PyObjectRef>` is not a root area, so the names cannot sit
        // there across `getattr` / `isabstractmethod_w`.  `RootedItems`
        // must own the shadow-stack top while it pushes, so the names stay
        // on this outer bracket until the frozenset constructor, which
        // opens `RootedItems` on top and copies from these slots.
        let mut abstract_slots: Vec<usize> = Vec::new();

        // `app_abc.py:74-77` — registry and both caches are per-class for the
        // same reason: resolved up the MRO, one base's would answer for every
        // descendant ABC, and a hit on `Rational` would satisfy `Integral`.
        for name in ["_abc_registry", "_abc_cache", "_abc_negative_cache"] {
            let mut fresh = new_simple_weak_set()?;
            pyre_object::with_roots!(fresh => pyre_interpreter::baseobjspace::setattr_str(
                roots.get(cls_slot),
                name,
                fresh,
            ))?;
        }
        // `space.newint` (`intobject.py wrapint`) allocates the version box.
        let mut version = w_int_new(INVALIDATION_COUNTER.load(Ordering::Relaxed) as i64);
        pyre_object::with_roots!(version => pyre_interpreter::baseobjspace::setattr_str(
            roots.get(cls_slot),
            "_abc_negative_cache_version",
            version,
        ))?;

        // `bases = cls.__bases__` is read once: a metaclass hook run by the
        // lookups below may rebind `cls.__bases__`, and the loop still walks
        // the original tuple.  It is a nursery object and `getattr` /
        // `collect_iterable` collect, so it is rooted and reloaded.
        let bases = unsafe { w_type_get_bases(roots.get(cls_slot)) };
        if !bases.is_null() && unsafe { is_tuple(bases) } {
            let n_bases = unsafe { w_tuple_len(bases) };
            let bases_slot = roots.pin_roots(&[bases]);
            for i in 0..n_bases {
                let bases = roots.get(bases_slot);
                let Some(mut base) = (unsafe { w_tuple_getitem(bases, i as i64) }) else {
                    continue;
                };
                // app_abc.py:67-71 — `getattr(..., set())` defaults only a
                // missing attribute.  A descriptor failure is observable and
                // any iterable, not just a set/frozenset, supplies names.
                let names = match pyre_object::with_roots!(base => {
                    pyre_interpreter::baseobjspace::getattr_str(base, "__abstractmethods__")
                }) {
                    Ok(names) => names,
                    Err(err) if err.kind == pyre_interpreter::PyErrorKind::AttributeError => {
                        w_set_new()
                    }
                    Err(err) => return Err(err),
                };
                let mut names = names;
                let collected = pyre_object::with_roots!(names => {
                    pyre_interpreter::builtins::collect_iterable(names)
                })?;
                let n_names = collected.len();
                let name_base = roots.pin_roots(&collected);
                for j in 0..n_names {
                    let mut name = roots.get(name_base + j);
                    // `_py_abc.py:69` — object-level getattr validates that
                    // every supplied abstract-method name is a string, then
                    // lets descriptors and metaclass attributes provide an
                    // implementation.  Only a missing attribute defaults.
                    let value = match pyre_object::with_roots!(name => {
                        pyre_interpreter::baseobjspace::getattr(roots.get(cls_slot), name)
                    }) {
                        Ok(value) => value,
                        Err(err) if err.kind == pyre_interpreter::PyErrorKind::AttributeError => {
                            w_none()
                        }
                        Err(err) => return Err(err),
                    };
                    let mut value = value;
                    if pyre_object::with_roots!(name, value => {
                        pyre_interpreter::baseobjspace::isabstractmethod_w(value)
                    })? {
                        abstract_slots.push(name_base + j);
                    }
                }
            }
        }
        // Re-read the namespace off the live class: the type dict moves
        // with the heap type, and the items vec is iterated across
        // `unwrap_cell` (`w_int_new` for an `IntMutableCell`) and
        // `isabstractmethod_w`.
        let mut namespace = unsafe { w_type_get_dict_ptr(roots.get(cls_slot)) as PyObjectRef };
        if !namespace.is_null() {
            let pairs = pyre_object::with_roots!(namespace => unsafe { w_dict_items(namespace) });
            let mut flat = Vec::with_capacity(pairs.len() * 2);
            for (name, value) in pairs {
                flat.push(name);
                flat.push(value);
            }
            let n_pairs = flat.len() / 2;
            let flat_base = roots.pin_roots(&flat);
            for j in 0..n_pairs {
                // `unwrap_cell` can allocate, so `name` is read after it.
                let mut value =
                    unsafe { pyre_object::celldict::unwrap_cell(roots.get(flat_base + 2 * j + 1)) };
                let mut name = roots.get(flat_base + 2 * j);
                if pyre_object::with_roots!(name, value => {
                    pyre_interpreter::baseobjspace::isabstractmethod_w(value)
                })? {
                    abstract_slots.push(flat_base + 2 * j);
                }
            }
        }
        // `app_abc.py _abc_init` `cls.__abstractmethods__ = frozenset(abstracts)`.
        // `w_frozenset_from_items` allocates; `RootedItems` owns the top while
        // it pushes copies of the already-pinned names (`gc_roots.rs`).
        {
            let mut abstract_names = pyre_object::gc_roots::RootedItems::new();
            for &slot in &abstract_slots {
                abstract_names.push(roots.get(slot));
            }
            let mut methods = w_frozenset_from_items(&abstract_names.take());
            pyre_object::with_roots!(methods => pyre_interpreter::baseobjspace::setattr_str(
                roots.get(cls_slot),
                "__abstractmethods__",
                methods,
            ))?;
        }

        // `app_abc.py _abc_init` — fold a `__abc_tpflags__` in the class body
        // into the structural-match marker, then drop the attribute.
        //
        // This runs here rather than in `type_new` because the marker is only
        // meant for a class going through `ABCMeta`: consuming it at class
        // creation gives a plain class the marker too, and `case [...]` would
        // then accept an object that is not a sequence.
        //
        // `_abc.c _abc__abc_init_impl` masks the value and applies whatever
        // collection bits survive, where `interp_abc.py set_collection_flag`
        // compares the whole value against one flag and raises `ValueError`
        // otherwise. The mask is the observable behaviour -- `__abc_tpflags__ =
        // (1 << 5) | 1` sets the sequence marker -- so mask here and hand the
        // primitive the single bit its own contract is written against.
        if unsafe { is_type(roots.get(cls_slot)) }
            && let Some(w_flags) =
                pyre_interpreter::type_dict_lookup(roots.get(cls_slot), "__abc_tpflags__")
        {
            // `PyDict_Pop(dict, &_Py_ID(__abc_tpflags__), &flags)` -- take the
            // entry out of the type dict itself, and take it before validating.
            // A metaclass `__delattr__` never sees this, and a value the checks
            // below reject is consumed all the same, so a subclass re-running
            // `_abc_init` does not inherit the same rejection.
            // `type_dict_lookup` unwraps an `IntMutableCell` through
            // `w_int_new`, so `cls` is re-read from the outer slot rather
            // than the copy captured before the lookup.
            let flag_roots = pyre_object::gc_roots::push_roots();
            let flag_base = flag_roots.pin_roots(&[w_flags, roots.get(cls_slot)]);
            let flags_slot = flag_base;
            let flag_cls_slot = flag_base + 1;
            pyre_interpreter::type_dict_delete(flag_roots.get(flag_cls_slot), "__abc_tpflags__");
            unsafe {
                pyre_interpreter::baseobjspace::mutated(
                    flag_roots.get(flag_cls_slot),
                    "__abc_tpflags__",
                )
            };
            let w_flags = flag_roots.get(flags_slot);
            // `PyLong_CheckExact` -- an `int` subclass, `bool` included, is
            // consumed and ignored, as is anything that is not an int at all.
            // Both int representations spell `int`, so both are exact.
            if unsafe { is_exact_type(w_flags, &INT_TYPE) || is_exact_type(w_flags, &LONG_TYPE) } {
                // `PyLong_AsLong` -- a value past the machine word raises
                // `OverflowError` rather than being skipped like a non-int.
                let flags = pyre_interpreter::baseobjspace::int_w(w_flags)?;
                if flags & COLLECTION_FLAGS == COLLECTION_FLAGS {
                    return Err(pyre_interpreter::PyError::type_error(
                        "__abc_tpflags__ cannot be both Py_TPFLAGS_SEQUENCE and Py_TPFLAGS_MAPPING",
                    ));
                }
                if flags & COLLECTION_FLAGS != 0 {
                    set_collection_flag_of(
                        flag_roots.get(flag_cls_slot),
                        flags & COLLECTION_FLAGS,
                    )?;
                }
            }
        }
    }
    Ok(w_none())
}

// `_abc_register` (`_abcmodule.c:_abc__abc_register_impl`) —
// `cls._abc_registry.add(subclass)`.  Pyre stores the registry as a list
// attribute (no WeakSet).
fn register(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.len() < 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "_abc_register() requires (cls, subclass)",
        ));
    }
    let mut cls = args[0];
    let mut subclass = args[1];
    // `_abc_register`: `if not isinstance(subclass, type): raise TypeError(
    // "Can only register classes")`.  Everything downstream reads a registry
    // entry as a class, so the rejection is what makes that sound.
    if !unsafe { is_type(subclass) } {
        return Err(pyre_interpreter::PyError::type_error(
            "Can only register classes",
        ));
    }
    // Already a subclass (`issubclass(subclass, cls)`) — nothing to register.
    // This also dedups: a previously registered `subclass` resolves through
    // `__subclasscheck__`'s registry walk.
    if pyre_object::with_roots!(cls, subclass => pyre_interpreter::baseobjspace::issubclass(subclass, cls))?
    {
        return Ok(subclass);
    }
    // Registering `subclass` would also make `cls` its subclass.  Tested after
    // the arm above, so `X.register(X)` stays a no-op rather than a cycle.
    if pyre_object::with_roots!(cls, subclass => pyre_interpreter::baseobjspace::issubclass(cls, subclass))?
    {
        return Err(pyre_interpreter::PyError::runtime_error(
            "Refusing to create an inheritance cycle",
        ));
    }
    // `app_abc.py cls._abc_registry.add(subclass)`.
    pyre_object::with_roots!(cls, subclass => weak_cache_add(cls, "_abc_registry", subclass))?;
    // `app_abc.py:100-101` — invalidate every negative cache.  A class this
    // registration now makes a subclass may already be recorded as a non-match
    // somewhere, and only the counter can reach those entries: they live on
    // arbitrary other ABCs, not on `cls`.
    INVALIDATION_COUNTER.fetch_add(1, Ordering::Relaxed);
    // `app_abc.py:102-105` — an ABC that carries a structural-match marker
    // hands it to the registered class and its descendants
    // (`_internal_set_collection_flag_recursive`).  Registering with
    // `Mapping` / `Sequence` is what makes `case {...}` / `case [...]` accept
    // a class that inherits from neither, so the marker has to travel with
    // the registration, not only with `__abc_tpflags__` at class creation.
    // `_abc_register` reads the flags under `if isinstance(cls, type):`.
    if unsafe { is_type(cls) } {
        let flag = unsafe { typeobject::w_type_get_flag_map_or_seq(cls) };
        if flag != b'?' {
            // Same livevar set as the `issubclass` / `weak_cache_add` arms
            // above: `interp_abc.py set_collection_flag_recursive` walks
            // `weak_subclasses`, so `subclass` is re-read from its slot
            // before the return.
            pyre_object::with_roots!(subclass => {
                set_collection_flag_recursive(subclass, flag);
            });
        }
    }
    Ok(subclass)
}

/// `interp_abc.py` takes both markers from `typeobject.py`, which is where the
/// bits are declared.
use pyre_object::typeobject::{PATMA_MAPPING, PATMA_SEQUENCE};

/// `app_abc.py COLLECTION_FLAGS`.
const COLLECTION_FLAGS: i64 = PATMA_SEQUENCE | PATMA_MAPPING;

/// `interp_abc.py set_collection_flag` — stamp one structural-match marker on
/// `w_type`, without touching its subclasses.
///
/// `_abc_init` masks `__abc_tpflags__` down to a single bit first, so the
/// strict test below only ever rejects a caller that spells the flag itself.
fn set_collection_flag_of(w_type: PyObjectRef, flag: i64) -> Result<(), pyre_interpreter::PyError> {
    let marker = collection_marker(w_type, flag, "_internal_set_collection_flag")?;
    unsafe { typeobject::w_type_set_flag_map_or_seq(w_type, marker) };
    Ok(())
}

/// The `flag_patma_collection` byte one collection flag stands for, with the
/// two rejections `set_collection_flag` makes before it stamps anything:
/// `space.interp_w(W_TypeObject, w_self)` on the receiver, and the strict
/// one-bit test on the flag.
fn collection_marker(
    w_type: PyObjectRef,
    flag: i64,
    who: &str,
) -> Result<u8, pyre_interpreter::PyError> {
    if !unsafe { is_type(w_type) } {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{who}() argument 1 must be a type"
        )));
    }
    match flag {
        PATMA_SEQUENCE => Ok(b'S'),
        PATMA_MAPPING => Ok(b'M'),
        _ => Err(pyre_interpreter::PyError::value_error(format!(
            "invalid value for __abc_tpflags__: {flag}"
        ))),
    }
}

/// `_abc._internal_set_collection_flag(cls, flag)`.
fn internal_set_collection_flag(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let [w_type, w_flag] = args else {
        return Err(pyre_interpreter::PyError::type_error(
            "_internal_set_collection_flag() requires (cls, flag)",
        ));
    };
    let mut w_type = *w_type;
    let flag = pyre_object::with_roots!(w_type => pyre_interpreter::baseobjspace::int_w(*w_flag))?;
    set_collection_flag_of(w_type, flag)?;
    Ok(w_none())
}

/// `_abc._internal_set_collection_flag_recursive(cls, flag)`.
fn internal_set_collection_flag_recursive(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let [w_type, w_flag] = args else {
        return Err(pyre_interpreter::PyError::type_error(
            "_internal_set_collection_flag_recursive() requires (cls, flag)",
        ));
    };
    let mut w_type = *w_type;
    let flag = pyre_object::with_roots!(w_type => pyre_interpreter::baseobjspace::int_w(*w_flag))?;
    let marker = collection_marker(w_type, flag, "_internal_set_collection_flag_recursive")?;
    // `_PyType_SetFlagsRecursive` starts the guarded walk at the argument
    // itself, not at its children: `set_collection_flag` stamps whatever it is
    // handed, so entering through that one would let this primitive mark an
    // immutable type -- `str` among them.
    set_collection_flag_recursive(w_type, marker);
    Ok(w_none())
}

// `typeobject.c set_flags_recursive` — stamp the marker on `w_type` and every
// class already deriving from it.  `interp_abc.py set_collection_flag_recursive`
// carries neither of the two stops below and would mark `str`.
fn set_collection_flag_recursive(w_type: PyObjectRef, flag: u8) {
    unsafe {
        // `Py_TPFLAGS_IMMUTABLETYPE`: a non-heap type's marker is fixed at
        // registration (`objspace.py StdObjSpace.initialize` marks exactly
        // dict / dictproxy / list / tuple). `_collections_abc` runs
        // `Sequence.register(str)` and `ByteString.register(bytes)`, so without
        // this stop `str` / `bytes` / `bytearray` would start matching
        // `case [...]` — the one thing a sequence pattern must never accept.
        //
        // `(tp_flags & mask) == flags`: a class already carrying the marker
        // passed it to its descendants at creation
        // (`inherit_flag_map_or_seq`), so that subtree is done.
        if !typeobject::w_type_is_heaptype(w_type)
            || typeobject::w_type_get_flag_map_or_seq(w_type) == flag
        {
            return;
        }
        typeobject::w_type_set_flag_map_or_seq(w_type, flag);
        // `only_real_subclasses` is False for every walk but
        // `descr___subclasses__` (typeobject.py).
        for child in typeobject::w_type_get_subclasses(w_type, false) {
            set_collection_flag_recursive(child, flag);
        }
    }
}

// `_py_abc.ABCMeta.__subclasscheck__` (`_py_abc.py`): the caches
// first, then the subclass hook, then a direct `__mro__` test, then the
// recursive registry and subclass walks.  `issubclass` re-dispatches through
// `__subclasscheck__` so a registered or descendant ABC applies its own hook
// in turn — which is also why the caches are load-bearing rather than a
// refinement: an uncached miss re-runs all three walks at every level of that
// recursion, so one `isinstance` against a deep ABC costs a walk of the whole
// ABC graph.
fn subclass_of(cls: PyObjectRef, subclass: PyObjectRef) -> Result<bool, pyre_interpreter::PyError> {
    // _py_abc.py:110-111 — `if not isinstance(subclass, type): raise
    // TypeError('issubclass() arg 1 must be a class')`.  The `__mro__`/registry
    // walks below dereference `subclass` as a type, so a non-type argument
    // (`issubclass({}, ABC)`) must be rejected up front, not read as garbage.
    if !unsafe { is_type(subclass) } {
        return Err(pyre_interpreter::PyError::type_error(
            "issubclass() arg 1 must be a class",
        ));
    }

    // The hook, registry walk, and `__subclasses__` walk below can all run
    // arbitrary Python.  Keep the two arguments live and reload them after
    // every such call, matching the translated livevars carried by PyPy's
    // `ABCMeta.__subclasscheck__`.
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.base();
    let _ = roots.pin_root(cls);
    let subclass_slot = cls_slot + 1;
    let _ = roots.pin_root(subclass);

    // `app_abc.py:130-131` — a positive hit is final: nothing invalidates it,
    // since `register` can only ever add subclasses.
    if weak_cache_contains(roots.get(cls_slot), "_abc_cache", roots.get(subclass_slot))? {
        return Ok(true);
    }
    // `app_abc.py:132-138` — a negative hit only holds for the registry it was
    // recorded against, so a bumped counter discards the whole cache rather
    // than trusting any entry in it.
    let counter = INVALIDATION_COUNTER.load(Ordering::Relaxed);
    if negative_cache_version_compare(
        roots.get(cls_slot),
        pyre_interpreter::objspace::descroperation::CompareOp::Lt,
    )? {
        let fresh = new_simple_weak_set()?;
        pyre_interpreter::baseobjspace::setattr_str(
            roots.get(cls_slot),
            "_abc_negative_cache",
            fresh,
        )?;
        let version = w_int_new(counter as i64);
        pyre_interpreter::baseobjspace::setattr_str(
            roots.get(cls_slot),
            "_abc_negative_cache_version",
            version,
        )?;
    } else if weak_cache_contains(
        roots.get(cls_slot),
        "_abc_negative_cache",
        roots.get(subclass_slot),
    )? {
        return Ok(false);
    }

    // Every arm below concludes the same question, so each one only decides the
    // verdict; the single recording site after the block puts it in the matching
    // cache (`app_abc.py:144-163`, which records at each of its own arms).
    let verdict = 'decide: {
        // _py_abc.py — `ok = cls.__subclasshook__(subclass)`.
        let hook =
            pyre_interpreter::baseobjspace::getattr_str(roots.get(cls_slot), "__subclasshook__")?;
        if !hook.is_null() {
            let hook_roots = pyre_object::gc_roots::push_roots();
            let hook_slot = hook_roots.base();
            let _ = hook_roots.pin_root(hook);
            let ok = pyre_interpreter::call::call_function_impl_result(
                hook_roots.get(hook_slot),
                &[roots.get(subclass_slot)],
            )?;
            if !unsafe { is_not_implemented(ok) } {
                break 'decide pyre_interpreter::baseobjspace::is_true(ok)?;
            }
        }
        // _py_abc.py — direct subclass via `__mro__`.
        unsafe {
            let mro_ptr = w_type_get_mro(roots.get(subclass_slot));
            if !mro_ptr.is_null() {
                for &t in (*mro_ptr).as_slice() {
                    if std::ptr::eq(t, roots.get(cls_slot)) {
                        break 'decide true;
                    }
                }
            }
        }
        // `app_abc.py _abc_subclasscheck for rcls in cls._abc_registry:` — subclass of a
        // registered class (recursive).  `SimpleWeakSet.__iter__` copies the
        // set before yielding and skips entries whose referent is gone, so the
        // walk survives a collection that fires the discard callback partway
        // through it.
        let registry = cache_attr(roots.get(cls_slot), "_abc_registry")?;
        {
            let registry_roots = pyre_object::gc_roots::push_roots();
            let registry_slot = registry_roots.base();
            let _ = registry_roots.pin_root(registry);
            let iterator = pyre_interpreter::baseobjspace::iter(registry_roots.get(registry_slot))?;
            let iterator_slot = registry_slot + 1;
            let _ = registry_roots.pin_root(iterator);
            loop {
                let rcls =
                    match pyre_interpreter::baseobjspace::next(registry_roots.get(iterator_slot)) {
                        Ok(rcls) => rcls,
                        Err(err) => {
                            let (stop, err) = err.matches_stop_iteration_keep();
                            if stop {
                                break;
                            }
                            return Err(err);
                        }
                    };
                let item_roots = pyre_object::gc_roots::push_roots();
                let rcls_slot = item_roots.base();
                let _ = item_roots.pin_root(rcls);
                if pyre_interpreter::baseobjspace::issubclass(
                    roots.get(subclass_slot),
                    item_roots.get(rcls_slot),
                )? {
                    break 'decide true;
                }
            }
        }
        // _py_abc.py — `for scls in cls.__subclasses__():`.  This must go
        // through normal attribute lookup, call, and iteration.  Reading the
        // internal type subclass vector directly hides user overrides and their
        // TypeError/custom exceptions, which are observable ABCMeta semantics.
        let subclasses_method =
            pyre_interpreter::baseobjspace::getattr_str(roots.get(cls_slot), "__subclasses__")?;
        let walk_roots = pyre_object::gc_roots::push_roots();
        let method_slot = walk_roots.base();
        let _ = walk_roots.pin_root(subclasses_method);
        let subclasses =
            pyre_interpreter::call::call_function_impl_result(walk_roots.get(method_slot), &[])?;
        let subclasses_slot = method_slot + 1;
        let _ = walk_roots.pin_root(subclasses);
        let iterator = pyre_interpreter::baseobjspace::iter(walk_roots.get(subclasses_slot))?;
        let iterator_slot = subclasses_slot + 1;
        let _ = walk_roots.pin_root(iterator);
        loop {
            let scls = match pyre_interpreter::baseobjspace::next(walk_roots.get(iterator_slot)) {
                Ok(scls) => scls,
                Err(err) => {
                    let (stop, err) = err.matches_stop_iteration_keep();
                    if stop {
                        break;
                    }
                    return Err(err);
                }
            };
            let item_roots = pyre_object::gc_roots::push_roots();
            let scls_slot = item_roots.base();
            let _ = item_roots.pin_root(scls);
            if pyre_interpreter::baseobjspace::issubclass(
                roots.get(subclass_slot),
                item_roots.get(scls_slot),
            )? {
                break 'decide true;
            }
        }
        false
    };

    // `app_abc.py:144-163` records at each of its own arms; one site here covers
    // all of them.
    let recorded = if verdict {
        "_abc_cache"
    } else {
        "_abc_negative_cache"
    };
    weak_cache_add(roots.get(cls_slot), recorded, roots.get(subclass_slot))?;
    Ok(verdict)
}

/// Interp-level `_abc_instancecheck` (`app_abc.py`).
///
/// `extra_init` stores the app-level function over this entry.
/// `abc.py ABCMeta.__instancecheck__` calls that one, which is what a tracer
/// can enter.  This body is the function the module table installs first.
///
/// The two classes are asked separately because they can differ: `__class__`
/// is an ordinary attribute an object may answer with something other than its
/// real type, and a proxy that does so is meant to pass the check for what it
/// claims to be.  Reading only `type(instance)` would ignore the claim;
/// reading only `__class__` would let it deny the real one.  A positive cache
/// hit on the claimed class is taken before the real type is even read, which
/// is the whole point of inlining the cache check here rather than leaving it
/// to `__subclasscheck__`.
fn instancecheck(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.len() < 2 {
        return Ok(w_bool_from(false));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[args[0]]);
    let instance_slot = roots.publish(&[args[1]]);

    // `_abc__abc_instancecheck_impl` reads `__class__` through
    // `PyObject_GetOptionalAttr` and, when that attribute is absent, uses
    // `Py_TYPE(instance)`. Only AttributeError is absence; a NameError from
    // the property must propagate.
    let subclass =
        match pyre_interpreter::baseobjspace::getattr_str(roots.get(instance_slot), "__class__") {
            Ok(cls) => cls,
            Err(err) if err.kind == pyre_interpreter::PyErrorKind::AttributeError => {
                pyre_interpreter::typedef::r#type(roots.get(instance_slot))
                    .map_or(std::ptr::null_mut(), |p| p.as_ptr())
            }
            Err(err) => return Err(err),
        };
    if subclass.is_null() {
        return Ok(w_bool_from(false));
    }
    let subclass_slot = roots.publish(&[subclass]);
    if weak_cache_contains(roots.get(cls_slot), "_abc_cache", roots.get(subclass_slot))? {
        return Ok(w_bool_from(true));
    }

    // `app_abc.py _abc_instancecheck subtype = type(instance)` — the instance's real class.
    // User-defined instances carry the generic layout marker in `ob_type` and
    // the real class in `w_class`, so reading `ob_type` directly would resolve
    // to `object`; `r#type` returns the class for both builtin and user
    // instances.
    let subtype = pyre_interpreter::typedef::r#type(roots.get(instance_slot))
        .map_or(std::ptr::null_mut(), |p| p.as_ptr());
    if subtype.is_null() {
        return Ok(w_bool_from(false));
    }
    let subtype_slot = roots.publish(&[subtype]);

    if std::ptr::eq(roots.get(subtype_slot), roots.get(subclass_slot)) {
        // `app_abc.py:115-117` — one class, so the negative cache can answer.
        // The version test is `==`, not `<`: a cache recorded against a
        // *later* counter than the one read here cannot describe this
        // registry either.
        if negative_cache_version_compare(
            roots.get(cls_slot),
            pyre_interpreter::objspace::descroperation::CompareOp::Eq,
        )? && weak_cache_contains(
            roots.get(cls_slot),
            "_abc_negative_cache",
            roots.get(subclass_slot),
        )? {
            return Ok(w_bool_from(false));
        }
        return Ok(w_bool_from(subclasscheck_of(
            roots.get(cls_slot),
            roots.get(subclass_slot),
        )?));
    }
    // `app_abc.py _abc_instancecheck any(cls.__subclasscheck__(c) for c in (subclass, subtype))`.
    for slot in [subclass_slot, subtype_slot] {
        if subclasscheck_of(roots.get(cls_slot), roots.get(slot))? {
            return Ok(w_bool_from(true));
        }
    }
    Ok(w_bool_from(false))
}

/// `cls.__subclasscheck__(subclass)` through attribute lookup, the way
/// `app_abc.py:118` and `:121` spell it, so an `ABCMeta` subclass that
/// overrides the hook is the one that answers.
fn subclasscheck_of(
    cls: PyObjectRef,
    subclass: PyObjectRef,
) -> Result<bool, pyre_interpreter::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[cls]);
    let subclass_slot = roots.publish(&[subclass]);
    let check =
        pyre_interpreter::baseobjspace::getattr_str(roots.get(cls_slot), "__subclasscheck__")?;
    let check_slot = roots.publish(&[check]);
    let result = pyre_interpreter::call::call_function_impl_result(
        roots.get(check_slot),
        &[roots.get(subclass_slot)],
    )?;
    let result_slot = roots.publish(&[result]);
    pyre_interpreter::baseobjspace::is_true(roots.get(result_slot))
}

fn subclasscheck(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.len() < 2 {
        return Ok(w_bool_from(false));
    }
    Ok(w_bool_from(subclass_of(args[0], args[1])?))
}

/// `_abc._reset_registry(cls)`: clear only this ABC's virtual-subclass
/// registry.  ABCMeta exposes it as `_abc_registry_clear`; a no-op leaks
/// registrations between independent users and test cases.
/// The invalidation counter stays put: only `_abc_register` advances it, so
/// an outstanding `get_cache_token` survives a registry reset.
fn reset_registry(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if let Some(&cls) = args.first() {
        weak_cache_clear(cls, "_abc_registry")?;
    }
    Ok(w_none())
}

/// `_abc._reset_caches(cls)` (`app_abc.py`): empty both of this ABC's
/// caches, leaving the registry and the invalidation counter untouched — a
/// cleared cache is answered by re-running the walks, which is not a change of
/// answer, so no token needs to expire.
///
/// Cleared in place rather than rebound, so anything already holding the set
/// sees the clear.
fn reset_caches(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if let Some(mut cls) = args.first().copied() {
        for name in ["_abc_cache", "_abc_negative_cache"] {
            pyre_object::with_roots!(cls => weak_cache_clear(cls, name))?;
        }
    }
    Ok(w_none())
}

/// `_abc._get_dump(cls)` (`app_abc.py`): shallow copies of the
/// registry, both caches, and the negative-cache version.  The three sets are
/// the collections' own `data`, which is why they are `SimpleWeakSet`s and not
/// bare sets — `ABC._dump_registry` prints what this returns.
fn get_dump(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let Some(&cls) = args.first() else {
        return Ok(w_tuple_new(vec![]));
    };
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[cls]);
    // `app_abc.py _get_dump` builds the four-tuple as a single expression, so
    // every `.data` it reads stays a live variable that the later reads reload.
    // Keep the slots rather than the raw pointers: `data` is whatever
    // `cache.data` answers, so it can be a list or a dict — the two kinds a
    // minor collection moves — and each further `cache_attr` / `getattr_str`
    // runs the descriptor protocol.
    let mut data_slots = Vec::with_capacity(3);
    for name in ["_abc_registry", "_abc_cache", "_abc_negative_cache"] {
        let cache = cache_attr(roots.get(cls_slot), name)?;
        let cache_slot = roots.publish(&[cache]);
        let data = pyre_interpreter::baseobjspace::getattr_str(roots.get(cache_slot), "data")?;
        data_slots.push(roots.publish(&[data]));
    }
    // `_get_dump` reports the generation attribute itself, not a number derived
    // from it.  The last read that can run Python; take it before the reloads
    // below so they answer with final addresses.
    let version = cache_attr(roots.get(cls_slot), "_abc_negative_cache_version")?;
    let mut items: Vec<PyObjectRef> = data_slots.iter().map(|&slot| roots.get(slot)).collect();
    items.push(version);
    Ok(w_tuple_new(items))
}

pyre_interpreter::py_module! {
    "_abc",
    functions: {
        "get_cache_token"     / 0 = |_| Ok(w_int_new(INVALIDATION_COUNTER.load(Ordering::Relaxed) as i64)),
        "_abc_init"           / 1 = abc_init,
        "_abc_register"       / 2 = register,
        "_abc_instancecheck"  / 2 = instancecheck,
        "_abc_subclasscheck"  / 2 = subclasscheck,
        "_get_dump"           / 1 = get_dump,
        "_reset_registry"     / 1 = reset_registry,
        "_reset_caches"       / 1 = reset_caches,
        "_internal_set_collection_flag"           / 2 = internal_set_collection_flag,
        "_internal_set_collection_flag_recursive" / 2 = internal_set_collection_flag_recursive,
    },
    extra_init: |ns| {
        let mut ns = ns;
        // `app_abc.py _abc_instancecheck` reads `get_cache_token` for the
        // negative-cache generation.  The app file runs before this module
        // can be imported, so the builtin stored above is seeded into the
        // app globals (`appleveldef_install_seeded`).  Storing the name
        // overwrites the interp-level entry the `functions` arm just installed.
        let mut get_cache_token = pyre_interpreter::module_ns_get(ns, "get_cache_token")
            .expect("_abc.get_cache_token installed before appleveldefs");
        pyre_object::with_roots!(ns, get_cache_token => pyre_interpreter::importing::appleveldef_install_seeded(
            ns,
            include_str!("app_abc.py"),
            "app_abc.py",
            "_abc",
            &["SimpleWeakSet", "_abc_instancecheck"],
            &[("get_cache_token", get_cache_token)],
        ))?;
        ns = pyre_object::gc_roots::pin_root(ns);
        let simple_weak_set = pyre_interpreter::module_ns_get(ns, "SimpleWeakSet")
            .expect("_abc.SimpleWeakSet must be installed by appleveldefs");
        SIMPLE_WEAK_SET_TYPE.set(simple_weak_set);
        let installed = simple_weak_set_contains_identity();
        if installed != (0, 0, 0, 0) {
            // Each word is an address the comparison keeps naming, so a match
            // has to mean the object is still the one installed rather than
            // that the allocator reused its address.  A weak reference settles
            // that without also deciding how long the object lives.  The
            // `TypeError` word is `0` unless a module entry shadows the
            // builtin, and zero names nothing to reference.
            let mut lifelines = [0usize; 4];
            let installed_words = [installed.0, installed.1, installed.2, installed.3];
            let recorded = lifelines
                .iter_mut()
                .zip(installed_words)
                .all(|(lifeline, word)| {
                    if word == 0 {
                        return true;
                    }
                    match weak_lifeline(word as PyObjectRef) {
                        Some(recorded) => {
                            *lifeline = recorded;
                            true
                        }
                        None => false,
                    }
                });
            // A referent that refuses a weak reference leaves nothing to
            // compare against later, so the fast path simply never installs.
            if recorded {
                let _ = SIMPLE_WEAK_SET_CONTAINS.set(lifelines);
            }
            record_builtin_type_error();
        }
    },
}
