//! _collections module — PyPy: `pypy/module/_collections/`.
//!
//! Provides the C-accelerated `deque` / `defaultdict` / `OrderedDict`
//! types.  `deque` is an interp-level typed-payload `#[pyre_class]` backed
//! by the same doubly-linked list of fixed-size blocks as PyPy's `W_Deque`.
//! `defaultdict` and `OrderedDict` are
//! app-level `dict` subclasses in `app_defaultdict.py` / `app_odict.py`,
//! mirroring PyPy (neither runtime can subclass the app-level `dict` from
//! interp-level).

use crate::objspace::std::mapdict::instance_getclass;
use pyre_object::*;
use std::cell::UnsafeCell;
use std::sync::LazyLock;

const BLOCKLEN: i64 = 62;
const CENTER: i64 = (BLOCKLEN - 1) / 2;

/// `interp_deque.py Block`: links and element storage are GC-visible object
/// references, exactly like the RPython object's three attributes.
pub mod deque_block {
    use pyre_object::*;

    #[crate::pyre_class("_collections._deque_block")]
    pub struct W_DequeBlock {
        pub leftlink: PyObjectRef,
        pub rightlink: PyObjectRef,
        pub data: PyObjectRef,
    }

    pub fn new(mut leftlink: PyObjectRef, mut rightlink: PyObjectRef) -> PyObjectRef {
        // `interp_deque.py Block.data = [None] * BLOCKLEN` is annotated by
        // RPython as a list of `W_Root` references.  It must therefore keep
        // ObjectListStrategy even when every live deque entry is an int or a
        // float; a typed list strategy would unbox and later rebox the entry,
        // breaking deque's shallow-copy identity semantics.
        let data = pyre_object::with_roots!(leftlink, rightlink => w_list_new_object(vec![w_none(); super::BLOCKLEN as usize]));
        W_DequeBlock::allocate_stable(W_DequeBlock {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            leftlink,
            rightlink,
            data,
        })
    }
}

use deque_block::W_DequeBlock;

/// `interp_deque.py Lock`: a fieldless GC object used only as an identity
/// token.  It is deliberately an object, not a counter — `W_Deque.lock` and
/// each iterator hold the same token until a mutation replaces the deque field
/// with `None`, exactly like PyPy.
pub mod deque_lock {
    use pyre_object::*;

    #[crate::pyre_class("_collections._deque_lock")]
    pub struct W_DequeLock {}

    pub fn new() -> PyObjectRef {
        W_DequeLock::allocate_stable(W_DequeLock {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
        })
    }
}

/// `pypy/module/_collections/interp_deque.py` — `W_Deque` surface
/// (append / appendleft / pop / popleft / clear / extend / extendleft /
/// rotate / count / remove / reverse / index / copy + container and repr
/// protocols, with `maxlen` bounding).
// CPython 3.14 Modules/_collectionsmodule.c:collections_exec ADD_TYPEs
// deque_spec through a module heap type; the spec is immutable.
#[crate::pyre_class("collections.deque", cpython_heaptype, user_layout, weakrefable)]
pub struct W_Deque {
    leftblock: PyObjectRef,
    rightblock: PyObjectRef,
    leftindex: i64,
    rightindex: i64,
    len: i64,
    /// Bound: `-1` = unbounded, `>= 0` = the maxlen. Validated non-negative at `__init__`.
    maxlen: i64,
    /// `interp_deque.py W_Deque.lock`: `None` until observed, then a unique
    /// fieldless `Lock` identity; every mutation stores `None` again.
    lock: PyObjectRef,
}

// PyPy's deque block/endpoint transitions execute atomically under its GIL.
// CPython 3.14's free-threaded deque preserves the same boundary with object
// critical sections.  Pyre keeps the semantic state on W_Deque and uses the
// same narrow address-striped reentrant synchronization adaptation as its
// list/dict/set ports.  The stripes are reset after fork because a vanished
// thread may have held one in the parent.
struct ForkDequeLock(UnsafeCell<parking_lot::ReentrantMutex<()>>);
unsafe impl Sync for ForkDequeLock {}

impl ForkDequeLock {
    fn new() -> Self {
        Self(UnsafeCell::new(parking_lot::ReentrantMutex::new(())))
    }

    fn get(&self) -> &parking_lot::ReentrantMutex<()> {
        unsafe { &*self.0.get() }
    }

    unsafe fn reinit_after_fork(&self) {
        unsafe { self.0.get().write(parking_lot::ReentrantMutex::new(())) };
    }
}

static DEQUE_LOCKS: LazyLock<Vec<ForkDequeLock>> =
    LazyLock::new(|| (0..256).map(|_| ForkDequeLock::new()).collect());

type DequeGuard = parking_lot::lock_api::ReentrantMutexGuard<
    'static,
    parking_lot::RawMutex,
    parking_lot::RawThreadId,
    (),
>;

#[majit_macros::dont_look_inside]
unsafe fn w_deque_lock(obj: PyObjectRef) -> DequeGuard {
    let lock = DEQUE_LOCKS[(obj as usize >> 4) & (DEQUE_LOCKS.len() - 1)].get();
    if let Some(guard) = lock.try_lock() {
        return guard;
    }
    let blocked = majit_gc::gc_sync::before_external_block();
    let guard = lock.lock();
    drop(blocked);
    guard
}

pub fn deque_locks_after_fork_child() {
    for lock in DEQUE_LOCKS.iter() {
        unsafe { lock.reinit_after_fork() };
    }
}

fn block(block: PyObjectRef) -> &'static mut W_DequeBlock {
    W_DequeBlock::from_obj(block).expect("deque block")
}

fn block_get(block_obj: PyObjectRef, index: i64) -> PyObjectRef {
    unsafe { w_list_getitem(block(block_obj).data, index) }.expect("deque block index")
}

fn block_set(block_obj: PyObjectRef, index: i64, value: PyObjectRef) {
    let stored = unsafe { w_list_setitem(block(block_obj).data, index, value) };
    debug_assert!(stored);
}

fn deque_len(self_obj: PyObjectRef) -> i64 {
    W_Deque::from_obj(self_obj).map(|d| d.len).unwrap_or(0)
}

/// Whether `obj` has PyPy's `W_Deque` layout, including a Python subclass
/// (`typedef.py` `_getusercls`).
pub(crate) fn is_deque(obj: PyObjectRef) -> bool {
    W_Deque::from_obj(obj).is_some()
}

/// The prologue of `_find_or_count` (interp_deque.py): `lock =
/// self.getlock()`, then the walk.  Pins the receiver, its lock and the
/// needle into the caller's open bracket before `getlock` -- which allocates
/// the lock token, and whose stripe may block, a safepoint -- and the
/// snapshot the walk reads after it.  Answers `(own_base, items_base,
/// items_len)`: `own_base` holds the receiver, `own_base + 1` the lock and
/// `own_base + 2` the needle.
fn pin_find_or_count(self_obj: PyObjectRef, x: PyObjectRef) -> (usize, usize, usize) {
    let own_base = pyre_object::gc_roots::pin_roots(&[self_obj, PY_NULL, x]);
    let lock = getlock(pyre_object::gc_roots::shadow_stack_get(own_base));
    pyre_object::gc_roots::shadow_stack_set(own_base + 1, lock);
    let items = snapshot(pyre_object::gc_roots::shadow_stack_get(own_base));
    let items_base = pyre_object::gc_roots::pin_roots(&items);
    (own_base, items_base, items.len())
}

/// Snapshot the backing list into a `Vec`.
///
/// The walk reads `leftblock`/`leftindex`/`len` and then follows `rightlink`,
/// so it has to exclude the chain replacement in `store`: a walk interleaved
/// with one reads endpoints belonging to two different chains and can reach a
/// `PY_NULL` link. The stripe is reentrant, so the `store`/`append_right`
/// nesting below costs nothing.
fn snapshot(self_obj: PyObjectRef) -> Vec<PyObjectRef> {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let self_obj = pyre_object::gc_roots::pin_root(self_obj);
    let _deque_guard = unsafe { w_deque_lock(self_obj) };
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let Some(deque) = W_Deque::from_obj(self_obj) else {
        return vec![];
    };
    let mut result = Vec::with_capacity(deque.len as usize);
    let mut block_obj = deque.leftblock;
    let mut index = deque.leftindex;
    for _ in 0..deque.len {
        result.push(block_get(block_obj, index));
        index += 1;
        if index >= BLOCKLEN {
            block_obj = block(block_obj).rightlink;
            index = 0;
        }
    }
    result
}

/// Replace the block chain with `items`.
///
/// Held across the whole replacement, not per `append_right`, so no other
/// thread observes the intermediate empty chain.
fn store(self_obj: PyObjectRef, items: Vec<PyObjectRef>) {
    let _roots = pyre_object::gc_roots::push_roots();
    // `append_right` allocates, so any of the appends below can collect and
    // move the entries `items` still names. They are livevars for the whole
    // replacement, not just `self_obj`. Publish both slices before the single
    // normalize: `pin_root` would normalize `self_obj` while the entries are
    // not yet on the stack, and that query is itself a safepoint.
    let root_base = pyre_object::gc_roots::publish_roots(&[self_obj]);
    let items_base = pyre_object::gc_roots::publish_roots(&items);
    pyre_object::gc_roots::normalize_roots(root_base, 1 + items.len());
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    // Reentrant: the two callees take the same stripe.
    let _deque_guard = unsafe { w_deque_lock(self_obj) };
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    clear_blocks(self_obj);
    for i in 0..items.len() {
        let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
        let item = pyre_object::gc_roots::shadow_stack_get(items_base + i);
        append_right(self_obj, item);
    }
}

fn clear_blocks(self_obj: PyObjectRef) {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let self_obj = pyre_object::gc_roots::pin_root(self_obj);
    let _deque_guard = unsafe { w_deque_lock(self_obj) };
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let new_block = deque_block::new(PY_NULL, PY_NULL);
    let deque = W_Deque::from_obj(self_obj).expect("deque");
    deque.leftblock = new_block;
    deque.rightblock = new_block;
    deque.leftindex = CENTER + 1;
    deque.rightindex = CENTER;
    deque.len = 0;
    modified(self_obj);
}

/// `interp_deque.py modified`: invalidate the current identity token.
fn modified(self_obj: PyObjectRef) {
    let base = self_obj as *mut W_Deque;
    if base.is_null() {
        return;
    }
    unsafe { (*base).lock = PY_NULL };
}

/// `interp_deque.py getlock`: lazily install and return one identity token.
///
/// This STORES into `self.lock` on the first call, exactly as upstream's
/// `self.lock = Lock()` does, so every method that reaches it owns the deque
/// mutably — `_find_or_count`, `index`, `iter` and `compare` all call
/// `self.getlock()` upstream.  A `&self` receiver here would let rustc mark
/// the parameter `noalias readonly` and then store through it.
fn getlock(self_obj: PyObjectRef) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let self_obj = pyre_object::gc_roots::pin_root(self_obj);
    let _deque_guard = unsafe { w_deque_lock(self_obj) };
    let base = self_obj as *mut W_Deque;
    if base.is_null() {
        return PY_NULL;
    }
    let current = unsafe { (*base).lock };
    if !current.is_null() {
        return current;
    }
    let token = deque_lock::new();
    unsafe { (*base).lock = token };
    token
}

/// `lock is not self.lock` — the identity test inside `interp_deque.py`'s
/// `checklock`, split out so the raise can stay in the caller.
// Keep the native interpreter's post-callback read out of the caller's alias
// scope.  This is not a JIT opacity hint: source translation still sees the
// ordinary field read below, exactly like PyPy's `lock is self.lock`.
#[inline(never)]
fn lock_valid(self_obj: PyObjectRef, lock: PyObjectRef) -> bool {
    let current = if self_obj.is_null() {
        PY_NULL
    } else {
        unsafe { (*(self_obj as *const W_Deque)).lock }
    };
    current == lock
}

/// `interp_deque.py checklock` — raise `RuntimeError` if the deque was
/// mutated since `lock` was taken.
fn checklock(self_obj: PyObjectRef, lock: PyObjectRef) -> Result<(), crate::PyError> {
    if !lock_valid(self_obj, lock) {
        return Err(crate::PyError::runtime_error(
            "deque mutated during iteration",
        ));
    }
    Ok(())
}

/// `self.maxlen`: `None` (unbounded) or a non-negative bound.  The
/// bound is validated non-negative at construction, so the stored
/// value is read back directly.
fn maxlen_bound(self_obj: PyObjectRef) -> Option<usize> {
    let m = W_Deque::from_obj(self_obj)?.maxlen;
    if m < 0 { None } else { Some(m as usize) }
}

fn maxlen_obj(self_obj: PyObjectRef) -> PyObjectRef {
    match W_Deque::from_obj(self_obj) {
        Some(d) if d.maxlen >= 0 => w_int_new(d.maxlen),
        _ => w_none(),
    }
}

// The two deque iterators are `.allocate()`-immortal `#[pyre_class]` wrappers
// holding a traced `deque` edge the marker never scans; re-export so `build_gc`
// can register their offsets with the immortal-root walker.
pub use deque_iter::W_DequeIter;
pub use deque_rev_iter::W_DequeRevIter;

// `W_DequeIter`: the iterator owns the deque, current position, number
// of remaining entries, and a snapshot of the deque mutation lock.
pub mod deque_iter {
    use super::{BLOCKLEN, W_Deque, block, block_get, checklock, getlock, instance_getclass};
    use pyre_object::*;

    // `collections_exec` ADD_TYPEs dequeiter_spec as an immutable heap type.
    #[crate::pyre_class("_collections._deque_iterator", cpython_heaptype)]
    pub struct W_DequeIter {
        deque: PyObjectRef,
        block: PyObjectRef,
        index: i64,
        counter: i64,
        lock: PyObjectRef,
    }

    fn constructor_args(args: &[PyObjectRef]) -> Result<(PyObjectRef, i64), crate::PyError> {
        // `__new__`'s descriptor ABI leaves the requested class at the front
        // of the varargs slice (the typed `_cls` slot is the descriptor
        // receiver).  PyPy's W_DequeIter__new__ starts parsing after it.
        let positional = args.get(1..).unwrap_or(&[]);
        if positional.is_empty() || positional.len() > 2 {
            return Err(crate::PyError::type_error(format!(
                "_deque_iterator expected 1 or 2 arguments, got {}",
                positional.len()
            )));
        }
        let mut deque = positional[0];
        if W_Deque::from_obj(deque).is_none() {
            let name = crate::type_methods::clinic_arg_type_name(deque);
            return Err(crate::PyError::type_error(format!(
                "must be collections.deque, not {name}"
            )));
        }
        let index = match positional.get(1) {
            Some(&value) => {
                pyre_object::with_roots!(deque => crate::builtins::space_index_w(value))?
            }
            None => 0,
        };
        Ok((deque, index))
    }

    #[crate::pyre_methods]
    impl W_DequeIter {
        #[staticmethod]
        fn __new__(_cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            let (deque, requested) = constructor_args(args)?;
            Ok(make(deque, requested))
        }

        fn __iter__(&self) -> PyObjectRef {
            self as *const W_DequeIter as PyObjectRef
        }

        fn __length_hint__(&self) -> i64 {
            self.counter.max(0)
        }

        fn __next__(&mut self) -> Result<PyObjectRef, crate::PyError> {
            if let Err(error) = checklock(self.deque, self.lock) {
                self.counter = 0;
                return Err(error);
            }
            if self.counter <= 0 {
                return Err(crate::PyError::stop_iteration());
            }
            self.counter -= 1;
            let item = block_get(self.block, self.index);
            self.index += 1;
            if self.index >= BLOCKLEN {
                self.block = block(self.block).rightlink;
                self.index = 0;
            }
            Ok(item)
        }

        fn __reduce__(&self) -> PyObjectRef {
            let self_obj = self as *const W_DequeIter as PyObjectRef;
            let _roots = pyre_object::gc_roots::push_roots();
            let self_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(self_obj);
            let ty =
                unsafe { instance_getclass(pyre_object::gc_roots::shadow_stack_get(self_slot)) };
            let deque = unsafe {
                (*(pyre_object::gc_roots::shadow_stack_get(self_slot) as *const W_DequeIter)).deque
            };
            let deque_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(deque);
            let counter = unsafe {
                (*(pyre_object::gc_roots::shadow_stack_get(self_slot) as *const W_DequeIter))
                    .counter
            };
            let consumed = W_Deque::from_obj(pyre_object::gc_roots::shadow_stack_get(deque_slot))
                .map(|deque| deque.len - counter)
                .unwrap_or(0);
            let consumed_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_int_new(consumed));
            let state = w_tuple_new(vec![
                pyre_object::gc_roots::shadow_stack_get(deque_slot),
                pyre_object::gc_roots::shadow_stack_get(consumed_slot),
            ]);
            let state_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(state);
            w_tuple_new(vec![
                ty,
                pyre_object::gc_roots::shadow_stack_get(state_slot),
            ])
        }
    }

    pub fn make(deque: PyObjectRef, requested: i64) -> PyObjectRef {
        let _ = type_object();
        // `W_DequeIter.__init__` receives an already-checked W_Deque and reads
        // these attributes directly; do not manufacture an Option/tuple
        // side-shape that has no counterpart in the RPython graph.
        let deque_ref = unsafe { &*(deque as *const W_Deque) };
        let len = deque_ref.len;
        let mut block_obj = deque_ref.leftblock;
        let mut index = deque_ref.leftindex;
        let mut counter = len;
        // `W_DequeIter._move_to`: a negative pickle index leaves the freshly
        // initialized iterator alone; a live position is located in one step.
        if requested >= 0 {
            counter = len - requested;
            if counter > 0 {
                (block_obj, index) = super::locate(deque, requested);
            }
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let deque = pyre_object::gc_roots::pin_root(deque);
        W_DequeIter::allocate_stable(W_DequeIter {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            deque,
            block: block_obj,
            index,
            counter,
            lock: getlock(deque),
        })
    }

    pub fn public_type() -> PyObjectRef {
        let ty = type_object();
        // `interp_deque.py`: W_DequeIter.typedef explicitly sets
        // acceptable_as_base_class=False.
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(ty, false) };
        ty
    }
}

// PyPy `W_DequeRevIter`, with the same state shape and reverse indexing into
// pyre's list-backed deque payload.
pub mod deque_rev_iter {
    use super::{BLOCKLEN, W_Deque, block, block_get, checklock, getlock, instance_getclass};
    use pyre_object::*;

    // `collections_exec` ADD_TYPEs dequereviter_spec as an immutable heap type.
    #[crate::pyre_class("_collections._deque_reverse_iterator", cpython_heaptype)]
    pub struct W_DequeRevIter {
        deque: PyObjectRef,
        block: PyObjectRef,
        index: i64,
        counter: i64,
        lock: PyObjectRef,
    }

    #[crate::pyre_methods]
    impl W_DequeRevIter {
        #[staticmethod]
        fn __new__(_cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            let positional = args.get(1..).unwrap_or(&[]);
            if positional.is_empty() || positional.len() > 2 {
                return Err(crate::PyError::type_error(format!(
                    "_deque_reverse_iterator expected 1 or 2 arguments, got {}",
                    positional.len()
                )));
            }
            let mut deque = positional[0];
            if W_Deque::from_obj(deque).is_none() {
                let name = crate::type_methods::clinic_arg_type_name(deque);
                return Err(crate::PyError::type_error(format!(
                    "must be collections.deque, not {name}"
                )));
            }
            let requested = match positional.get(1) {
                Some(&value) => {
                    pyre_object::with_roots!(deque => crate::builtins::space_index_w(value))?
                }
                None => 0,
            };
            Ok(make(deque, requested))
        }

        fn __iter__(&self) -> PyObjectRef {
            self as *const W_DequeRevIter as PyObjectRef
        }

        fn __length_hint__(&self) -> i64 {
            self.counter.max(0)
        }

        fn __next__(&mut self) -> Result<PyObjectRef, crate::PyError> {
            if let Err(error) = checklock(self.deque, self.lock) {
                self.counter = 0;
                return Err(error);
            }
            if self.counter <= 0 {
                return Err(crate::PyError::stop_iteration());
            }
            self.counter -= 1;
            let item = block_get(self.block, self.index);
            self.index -= 1;
            if self.index < 0 {
                self.block = block(self.block).leftlink;
                self.index = BLOCKLEN - 1;
            }
            Ok(item)
        }

        fn __reduce__(&self) -> PyObjectRef {
            let self_obj = self as *const W_DequeRevIter as PyObjectRef;
            let _roots = pyre_object::gc_roots::push_roots();
            let self_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(self_obj);
            let ty =
                unsafe { instance_getclass(pyre_object::gc_roots::shadow_stack_get(self_slot)) };
            let deque = unsafe {
                (*(pyre_object::gc_roots::shadow_stack_get(self_slot) as *const W_DequeRevIter))
                    .deque
            };
            let deque_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(deque);
            let counter = unsafe {
                (*(pyre_object::gc_roots::shadow_stack_get(self_slot) as *const W_DequeRevIter))
                    .counter
            };
            let consumed = W_Deque::from_obj(pyre_object::gc_roots::shadow_stack_get(deque_slot))
                .map(|deque| deque.len - counter)
                .unwrap_or(0);
            let consumed_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_int_new(consumed));
            let state = w_tuple_new(vec![
                pyre_object::gc_roots::shadow_stack_get(deque_slot),
                pyre_object::gc_roots::shadow_stack_get(consumed_slot),
            ]);
            let state_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(state);
            w_tuple_new(vec![
                ty,
                pyre_object::gc_roots::shadow_stack_get(state_slot),
            ])
        }
    }

    pub fn make(deque: PyObjectRef, requested: i64) -> PyObjectRef {
        let _ = type_object();
        // PyPy's reverse iterator likewise reads its checked deque directly.
        let deque_ref = unsafe { &*(deque as *const W_Deque) };
        let len = deque_ref.len;
        let mut block_obj = deque_ref.rightblock;
        let mut index = deque_ref.rightindex;
        let mut counter = len;
        // `W_DequeRevIter._move_to` locates from the left using the remaining
        // length, rather than maintaining a second reverse block walk.
        if requested >= 0 {
            counter = len - requested;
            if counter > 0 {
                (block_obj, index) = super::locate(deque, counter - 1);
            }
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let deque = pyre_object::gc_roots::pin_root(deque);
        W_DequeRevIter::allocate_stable(W_DequeRevIter {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            deque,
            block: block_obj,
            index,
            counter,
            lock: getlock(deque),
        })
    }

    pub fn public_type() -> PyObjectRef {
        let ty = type_object();
        // `interp_deque.py`: W_DequeRevIter.typedef explicitly sets
        // acceptable_as_base_class=False.
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(ty, false) };
        ty
    }
}

/// Append every element of `iterable` to `self_obj` through `append`.
///
/// `collect_iterable`'s own root scope has popped by the time it returns, so
/// the elements sit in a native Vec the collector does not walk while each
/// append allocates the next block.  Publish the whole run up front and read
/// the deque and each element back at the turn that consumes them.
fn extend_from_iterable(
    mut self_obj: PyObjectRef,
    iterable: PyObjectRef,
    is_extend_right: bool,
) -> Result<(), crate::PyError> {
    let items = pyre_object::with_roots!(self_obj => crate::builtins::collect_iterable(iterable))?;
    let _roots = pyre_object::gc_roots::push_roots();
    let deque_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(self_obj);
    let item_base = pyre_object::gc_roots::pin_roots(&items);
    for index in 0..items.len() {
        let deque = pyre_object::gc_roots::shadow_stack_get(deque_slot);
        let item = pyre_object::gc_roots::shadow_stack_get(item_base + index);
        // interp_deque.py:_extend keeps `is_extend_right` as the green
        // direction and performs the direct append/appendleft dispatch in
        // the loop.  Preserve that shape instead of passing a Rust function
        // pointer, which invents an indirect call that RPython never has.
        if is_extend_right {
            append_right(deque, item);
        } else {
            append_left(deque, item);
        }
    }
    Ok(())
}

/// `W_Deque.append` + `trimleft`, ported from `interp_deque.py`.
fn append_right(self_obj: PyObjectRef, item: PyObjectRef) {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(self_obj);
    let _ = pyre_object::gc_roots::pin_root(item);
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let _deque_guard = unsafe { w_deque_lock(self_obj) };
    let mut self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let item = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
    let mut ri = W_Deque::from_obj(self_obj).expect("deque").rightindex + 1;
    if ri >= BLOCKLEN {
        let old_right = W_Deque::from_obj(self_obj).expect("deque").rightblock;
        let old_right = pyre_object::gc_roots::pin_root(old_right);
        let new_right = deque_block::new(old_right, PY_NULL);
        self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
        let old_right = pyre_object::gc_roots::shadow_stack_get(root_base + 2);
        block(old_right).rightlink = new_right;
        W_Deque::from_obj(self_obj).expect("deque").rightblock = new_right;
        ri = 0;
    }
    let deque = W_Deque::from_obj(self_obj).expect("deque");
    deque.rightindex = ri;
    block_set(
        deque.rightblock,
        ri,
        pyre_object::gc_roots::shadow_stack_get(root_base + 1),
    );
    deque.len += 1;
    if maxlen_bound(self_obj).is_some_and(|maxlen| deque.len > maxlen as i64) {
        let _ = pop_left(self_obj).expect("trimleft on non-empty deque");
    }
    modified(self_obj);
}

fn append_left(self_obj: PyObjectRef, item: PyObjectRef) {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(self_obj);
    let _ = pyre_object::gc_roots::pin_root(item);
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let _deque_guard = unsafe { w_deque_lock(self_obj) };
    let mut self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let item = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
    let mut li = W_Deque::from_obj(self_obj).expect("deque").leftindex - 1;
    if li < 0 {
        let old_left = W_Deque::from_obj(self_obj).expect("deque").leftblock;
        let old_left = pyre_object::gc_roots::pin_root(old_left);
        let new_left = deque_block::new(PY_NULL, old_left);
        self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
        let old_left = pyre_object::gc_roots::shadow_stack_get(root_base + 2);
        block(old_left).leftlink = new_left;
        W_Deque::from_obj(self_obj).expect("deque").leftblock = new_left;
        li = BLOCKLEN - 1;
    }
    let deque = W_Deque::from_obj(self_obj).expect("deque");
    deque.leftindex = li;
    block_set(
        deque.leftblock,
        li,
        pyre_object::gc_roots::shadow_stack_get(root_base + 1),
    );
    deque.len += 1;
    if maxlen_bound(self_obj).is_some_and(|maxlen| deque.len > maxlen as i64) {
        let _ = pop_right(self_obj).expect("trimright on non-empty deque");
    }
    modified(self_obj);
}

fn pop_right(self_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let self_obj = pyre_object::gc_roots::pin_root(self_obj);
    let _deque_guard = unsafe { w_deque_lock(self_obj) };
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let deque = W_Deque::from_obj(self_obj).expect("deque");
    if deque.len == 0 {
        return Err(crate::PyError::index_error("pop from an empty deque"));
    }
    deque.len -= 1;
    let item = block_get(deque.rightblock, deque.rightindex);
    block_set(deque.rightblock, deque.rightindex, w_none());
    let mut ri = deque.rightindex - 1;
    if ri < 0 {
        if deque.len == 0 {
            deque.leftindex = CENTER + 1;
            ri = CENTER;
        } else {
            let old_right = deque.rightblock;
            let new_right = block(old_right).leftlink;
            deque.rightblock = new_right;
            block(new_right).rightlink = PY_NULL;
            ri = BLOCKLEN - 1;
        }
    }
    deque.rightindex = ri;
    modified(self_obj);
    Ok(item)
}

fn pop_left(self_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let self_obj = pyre_object::gc_roots::pin_root(self_obj);
    let _deque_guard = unsafe { w_deque_lock(self_obj) };
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let deque = W_Deque::from_obj(self_obj).expect("deque");
    if deque.len == 0 {
        return Err(crate::PyError::index_error("pop from an empty deque"));
    }
    deque.len -= 1;
    let item = block_get(deque.leftblock, deque.leftindex);
    block_set(deque.leftblock, deque.leftindex, w_none());
    let mut li = deque.leftindex + 1;
    if li >= BLOCKLEN {
        if deque.len == 0 {
            li = CENTER + 1;
            deque.rightindex = CENTER;
        } else {
            let old_left = deque.leftblock;
            let new_left = block(old_left).rightlink;
            deque.leftblock = new_left;
            block(new_left).leftlink = PY_NULL;
            li = 0;
        }
    }
    deque.leftindex = li;
    modified(self_obj);
    Ok(item)
}

/// `space.decode_index4` index-only case — reject a slice with a
/// `TypeError`, route any `__index__`-able object through
/// `getindex_w`, apply the negative-index wrap and the in-range
/// check, and return the resolved element position.
fn deque_index(index: PyObjectRef, mut self_obj: PyObjectRef) -> Result<usize, crate::PyError> {
    if unsafe { pyre_object::is_slice(index) } {
        return Err(crate::PyError::type_error("deque[:] is not supported"));
    }
    let mut idx = pyre_object::with_roots!(self_obj => crate::builtins::getindex_w(index))?;
    // PyPy `decode_index4(w_index, self)` bounds the converted index against
    // the deque as it stands *after* `__index__` ran.  Read the owner directly
    // at that point, rather than manufacturing a closure for the length read,
    // so the translated graph keeps PyPy's ordinary owner/value shape.
    let len = deque_len(self_obj);
    if idx < 0 {
        idx += len;
    }
    if idx < 0 || idx >= len {
        return Err(crate::PyError::index_error("index out of range"));
    }
    Ok(idx as usize)
}

/// Re-bound a converted index against the deque as it stands *under* the
/// stripe guard.
///
/// `deque_index` bounds against the length it read while `__index__` was still
/// running, so by the time the guard is held another thread may have shortened
/// the deque. `locate` would then walk past the end of the chain and
/// `Vec::remove` would panic instead of raising, so every indexed accessor
/// re-checks here before it addresses storage.
fn check_index_under_guard(self_obj: PyObjectRef, idx: i64) -> Result<(), crate::PyError> {
    if idx < 0 || idx >= deque_len(self_obj) {
        return Err(crate::PyError::index_error("index out of range"));
    }
    Ok(())
}

/// `W_Deque.locate`: choose the nearer endpoint and walk whole blocks.
fn locate(self_obj: PyObjectRef, mut index: i64) -> (PyObjectRef, i64) {
    let deque = W_Deque::from_obj(self_obj).expect("deque");
    if index < (deque.len >> 1) {
        index += deque.leftindex;
        let mut block_obj = deque.leftblock;
        while index >= BLOCKLEN {
            block_obj = block(block_obj).rightlink;
            index -= BLOCKLEN;
        }
        (block_obj, index)
    } else {
        index = index - deque.len + 1 + deque.rightindex;
        let mut block_obj = deque.rightblock;
        while index < 0 {
            block_obj = block(block_obj).leftlink;
            index += BLOCKLEN;
        }
        (block_obj, index)
    }
}

/// `W_Deque.compare` — element-wise (`compare_by_iteration`) ordering
/// against another deque, ignoring `maxlen`; `NotImplemented` when the
/// other operand is not a deque.  Delegates to list comparison over
/// snapshots of both backings.
fn deque_compare(
    mut self_obj: PyObjectRef,
    mut other: PyObjectRef,
    op: crate::baseobjspace::CompareOp,
) -> Result<PyObjectRef, crate::PyError> {
    if !pyre_object::with_roots!(other, self_obj => crate::baseobjspace::isinstance(other, type_object()))?
    {
        return Ok(pyre_object::w_not_implemented());
    }
    // `compare_by_iteration` walks both deques
    // through their iterators; each `W_DequeIter.next` checks the lock
    // before yielding, so a mutation is detected
    // up to — but not past — the element that decides the result.  pyre
    // snapshots, so check each deque's lock before consuming its element
    // and stop as soon as the result is determined.
    use crate::baseobjspace::CompareOp;
    // `getlock` allocates the lock token, `snapshot` takes the stripe and
    // `eq_w` runs Python: both deques, their locks and the first snapshot are
    // read back from their slots.
    let _roots = pyre_object::gc_roots::push_roots();
    let own_base = pyre_object::gc_roots::pin_roots(&[self_obj, other, PY_NULL, PY_NULL]);
    let lock_a = getlock(pyre_object::gc_roots::shadow_stack_get(own_base));
    pyre_object::gc_roots::shadow_stack_set(own_base + 2, lock_a);
    let lock_b = getlock(pyre_object::gc_roots::shadow_stack_get(own_base + 1));
    pyre_object::gc_roots::shadow_stack_set(own_base + 3, lock_b);
    let snap_a = snapshot(pyre_object::gc_roots::shadow_stack_get(own_base));
    let a_base = pyre_object::gc_roots::pin_roots(&snap_a);
    let snap_b = snapshot(pyre_object::gc_roots::shadow_stack_get(own_base + 1));
    let b_base = pyre_object::gc_roots::pin_roots(&snap_b);
    let mut i = 0usize;
    loop {
        // next(w_it1): lock-check precedes the element.
        checklock(
            pyre_object::gc_roots::shadow_stack_get(own_base),
            pyre_object::gc_roots::shadow_stack_get(own_base + 2),
        )?;
        let x1 = (i < snap_a.len()).then(|| pyre_object::gc_roots::shadow_stack_get(a_base + i));
        // next(w_it2): lock-check precedes the element.
        checklock(
            pyre_object::gc_roots::shadow_stack_get(own_base + 1),
            pyre_object::gc_roots::shadow_stack_get(own_base + 3),
        )?;
        let x2 = (i < snap_b.len()).then(|| pyre_object::gc_roots::shadow_stack_get(b_base + i));
        match (x1, x2) {
            (Some(_), Some(_)) => {
                let a = pyre_object::gc_roots::shadow_stack_get(a_base + i);
                let b = pyre_object::gc_roots::shadow_stack_get(b_base + i);
                if !crate::baseobjspace::eq_w(a, b)? {
                    // First differing pair decides the result; no further
                    // `next`, so no further lock check.  Reload after `eq_w`,
                    // which can collect and move both operands.
                    let a = pyre_object::gc_roots::shadow_stack_get(a_base + i);
                    let b = pyre_object::gc_roots::shadow_stack_get(b_base + i);
                    return match op {
                        CompareOp::Eq => Ok(pyre_object::w_bool_from(false)),
                        CompareOp::Ne => Ok(pyre_object::w_bool_from(true)),
                        _ => crate::baseobjspace::compare(a, b, op),
                    };
                }
            }
            // One or both deques exhausted — decide by length.
            _ => {
                let res = match op {
                    CompareOp::Eq => x1.is_none() && x2.is_none(),
                    CompareOp::Ne => !(x1.is_none() && x2.is_none()),
                    CompareOp::Lt => x2.is_some(),
                    CompareOp::Le => x1.is_none(),
                    CompareOp::Gt => x1.is_some(),
                    CompareOp::Ge => x2.is_none(),
                };
                return Ok(pyre_object::w_bool_from(res));
            }
        }
        i += 1;
    }
}

/// `W_Deque.mul` — repeat the elements `num` times, then re-bound by
/// `maxlen` by routing through the constructor (which trims).
pub(crate) fn deque_repeat(
    self_obj: PyObjectRef,
    n: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    // PyPy's W_Deque.mul receives `w_int` and starts with `space.int_w`.
    // CPython v3.14.6 routes the public sq_repeat slot through
    // abstract.c sequence_repeat first, so only __index__ is admitted and an
    // oversized result names an index-sized integer.  Keep PyPy's body below
    // after applying that observable gateway contract.
    let _roots = pyre_object::gc_roots::push_roots();
    let roots = pyre_object::gc_roots::pin_roots(&[self_obj, n]);
    let w_count =
        crate::baseobjspace::getindex_repeat(pyre_object::gc_roots::shadow_stack_get(roots + 1))?;
    let num = crate::baseobjspace::int_w(w_count)?.max(0) as usize;
    let self_obj = pyre_object::gc_roots::shadow_stack_get(roots);
    let base = snapshot(self_obj);
    // interp_deque.py W_Deque.mul: ovfcheck(self.len * num) raises
    // MemoryError.  `ovfcheck` guards a machine-signed multiplication, so the
    // edge is `isize` rather than the `usize` the accumulator counts in.
    if num > 0 && base.len() > (isize::MAX as usize) / num {
        return Err(crate::PyError::memory_error(""));
    }
    // `copied = W_Deque(space); copied.maxlen = self.maxlen`, then
    // `for _ in range(num): copied.extend(self)`.  Every `append` behind that
    // `extend` runs `trimleft`, so the copy never holds more than `maxlen`
    // items at any point; materialising the whole product and trimming once
    // at the end is what makes a bounded deque times a large count exhaust
    // memory instead of answering.
    let maxlen = maxlen_bound(self_obj);
    let total = base.len() * num;
    let mut items = Vec::new();
    items
        .try_reserve_exact(match maxlen {
            Some(m) => total.min(m.saturating_add(base.len())),
            None => total,
        })
        .map_err(|_| crate::PyError::memory_error(""))?;
    // An empty receiver copies nothing, so the count is not walked as a trip
    // count.
    if !base.is_empty() {
        for _ in 0..num {
            items.extend_from_slice(&base);
            if let Some(m) = maxlen
                && items.len() > m
            {
                items.drain(0..items.len() - m);
            }
        }
    }
    let items_base = pyre_object::gc_roots::publish_roots(&items);
    pyre_object::gc_roots::normalize_roots(items_base, items.len());
    let list = w_list_new(
        (0..items.len())
            .map(|i| pyre_object::gc_roots::shadow_stack_get(items_base + i))
            .collect(),
    );
    let list_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(list);
    let m = maxlen_obj(pyre_object::gc_roots::shadow_stack_get(roots));
    // `w_list_new` and `maxlen_obj` allocate; the type is read off the
    // rooted receiver after them.
    let ty = unsafe { instance_getclass(pyre_object::gc_roots::shadow_stack_get(roots)) };
    if unsafe { is_none(m) } {
        crate::call::call_function_impl_result(
            ty,
            &[pyre_object::gc_roots::shadow_stack_get(list_slot)],
        )
    } else {
        let _ = pyre_object::gc_roots::pin_root(m);
        crate::call::call_function_impl_result(
            ty,
            &[
                pyre_object::gc_roots::shadow_stack_get(list_slot),
                pyre_object::gc_roots::shadow_stack_get(list_slot + 1),
            ],
        )
    }
}

// [3.14-spec] PyPy `W_Deque.typedef` owns the older constructor-oriented doc;
// CPython 3.14 `_collectionsmodule.c` `deque_init__doc__` is the observable
// type doc.  Keep PyPy's TypeDef owner and project only that string.
#[crate::pyre_methods(
    doc = "A list-like sequence optimized for data accesses near its endpoints.",
    weakrefable,
    unhashable
)]
impl W_Deque {
    // `deque_new` allocates an empty unbounded payload; the construction
    // arguments (`iterable`, `maxlen`) are consumed by `__init__`, so they
    // are accepted and ignored here — the type-call protocol forwards them
    // (and any keywords) to `__new__` as well.
    #[staticmethod]
    fn __new__(cls: PyObjectRef, _args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // Construction arguments are consumed/validated by `__init__`; accept
        // (and ignore) any positional or keyword args here via the whole-slice
        // catch-all so a subclass with its own `__init__` keyword parameters
        // does not trip an unknown-keyword error in `__new__`.
        let _ = _args;
        // `objspace.py` `allocate_instance` validates the requested subtype.
        // The builtin is the base layout; a subclass is `typedef.py`
        // `_getusercls` (`interp_deque.py` `W_Deque.descr__new__`).
        // `check_user_subclass` can collect. The slot is the live word;
        // this pin's argument is not read again.
        let _roots = pyre_object::gc_roots::push_roots();
        let cls_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(cls);
        crate::typedef::check_user_subclass(
            type_object(),
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        )?;
        // `deque_block::new` allocates the block and its data list. The
        // requested class stays rooted across that allocation. The block
        // itself is `allocate_stable`, so the exact deque path can store it
        // without a reload; a subclass `allocate_instance` pins it.
        let initial_block = deque_block::new(PY_NULL, PY_NULL);
        Ok(W_Deque::allocate_instance(
            W_Deque {
                ob: pyre_object::PyObject {
                    ob_type: std::ptr::null(),
                    w_class: std::ptr::null_mut(),
                },
                leftblock: initial_block,
                rightblock: initial_block,
                leftindex: CENTER + 1,
                rightindex: CENTER,
                len: 0,
                maxlen: -1,
                lock: PY_NULL,
                lifeline: PY_NULL,
            },
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        ))
    }

    // interp_deque.py:619-620 — exact classmethod binding preserves a deque
    // subclass as the GenericAlias origin.
    #[classmethod]
    fn __class_getitem__(
        cls: PyObjectRef,
        item: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        crate::_pypy_generic_alias::generic_alias_class_getitem(&[cls, item])
    }

    // `init(iterable=None, maxlen=None)` — remember maxlen, then
    // extend so the bound is enforced while filling.
    fn __init__(
        &mut self,
        #[default(None)] iterable: Option<PyObjectRef>,
        #[default(None)] maxlen: Option<PyObjectRef>,
    ) -> Result<(), crate::PyError> {
        // `interp_deque.py init`: `gateway_nonnegint_w(w_maxlen)` after the
        // `is_none` arm.  Conversion goes through `gateway_int_w`.
        // [3.15-spec] a negative bound is `maxlen must be non-negative`
        // (`_collectionsmodule.c deque_init`), not gateway_nonnegint_w's
        // `expected a non-negative integer`.
        let mut self_obj = self as *mut W_Deque as PyObjectRef;
        let has_iterable = iterable.is_some();
        let mut iterable_obj = iterable.unwrap_or(PY_NULL);
        let mut maxlen_obj = maxlen.unwrap_or(PY_NULL);
        let maxlen_val = if maxlen_obj.is_null() || unsafe { is_none(maxlen_obj) } {
            -1
        } else {
            let v = pyre_object::with_roots!(self_obj, iterable_obj, maxlen_obj => {
                crate::baseobjspace::gateway_int_w(maxlen_obj)
            })?;
            if v < 0 {
                return Err(crate::PyError::value_error("maxlen must be non-negative"));
            }
            v
        };
        W_Deque::from_obj(self_obj).expect("deque").maxlen = maxlen_val;
        // Reinitialize by clearing, but only when non-empty (`init`:
        // `if self.len > 0: self.clear()`). `store` bumps the iteration lock
        // (`modified`) so an outstanding scan of an existing deque detects the
        // reset; an already-empty deque leaves `lock` untouched, so re-init
        // while iterating an empty deque yields StopIteration, not a mutation
        // RuntimeError.
        if deque_len(self_obj) > 0 {
            pyre_object::with_roots!(self_obj, iterable_obj => clear_blocks(self_obj));
        }
        if has_iterable {
            extend_from_iterable(self_obj, iterable_obj, true)?;
        }
        Ok(())
    }
    fn append(&mut self, item: PyObjectRef) {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        append_right(self_obj, item);
    }
    fn appendleft(&mut self, item: PyObjectRef) {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        append_left(self_obj, item);
    }
    fn pop(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        pop_right(self_obj)
    }
    fn popleft(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        pop_left(self_obj)
    }
    fn clear(&mut self) {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        clear_blocks(self_obj);
    }
    fn extend(&mut self, iterable: PyObjectRef) -> Result<(), crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        extend_from_iterable(self_obj, iterable, true)
    }
    fn extendleft(&mut self, iterable: PyObjectRef) -> Result<(), crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        // Each element is appended on the left, so the result is
        // the reverse of `iterable`.
        extend_from_iterable(self_obj, iterable, false)
    }
    fn count(&mut self, x: PyObjectRef) -> Result<i64, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        let (own_base, items_base, items_len) = pin_find_or_count(self_obj, x);
        let x_slot = own_base + 2;
        let mut n = 0i64;
        for i in 0..items_len {
            let equal = crate::baseobjspace::eq_w(
                pyre_object::gc_roots::shadow_stack_get(items_base + i),
                pyre_object::gc_roots::shadow_stack_get(x_slot),
            )?;
            checklock(
                pyre_object::gc_roots::shadow_stack_get(own_base),
                pyre_object::gc_roots::shadow_stack_get(own_base + 1),
            )?;
            if equal {
                n += 1;
            }
        }
        Ok(n)
    }
    fn remove(&mut self, x: PyObjectRef) -> Result<(), crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        let (own_base, items_base, items_len) = pin_find_or_count(self_obj, x);
        let x_slot = own_base + 2;
        let mut pos = None;
        for i in 0..items_len {
            let equal = crate::baseobjspace::eq_w(
                pyre_object::gc_roots::shadow_stack_get(items_base + i),
                pyre_object::gc_roots::shadow_stack_get(x_slot),
            )?;
            // [3.15-spec] `deque_remove_impl` raises IndexError
            // ("deque mutated during iteration") at both mutation
            // checks. `interp_deque.py checklock` / `_find_or_count`
            // raise RuntimeError. `find_jmp` greens are `is_find` and
            // `tp`; they do not govern the exception type.
            // lib-python/3/test/test_deque.py `test_remove` accepts both.
            if !lock_valid(
                pyre_object::gc_roots::shadow_stack_get(own_base),
                pyre_object::gc_roots::shadow_stack_get(own_base + 1),
            ) {
                return Err(crate::PyError::index_error(
                    "deque mutated during iteration",
                ));
            }
            if equal {
                pos = Some(i);
                break;
            }
        }
        match pos {
            Some(pos) => {
                // The guard cannot span the loop above: `eq_w` runs arbitrary
                // Python, and holding a stripe across it would let user code
                // reach a second deque and invert the acquisition order. It
                // does cover the write-back, and the epoch is re-read under it
                // so a mutation that landed after the last comparison is
                // reported rather than silently overwritten by the snapshot.
                let _deque_guard =
                    unsafe { w_deque_lock(pyre_object::gc_roots::shadow_stack_get(own_base)) };
                let self_obj = pyre_object::gc_roots::shadow_stack_get(own_base);
                let lock = pyre_object::gc_roots::shadow_stack_get(own_base + 1);
                if !lock_valid(self_obj, lock) {
                    return Err(crate::PyError::index_error(
                        "deque mutated during iteration",
                    ));
                }
                let mut items: Vec<PyObjectRef> = (0..items_len)
                    .map(|i| pyre_object::gc_roots::shadow_stack_get(items_base + i))
                    .collect();
                items.remove(pos);
                store(self_obj, items);
                Ok(())
            }
            None => Err(crate::PyError::value_error(
                "deque.remove(x): x not in deque",
            )),
        }
    }
    fn __contains__(&mut self, x: PyObjectRef) -> Result<bool, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        let (own_base, items_base, items_len) = pin_find_or_count(self_obj, x);
        let x_slot = own_base + 2;
        for i in 0..items_len {
            let equal = crate::baseobjspace::eq_w(
                pyre_object::gc_roots::shadow_stack_get(items_base + i),
                pyre_object::gc_roots::shadow_stack_get(x_slot),
            )?;
            checklock(
                pyre_object::gc_roots::shadow_stack_get(own_base),
                pyre_object::gc_roots::shadow_stack_get(own_base + 1),
            )?;
            if equal {
                return Ok(true);
            }
        }
        Ok(false)
    }
    fn reverse(&mut self) {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        // `interp_deque.py reverse`: swap entries from both endpoints without
        // invalidating iterators. No Python runs inside the swap loop, so the
        // stripe covers the whole transition.
        let _deque_guard = unsafe { w_deque_lock(self_obj) };
        let len = deque_len(self_obj);
        for index in 0..(len >> 1) {
            let (left_block, left_index) = locate(self_obj, index);
            let (right_block, right_index) = locate(self_obj, len - index - 1);
            let left = block_get(left_block, left_index);
            let right = block_get(right_block, right_index);
            block_set(left_block, left_index, right);
            block_set(right_block, right_index, left);
        }
    }
    fn rotate(&mut self, n: Option<PyObjectRef>) -> Result<(), crate::PyError> {
        let mut self_obj = self as *mut W_Deque as PyObjectRef;
        // Rotate right by n (negative rotates left).  The count goes
        // through `__index__` so non-integers raise `TypeError`.
        let n = match n {
            Some(v) => pyre_object::with_roots!(self_obj => crate::builtins::getindex_w(v))?,
            None => 1,
        };
        // Read-modify-write: `snapshot` and `store` are individually atomic,
        // and the stripe held across both is what stops a concurrent mutation
        // in between from being overwritten by the stale snapshot. No Python
        // runs from here on.
        let _deque_guard = unsafe { w_deque_lock(self_obj) };
        let mut items = snapshot(self_obj);
        let len = items.len() as i64;
        if len <= 1 {
            return Ok(());
        }
        let shift = ((n % len) + len) % len;
        if shift != 0 {
            items.rotate_right(shift as usize);
            store(self_obj, items);
        }
        Ok(())
    }
    fn insert(&mut self, i: PyObjectRef, mut x: PyObjectRef) -> Result<(), crate::PyError> {
        let mut self_obj = self as *mut W_Deque as PyObjectRef;
        // `W_Deque.insert(i, x)` — the index goes through `__index__` (converted
        // before the deque is inspected), then a bounded deque that is already
        // full raises before the element is placed.
        let index = pyre_object::with_roots!(self_obj, x => crate::builtins::getindex_w(i))?;
        // See `rotate`: the snapshot/store pair is the read-modify-write that
        // the stripe has to cover as one.
        let _deque_guard = unsafe { w_deque_lock(self_obj) };
        let mut items = snapshot(self_obj);
        if maxlen_bound(self_obj).is_some_and(|m| items.len() >= m) {
            return Err(crate::PyError::index_error(
                "deque already at its maximum size",
            ));
        }
        let len = items.len() as i64;
        // list.insert clamping: a negative index counts from the end and an
        // out-of-range index clamps to the near edge.
        let pos = if index < 0 {
            (index + len).max(0)
        } else {
            index.min(len)
        } as usize;
        items.insert(pos, x);
        store(self_obj, items);
        Ok(())
    }
    fn index(
        &mut self,
        x: PyObjectRef,
        start: Option<PyObjectRef>,
        stop: Option<PyObjectRef>,
    ) -> Result<i64, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        // Everything live is published before `snapshot`: taking the deque
        // lock may block, and a blocked thread is a safepoint.
        let has_start = start.is_some();
        let has_stop = stop.is_some();
        let _roots = pyre_object::gc_roots::push_roots();
        let own_base = pyre_object::gc_roots::pin_roots(&[
            self_obj,
            PY_NULL,
            start.unwrap_or(PY_NULL),
            stop.unwrap_or(PY_NULL),
            x,
        ]);
        let x_slot = own_base + 4;
        let items = snapshot(pyre_object::gc_roots::shadow_stack_get(own_base));
        let items_len = items.len();
        let items_base = pyre_object::gc_roots::pin_roots(&items);
        let len = items_len as i64;
        // `space.iter(self)` takes the lock before `unwrap_start_stop`,
        // so a `__index__` on start/stop that mutates the deque is caught
        // by the first `checklock`.
        let lock = getlock(pyre_object::gc_roots::shadow_stack_get(own_base));
        pyre_object::gc_roots::shadow_stack_set(own_base + 1, lock);
        let clamp = |i: i64| if i < 0 { (i + len).max(0) } else { i.min(len) };
        // `start` and `stop` are read back from their slots: `getindex_w`
        // on the first may run `__index__` and collect.
        let start = clamp(if has_start {
            crate::builtins::getindex_w(pyre_object::gc_roots::shadow_stack_get(own_base + 2))?
        } else {
            0
        });
        let stop = clamp(if has_stop {
            crate::builtins::getindex_w(pyre_object::gc_roots::shadow_stack_get(own_base + 3))?
        } else {
            len
        });
        let upper = stop.min(len);
        let mut i = 0i64;
        while i < upper {
            // `space.next(w_iter)` checks the lock before each element.
            checklock(
                pyre_object::gc_roots::shadow_stack_get(own_base),
                pyre_object::gc_roots::shadow_stack_get(own_base + 1),
            )?;
            if i >= start {
                if crate::baseobjspace::eq_w(
                    pyre_object::gc_roots::shadow_stack_get(items_base + i as usize),
                    pyre_object::gc_roots::shadow_stack_get(x_slot),
                )? {
                    // Match returns immediately, before the post-match
                    // `checklock`.
                    return Ok(i);
                }
                // Re-check the lock after a non-match.
                checklock(
                    pyre_object::gc_roots::shadow_stack_get(own_base),
                    pyre_object::gc_roots::shadow_stack_get(own_base + 1),
                )?;
            }
            i += 1;
        }
        Err(crate::PyError::value_error(crate::display::wtf8_format!(
            unsafe {
                crate::display::py_repr_wtf8(pyre_object::gc_roots::shadow_stack_get(x_slot))?
            },
            " is not in deque"
        )))
    }
    fn copy(&self) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        // `type(self)(self)` or `type(self)(self, maxlen)`.
        let ty = unsafe { instance_getclass(self_obj) };
        let m = maxlen_obj(self_obj);
        if unsafe { is_none(m) } {
            crate::call::call_function_impl_result(ty, &[self_obj])
        } else {
            crate::call::call_function_impl_result(ty, &[self_obj, m])
        }
    }
    fn __copy__(&self) -> Result<PyObjectRef, crate::PyError> {
        // interp_deque.py typedef: `__copy__ = interp2app(W_Deque.copy)`.
        self.copy()
    }
    fn __reduce__(&self) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        // `interp_deque.py W_Deque.reduce` / `deque___reduce___impl`:
        // `__getstate__` (method lookup), then `iter(self)`, then the
        // `((), maxlen)` / `()` argument tuple.  `args` is `((), maxlen)`
        // when the deque is bounded so the bound survives the round-trip.
        // Items ride the 4th-element iterator.
        //
        // `call_method` / `iter` run Python; the deque and every nursery
        // tuple built here have to sit on one open set.
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(self_obj);
        let self_obj = || pyre_object::gc_roots::shadow_stack_get(self_slot);
        let ty_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(unsafe { instance_getclass(self_obj()) });
        let state = crate::baseobjspace::call_method_result(self_obj(), "__getstate__", &[])?;
        let state_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(state);
        // `W_Deque.reduce` calls `space.iter(self)`, preserving subclass
        // `__iter__` overrides (including an override that raises).
        let items = crate::baseobjspace::iter(self_obj())?;
        let items_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(items);
        let m = maxlen_obj(self_obj());
        let args_slot;
        if unsafe { is_none(m) } {
            args_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_tuple_new(vec![]));
        } else {
            let empty_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_tuple_new(vec![]));
            let m_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(m);
            let args = w_tuple_new(vec![
                pyre_object::gc_roots::shadow_stack_get(empty_slot),
                pyre_object::gc_roots::shadow_stack_get(m_slot),
            ]);
            args_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(args);
        }
        Ok(w_tuple_new(vec![
            pyre_object::gc_roots::shadow_stack_get(ty_slot),
            pyre_object::gc_roots::shadow_stack_get(args_slot),
            pyre_object::gc_roots::shadow_stack_get(state_slot),
            pyre_object::gc_roots::shadow_stack_get(items_slot),
        ]))
    }
    fn __len__(&self) -> i64 {
        let self_obj = self as *const W_Deque as PyObjectRef;
        deque_len(self_obj)
    }
    fn __iter__(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        Ok(deque_iter::make(self_obj, 0))
    }
    fn __reversed__(&mut self) -> PyObjectRef {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        deque_rev_iter::make(self_obj, 0)
    }
    fn __getitem__(&self, index: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let mut self_obj = self as *const W_Deque as PyObjectRef;
        let idx = pyre_object::with_roots!(self_obj => deque_index(index, self_obj))? as i64;
        // `locate` walks the chain and the caller then reads through the block
        // it returns, so the pair has to exclude a concurrent `store`. The
        // index conversion runs Python and stays outside.
        let _deque_guard = unsafe { w_deque_lock(self_obj) };
        check_index_under_guard(self_obj, idx)?;
        let (block_obj, block_index) = locate(self_obj, idx);
        Ok(block_get(block_obj, block_index))
    }
    fn __setitem__(
        &mut self,
        index: PyObjectRef,
        mut value: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        let mut self_obj = self as *mut W_Deque as PyObjectRef;
        let idx = pyre_object::with_roots!(self_obj, value => deque_index(index, self_obj))? as i64;
        // interp_deque.py W_Deque.setitem replaces the block entry in place
        // and deliberately does not call `modified()`: deque iterators remain
        // valid and observe subsequently assigned elements (including after
        // pickle reconstruction).
        let _deque_guard = unsafe { w_deque_lock(self_obj) };
        check_index_under_guard(self_obj, idx)?;
        let (block_obj, block_index) = locate(self_obj, idx);
        block_set(block_obj, block_index, value);
        Ok(())
    }
    fn __delitem__(&mut self, index: PyObjectRef) -> Result<(), crate::PyError> {
        let mut self_obj = self as *mut W_Deque as PyObjectRef;
        let idx = pyre_object::with_roots!(self_obj => deque_index(index, self_obj))?;
        // See `rotate`: the snapshot/store pair is the read-modify-write that
        // the stripe has to cover as one.
        let _deque_guard = unsafe { w_deque_lock(self_obj) };
        check_index_under_guard(self_obj, idx as i64)?;
        let mut items = snapshot(self_obj);
        items.remove(idx);
        store(self_obj, items);
        Ok(())
    }
    fn __eq__(&mut self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Eq)
    }
    fn __ne__(&mut self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Ne)
    }
    fn __lt__(&mut self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Lt)
    }
    fn __le__(&mut self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Le)
    }
    fn __gt__(&mut self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Gt)
    }
    fn __ge__(&mut self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Ge)
    }
    fn __add__(&self, mut other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        if W_Deque::from_obj(other).is_none() {
            return Err(crate::PyError::type_error(format!(
                "can only concatenate deque (not \"{}\") to deque",
                crate::error::type_name_of(other),
            )));
        }

        // CPython 3.14 deque_concat copies the left operand, preserving its
        // concrete subtype, then extends that copy with the right deque.
        let copy = pyre_object::with_roots!(other => self.copy())?;
        if W_Deque::from_obj(copy).is_none() {
            return Err(crate::PyError::type_error(
                "deque.__add__ returned a non-deque",
            ));
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let root_base = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(copy);
        for item in snapshot(other) {
            append_right(pyre_object::gc_roots::shadow_stack_get(root_base), item);
        }
        Ok(pyre_object::gc_roots::shadow_stack_get(root_base))
    }
    fn __iadd__(&mut self, iterable: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        // interp_deque.py W_Deque.iadd: extend in place and return self.
        let mut self_obj = self as *mut W_Deque as PyObjectRef;
        pyre_object::with_roots!(self_obj => self.extend(iterable))?;
        Ok(self_obj)
    }
    fn __mul__(&self, n: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        deque_repeat(self_obj, n)
    }
    fn __rmul__(&self, n: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        deque_repeat(self_obj, n)
    }
    fn __imul__(&mut self, n: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        // `W_Deque.imul` — empty or *1 is self; *<=0 clears; else
        // repeat in place, trimmed by maxlen.
        // As in `deque_repeat`, the public descriptor applies CPython 3.14's
        // sq_repeat __index__ contract before the PyPy W_Deque.imul body.
        let _roots = pyre_object::gc_roots::push_roots();
        let roots = pyre_object::gc_roots::pin_roots(&[self_obj, n]);
        let w_count = crate::baseobjspace::getindex_repeat(
            pyre_object::gc_roots::shadow_stack_get(roots + 1),
        )?;
        let num = crate::baseobjspace::int_w(w_count)?;
        let self_obj = pyre_object::gc_roots::shadow_stack_get(roots);
        // Nothing between the snapshot and the write-back runs Python, so the
        // whole read-modify-write fits under one stripe — without it a
        // concurrent `append` lands between them and is dropped by `store`.
        let _deque_guard = unsafe { w_deque_lock(self_obj) };
        let base = snapshot(self_obj);
        if base.is_empty() || num == 1 {
            return Ok(self_obj);
        }
        if num <= 0 {
            store(self_obj, vec![]);
            return Ok(self_obj);
        }
        // interp_deque.py W_Deque.imul: ovfcheck(self.len * num), whose
        // machine-signed multiplication puts the edge at `isize`.
        let repeat = num as usize;
        if base.len() > (isize::MAX as usize) / repeat {
            return Err(crate::PyError::memory_error(""));
        }
        // `copy` is a `maxlen`-bounded copy of self and `self.extend(copy)`
        // runs `num - 1` times.  self is already trimmed, and every `append`
        // behind `extend` trims again, so it never holds more than `maxlen`
        // items — the product is never materialised.
        let maxlen = maxlen_bound(self_obj);
        let total = base.len() * repeat;
        let mut items = Vec::new();
        items
            .try_reserve_exact(match maxlen {
                Some(m) => total.min(m.saturating_add(base.len())),
                None => total,
            })
            .map_err(|_| crate::PyError::memory_error(""))?;
        items.extend_from_slice(&base);
        for _ in 0..repeat - 1 {
            items.extend_from_slice(&base);
            if let Some(m) = maxlen
                && items.len() > m
            {
                items.drain(0..items.len() - m);
            }
        }
        store(self_obj, items);
        Ok(self_obj)
    }
    fn __repr__(&self) -> Result<PyObjectRef, crate::PyError> {
        let mut self_obj = self as *const W_Deque as PyObjectRef;
        // `dequerepr` — a deque reachable from its own items renders
        // the inner reference as `[...]` instead of recursing.
        let Some(_guard) = crate::display::ReprGuard::enter(self_obj) else {
            return Ok(pyre_object::w_str_new_managed("[...]"));
        };
        // The repr uses the short class name, so strip any dotted
        // module prefix from the builtin tp_name (`collections.deque`
        // → `deque`); a user subclass name has no dot.
        let full = unsafe { w_type_get_name(instance_getclass(self_obj)) };
        let name = full.rsplit('.').next().unwrap_or(full);
        let mut out = rustpython_wtf8::Wtf8Buf::new();
        out.push_str(name);
        out.push_str("([");
        for (i, item) in snapshot(self_obj).into_iter().enumerate() {
            if i != 0 {
                out.push_str(", ");
            }
            out.push_wtf8(
                &pyre_object::with_roots!(self_obj => unsafe { crate::py_repr_wtf8(item) })?,
            );
        }
        match maxlen_bound(self_obj) {
            Some(m) => out.push_str(&format!("], maxlen={m})")),
            None => out.push_str("])"),
        }
        Ok(pyre_object::w_str_from_wtf8_managed(out))
    }

    #[getter]
    fn maxlen(&self) -> PyObjectRef {
        let self_obj = self as *const W_Deque as PyObjectRef;
        maxlen_obj(self_obj)
    }
}

crate::py_module! {
    "_collections",
    interpleveldefs: {
        "deque"                   => {
            let w_deque_type = type_object();
            // [3.15-spec] `_collectionsmodule.c` `deque_spec` has
            // `Py_TPFLAGS_SEQUENCE`; PyPy's deque starts "X" and `MATCH_SEQUENCE`
            // falls back to `issequence_w`.  Evidence:
            // `parity_tests/abc_weak_cache_membership.py`
            // `collection_markers_follow_the_type_flags` passes on 3.15.2t and
            // fails on pypy3.  The `flag_patma_collection?` hint is untouched:
            // this store runs once when the module builds the type, before any
            // trace can read the field.
            unsafe {
                pyre_object::typeobject::w_type_set_flag_patma_collection(w_deque_type, b'S');
            }
            w_deque_type
        },
        "_deque_iterator"         => deque_iter::public_type(),
        "_deque_reverse_iterator" => deque_rev_iter::public_type(),
    },
    // `defaultdict` and `OrderedDict` are app-level `dict` subclasses —
    // see the module header and `app_defaultdict.py` / `app_odict.py`
    // (PyPy `app_defaultdict.defaultdict` / `app_odict.OrderedDict`).
    appleveldefs: {
        "app_defaultdict.py" => ["defaultdict"],
        "app_odict.py" => ["OrderedDict"],
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyre_object::PY_NULL;

    #[test]
    fn deque_subclass_instance_uses_user_typeptr() {
        // A non-null subtype that is not the builtin type object takes
        // `typedef.py` `_getusercls`. No live VM: the finalizer hook is unset.
        static SUB: pyre_object::PyObject = pyre_object::PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        };
        let sub = &SUB as *const pyre_object::PyObject as PyObjectRef;
        let block = deque_block::new(PY_NULL, PY_NULL);
        let obj = W_Deque::allocate_instance(
            W_Deque {
                ob: pyre_object::PyObject {
                    ob_type: std::ptr::null(),
                    w_class: std::ptr::null_mut(),
                },
                leftblock: block,
                rightblock: block,
                leftindex: super::CENTER + 1,
                rightindex: super::CENTER,
                len: 0,
                maxlen: -1,
                lock: PY_NULL,
                lifeline: PY_NULL,
            },
            sub,
        );
        unsafe {
            assert!(std::ptr::eq((*obj).ob_type, &DEQUE_USER_TYPE));
            assert!(crate::objspace::std::mapdict::has_mapdict_layout(obj));
        }
        let exact_block = deque_block::new(PY_NULL, PY_NULL);
        let exact = W_Deque::allocate_instance(
            W_Deque {
                ob: pyre_object::PyObject {
                    ob_type: std::ptr::null(),
                    w_class: std::ptr::null_mut(),
                },
                leftblock: exact_block,
                rightblock: exact_block,
                leftindex: super::CENTER + 1,
                rightindex: super::CENTER,
                len: 0,
                maxlen: -1,
                lock: PY_NULL,
                lifeline: PY_NULL,
            },
            PY_NULL,
        );
        unsafe {
            assert!(std::ptr::eq((*exact).ob_type, &DEQUE_TYPE));
            assert!(!crate::objspace::std::mapdict::has_mapdict_layout(exact));
        }
    }

    #[test]
    fn weakref_lifeline_roundtrip_set_type_and_deque_subclass() {
        crate::typedef::init_typeobjects();
        let set_obj = pyre_object::w_set_new();
        let type_obj = pyre_object::w_type_new("LifelineRoundTrip", PY_NULL, std::ptr::null_mut());
        let w_class = pyre_object::w_type_new("DequeSubLifeline", PY_NULL, std::ptr::null_mut());
        unsafe { pyre_object::w_type_set_weakrefable(w_class, true) };
        let block = deque_block::new(PY_NULL, PY_NULL);
        let deque_obj = W_Deque::allocate_instance(
            W_Deque {
                ob: pyre_object::PyObject {
                    ob_type: std::ptr::null(),
                    w_class: std::ptr::null_mut(),
                },
                leftblock: block,
                rightblock: block,
                leftindex: super::CENTER + 1,
                rightindex: super::CENTER,
                len: 0,
                maxlen: -1,
                lock: PY_NULL,
                lifeline: PY_NULL,
            },
            w_class,
        );
        for obj in [set_obj, type_obj, deque_obj] {
            unsafe {
                assert_ne!((*(*obj).ob_type).weakref_offset, 0);
            }
            let lifeline = pyre_object::w_instance_new(PY_NULL);
            assert!(crate::baseobjspace::getweakref(obj).is_none());
            crate::baseobjspace::setweakref(obj, lifeline).unwrap();
            assert_eq!(crate::baseobjspace::getweakref(obj), Some(lifeline));
            crate::baseobjspace::delweakref(obj);
            assert!(crate::baseobjspace::getweakref(obj).is_none());
        }
    }
}
