//! RPython `rpython/tool/flattenrec.py` (32 LOC) — queueing helper
//! that flattens deeply-recursive algorithms by deferring nested
//! invocations.
//!
//! ```python
//! class FlattenRecursion(TlsClass):
//!     def __init__(self):
//!         self.later = None
//!
//!     def __call__(self, func, *args, **kwds):
//!         """Call func(*args, **kwds), either now, or, if we're
//!         recursing, then the call will be done later by the first
//!         level.
//!         """
//!         if self.later is not None:
//!             self.later.append((func, args, kwds))
//!         else:
//!             self.later = lst = []
//!             try:
//!                 func(*args, **kwds)
//!                 for func, args, kwds in lst:
//!                     func(*args, **kwds)
//!             finally:
//!                 self.later = None
//! ```
//!
//! The Rust port keeps the same semantics: if a call nests inside an
//! outer [`Self::call`], the callable is queued; when the outer call
//! returns, the queue is drained. If the top-level call returns an
//! error, the queued callables are discarded (matching upstream's
//! exception-unwinds-the-try behaviour).
//!
//! `TlsClass` (thread-local storage) is collapsed into per-instance
//! `RefCell` state: the single annotator thread makes TLS-vs-instance
//! storage observationally identical, and RefCell provides the same
//! interior-mutability surface as Python's `self.later =` assignments.

use std::cell::RefCell;
use std::collections::VecDeque;

/// Callable queued by [`FlattenRecursion::call`]. Returning an error
/// short-circuits the drain loop and propagates the error back to the
/// outermost caller — matches upstream's `raise` through the
/// `try: ... finally:` block.
pub type DeferredCall<E> = Box<dyn FnOnce() -> Result<(), E>>;

/// RPython `class FlattenRecursion(TlsClass)` (tool/flattenrec.py).
pub struct FlattenRecursion<E> {
    /// RPython `self.later: list | None` — `None` when not currently
    /// draining; `Some(queue)` while an outer call holds the queue.
    later: RefCell<Option<VecDeque<DeferredCall<E>>>>,
}

impl<E> FlattenRecursion<E> {
    /// RPython `FlattenRecursion.__init__(self): self.later = None`
    /// (flattenrec.py).
    pub fn new() -> Self {
        FlattenRecursion {
            later: RefCell::new(None),
        }
    }

    /// RPython `FlattenRecursion.__call__(self, func, *args, **kwds)`
    /// (flattenrec.py).
    pub fn call(&self, func: DeferredCall<E>) -> Result<(), E> {
        // upstream: `if self.later is not None: self.later.append(...)`.
        let is_outer = self.later.borrow().is_none();
        if !is_outer {
            self.later.borrow_mut().as_mut().unwrap().push_back(func);
            return Ok(());
        }
        // upstream: `self.later = lst = []`.
        *self.later.borrow_mut() = Some(VecDeque::new());
        // flattenrec.py:FlattenRecursion.__call__ resets `later` in finally.
        // Rust unwinding must discard pending work as well as Result::Err.
        struct ResetLater<'a, E>(&'a FlattenRecursion<E>);
        impl<E> Drop for ResetLater<'_, E> {
            fn drop(&mut self) {
                *self.0.later.borrow_mut() = None;
            }
        }
        let _reset_later = ResetLater(self);
        // upstream: `try: func(); for func, args, kwds in lst: func(...)`.
        // Python's for-loop over `lst` re-reads `len(lst)` each step, so
        // appends *during* the drain (via nested `call()`) are visited
        // after the existing head. VecDeque::pop_front matches that
        // FIFO order in O(1).

        (|| -> Result<(), E> {
            func()?;
            loop {
                let next = self
                    .later
                    .borrow_mut()
                    .as_mut()
                    .expect("later invariant")
                    .pop_front();
                match next {
                    Some(f) => f()?,
                    None => break,
                }
            }
            Ok(())
        })()
    }
}

impl<E> Default for FlattenRecursion<E> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    #[derive(Debug, PartialEq, Eq)]
    struct TestError(String);

    #[test]
    fn call_runs_immediately_when_not_nested() {
        let flat: FlattenRecursion<TestError> = FlattenRecursion::new();
        let fired = Rc::new(Cell::new(false));
        let fired_cl = fired.clone();
        flat.call(Box::new(move || {
            fired_cl.set(true);
            Ok(())
        }))
        .unwrap();
        assert!(fired.get());
    }

    #[test]
    fn nested_calls_are_queued_and_drained_after_outer_returns() {
        // upstream parity: an inner call during the outer func body
        // appends to `later` and does NOT run until the outer body
        // returns. We observe the ordering via the shared counter.
        let flat: Rc<FlattenRecursion<TestError>> = Rc::new(FlattenRecursion::new());
        let order = Rc::new(RefCell::new(Vec::<&'static str>::new()));

        let flat_cl = flat.clone();
        let order_cl = order.clone();
        flat.call(Box::new(move || {
            order_cl.borrow_mut().push("outer-start");
            // Nested call inside the outer body — must queue.
            let order_cl2 = order_cl.clone();
            flat_cl
                .call(Box::new(move || {
                    order_cl2.borrow_mut().push("inner");
                    Ok(())
                }))
                .unwrap();
            order_cl.borrow_mut().push("outer-end");
            Ok(())
        }))
        .unwrap();

        // Inner runs AFTER the outer body finishes.
        assert_eq!(*order.borrow(), vec!["outer-start", "outer-end", "inner"]);
    }

    #[test]
    fn error_from_outer_skips_queued_work_and_unwinds() {
        let flat: Rc<FlattenRecursion<TestError>> = Rc::new(FlattenRecursion::new());
        let queued_ran = Rc::new(Cell::new(false));

        let flat_cl = flat.clone();
        let queued_ran_cl = queued_ran.clone();
        let result = flat.call(Box::new(move || {
            // Queue a nested call, then error out.
            let queued_ran_cl2 = queued_ran_cl.clone();
            flat_cl
                .call(Box::new(move || {
                    queued_ran_cl2.set(true);
                    Ok(())
                }))
                .unwrap();
            Err(TestError("boom".to_string()))
        }));

        assert_eq!(result, Err(TestError("boom".to_string())));
        assert!(
            !queued_ran.get(),
            "queued work must be discarded after outer error"
        );
    }

    #[test]
    fn multiple_nested_calls_drain_in_fifo_order() {
        // upstream parity: `for func, args, kwds in lst: func(...)`
        // iterates in insertion order. Appending A then B during the
        // outer body must execute A before B.
        let flat: Rc<FlattenRecursion<TestError>> = Rc::new(FlattenRecursion::new());
        let order = Rc::new(RefCell::new(Vec::<&'static str>::new()));

        let flat_cl = flat.clone();
        let order_cl = order.clone();
        flat.call(Box::new(move || {
            order_cl.borrow_mut().push("outer");
            let order_a = order_cl.clone();
            flat_cl
                .call(Box::new(move || {
                    order_a.borrow_mut().push("A");
                    Ok(())
                }))
                .unwrap();
            let order_b = order_cl.clone();
            flat_cl
                .call(Box::new(move || {
                    order_b.borrow_mut().push("B");
                    Ok(())
                }))
                .unwrap();
            Ok(())
        }))
        .unwrap();

        assert_eq!(*order.borrow(), vec!["outer", "A", "B"]);
    }

    #[test]
    fn nested_append_during_drain_is_visited_in_order() {
        // upstream parity: Python's for-loop re-reads the list length
        // each step, so appends *during* the drain are visited too.
        let flat: Rc<FlattenRecursion<TestError>> = Rc::new(FlattenRecursion::new());
        let order = Rc::new(RefCell::new(Vec::<&'static str>::new()));

        let flat_cl = flat.clone();
        let order_cl = order.clone();
        flat.call(Box::new(move || {
            order_cl.borrow_mut().push("outer");
            let flat_a = flat_cl.clone();
            let order_a = order_cl.clone();
            flat_cl
                .call(Box::new(move || {
                    order_a.borrow_mut().push("A");
                    // A appends C during the drain — must run after B.
                    let order_c = order_a.clone();
                    flat_a
                        .call(Box::new(move || {
                            order_c.borrow_mut().push("C");
                            Ok(())
                        }))
                        .unwrap();
                    Ok(())
                }))
                .unwrap();
            let order_b = order_cl.clone();
            flat_cl
                .call(Box::new(move || {
                    order_b.borrow_mut().push("B");
                    Ok(())
                }))
                .unwrap();
            Ok(())
        }))
        .unwrap();

        assert_eq!(*order.borrow(), vec!["outer", "A", "B", "C"]);
    }

    #[test]
    fn sequential_calls_after_drain_run_immediately_again() {
        let flat: FlattenRecursion<TestError> = FlattenRecursion::new();
        let counter = Rc::new(Cell::new(0u32));

        for _ in 0..3 {
            let counter_cl = counter.clone();
            flat.call(Box::new(move || {
                counter_cl.set(counter_cl.get() + 1);
                Ok(())
            }))
            .unwrap();
        }
        assert_eq!(counter.get(), 3);
    }

    #[test]
    fn unwind_discards_deferred_work_and_resets_outer_state() {
        let flat: Rc<FlattenRecursion<TestError>> = Rc::new(FlattenRecursion::new());
        let nested = flat.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            flat.call(Box::new(move || {
                nested.call(Box::new(|| panic!("discarded pending work")))?;
                panic!("outer initialization failed")
            }))
        }));
        assert!(outcome.is_err());
        assert!(flat.later.borrow().is_none());
        let called = Rc::new(Cell::new(false));
        let called_inner = called.clone();
        flat.call(Box::new(move || {
            called_inner.set(true);
            Ok(())
        }))
        .unwrap();
        assert!(called.get());
    }
}
