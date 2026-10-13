//! PyFrame — execution frame for Python bytecode.
#![allow(non_snake_case)]
//!
//! Each function call creates a new frame with its own value stack,
//! local variables, and instruction pointer. The JIT virtualizes
//! these fields so they live in registers instead of memory.

use std::rc::Rc;

use crate::PyExecutionContext;
use crate::{CodeFlags, CodeObject};
use pyre_object::FixedObjectArray;
use pyre_object::*;

// Ensure *const PyExecutionContext and Rc<PyExecutionContext> have the same
// size so that PyFrame field offsets are preserved after the switch.
const _: () = assert!(
    std::mem::size_of::<*const PyExecutionContext>()
        == std::mem::size_of::<Rc<PyExecutionContext>>()
);

/// Borrow a frame's `locals_cells_stack_w` array.
///
/// A macro and not an accessor method on purpose.  The codewriter only
/// lowers an array access through the virtualizable protocol —
/// `getarrayitem_vable_r` / `setarrayitem_vable_r` — when the `getfield` of
/// the virtualizable array and the array access itself sit in the same basic
/// block (`rpython/jit/codewriter/jtransform.py`).  A method that returns the
/// array reference puts the `getfield` in its own graph, so every access
/// through it falls back to a plain `getarrayitem_gc`.  Expanding the field
/// read at the call site is what keeps the pair together, which is also why
/// `pyframe.py` writes `self.locals_cells_stack_w[i]` inline at every use
/// instead of going through an accessor.
///
/// Accepts anything that derefs to a `PyFrame`, so `FrameBox` works too.
#[macro_export]
macro_rules! locals_w {
    ($frame:expr) => {
        (unsafe { &*$frame.locals_cells_stack_w })
    };
}

/// Mutably borrow a frame's `locals_cells_stack_w` array. See [`locals_w!`].
///
/// The `&mut *` on the receiver is load-bearing: the field it projects is a
/// raw pointer, which reads fine through a shared `&PyFrame`, so without it a
/// safe caller holding a shared frame could mint `&mut` to the array.  The
/// accessor this replaced took `&mut self` and this restores that requirement.
///
/// It does not bound the result's lifetime — `&mut *ptr` has an unconstrained
/// one — so two overlapping calls still produce aliasing the borrow checker
/// cannot see.  Tying the lifetime needs a function signature to tie it to,
/// which is the accessor form that puts the `getfield` in its own graph and
/// defeats the pairing described on [`locals_w!`].
#[macro_export]
macro_rules! locals_w_mut {
    ($frame:expr) => {
        (unsafe { &mut *(&mut *$frame).locals_cells_stack_w })
    };
}

/// `types.FrameType` — the PyType every `PyFrame`'s `ob_header` points at.
/// pyframe.py `class PyFrame(W_Root)` with `typedef.py:736 PyFrame.typedef
/// = TypeDef("frame", ...)`. The descriptors (`f_back`, `f_locals`, …) are
/// attached to this type by the `frame` typedef; until then it is a bare
/// identity tag so a frame carries a valid `ob_type` like every other
/// W_Root (mirrors `pytraceback::PYTRACEBACK_TYPE`).
pub static FRAME_TYPE: PyType = new_pytype("frame");

/// CPython 3.14 `FrameLocalsProxy` — the write-through mapping exposed by
/// `frame.f_locals` for optimized frames.  PyPy's older `getdictscope` keeps
/// the canonical locals mapping on `FrameDebugData.w_locals`; retain that
/// owner and make the proxy synchronize through `fast2locals` / `locals2fast`
/// rather than inventing a second per-frame store.
pub mod frame_locals_proxy {
    use super::*;

    #[crate::pyre_class("FrameLocalsProxy")]
    pub struct FrameLocalsProxy {
        w_frame: PyObjectRef,
    }

    /// Offset of `w_frame` for the JIT field descr.
    pub const W_FRAME_OFFSET: usize = std::mem::offset_of!(FrameLocalsProxy, w_frame);

    /// The frame `obj` is a live proxy onto, or `None` for anything else.
    ///
    /// The proxy reads the frame's array lazily rather than copying out of it,
    /// so a reader has to know whether the frame behind it is one whose locals
    /// currently live somewhere else -- a virtualizable's boxes while a trace
    /// is recording. Only the walker asks; the interpreter always reads the
    /// array it writes.
    pub fn viewed_frame(obj: PyObjectRef) -> Option<PyObjectRef> {
        unsafe { FrameLocalsProxy::from_obj(obj) }.map(|proxy| proxy.w_frame)
    }

    /// Whether `obj` is a `FrameLocalsProxy`.  A class-pointer test, so
    /// `list.__init__` can branch on it without entering `keys`.
    pub fn is_frame_locals_proxy(obj: PyObjectRef) -> bool {
        unsafe { FrameLocalsProxy::from_obj(obj) }.is_some()
    }

    /// `keys()` of a `FrameLocalsProxy`, or `None` when `obj` is not one.
    ///
    /// `list(f_locals)` reaches this from `ListStrategy.extend` instead of
    /// `space.iter`: the proxy's `__iter__` is `iter(self.keys())`, and that
    /// method call re-enters the interpreter once per extend.
    pub fn keys_list(obj: PyObjectRef) -> Option<Result<PyObjectRef, crate::PyError>> {
        let proxy = unsafe { FrameLocalsProxy::from_obj(obj) }?;
        Some(proxy.keys())
    }

    /// Pin every entry of `extra` as a `(key, value)` pair on top of the
    /// caller's bracket, and report how many.
    ///
    /// Pins into `roots` on purpose — a nested `push_roots` would rewind
    /// those slots on drop before the caller could read them.  Only the
    /// residual helpers below call it: this loop is bounded by the dict's
    /// length, which is red.
    fn pin_extra_locals_entries(
        roots: &pyre_object::gc_roots::RootScope,
        extra: PyObjectRef,
    ) -> usize {
        // `pin_root` itself is a forwarding query, so a sequential pin of
        // the remaining pairs would leave them unrooted across that
        // safepoint. Publish the whole snapshot first.
        let items = unsafe { pyre_object::dictmultiobject::w_dict_items(extra) };
        let mut live = Vec::with_capacity(items.len() * 2);
        for (key, value) in items {
            live.push(key);
            live.push(value);
        }
        let _ = roots.pin_roots(&live);
        live.len() / 2
    }

    /// The `f_extra_locals` half of `framelocalsproxy_keys` (`part` 0) or
    /// `_values` (`part` 1): append one entry per `extra` pair to `w_list`
    /// and return the list.
    ///
    /// `dont_look_inside` because the loop is bounded by the dict's length,
    /// which is red: it stays one residual call, so the hinted collect in
    /// [`FrameLocalsProxy::collect_entries`] is bounded by the green slot
    /// scan alone.
    #[majit_macros::dont_look_inside]
    fn append_extra_locals(w_list: PyObjectRef, extra: PyObjectRef, part: usize) -> PyObjectRef {
        let roots = pyre_object::gc_roots::push_roots();
        let list_slot = roots.pin_roots(&[w_list]);
        let pairs_base = list_slot + 1;
        let count = pin_extra_locals_entries(&roots, extra);
        for index in 0..count {
            unsafe {
                pyre_object::w_list_append(
                    roots.get(list_slot),
                    roots.get(pairs_base + index * 2 + part),
                )
            };
        }
        roots.get(list_slot)
    }

    /// [`append_extra_locals`] for `framelocalsproxy_items`: one
    /// `(key, value)` tuple per `extra` pair.
    #[majit_macros::dont_look_inside]
    fn append_extra_locals_items(w_list: PyObjectRef, extra: PyObjectRef) -> PyObjectRef {
        let roots = pyre_object::gc_roots::push_roots();
        let list_slot = roots.pin_roots(&[w_list]);
        let pairs_base = list_slot + 1;
        let count = pin_extra_locals_entries(&roots, extra);
        for index in 0..count {
            let item = pyre_object::w_tuple_new(vec![
                roots.get(pairs_base + index * 2),
                roots.get(pairs_base + index * 2 + 1),
            ]);
            unsafe { pyre_object::w_list_append(roots.get(list_slot), item) };
        }
        roots.get(list_slot)
    }

    /// `framelocalsproxy_reversed`'s in-place `list.reverse` of the keys
    /// list, returning the list reloaded from its root.
    #[majit_macros::dont_look_inside]
    fn reverse_list_in_place(w_list: PyObjectRef) -> PyObjectRef {
        let roots = pyre_object::gc_roots::push_roots();
        let list_slot = roots.pin_roots(&[w_list]);
        unsafe { pyre_object::listobject::w_list_reverse(roots.get(list_slot)) };
        roots.get(list_slot)
    }

    /// Empty list the proxy collectors append into.
    ///
    /// `w_list_new_empty` stays inside this residual: `descr_init` records
    /// the call, and `Vec::new` does not appear in that graph.
    #[majit_macros::dont_look_inside_cannot_raise]
    fn proxy_list_new() -> PyObjectRef {
        pyre_object::w_list_new_empty()
    }

    /// One `w_list_append` of `item` onto `list`, returning the list.
    ///
    /// The append wrapper stays look-inside so `list.append` can descend,
    /// and that body takes `w_list_lock`. Calling it from the slot scan
    /// would follow the lock into `list.__init__`. The lock stays in this
    /// residual; the scan records the call.
    #[majit_macros::dont_look_inside_cannot_raise]
    fn proxy_list_append(list: PyObjectRef, item: PyObjectRef) -> PyObjectRef {
        let roots = pyre_object::gc_roots::push_roots();
        let list_slot = roots.pin_roots(&[list, item]);
        unsafe {
            pyre_object::w_list_append(roots.get(list_slot), roots.get(list_slot + 1));
        }
        roots.get(list_slot)
    }

    /// Interned `co_localsplusnames` entry at `slot`.
    ///
    /// `pyframe.py fast2locals` reads `getcode().getvarnames()[i]`. The
    /// proxy key is that name's canonical exact `str` (`names_tuple` /
    /// `intern_str_value`), not a fresh string. The code object and the
    /// unrolled index are green, so this pure call folds; `intern_str_value`
    /// itself stays opaque because of the intern table.
    ///
    /// Slot order matches `locals_plus_names`: varnames, then pure cellvars,
    /// then freevars. An overlapping cellvar keeps the varname slot.
    #[majit_macros::elidable_cannot_raise]
    fn interned_localsplus_name(code: &crate::CodeObject, slot: usize) -> PyObjectRef {
        let nvar = code.varnames.len();
        let name: &str = if slot < nvar {
            code.varnames[slot].as_ref()
        } else {
            let band = slot - nvar;
            let npure = npure_cellvars(code);
            if band < npure {
                code.cellvars[nth_pure_cellvar_index(code, band)].as_ref()
            } else {
                let free_idx = band - npure;
                if free_idx < code.freevars.len() {
                    code.freevars[free_idx].as_ref()
                } else {
                    ""
                }
            }
        };
        pyre_object::intern_str_value(name)
    }

    /// `pyframe.py fast2locals` slot walk, appending each bound name
    /// (`part` 0) or value (`part` 1) onto `list`.
    ///
    /// `list(FrameLocalsProxy)` and `tuple(FrameLocalsProxy)` enter this
    /// body. The induction is `ll_rangenext` (`next < stop`, then
    /// `next += step`): the header carries the slot index. A
    /// `for slot in 0..nslots` is `rrange.py` `RANGEITER`
    /// (`GcStruct("range", next, stop)`), and this body's entry edge
    /// passed the `nslots` add into that struct. A `Vec` from
    /// `locals_plus_names` puts the iterator and the allocation on
    /// `descr_init`'s descent frontier. Hidden slots stay, and the scan
    /// does not consult `CodeFlags` — `fast2locals` skips a hidden slot
    /// only on a non-optimized frame, and that test is the one symbolic
    /// call in its graph.
    ///
    /// The frame is re-read from its root slot each iteration. The list slot
    /// is that same object: `proxy_list_append` pins it and returns the
    /// forwarded word. The code pointer comes from `PyFrame::getcode` once;
    /// it is not a GC object. `f_extra_locals` is red and stays
    /// [`append_extra_locals`].
    #[majit_macros::unroll_safe]
    fn append_bound_entries(frame: PyObjectRef, list: PyObjectRef, part: usize) -> PyObjectRef {
        let roots = pyre_object::gc_roots::push_roots();
        // Frame and list only. `RootScope::set` is not erased with the
        // bracket (`lower_erased_root_bracket_call` answers `pin_roots` /
        // `pin_root` / `get`), and a method receiver has no fnaddr, so a
        // slot rewrite would stay a symbolic residual inside `list.__init__`.
        let frame_slot = roots.pin_roots(&[frame, list]);
        // Slot 0 is the frame and slot 1 the list. `expand_pop_roots`
        // erases `get(base + k)` when the index operand is that sum.
        // `getcode` is `hint(self.pycode, promote=True)`. The returned
        // reference addresses the code object, which is not the frame, so
        // it stays valid when the frame is re-read below. A `*const`
        // round-trip is an Int in the codewriter, and `interned_localsplus_name`
        // takes the reference.
        let frame0 = unsafe { &*(roots.get(frame_slot) as *const PyFrame) };
        let code = frame0.getcode();
        let numlocals = code.varnames.len();
        let nslots = numlocals + npure_cellvars(code) + code.freevars.len();
        let mut slot = 0usize;
        while slot < nslots {
            let frame = unsafe { &*(roots.get(frame_slot) as *const PyFrame) };
            if slot < locals_w!(frame).len() {
                let raw = locals_w!(frame)[slot];
                let cellish = if slot < numlocals {
                    slot < code.localspluskinds.len()
                        && code.localspluskinds[slot] & crate::bytecode::CO_FAST_CELL != 0
                } else {
                    true
                };
                let value = if cellish && !raw.is_null() && unsafe { pyre_object::is_cell(raw) } {
                    unsafe { pyre_object::w_cell_get(raw) }
                } else {
                    raw
                };
                if !value.is_null() {
                    let item = if part == 0 {
                        interned_localsplus_name(code, slot)
                    } else {
                        value
                    };
                    // `proxy_list_append` publishes both words before `w_list_append`
                    // can collect. The frame and list slots above are forwarded by
                    // that collection; the returned list is the same object.
                    let _ = proxy_list_append(roots.get(frame_slot + 1), item);
                }
            }
            slot += 1;
        }
        let frame = unsafe { &*(roots.get(frame_slot) as *const PyFrame) };
        let extra = frame.get_extra_locals();
        if extra.is_null() {
            return roots.get(frame_slot + 1);
        }
        append_extra_locals(roots.get(frame_slot + 1), extra, part)
    }

    /// `keys` (`part` 0) or `values` (`part` 1) of a proxy, or an empty
    /// list when `obj` is not one.
    ///
    /// Hinted so the slot loop, if inlined from `append_bound_entries`,
    /// does not make `contains_loop` decline `tuple.__new__`.
    #[majit_macros::unroll_safe]
    pub(crate) fn bound_part_list(obj: PyObjectRef, part: usize) -> PyObjectRef {
        if !is_frame_locals_proxy(obj) {
            return proxy_list_new();
        }
        let roots = pyre_object::gc_roots::push_roots();
        let obj_slot = roots.pin_roots(&[obj]);
        let frame = unsafe { (*(roots.get(obj_slot) as *const FrameLocalsProxy)).w_frame };
        // The frame is the next pin after `obj`. `expand_pop_roots` reads
        // it as `get(base + 1)` after `pin_root` has published that slot.
        let _ = roots.pin_root(frame);
        let list = proxy_list_new();
        append_bound_entries(roots.get(obj_slot + 1), list, part)
    }

    /// Append the proxy's bound names (`part` 0) or values (`part` 1) onto
    /// `list`. `false` when `obj` is not a proxy.
    ///
    /// Hinted so the slot loop, if inlined from `append_bound_entries`,
    /// does not make `contains_loop` decline `list.__init__`.
    #[majit_macros::unroll_safe]
    pub(crate) fn extend_bound_entries(list: PyObjectRef, obj: PyObjectRef, part: usize) -> bool {
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.pin_roots(&[list, obj]);
        let obj = roots.get(base + 1);
        if !is_frame_locals_proxy(obj) {
            return false;
        }
        let frame = unsafe { (*(obj as *const FrameLocalsProxy)).w_frame };
        let _ = append_bound_entries(frame, roots.get(base), part);
        true
    }

    impl FrameLocalsProxy {
        #[inline]
        fn frame(&self) -> &mut PyFrame {
            unsafe { &mut *(self.w_frame as *mut PyFrame) }
        }

        /// Materialize the viewed frame's `locals_cells_stack_w`.
        ///
        /// There is a write to materialize for at all because the proxy
        /// implements PEP 667: a `FrameLocalsProxy` store updates the fast
        /// local, so this force is what gives that store somewhere real to
        /// land.  Upstream has no such store to serve -- `pyframe.py
        /// getdictscope` hands out the `fast2locals` snapshot dict, and only
        /// `setdictscope` pushes a dict back through `locals2fast`, so
        /// mutating what `f_locals` returned reaches no fast local there.
        /// That is a version difference, not a porting gap: PEP 667 postdates
        /// the language version upstream implements.  The parity corpus
        /// declares it on `framelocalsproxy_write_coherence` and
        /// `framelocalsproxy_gc_roots` rather than carrying it as a defect.
        ///
        /// `rvirtualizable.py hook_access_field` puts the force at the field
        /// access.  The `f_locals` getter forces once, when it builds the
        /// proxy, and a proxy is a live view that outlives that force: bind one
        /// before a loop the JIT afterwards compiles and the two sides stop
        /// agreeing.  Both directions diverge, and each needs a store in the
        /// loop *body* to show, because a store outside it precedes the compile:
        ///
        /// - a write through the proxy lands in the frame array while the
        ///   compiled loop answers that local from its shadow;
        /// - a `STORE_FAST` in the body lands in the shadow while a read
        ///   through the proxy answers from the array.
        ///
        /// This call site is the write direction, and it is the one that wants
        /// an UNDELETED force: a store has to reach the shadow the compiled
        /// loop answers from, so the force has to happen in the traced copy
        /// too.  The read direction wants the opposite -- `hook_access_field`
        /// runs in residual code, while inside a trace the same accesses are
        /// *redirected* to the shadow, so upstream never forces from a traced
        /// read; `pyframe.py fast2locals` is `@jit.unroll_safe` for exactly
        /// that reason.  Forcing unconditionally on every read forces the
        /// traced ones too, and measured, that aborts the loop where a hot
        /// `len(fr.f_locals)` or `fr.f_locals["x"]` sits in the traced body.
        /// What would tell the two apart is the marker, not the call site: a
        /// read force spelled [`crate::executioncontext::jit_force_virtualizable`]
        /// is deleted by `jtransform.py rewrite_op_jit_force_virtualizable` in
        /// the copy the codewriter looks inside and kept in the residual one.
        /// Putting one on [`PyFrame::fast2locals`] and
        /// [`PyFrame::frame_locals_proxy_snapshot`] was measured and reverted:
        /// it changed no output and lost five fixtures' compilations, because
        /// `vable_and_vrefs_before_residual_call` already stores the boxes
        /// back before the residual read.  See that function's own doc.
        ///
        /// [`Self::frame`] does not take it either: [`Self::fast_local_index`]
        /// goes through it only for `code`, which is not a redirected field.
        ///
        /// The `vable_token` test is `virtualizable.py`'s, and it is required
        /// rather than a shortcut:
        ///
        /// ```text
        /// def force_virtualizable_if_necessary(virtualizable):
        ///     if virtualizable.vable_token:
        ///         force_now(virtualizable)
        /// ```
        ///
        /// The backend hook behind `force_frame` signals a frame escape before
        /// it tests any token, and a proxy write reached while this frame is
        /// being traced holds no token.  Calling it there raises an escape with
        /// no committed resume pc, so the walk replays from entry and
        /// double-applies the residual's body effects; measured, that aborts
        /// the loop and leaves a name string where the operand stack expects
        /// its iterator.  The token is set only once compiled code owns the
        /// frame, which is exactly when the write has a shadow to reach.
        fn force_locals(&self) {
            let frame = self.w_frame as *mut PyFrame;
            let token = unsafe { (*frame).vable_token };
            // `virtualizable.py force_virtualizable_if_necessary`: force only
            // when a compiled loop owns the frame. TOKEN_NONE (0) is idle.
            if token == 0 {
                return;
            }
            // `virtualizable.py force_now`: TOKEN_TRACING_RESCALL values are
            // already correct during tracing; reset to TOKEN_NONE as the
            // escape marker `tracing_after_residual_call` reads.
            if token == majit_metainterp::virtualref::token_tracing_rescall() as usize {
                unsafe { (*frame).vable_token = 0 };
                return;
            }
            // The force materializes through a backend hook this crate cannot
            // follow and is judged as able to collect, so the frame is
            // anchored across it.
            let anchor = unsafe { crate::eval::FrameAnchor::from_raw(frame) };
            crate::executioncontext::force_frame_before_locals_read(anchor.live());
        }

        #[inline]
        fn mapping(&self) -> Result<PyObjectRef, crate::PyError> {
            // No force here or in the callee; see [`Self::force_locals`] on
            // why the read direction takes none.
            self.frame().frame_locals_proxy_snapshot()
        }

        fn call_mapping_method(
            &self,
            name: &str,
            args: &[PyObjectRef],
        ) -> Result<PyObjectRef, crate::PyError> {
            // `mapping` materializes a fresh dict, so every element of the
            // caller's slice is a pre-allocation copy by the time the call
            // reads it.  Publish the slice, take the snapshot, then rebuild it.
            // The proxy payload itself is born non-moving, so `self` and the
            // frame it reaches stay valid across the allocation.
            let roots = pyre_object::gc_roots::push_roots();
            let args_base = roots.publish(args);
            roots.normalize(args_base, args.len());
            let mapping = self.mapping()?;
            let mut reloaded_args: Vec<PyObjectRef> = Vec::with_capacity(args.len());
            for i in 0..args.len() {
                reloaded_args.push(roots.get(args_base + i));
            }
            let args = reloaded_args;
            let result = crate::baseobjspace::call_method(mapping, name, &args);
            if result.is_null() {
                Err(crate::call::take_call_error()
                    .unwrap_or_else(|| crate::PyError::runtime_error("method call failed")))
            } else {
                Ok(result)
            }
        }

        fn fast_local_index(&self, key: PyObjectRef) -> Result<Option<usize>, crate::PyError> {
            // Every candidate name allocates its own string and both
            // `__hash__` and `__eq__` can run a Python method, so the caller's
            // key is a pre-allocation copy from the first of them onward.
            // Publish it and reload it for each compare; the candidate reuses
            // one slot rather than growing the stack once per local.
            let roots = pyre_object::gc_roots::push_roots();
            let key_slot = roots.base();
            let _ = roots.pin_root(key);
            let candidate_slot = key_slot + 1;
            let _ = roots.pin_root(pyre_object::PY_NULL);
            // The scan hashes the key first whether it was reached to write or
            // to read, so an unhashable key is a `TypeError` here rather than
            // whatever the extras dict would go on to say about it.
            let key_hash = crate::baseobjspace::hash_w_strict(roots.get(key_slot))?;
            // `code` addresses the compiler code object, which lives outside
            // the GC heap and so stays valid across those collections.
            let code = self.frame().code();
            let matches = |name: &str| -> Result<bool, crate::PyError> {
                roots.set(candidate_slot, pyre_object::w_str_new_managed(name));
                // A name whose hash differs is never compared, so a key that
                // claims equality with a name it does not hash like does not
                // reach that name's slot.
                Ok(
                    crate::baseobjspace::hash_w_strict(roots.get(candidate_slot))? == key_hash
                        && crate::baseobjspace::eq_w(
                            roots.get(candidate_slot),
                            roots.get(key_slot),
                        )?,
                )
            };
            // `range` + `ll_getitem_fast`, not `Enumerate`.
            let mut index = 0usize;
            for name in code.varnames.iter() {
                if hidden_local(code, index) {
                    index += 1;
                    continue;
                }
                if matches(name.as_ref())? {
                    return Ok(Some(index));
                }
                index += 1;
            }
            let mut index = code.varnames.len();
            for name in code.cellvars.iter() {
                if code.varnames.iter().any(|local| local == name) {
                    continue;
                }
                if matches(name.as_ref())? {
                    return Ok(Some(index));
                }
                index += 1;
            }
            for name in code.freevars.iter() {
                if matches(name.as_ref())? {
                    return Ok(Some(index));
                }
                index += 1;
            }
            Ok(None)
        }

        fn setitem_value(
            &mut self,
            key: PyObjectRef,
            value: PyObjectRef,
        ) -> Result<(), crate::PyError> {
            // `fast_local_index` compares against a freshly allocated string
            // per local and `get_or_create_extra_locals` allocates the dict, so
            // both stores below would otherwise write pre-allocation copies.
            let roots = pyre_object::gc_roots::push_roots();
            let key_slot = roots.publish(&[key, value]);
            roots.normalize(key_slot, 2);
            // The stores below target `locals_cells_stack_w`, so materialize it
            // first.  The force follows the publish rather than preceding the
            // region: it can collect, and `key` and `value` are pre-allocation
            // copies until they are rooted.
            self.force_locals();
            let value_slot = key_slot + 1;
            if let Some(index) = self.fast_local_index(roots.get(key_slot))? {
                let frame = self.frame();
                let slot = locals_w!(frame)[index];
                if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
                    unsafe { pyre_object::w_cell_set(slot, roots.get(value_slot)) };
                } else {
                    frame.set_locals_w(index, roots.get(value_slot));
                }
                return Ok(());
            }
            // CPython 3.14 `framelocalsproxy_setitem`: a key which is not a
            // writable fast local goes into the frame object's separate
            // `f_extra_locals` exact dict.  In particular, a PEP 709 hidden
            // local is read from the fast slot but proxy writes of the same
            // name do not mutate that slot or the module/eval locals mapping.
            let extra = self.frame().get_or_create_extra_locals();
            crate::baseobjspace::setitem(extra, roots.get(key_slot), roots.get(value_slot))
                .map(|_| ())
        }

        /// `framelocalsproxy_getkeyindex` with `read == true`, followed by
        /// `framelocalsproxy_getval`: the value of the first locals-plus slot
        /// whose name hashes and compares equal to `key` and which is bound.
        ///
        /// A name that matches only unbound slots reads as absent, which is
        /// what sends `__getitem__` on to `f_extra_locals`.  The write
        /// direction skips a PEP 709 hidden slot and this one does not, so it
        /// cannot share [`Self::fast_local_index`]: a running comprehension's
        /// iteration variable is readable through the proxy even though
        /// assigning to that name goes to the extras dict instead.
        ///
        /// `@jit.unroll_safe` for the same reason as [`PyFrame::fast2locals`]
        /// and [`PyFrame::frame_locals_proxy_snapshot`]: the scan is bounded by
        /// the code object's varnames, pure cellvars and freevars, which is
        /// green.  The loops index those arrays in place, the way
        /// [`PyFrame::fast2locals`] does, so the bound does not collect a
        /// `Vec`.  Without the hint `contains_loop` declines the graph, and
        /// then a `fr.f_locals["x"]` in a traced body is one residual call --
        /// the array read behind it never reaches the virtualizable lowering,
        /// so it answers from memory instead of from the shadow the compiled
        /// loop is writing.  What the body does per candidate is not what the
        /// hint is about: `hash_w_strict` and `eq_w` can run application code,
        /// but they are calls, not the loop bound.
        #[majit_macros::unroll_safe]
        fn locals_plus_value(
            &self,
            key: PyObjectRef,
        ) -> Result<Option<PyObjectRef>, crate::PyError> {
            // A candidate name allocates its own string and both `__hash__`
            // and `__eq__` can run Python, so the key is rooted before the
            // first of them and read back from that root afterwards.
            let roots = pyre_object::gc_roots::push_roots();
            let key_slot = roots.base();
            let _ = roots.pin_root(key);
            let candidate_slot = key_slot + 1;
            let _ = roots.pin_root(pyre_object::PY_NULL);
            // No force here; see [`Self::force_locals`] on the read direction.
            // An exact `str` is decided by WTF-8 bytes, so its hash is not
            // read and cannot fail.  Every other key is hashed before the
            // scan, including a frame with no locals, so an unhashable key
            // is still a `TypeError`.
            let exact_str_key = unsafe {
                pyre_object::pyobject::is_exact_type(
                    roots.get(key_slot),
                    &pyre_object::pyobject::STR_TYPE,
                )
            };
            let key_hash = if exact_str_key {
                0
            } else {
                crate::baseobjspace::hash_w_strict(roots.get(key_slot))?
            };
            // `RootScope::set` stays in this closure, the same split as
            // [`Self::fast_local_index`]'s `matches`. A `set` in this body
            // marks every collecting call unread (`ScanStats::withheld_contents_opaque`,
            // reason `slot-set`), and each of the three loops below would
            // carry its own `hash_w_strict` / `eq_w`. The exact-`str` compare
            // and the virtualizable array read stay here: a closure over
            // `self` would capture the array field (`frame_locals_proxy_snapshot`).
            let nonexact_same = |name: &str| -> Result<bool, crate::PyError> {
                roots.set(candidate_slot, pyre_object::w_str_new_managed(name));
                // A name whose hash differs is never compared: a key that
                // claims equality with a name it does not hash like reads as
                // absent rather than as that name's slot.
                Ok(
                    crate::baseobjspace::hash_w_strict(roots.get(candidate_slot))? == key_hash
                        && crate::baseobjspace::eq_w(
                            roots.get(candidate_slot),
                            roots.get(key_slot),
                        )?,
                )
            };
            // `code` addresses the compiler code object, which lives outside
            // the GC heap and so stays valid across those collections.
            // Same slot order as [`locals_plus_names`]: varnames, then the
            // pure-cellvar band, then freevars.  Positional indexing rather
            // than that `Vec` because this graph is `unroll_safe`: a
            // collected `Vec` puts the iterator and the allocation on the
            // descent's blocker frontier (`fast2locals`).
            let code = self.frame().code();
            let nvar = code.varnames.len();
            // The name borrow ends before `self.frame()`.  `locals_w!` stays
            // in this function so the virtualizable array getfield and the
            // access share a basic block.
            macro_rules! consider_local {
                ($index:expr, $name:expr, $is_cell:expr) => {{
                    let same = {
                        let name: &str = $name;
                        if exact_str_key {
                            // Comparing WTF-8 bytes decides an exact `str`
                            // key outright, so it needs neither the hash nor
                            // a candidate allocation.
                            unsafe { pyre_object::w_str_get_wtf8(roots.get(key_slot)) }.as_bytes()
                                == name.as_bytes()
                        } else {
                            nonexact_same(name)?
                        }
                    };
                    if same {
                        let frame = self.frame();
                        if $index < locals_w!(frame).len() {
                            let slot = locals_w!(frame)[$index];
                            let value = if $is_cell
                                && !slot.is_null()
                                && unsafe { pyre_object::is_cell(slot) }
                            {
                                unsafe { pyre_object::w_cell_get(slot) }
                            } else {
                                slot
                            };
                            if !value.is_null() {
                                return Ok(Some(value));
                            }
                        }
                    }
                }};
            }
            for index in 0..nvar {
                let is_cell = super::cell_slot(code, index);
                consider_local!(index, code.varnames[index].as_str(), is_cell);
            }
            let npure = super::npure_cellvars(code);
            for i in 0..npure {
                consider_local!(
                    nvar + i,
                    code.cellvars[super::nth_pure_cellvar_index(code, i)].as_str(),
                    true
                );
            }
            let nfree = code.freevars.len();
            for i in 0..nfree {
                consider_local!(nvar + npure + i, code.freevars[i].as_str(), true);
            }
            Ok(None)
        }

        fn key_is_fast_local(&self, key: PyObjectRef) -> Result<bool, crate::PyError> {
            Ok(self.fast_local_index(key)?.is_some())
        }

        /// `framelocalsproxy_keys`, `_values` and `_items` walk the same two
        /// halves in the same order: every bound locals-plus slot in
        /// `co_localsplusnames` order, then every `f_extra_locals` entry.
        /// This scan is the first half and serves all three.  Pins `2 * n`
        /// shadow-stack slots from the caller's bracket base upwards,
        /// `(key, value)` interleaved, and returns `n`; the second half is
        /// [`append_extra_locals`] / [`append_extra_locals_items`].
        ///
        /// The two halves are reported INDEPENDENTLY, so a name that is both a
        /// bound slot and an `f_extra_locals` key yields two entries — which is
        /// what makes `len`, `keys`, `values` and `items` all count a PEP 709
        /// hidden slot and a proxy write of the same name as two.  The
        /// deduplicating view is [`PyFrame::frame_locals_proxy_snapshot`], the
        /// `dict(proxy)` these are not.
        ///
        /// Each pair is pinned as it is produced: a slot's key is a freshly
        /// allocated string, so an earlier pair left in a native `Vec` would
        /// not survive a later name's allocation.  Pins into `roots` on
        /// purpose — a nested `push_roots` would rewind those slots on
        /// drop before the caller could read them.  The pairs start at the
        /// bracket's top as this scan finds it, so a caller that pinned
        /// something first reads its own entries from there rather than from
        /// `roots.base()`.
        ///
        /// `@jit.unroll_safe` for the same reason as
        /// [`Self::locals_plus_value`]: the slot scan is bounded by
        /// `locals_plus_names`, i.e. the code object's varnames plus cellvars
        /// plus freevars, which is green.  The extras half is bounded by the
        /// dict's length instead, so it is not part of this scan.
        #[majit_macros::unroll_safe]
        fn pin_entries(
            &self,
            roots: &pyre_object::gc_roots::RootScope,
            frame_slot: usize,
        ) -> usize {
            let mut count = 0;
            // `code` addresses the compiler code object, which lives outside
            // the GC heap and so stays valid across those collections.
            // The frame itself is nursery-allocated and `intern_str_value`
            // below can collect, so each iteration re-reads `frame_slot`
            // (`shadowstack.py` restores the livevar after the call).
            let frame = || unsafe { &mut *(roots.get(frame_slot) as *mut PyFrame) };
            // `pycode` is outside the GC heap. Take the pointer once so the
            // `CodeObject` borrow does not keep the frame temporary alive.
            let code_ptr = unsafe { crate::pyframe::pyframe_get_pycode(frame()) };
            let code = unsafe { &*code_ptr };
            // The first slot this scan claims. Reading it from the bracket
            // rather than from `roots.base()` keeps the pair arithmetic below
            // correct for a caller that had already pinned something.
            let mut next_slot = roots.publish(&[]);
            // A plain loop rather than a closure over `self`, for the reason
            // [`PyFrame::frame_locals_proxy_snapshot`] gives.
            for (index, name, cell_slot) in locals_plus_names(code) {
                let frame = frame();
                if index >= locals_w!(frame).len() {
                    continue;
                }
                let slot = locals_w!(frame)[index];
                let value = if cell_slot && !slot.is_null() && unsafe { pyre_object::is_cell(slot) }
                {
                    unsafe { pyre_object::w_cell_get(slot) }
                } else {
                    slot
                };
                // `framelocalsproxy_hasval`: an unbound slot names no entry.
                if value.is_null() {
                    continue;
                }
                // The key allocates and the value does not, so claim the key's
                // slot first, pin the value, and only then build the name.
                let key_slot = next_slot;
                next_slot += 2;
                let _ = roots.pin_root(pyre_object::PY_NULL);
                let _ = roots.pin_root(value);
                // `co_localsplusnames` entries are the code object's own strings.
                // A fresh managed string per `keys` call allocated one object
                // per bound local on every `list(f_locals)`.
                roots.set(key_slot, pyre_object::intern_str_value(name));
                count += 1;
            }
            count
        }

        /// `framelocalsproxy_length` — the `f_extra_locals` size plus one per
        /// bound locals-plus slot.  A name held in both halves counts twice,
        /// and nothing is materialized to count it; answering from
        /// [`Self::mapping`] instead reported the deduplicated dict's size,
        /// and built that dict to do it.
        ///
        /// Named for what it counts rather than `length`, because
        /// `REVIEWED_UNROLL_SAFE` in `test_unroll_safe_inventory.rs` matches a
        /// hint by its leaf and `length` is already the leaf of
        /// `bufferview::BufferView::length`.
        ///
        /// `@jit.unroll_safe` for the same reason as [`Self::pin_entries`].
        #[majit_macros::unroll_safe]
        fn entry_count(&self) -> i64 {
            let extra = self.frame().get_extra_locals();
            let mut size = if extra.is_null() {
                0
            } else {
                unsafe { pyre_object::dictmultiobject::w_dict_len(extra) as i64 }
            };
            // `code` addresses the compiler code object, which lives outside
            // the GC heap and so stays valid across those collections.
            let code = self.frame().code();
            for (index, _name, cell_slot) in locals_plus_names(code) {
                let frame = self.frame();
                if index >= locals_w!(frame).len() {
                    continue;
                }
                let slot = locals_w!(frame)[index];
                let value = if cell_slot && !slot.is_null() && unsafe { pyre_object::is_cell(slot) }
                {
                    unsafe { pyre_object::w_cell_get(slot) }
                } else {
                    slot
                };
                // `framelocalsproxy_hasval`.
                if !value.is_null() {
                    size += 1;
                }
            }
            size
        }

        /// `framelocalsproxy_richcompare`.
        ///
        /// Two proxies compare by FRAME IDENTITY, not by contents: distinct
        /// frames with equal locals are unequal.  A `dict` on the right is
        /// compared against the materialized mapping, and every other operand
        /// answers `NotImplemented`.
        fn richcompare(
            &self,
            other: PyObjectRef,
            op: crate::baseobjspace::CompareOp,
        ) -> Result<PyObjectRef, crate::PyError> {
            use crate::baseobjspace::CompareOp;
            if let Some(other_frame) = viewed_frame(other) {
                let same = std::ptr::eq(self.w_frame, other_frame);
                return Ok(match op {
                    CompareOp::Eq => pyre_object::w_bool_from(same),
                    CompareOp::Ne => pyre_object::w_bool_from(!same),
                    _ => pyre_object::w_not_implemented(),
                });
            }
            // The gate is `PyDict_Check`, which admits a subclass, so the
            // materialized mapping is what answers the comparison; `is_dict`
            // is the exact-layout test and would decline it.
            if unsafe { crate::baseobjspace::isinstance_dict_w(other) } {
                let roots = pyre_object::gc_roots::push_roots();
                let other_slot = roots.base();
                let _ = roots.pin_root(other);
                let mapping = self.mapping()?;
                let mapping_slot = other_slot + 1;
                let _ = roots.pin_root(mapping);
                return crate::baseobjspace::compare(
                    roots.get(mapping_slot),
                    roots.get(other_slot),
                    op,
                );
            }
            Ok(pyre_object::w_not_implemented())
        }

        /// `framelocalsproxy_merge` — apply `other`'s entries through the
        /// MAPPING protocol: `keys()` on the argument, then `other[key]` per
        /// key, then `framelocalsproxy_setitem`.  A `dict` subclass that
        /// overrides either one is therefore honoured; routing the argument
        /// through a temporary `dict.update` was not, because `dict_merge`
        /// takes its fast path off the subclass's stored entries.  Starting
        /// from the proxy's own snapshot would additionally replay every
        /// existing fast local into `f_extra_locals`.
        ///
        /// `Ok(false)` is the `-1` upstream returns with no exception set:
        /// `other` is neither a `dict` nor another proxy.  The two callers
        /// turn that into their own answer — a `TypeError` for `update`,
        /// `NotImplemented` for `|=` — and neither iterates the operand, so a
        /// list of pairs is rejected rather than silently accepted.
        fn merge(&mut self, other: PyObjectRef) -> Result<bool, crate::PyError> {
            // The gate is `PyDict_Check`, which admits a subclass — that is
            // the whole point of reading the argument through `keys()` and
            // `__getitem__` afterwards.  `is_dict` is the exact-layout test
            // and would decline the very case this port exists for.
            if !unsafe { crate::baseobjspace::isinstance_dict_w(other) }
                && viewed_frame(other).is_none()
            {
                return Ok(false);
            }
            // Every step below runs the argument's Python and collects, so
            // each operand is read back out of its root at each use.
            let roots = pyre_object::gc_roots::push_roots();
            let other_slot = roots.base();
            let _ = roots.pin_root(other);
            // `PyMapping_Keys`.
            let keys = crate::baseobjspace::call_method(roots.get(other_slot), "keys", &[]);
            if keys.is_null() {
                return Err(crate::call::take_call_error()
                    .unwrap_or_else(|| crate::PyError::runtime_error("keys() failed")));
            }
            let keys_slot = other_slot + 1;
            let _ = roots.pin_root(keys);
            let iter = crate::baseobjspace::iter(roots.get(keys_slot))?;
            let iter_slot = keys_slot + 1;
            let _ = roots.pin_root(iter);
            let key_slot = iter_slot + 1;
            let _ = roots.pin_root(pyre_object::PY_NULL);
            let value_slot = key_slot + 1;
            let _ = roots.pin_root(pyre_object::PY_NULL);
            loop {
                match crate::baseobjspace::next(roots.get(iter_slot)) {
                    Ok(key) => roots.set(key_slot, key),
                    Err(err) => {
                        let (stop, err) = err.matches_stop_iteration_keep();
                        if stop {
                            break;
                        }
                        return Err(err);
                    }
                }
                let value =
                    crate::baseobjspace::getitem(roots.get(other_slot), roots.get(key_slot))?;
                roots.set(value_slot, value);
                self.setitem_value(roots.get(key_slot), roots.get(value_slot))?;
            }
            Ok(true)
        }
    }

    #[crate::pyre_methods(doc = "A write-through view of a frame's optimized locals.")]
    impl FrameLocalsProxy {
        #[staticmethod]
        fn __new__(_cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            // The gateway includes the type object in the raw argument slice
            // passed to a static __new__ wrapper.
            if args.len() != 2 {
                return Err(crate::PyError::type_error(format!(
                    "FrameLocalsProxy expected 1 argument, got {}",
                    args.len().saturating_sub(1)
                )));
            }
            let w_frame = args[1];
            if !unsafe { pyre_object::py_type_check(w_frame, &FRAME_TYPE) } {
                // `framelocalsproxy_new` names the rejected operand's type
                // with `%T`, i.e. `PyType_GetFullyQualifiedName`.
                let received = crate::typedef::r#type(w_frame)
                    .map(|tp| unsafe {
                        crate::baseobjspace::type_fully_qualified_name(tp.as_ptr())
                    })
                    .unwrap_or_else(|| rustpython_wtf8::Wtf8Buf::from("object"));
                return Err(crate::PyError::type_error(crate::display::wtf8_format!(
                    "expect frame, not ",
                    received
                )));
            }
            let _roots = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(w_frame);
            let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let proxy = FrameLocalsProxy::allocate_stable(FrameLocalsProxy {
                ob: PyObject {
                    ob_type: std::ptr::null(),
                    w_class: std::ptr::null_mut(),
                },
                w_frame: pyre_object::gc_roots::shadow_stack_get(slot),
            });
            Ok(proxy)
        }

        fn __getitem__(&self, key: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            // `framelocalsproxy_getitem` reads the one slot the key names.
            // Materializing the whole mapping to answer a single lookup costs a
            // dict plus an entry per bound local, and reports the miss as the
            // dict's own `KeyError(key)` rather than the message below.
            //
            // `locals_plus_value` and the extras lookup both allocate, so
            // reload the key from its root between them.
            let roots = pyre_object::gc_roots::push_roots();
            let key_slot = roots.base();
            let _ = roots.pin_root(key);
            if let Some(value) = self.locals_plus_value(roots.get(key_slot))? {
                return Ok(value);
            }
            let extra = self.frame().get_extra_locals();
            if !extra.is_null() {
                let extra_slot = key_slot + 1;
                let _ = roots.pin_root(extra);
                if let Some(value) =
                    crate::baseobjspace::finditem(roots.get(extra_slot), roots.get(key_slot))?
                {
                    return Ok(value);
                }
            }
            let key_repr = unsafe { crate::display::py_repr_wtf8(roots.get(key_slot))? };
            Err(crate::PyError::key_error(crate::display::wtf8_format!(
                "local variable '",
                key_repr,
                "' is not defined"
            )))
        }

        fn __setitem__(
            &mut self,
            key: PyObjectRef,
            value: PyObjectRef,
        ) -> Result<(), crate::PyError> {
            self.setitem_value(key, value)
        }

        fn __delitem__(&mut self, key: PyObjectRef) -> Result<(), crate::PyError> {
            // `framelocalsproxy_setitem` with no value: the scan decides, and
            // only a key it places nowhere goes on to the extras dict.  Probing
            // the snapshot first instead built a whole mapping to reject one
            // name, and reported an unhashable key in that dict's terms rather
            // than as the hash's own `TypeError`.
            //
            // The extras lookup allocates, so reload the key from its root.
            let roots = pyre_object::gc_roots::push_roots();
            let key_slot = roots.base();
            let _ = roots.pin_root(key);
            if self.key_is_fast_local(roots.get(key_slot))? {
                return Err(crate::PyError::value_error(
                    "cannot remove local variables from FrameLocalsProxy",
                ));
            }
            // A delete does not create the extras dict that a store would.
            let extra = self.frame().get_extra_locals();
            if extra.is_null() {
                return Err(crate::PyError::key_error_with_key(roots.get(key_slot)));
            }
            let extra_slot = key_slot + 1;
            let _ = roots.pin_root(extra);
            crate::baseobjspace::delitem(roots.get(extra_slot), roots.get(key_slot))
        }

        fn __len__(&self) -> Result<i64, crate::PyError> {
            Ok(self.entry_count())
        }

        fn __iter__(&self) -> Result<PyObjectRef, crate::PyError> {
            // `framelocalsproxy_iter` is `iter(self.keys())`, so a name held
            // by both a slot and `f_extra_locals` is yielded twice.
            crate::baseobjspace::iter(self.keys()?)
        }

        fn __reversed__(&self) -> Result<PyObjectRef, crate::PyError> {
            // `framelocalsproxy_reversed` reverses the key LIST in place and
            // hands that back; it does not build a cursor.
            let keys = self.keys()?;
            Ok(reverse_list_in_place(keys))
        }

        fn __contains__(&self, key: PyObjectRef) -> Result<bool, crate::PyError> {
            // `framelocalsproxy_contains`: the first bound locals-plus slot
            // whose name hashes and compares equal to `key`, else
            // `f_extra_locals`.  Materializing the mapping allocated a string
            // per bound local for a yes/no.
            //
            // `locals_plus_value` and the extras lookup both allocate, so
            // reload the key from its root between them.
            let roots = pyre_object::gc_roots::push_roots();
            let key_slot = roots.base();
            let _ = roots.pin_root(key);
            if self.locals_plus_value(roots.get(key_slot))?.is_some() {
                return Ok(true);
            }
            let extra = self.frame().get_extra_locals();
            if extra.is_null() {
                return Ok(false);
            }
            let extra_slot = key_slot + 1;
            let _ = roots.pin_root(extra);
            crate::baseobjspace::contains(roots.get(extra_slot), roots.get(key_slot))
        }

        fn keys(&self) -> Result<PyObjectRef, crate::PyError> {
            self.collect_entries(0)
        }

        fn values(&self) -> Result<PyObjectRef, crate::PyError> {
            self.collect_entries(1)
        }

        /// [`Self::keys`] (`part` 0) or [`Self::values`] (`part` 1).
        ///
        /// The slot walk is [`append_bound_entries`], the same positional
        /// loops as `fast2locals`. The `f_extra_locals` half stays the
        /// residual [`append_extra_locals`]. The frame word is pinned
        /// before the empty list is built: `proxy_list_new` can collect,
        /// and this `&self` is not a root.
        #[majit_macros::unroll_safe]
        fn collect_entries(&self, part: usize) -> Result<PyObjectRef, crate::PyError> {
            let roots = pyre_object::gc_roots::push_roots();
            let frame_slot = roots.pin_roots(&[self.w_frame]);
            let list = proxy_list_new();
            Ok(append_bound_entries(roots.get(frame_slot), list, part))
        }

        /// Same bound as [`Self::collect_entries`], and the extras half is
        /// the residual [`append_extra_locals_items`].
        #[majit_macros::unroll_safe]
        fn items(&self) -> Result<PyObjectRef, crate::PyError> {
            // Each `w_tuple_new` allocates, so the pairs still queued in the
            // scan and the tuples already built are pre-allocation copies by
            // the next iteration.  Publish both and read them back.
            let roots = pyre_object::gc_roots::push_roots();
            let frame_slot = roots.base();
            let _ = roots.pin_root(self.w_frame);
            let pairs_base = pyre_object::gc_roots::shadow_stack_len();
            let count = self.pin_entries(&roots, frame_slot);
            let out_base = roots.publish(&[]);
            for index in 0..count {
                let _ = roots.pin_root(pyre_object::w_tuple_new(vec![
                    roots.get(pairs_base + index * 2),
                    roots.get(pairs_base + index * 2 + 1),
                ]));
            }
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                out.push(roots.get(out_base + i));
            }
            let items = pyre_object::w_list_new(out);
            let frame = unsafe { &*(roots.get(frame_slot) as *const PyFrame) };
            let extra = frame.get_extra_locals();
            if extra.is_null() {
                return Ok(items);
            }
            Ok(append_extra_locals_items(items, extra))
        }

        fn copy(&self) -> Result<PyObjectRef, crate::PyError> {
            self.call_mapping_method("copy", &[])
        }

        fn get(
            &self,
            key: PyObjectRef,
            #[default(pyre_object::w_none())] default: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            // `framelocalsproxy_get` is `framelocalsproxy_getitem` with a miss
            // turned into `default`.  Routing through the snapshot dict paid a
            // string per bound local for a single lookup.
            let roots = pyre_object::gc_roots::push_roots();
            let key_slot = roots.publish(&[key, default]);
            roots.normalize(key_slot, 2);
            let default_slot = key_slot + 1;
            match self.__getitem__(roots.get(key_slot)) {
                Ok(value) => Ok(value),
                Err(err) if err.kind == crate::PyErrorKind::KeyError => Ok(roots.get(default_slot)),
                Err(err) => Err(err),
            }
        }

        fn update(&mut self, other: PyObjectRef) -> Result<(), crate::PyError> {
            // `framelocalsproxy_update` reports EVERY merge failure as this
            // one message, the argument's own exception included.
            match self.merge(other) {
                Ok(true) => Ok(()),
                _ => Err(crate::PyError::type_error(
                    "update() argument must be dict or another FrameLocalsProxy",
                )),
            }
        }

        fn setdefault(
            &mut self,
            key: PyObjectRef,
            #[default(pyre_object::w_none())] default: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            // The lookup allocates its snapshot and can run a Python `__eq__`
            // before the miss, so both arguments are stale by the store.
            let roots = pyre_object::gc_roots::push_roots();
            let key_slot = roots.publish(&[key, default]);
            roots.normalize(key_slot, 2);
            let default_slot = key_slot + 1;
            match self.__getitem__(roots.get(key_slot)) {
                Ok(value) => Ok(value),
                Err(err) if err.kind == crate::PyErrorKind::KeyError => {
                    self.setitem_value(roots.get(key_slot), roots.get(default_slot))?;
                    Ok(roots.get(default_slot))
                }
                Err(err) => Err(err),
            }
        }

        fn pop(&mut self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            // Raw slice methods receive the bound receiver in args[0].
            if args.len() < 2 || args.len() > 3 {
                return Err(crate::PyError::type_error("pop expected 1 or 2 arguments"));
            }
            // `framelocalsproxy_pop`: the scan decides, and only a key it
            // places nowhere reaches the extras dict.  Probing the snapshot
            // first discarded the lookup's error, so an unhashable key came
            // back as a `KeyError` naming that key instead of the `TypeError`
            // the hash raises.
            //
            // The extras lookup allocates, so publish the incoming arguments
            // and rebuild the slice for the delegated call.
            let roots = pyre_object::gc_roots::push_roots();
            let args_base = roots.publish(&args[1..]);
            let nargs = args.len() - 1;
            roots.normalize(args_base, nargs);
            if self.key_is_fast_local(roots.get(args_base))? {
                return Err(crate::PyError::value_error(
                    "cannot remove local variables from FrameLocalsProxy",
                ));
            }
            let extra = self.frame().get_extra_locals();
            if extra.is_null() {
                // A frame with no extras dict answers the default if one was
                // passed and reports the key itself otherwise; it does not
                // gain a dict from a pop that finds nothing.
                return if nargs == 2 {
                    Ok(roots.get(args_base + 1))
                } else {
                    Err(crate::PyError::key_error_with_key(roots.get(args_base)))
                };
            }
            let extra_slot = args_base + nargs;
            let _ = roots.pin_root(extra);
            let mut call_args = Vec::with_capacity(nargs);
            for i in 0..nargs {
                call_args.push(roots.get(args_base + i));
            }
            let result = crate::baseobjspace::call_method(roots.get(extra_slot), "pop", &call_args);
            if result.is_null() {
                Err(crate::call::take_call_error()
                    .unwrap_or_else(|| crate::PyError::runtime_error("pop failed")))
            } else {
                Ok(result)
            }
        }

        /// `framelocalsproxy_or` — build a fresh dict, fill it from the
        /// proxy, then `PyDict_Update` it from the operand.  The gate admits
        /// another proxy as well as a `dict`, and reading the operand through
        /// `keys()` / `__getitem__` is what lets one in.  Delegating to
        /// `dict.__or__` declined a proxy on the right, and because both
        /// operands then share a type the reflected slot is never tried, so
        /// `p | q` raised a `TypeError`.
        fn __or__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            if !unsafe { crate::baseobjspace::isinstance_dict_w(other) }
                && viewed_frame(other).is_none()
            {
                return Ok(pyre_object::w_not_implemented());
            }
            let roots = pyre_object::gc_roots::push_roots();
            let other_slot = roots.base();
            let _ = roots.pin_root(other);
            // `PyDict_New` then `PyDict_Update(result, self)`: the snapshot is
            // already a fresh dict holding exactly the proxy's entries.
            let result = self.mapping()?;
            let result_slot = other_slot + 1;
            let _ = roots.pin_root(result);
            crate::opcode_ops::dict_update(roots.get(result_slot), roots.get(other_slot))?;
            Ok(roots.get(result_slot))
        }

        /// `nb_or` serves both directions upstream, so `__ror__` is the
        /// `wrap_binaryfunc_r` wrapper over it and reaches
        /// `framelocalsproxy_or` with the operands swapped — the type check
        /// then guards the PROXY, which always passes, while `PyDict_Update`
        /// runs against the other operand.  So there is no gate on this side,
        /// and `3 | proxy` raises `AttributeError: 'int' object has no
        /// attribute 'keys'` rather than the generic operator TypeError.
        ///
        /// `binary_op1` invokes the slot in `(left, right)` order both ways,
        /// so the operand is the one filled in first and the proxy's own
        /// entries win — and the proxy is snapshotted only afterwards,
        /// because the operand's `keys()` / `__getitem__` may have written to
        /// the frame.
        fn __ror__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            let roots = pyre_object::gc_roots::push_roots();
            let other_slot = roots.base();
            let _ = roots.pin_root(other);
            let result_slot = other_slot + 1;
            let _ = roots.pin_root(pyre_object::w_dict_new());
            crate::opcode_ops::dict_update(roots.get(result_slot), roots.get(other_slot))?;
            let snapshot_slot = result_slot + 1;
            let _ = roots.pin_root(self.mapping()?);
            crate::opcode_ops::dict_update(roots.get(result_slot), roots.get(snapshot_slot))?;
            Ok(roots.get(result_slot))
        }

        fn __ior__(&mut self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            // `framelocalsproxy_inplace_or` declines anything that is neither
            // a `dict` nor another proxy, so the operand's `__ror__` runs.
            // Routing it into `update` instead iterated the operand: a list of
            // pairs was accepted and a `str` reported a `dict.update` error.
            //
            // A merge that fails AFTER that type check does not surface its own
            // exception: `framelocalsproxy_inplace_or` returns `NotImplemented`
            // with the exception still set, `binary_iop1` discards that and
            // falls through to `framelocalsproxy_or`, and the `PyDict_Update`
            // there calls the proxy's `keys` — whose `_Py_CheckFunctionResult`
            // sees the pending exception and raises the chained `SystemError`.
            //
            // pyre cannot tell an explicit `p.__ior__(d)` from `|=`, so the
            // message names `keys` in both, where the explicit call names the
            // `__ior__` slot wrapper instead.
            match self.merge(other) {
                Ok(true) => Ok(self as *mut FrameLocalsProxy as PyObjectRef),
                Ok(false) => Ok(pyre_object::w_not_implemented()),
                Err(error) => Err(crate::error::system_error_from_cause(
                    "<method 'keys' of 'FrameLocalsProxy' objects> returned a result with an exception set"
                        .to_string(),
                    error,
                )),
            }
        }

        fn __eq__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            self.richcompare(other, crate::baseobjspace::CompareOp::Eq)
        }

        fn __ne__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            self.richcompare(other, crate::baseobjspace::CompareOp::Ne)
        }

        fn __lt__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            self.richcompare(other, crate::baseobjspace::CompareOp::Lt)
        }

        fn __le__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            self.richcompare(other, crate::baseobjspace::CompareOp::Le)
        }

        fn __gt__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            self.richcompare(other, crate::baseobjspace::CompareOp::Gt)
        }

        fn __ge__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            self.richcompare(other, crate::baseobjspace::CompareOp::Ge)
        }

        fn __repr__(&self) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
            // CPython 3.14 `framelocalsproxy_repr` brackets the materialized
            // dict repr with `Py_ReprEnter/Leave(self)`.  The guard must name
            // the proxy, not the temporary dict: `proxy[1] = proxy` then
            // renders as `{1: {...}}` instead of recursively materializing a
            // fresh dict forever.
            let self_obj = self as *const FrameLocalsProxy as PyObjectRef;
            let Some(_guard) = crate::display::ReprGuard::enter(self_obj) else {
                return Ok(rustpython_wtf8::Wtf8Buf::from_string("{...}".to_string()));
            };
            unsafe { crate::display::py_repr_wtf8(self.mapping()?) }
        }
    }

    pub fn new(w_frame: PyObjectRef) -> PyObjectRef {
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(w_frame);
        let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let w_type = type_object();
        unsafe { pyre_object::typeobject::w_type_set_flag_map_or_seq(w_type, b'M') };
        FrameLocalsProxy::allocate_stable(FrameLocalsProxy {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_frame: pyre_object::gc_roots::shadow_stack_get(slot),
        })
    }
}

/// The `f_extra_locals` half of [`PyFrame::frame_locals_proxy_snapshot`]:
/// store every entry of `extra` that `snapshot` does not already carry.
///
/// A key a bound slot already supplied is skipped because
/// `framelocalsproxy_getitem` answers it from the slot — the store would only
/// rewrite the entry with the value it already holds, and the key keeps its
/// slot position either way.
///
/// Its own function, and deliberately UNHINTED, for the reason
/// `pin_extra_locals_entries` gives: this loop is bounded by the dict's
/// length, which is red, while the slot loop that precedes it is bounded by
/// the green `locals_plus_names` and is what lets the caller carry
/// `@jit.unroll_safe`.
fn merge_extra_locals(snapshot: PyObjectRef, extra: PyObjectRef) -> Result<(), crate::PyError> {
    // `setitem` allocates and can run the key's `__hash__`, so the pairs still
    // queued in this native snapshot are published first and read back per
    // store.
    let roots = pyre_object::gc_roots::push_roots();
    let items = unsafe { pyre_object::dictmultiobject::w_dict_items(extra) };
    let mut live = Vec::with_capacity(1 + items.len() * 2);
    live.push(snapshot);
    for &(key, value) in &items {
        live.push(key);
        live.push(value);
    }
    let snapshot_slot = roots.pin_roots(&live);
    let items_base = snapshot_slot + 1;
    for index in 0..items.len() {
        let key = roots.get(items_base + index * 2);
        if crate::baseobjspace::contains(roots.get(snapshot_slot), key)? {
            continue;
        }
        crate::baseobjspace::setitem(
            roots.get(snapshot_slot),
            roots.get(items_base + index * 2),
            roots.get(items_base + index * 2 + 1),
        )?;
    }
    Ok(())
}

/// Build the `ob_header` for a freshly-created `PyFrame` — `ob_type`
/// pinned to [`FRAME_TYPE`], `w_class` the cached `W_TypeObject`
/// (null during bootstrap before `init_typeobjects`). Mirrors
/// `pytraceback::pytraceback_new`'s header construction.
#[inline]
fn frame_ob_header() -> PyObject {
    PyObject {
        ob_type: &FRAME_TYPE as *const PyType,
        w_class: get_instantiate(&FRAME_TYPE),
    }
}

/// Execution frame for a single Python code block.
///
/// Unified `locals_cells_stack_w` array layout:
///   - indices `0..nlocals` — local variables
///   - indices `nlocals..nlocals+ncells` — cell/free variable slots
///   - indices `nlocals+ncells..` — operand stack
///
/// `valuestackdepth` is the absolute index into this array; it starts at
/// `nlocals + ncells` (empty stack) and grows upward on push.
///
/// The JIT's Virtualize pass keeps `locals_cells_stack_w` slots in CPU
/// registers during compiled code execution, eliminating heap reads/writes
/// for the hottest interpreter state.
///
/// The `vable_token` field coordinates ownership: when JIT code is
/// running, the token is nonzero and the canonical field values live
/// in registers. A "force" flushes them back to the heap.
#[repr(C)]
pub struct PyFrame {
    /// `PyObject` prefix (`ob_type` / `w_class`) making the frame a
    /// Python-visible `W_Root` — `pyframe.py class PyFrame(W_Root)`.
    /// `ob_type` points at [`FRAME_TYPE`]. Kept at offset 0 like every
    /// other `PyObject`-layout struct so `ob_type` reads land on the
    /// typeptr the JIT `GuardClass` / `type()` expect.
    pub ob_header: PyObject,
    /// Pointer to the Code object (PyCode).
    ///
    /// PyPy: pyframe.py `self.pycode = code` — stores the PyCode instance.
    /// Same pointer as `func.getcode()`, so `getcode(func) == frame.pycode`.
    pub pycode: *const (),
    /// pypy/interpreter/pyframe.py:84,110-112 locals_cells_stack_w
    /// `[None] * size; make_sure_not_resized(...)` → fixed-length GcArray.
    pub locals_cells_stack_w: *mut FixedObjectArray,
    /// Absolute index into `locals_cells_stack_w` marking the top of the
    /// operand stack. Starts at `nlocals + ncells` (empty stack), grows upward.
    pub valuestackdepth: usize,
    /// `pyframe.py:72 last_instr` — index of the currently-dispatching (or
    /// just-dispatched) instruction. Initialized to `-1` (frame not yet
    /// entered). `get_last_lineno` uses this for `offset2lineno`.
    ///
    /// Storage semantic matches RPython.  `pypy/interpreter/pyopcode.py:172
    /// self.last_instr = intmask(next_instr)` writes the to-be-dispatched
    /// byte-offset at the top of every dispatch iteration, and pyre's
    /// `eval.rs::eval_loop` writes `frame.last_instr = pc` at the same
    /// point.  At any inspection boundary (between opcodes, at a guard,
    /// during a handler) both sides hold "current/just-completed opcode"
    /// modulo bytes-vs-instruction units.
    ///
    /// TODO(pattern-level, not semantic): RPython's
    /// dispatch carries the next-to-execute pc as a separate `next_instr`
    /// local variable advanced by `next_instr += 2` after the opcode read.
    /// pyre packs the same information into `last_instr` plus the
    /// `next_instr()` accessor below, which returns `last_instr + 1`.
    /// `set_last_instr_from_next_instr` is the inverse setter, and the
    /// JIT vable shadow `vable_last_instr` mirrors the same field. The
    /// `±1` arithmetic on either side of the accessor cancels — the
    /// runtime never observes a divergence — so this remains a structural
    /// adaptation, not a parity bug.
    pub last_instr: isize,
    /// Packed frame status bits, one byte in place of two RPython
    /// class-level bools: `FLAG_ESCAPED` (pyframe.py:80 escaped — see
    /// mark_as_escaped()) and `FLAG_FRAME_FINISHED`
    /// (frame_finished_execution).  Access through the
    /// `escaped()`/`set_escaped()`/`frame_finished_execution()`/
    /// `set_frame_finished_execution()` methods.
    pub flags: u8,
    /// CPython 3.14 observable finalizer timing adaptation for a temporary
    /// receiver consumed by a failing `LOAD_ATTR`.  PyPy's tracing GC does
    /// not need this state, but pyre must retain it on the owning red frame
    /// until the exception stack has discarded the receiver.  It is a
    /// separate ordinary byte beside `flags`.  PyPy represents the adjacent
    /// frame-status booleans as byte fields; this CPython-only state has only
    /// six values, so widening it to a machine word would make every hot call
    /// frame pay for cold exception cleanup.  It remains a concrete red-frame
    /// getfield/setfield, just with the narrow storage PyPy's rtyper would
    /// select for the same finite state.
    pub failed_attr_cleanup: u8,
    /// pyframe.py:82 debugdata — lazily allocated tracing/debug payload.
    /// Virtualizable static field (interp_jit.py:28).
    pub debugdata: *mut FrameDebugData,
    /// Virtualizable token — set by JIT when this frame is virtualized.
    /// 0 = not virtualized, nonzero = pointer to JIT state.
    pub vable_token: usize,
    /// pyframe.py `f_generator_wref` — a strong edge to the WEAKREF struct
    /// (`pyre_object::weakref::Weakref`) whose `weakptr` is weak; NULL is
    /// `rweakref.dead_ref`.  `initialize_as_generator` stores
    /// `rweakref.ref(gen)`; `get_generator` dereferences it.
    pub f_generator_wref: PyObjectRef,
    /// PyPy: `w_yielding_from = None`.
    pub w_yielding_from: PyObjectRef,
    /// PyPy: `f_backref = jit.vref_None`.
    pub f_backref: *mut PyFrame,
    /// pyframe.py:115-116 — `self.builtin = space.builtin
    /// .pick_builtin(w_globals)`, gated under `honor__builtins__`
    /// (`baseobjspace::HONOR_BUILTINS`, default False).  With the option
    /// off (the default) `frame_builtin*` returns `space.builtin`, ignoring
    /// a custom `__builtins__` in globals; with it on the `pick_builtin*`
    /// family honors `globals['__builtins__']`.  Consulted by LOAD_GLOBAL's
    /// builtins fallback (`pyopcode.py`) via
    /// `space.finditem_str(self.w_builtin.w_dict, name)` — honouring
    /// dict subclass `__getitem__` overrides (`moduledef.py:102-103`)
    /// without an extra storage-keyed fast path.  `frame.get_builtin()`
    /// returns this directly so callers (`exec`'s
    /// `setdefault('__builtins__', self.get_builtin())` at
    /// `pyopcode.py`) see the picked Module, not the EC's
    /// default builtin.
    pub w_builtin: PyObjectRef,
}

/// GC type id for `PyFrame`. Reserved ahead of any callsite that allocates
/// frames through `NewWithVtable` / `New` in trace IR — the GC consults
/// the registered `TypeInfo` to write the type tag into the header. Today
/// every `PyFrame` is heap-allocated outside the nursery (`std::alloc`
/// + a leaked `Box`) and roots are visited by the custom walker
/// (`pyre-interpreter::eval::walk_pyframe_roots`); this id is therefore
/// metadata-only at registration time.
/// `emit_new_pyframe_inline_self_recursive` will be the first writer.
///
/// Asserts the same id is returned by `gc.register_type(...)` so any
/// drift panics on startup.
pub const PYFRAME_GC_TYPE_ID: u32 = 37;

/// GC type id appended after the existing runtime registration census.
/// `FrameDebugData` is stationary old-gen for a GC-owned frame.  Keep this
/// at the tail of `pyre-jit::eval::build_gc` so older type ids never shift.
pub const FRAME_DEBUG_DATA_GC_TYPE_ID: u32 = 98;

/// GC header size in bytes — single source of truth is
/// [`majit_gc::header::GcHeader::SIZE`]. Every `FixedObjectArray` and
/// `PyFrame` allocation prepends this many zero bytes so RPython-style
/// write barriers (`*(obj + wb_byteofs) & mask`) read a valid header with
/// `TRACK_YOUNG_PTRS=0` and skip the slow path. Scalar `value`
/// allocations route through [`majit_gc::header::alloc_with_gc_header`].
pub const GC_HEADER_SIZE: usize = majit_gc::header::GcHeader::SIZE;

/// Ownership selected by the caller that decides a frame's lifetime.
/// Normal call frames use `NurseryGc`, matching `PyFrame.__init__`'s fresh
/// `[None] * size`; frame-owned auxiliary snapshots use `OldGenGc`, and
/// tracer-private snapshots use `StdAlloc` until deterministic `Drop`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameLocalsArrayAllocation {
    NurseryGc,
    OldGenGc,
    StdAlloc,
}

/// Allocation size (in bytes, including the GC header) for a
/// `FixedObjectArray` of the given length.
#[inline]
fn fixed_array_alloc_size(len: usize) -> usize {
    GC_HEADER_SIZE
        + pyre_object::FIXED_ARRAY_ITEMS_OFFSET
        + len * std::mem::size_of::<pyre_object::PyObjectRef>()
}

#[inline]
fn fixed_array_layout(len: usize) -> std::alloc::Layout {
    // Track the header alignment (the block leads with a GcHeader the barrier
    // reads at `obj - SIZE`), not a hardcoded 8.
    let align = std::mem::align_of::<majit_gc::header::GcHeader>()
        .max(std::mem::align_of::<FixedObjectArray>());
    std::alloc::Layout::from_size_align(fixed_array_alloc_size(len), align).unwrap()
}

/// Allocate a fixed-length GcArray-shaped `FixedObjectArray` with all
/// slots initialised to `fill`. The allocation is prefixed with a
/// zeroed GC header ([`GC_HEADER_SIZE`] bytes) so the write barrier
/// fast-path sees `TRACK_YOUNG_PTRS=0`.
///
/// The returned pointer points at the length prefix; items follow
/// immediately in memory at `FIXED_ARRAY_ITEMS_OFFSET` — matching
/// RPython `Ptr(GcArray(PyObjectRef))`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn alloc_fixed_array_with_header(
    len: usize,
    fill: pyre_object::PyObjectRef,
) -> *mut FixedObjectArray {
    unsafe {
        let layout = fixed_array_layout(len);
        let raw = std::alloc::alloc_zeroed(layout);
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let arr = raw.add(GC_HEADER_SIZE) as *mut FixedObjectArray;
        (*arr).len = len;
        let items = (arr as *mut u8).add(pyre_object::FIXED_ARRAY_ITEMS_OFFSET)
            as *mut pyre_object::PyObjectRef;
        for i in 0..len {
            items.add(i).write(fill);
        }
        arr
    }
}

/// Allocate a frame locals array in the lifetime regime selected by its owner.
/// The old-gen form is the type-9 GcArray layout `[GcHeader | len | items]`.
unsafe fn alloc_frame_locals_array(
    len: usize,
    fill: pyre_object::PyObjectRef,
    allocation: FrameLocalsArrayAllocation,
) -> *mut FixedObjectArray {
    let payload = pyre_object::FIXED_ARRAY_ITEMS_OFFSET
        + len * std::mem::size_of::<pyre_object::PyObjectRef>();
    let raw = match allocation {
        FrameLocalsArrayAllocation::NurseryGc => pyre_object::gc_hook::GcAllocOutcome::from_hook(
            pyre_object::gc_hook::try_gc_alloc(pyre_object::PY_OBJECT_ARRAY_GC_TYPE_ID, payload),
        )
        .allocated_or_abort(payload)
        .unwrap_or(std::ptr::null_mut()),
        FrameLocalsArrayAllocation::OldGenGc => pyre_object::gc_hook::try_gc_alloc_stable_raw(
            pyre_object::PY_OBJECT_ARRAY_GC_TYPE_ID,
            payload,
        ),
        FrameLocalsArrayAllocation::StdAlloc => std::ptr::null_mut(),
    };
    if !raw.is_null() {
        // Nursery memory is not zero-filled. The length and every item are
        // written before the next collection point, so the array is never
        // reachable from a root while it still holds allocator garbage; the
        // caller publishes the completed array.
        let arr = raw as *mut FixedObjectArray;
        unsafe {
            (*arr).len = len;
            let items = (*arr).items_mut_ptr();
            for i in 0..len {
                items.add(i).write(fill);
            }
        }
        // Nursery-full spill is old-gen; remember young `fill` values.
        remember_frame_locals_array(arr);
        return arr;
    }
    unsafe { alloc_fixed_array_with_header(len, fill) }
}

/// Write barrier for a batch of stores into a frame's
/// `locals_cells_stack_w`: the array is its own GC object, so an old array
/// receiving young values must join the remembered set.
///
/// `incminimark.py write_barrier` remembers an object only when
/// `GCFLAG_TRACK_YOUNG_PTRS` is set. Nursery objects are born without that
/// flag, and a raw `alloc_fixed_array_with_header` block has a zeroed header
/// (`GcFramePrefix`), so the flag test covers both regimes. The membership
/// query would only reject the same flag-clear addresses.
///
/// `dont_look_inside`: the flag is mutable GC state. A traced load can
/// keep the clear bit from the young birth and skip the barrier after the
/// array is promoted (`incminimark.py write_barrier` reads it at the store).
#[inline]
#[majit_macros::dont_look_inside]
pub fn remember_frame_locals_array(array: *mut FixedObjectArray) {
    if array.is_null() {
        return;
    }
    let tracks_young = unsafe {
        (*majit_gc::header::header_of(array as usize))
            .has_flag(majit_gc::GcFlags::GCFLAG_TRACK_YOUNG_PTRS)
    };
    if tracks_young {
        pyre_object::gc_hook::try_gc_write_barrier(array as pyre_object::gc_hook::GCREF);
    }
}

/// Deallocate a `FixedObjectArray` allocated with
/// [`alloc_fixed_array_with_header`].
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn dealloc_array_with_gc_header(ptr: *mut FixedObjectArray) {
    if ptr.is_null() {
        return;
    }
    debug_assert!(!pyre_object::gc_hook::try_gc_owns_object(
        ptr as pyre_object::gc_hook::GCREF
    ));
    unsafe {
        let len = (*ptr).len;
        let raw = (ptr as *mut u8).sub(GC_HEADER_SIZE);
        std::alloc::dealloc(raw, fixed_array_layout(len));
    }
}

/// Heap layout for a managed `PyFrame`: a leading GC header followed by
/// the frame body. The header occupies [`GC_HEADER_SIZE`] bytes and stays
/// zeroed (`type_id = 0`, `flags = 0`) so the write-barrier fast-path reads
/// `TRACK_YOUNG_PTRS = 0` at `frame - GC_HEADER_SIZE` and skips the slow
/// path. Mirrors `FixedObjectArray`'s header prefix and the callee-frame
/// allocation in `pyre-jit::call_jit`, realising the documented invariant
/// that every heap `PyFrame` is a header-bearing GC object.
#[repr(C)]
struct GcFramePrefix {
    gc_header: u64,
    frame: PyFrame,
}

// The frame must sit exactly GC_HEADER_SIZE bytes into the prefix: the write
// barrier reads the header at `frame - GC_HEADER_SIZE`, and FrameBox::drop
// reconstructs the owning Box from that same address. A future PyFrame field
// with alignment greater than the header word would make `#[repr(C)]` pad the
// frame to a larger offset and silently break both, so pin the offset here.
const _: () = assert!(std::mem::offset_of!(GcFramePrefix, frame) == GC_HEADER_SIZE);

/// Owning handle to a heap [`PyFrame`] allocated with a leading GC header.
/// Dereferences to the inner `PyFrame`; the header lives at
/// `self.ptr - GC_HEADER_SIZE`. Replaces the bare `Box<PyFrame>` so the
/// frame the JIT treats as the virtualizable is a real GC object with a
/// valid header at `frame - GC_HEADER_SIZE`.
pub struct FrameBox {
    ptr: *mut PyFrame,
    /// RPython's translated Rust local is a shadow-stack livevar for the
    /// complete lifetime of the owning handle.  In particular it remains a
    /// root after `FrameBox::new` returns and before `execute_frame` installs
    /// the frame on the ExecutionContext chain.
    owner_root: Option<majit_gc::shadow_stack::OwnerRootGuard>,
}

/// `FrameBox::new` takes the frame **by value**, and the codewriter depends on
/// it: a virtualizable-field projection whose base is a non-dereferenced local
/// aggregate is not treated as reaching a live virtualizable, which is what
/// keeps the twelve frame-construction projections in `new` off the
/// `getfield_vable_*` / `setfield_vable_*` path.  Allocating first and
/// initialising through a pointer turns every one of them into a deref and
/// withdraws that suppression silently — nothing fails where the edit is made.
///
/// Changing this signature is therefore a decision, not a refactor.
/// `pyre-jit-trace`'s `vable_projection_census` test is the corpus-level check.
const _: fn(PyFrame) -> FrameBox = FrameBox::new;

/// First published-root index [`FrameBox::new`] gives to the interior of a
/// `std::alloc` `FrameDebugData`, after the frame's own eight GCREF fields.
const FRAME_BOX_DEBUG_INPUT: usize = 8;

impl FrameBox {
    /// Move `frame` onto the heap behind a GC header.
    ///
    /// `pyframe.py class PyFrame(W_Root)` — an executing frame is a normal
    /// GC object whose lifetime is its reachability.  When the GC hook is
    /// installed this allocates a non-moving YOUNG `PYFRAME_GC_TYPE_ID`
    /// block (`external_malloc(..., alloc_young=True)`, through
    /// `try_gc_alloc_young_nonmoving_raw`); the collector reclaims the frame
    /// and its GC-managed locals and debug data once no root
    /// (the handle's own owner root while it lives, then `walk_pyframe_roots`
    /// over the `CURRENT_FRAME` / `f_backref` chain or a traced edge such as
    /// a traceback's `tb_frame`) reaches it, so `Drop` performs no manual
    /// free (`executioncontext.py leave` frees nothing either).
    ///
    /// Born young rather than old because upstream's frame is a nursery
    /// object: a returned frame nothing retains dies at the next minor
    /// collection, and until then its slots are not traced.  A frame born
    /// old instead joins the remembered set the moment it holds a young
    /// argument box, and the next minor traces it whether or not it is still
    /// reachable, promoting everything its dead slots name — measured
    /// 92-99.7% of the nursery surviving each minor on a module-level loop
    /// calling a compiled function that raises, against 15% with the JIT off.
    ///
    /// The non-moving part is pyre's, not upstream's: `pyframe.py class
    /// PyFrame(W_Root)` declares no placement hint and a minor collection
    /// relocates it, rewriting every referring slot
    /// (`rpython/memory/gc/incminimark.py` / `:2252`) because the
    /// translator delivers roots as slot addresses and rewrites live GCREF
    /// locals in them (`rpython/memory/gctransform/shadowstack.py:43-46`).
    /// Rust has no such pass, and a running opcode holds a `&mut PyFrame`
    /// across its own allocations, so this allocation standing still is what
    /// takes the place of that rewrite.  It is not a rule about every frame —
    /// a compiled trace's inlined-callee frame is a nursery allocation
    /// (`emit_new_pyframe_inline_with_params` -> `NewWithVtable`), so the
    /// crossing into raw-pointer territory is guarded at the seams instead,
    /// by `crate::eval::FrameAnchor` and the JIT layer's `FrameRoot`.
    ///
    /// Making it uniform is one line —
    /// `PYFRAME_DESCR_GROUP.size_descr.set_non_moving(true)` sends
    /// `NewWithVtable` down `gen_malloc_fixedsize` to the old generation —
    /// and it costs 3.6x on `bench/fib_recursive.py`: 0.39s against 1.43s,
    /// user CPU, five samples per arm, arm64 darwin, 2026-08-20.  An old-gen
    /// frame holding young argument boxes joins the remembered set and is
    /// re-traced every minor collection, and each returned frame is old-gen
    /// garbage only a major reclaims.  Pinning is not an alternative:
    /// `GcAllocator::pin` ports `gctypelayout.py q_cannot_pin` and
    /// refuses any type carrying GC pointers or a custom trace, which a
    /// `PyFrame` does both.
    ///
    /// Before the hook is wired (bootstrap, tests) `try_gc_alloc_stable`
    /// returns `None`; fall back to the `std::alloc` `GcFramePrefix` box,
    /// which `Drop` frees manually.  The two regimes share one memory
    /// layout — an 8-byte GC header immediately before the frame body — so
    /// every reader (`frame - GC_HEADER_SIZE` write-barrier header,
    /// `pyframe_object_custom_trace`) is regime-independent; `Drop`
    /// distinguishes them with `try_gc_owns_object`.
    pub fn new(mut frame: PyFrame) -> Self {
        // `frame` is an RPython aggregate local whose GCREF fields are all
        // translated livevars across the frame allocation. Publish every field
        // before it, the way `gct_fv_gc_malloc` brackets a malloc with the
        // livevars it holds: old-gen buys immobility, not survival, so a field
        // no heap edge names yet is what the next cycle sweeps.
        //
        // One `push_roots(hop)` bracket over the whole set, not ten
        // independently-owned slots: their lifetime is exactly this function
        // body, which is the lexical shape `gc_enter_roots_frame` /
        // `gc_leave_roots_frame` express (`shadowcolor.py add_enter_leave_roots_frame`), and
        // `pin_roots` is the one-phase publish `push_roots` performs — pinning
        // them one at a time would query for forwarding after the first write
        // and let a foreign collection run before the later values were
        // visible.
        //
        // Publish and normalize through the bracket's own cell: the free
        // `pin_roots` resolves the thread-local once per phase, which on
        // Darwin is an out-of-line `_tlv_get_addr` each time.
        //
        // A `FrameDebugData` the collector owns needs nothing beyond the
        // pointer: `FRAME_DEBUG_DATA_GC_TYPE_ID` declares every `PyObjectRef`
        // offset in it, so the ordinary walker forwards the interior.  A
        // `std::alloc` block does not — `finish_for_call_with_globals_obj`
        // creates the debug block on a stack `PyFrame`, where `aux_allocation`
        // answers `StdAlloc`, and the globals override a shared code object
        // stores there is then named by nothing the collector scans.  Publish
        // that block's own fields alongside the frame's and write them back.
        let frame_root = pyre_object::gc_roots::push_roots();
        let unscanned_debugdata = !frame.debugdata.is_null()
            && !pyre_object::gc_hook::try_gc_owns_object(
                frame.debugdata as pyre_object::gc_hook::GCREF,
            );
        let mut published = [
            frame.ob_header.w_class,
            frame.pycode as pyre_object::PyObjectRef,
            frame.locals_cells_stack_w as pyre_object::PyObjectRef,
            frame.debugdata as pyre_object::PyObjectRef,
            frame.f_generator_wref,
            frame.w_yielding_from,
            frame.f_backref as pyre_object::PyObjectRef,
            frame.w_builtin,
            PY_NULL,
            PY_NULL,
            PY_NULL,
            PY_NULL,
            PY_NULL,
        ];
        if unscanned_debugdata {
            let debug = unsafe { &*frame.debugdata };
            published[FRAME_BOX_DEBUG_INPUT..].copy_from_slice(&[
                debug.w_globals,
                debug.w_locals,
                debug.w_extra_locals,
                debug.w_f_trace,
                debug.hidden_operationerr,
            ]);
        }
        let published: &[pyre_object::PyObjectRef] = if unscanned_debugdata {
            &published
        } else {
            &published[..FRAME_BOX_DEBUG_INPUT]
        };
        let inputs = frame_root.publish(published);
        frame_root.normalize(inputs, published.len());
        let raw = pyre_object::gc_hook::try_gc_alloc_young_nonmoving_raw(
            PYFRAME_GC_TYPE_ID,
            std::mem::size_of::<PyFrame>(),
        );
        if !raw.is_null() {
            // RPython allocation results stay live in the translated shadow
            // stack while the caller initializes their fields.  Free-threaded
            // pyre must expose the same root before entering the write-barrier
            // GC operation: a contended barrier is an entry safepoint, so an
            // unrooted fresh frame could otherwise be swept by another
            // collector between this allocation and the barrier below.
            let _ = frame_root.pin_root(raw as pyre_object::PyObjectRef);
            // Read back through the bracket's own cell rather than
            // `shadow_stack_get`, which resolves the thread-local per slot.
            frame.ob_header.w_class = frame_root.get(inputs);
            frame.pycode = frame_root.get(inputs + 1) as *const ();
            frame.locals_cells_stack_w =
                frame_root.get(inputs + 2) as *mut pyre_object::FixedObjectArray;
            frame.debugdata = frame_root.get(inputs + 3) as *mut FrameDebugData;
            frame.f_generator_wref = frame_root.get(inputs + 4);
            frame.w_yielding_from = frame_root.get(inputs + 5);
            frame.f_backref = frame_root.get(inputs + 6) as *mut PyFrame;
            frame.w_builtin = frame_root.get(inputs + 7);
            if unscanned_debugdata {
                let debug = unsafe { &mut *frame.debugdata };
                debug.w_globals = frame_root.get(inputs + FRAME_BOX_DEBUG_INPUT);
                debug.w_locals = frame_root.get(inputs + FRAME_BOX_DEBUG_INPUT + 1);
                debug.w_extra_locals = frame_root.get(inputs + FRAME_BOX_DEBUG_INPUT + 2);
                debug.w_f_trace = frame_root.get(inputs + FRAME_BOX_DEBUG_INPUT + 3);
                debug.hidden_operationerr = frame_root.get(inputs + FRAME_BOX_DEBUG_INPUT + 4);
            }
            debug_assert!(pyre_object::gc_hook::try_gc_owns_object(
                frame.locals_cells_stack_w as pyre_object::gc_hook::GCREF
            ));
            let ptr = raw as *mut PyFrame;
            unsafe {
                std::ptr::write(ptr, frame);
            }
            // A young frame needs no barrier for the young argument / locals
            // objects it names, and the barrier's TRACK_YOUNG_PTRS test
            // answers so.  A refused young birth is old-gen, and that frame
            // must be remembered for the next minor tracer, exactly as
            // `generator.rs`'s `w_generator_or_coroutine_new` does for a
            // stable generator wrapping young frame contents.
            pyre_object::gc_hook::try_gc_write_barrier_managed(raw);
            let owner_root =
                majit_gc::shadow_stack::OwnerRootGuard::new(majit_ir::GcRef(ptr as usize));
            drop(frame_root);
            return FrameBox {
                ptr,
                owner_root: Some(owner_root),
            };
        }
        debug_assert!(!pyre_object::gc_hook::try_gc_owns_object(
            frame.locals_cells_stack_w as pyre_object::gc_hook::GCREF
        ));
        FrameBox::new_boxed(frame)
    }

    /// Allocate a frame outside the GC — a plain `std::alloc`
    /// `GcFramePrefix` box reclaimed by `Drop`.
    ///
    /// The sole caller is [`Self::new`]'s fallback for the window before the
    /// GC hook is installed (bootstrap, unit tests).  Nothing is collected in
    /// that window, so a frame allocated here is never swept and never moves,
    /// which is what lets the rest of the interpreter treat every frame
    /// address as stable.
    fn new_boxed(frame: PyFrame) -> Self {
        let raw = Box::into_raw(Box::new(GcFramePrefix {
            gc_header: 0,
            frame,
        }));
        let ptr = unsafe { std::ptr::addr_of_mut!((*raw).frame) };
        FrameBox {
            ptr,
            owner_root: None,
        }
    }

    /// The frame's current address.
    ///
    /// `owner_root` is this handle's translated livevar slot.  A collection
    /// rewrites the slot in place (`rpython/memory/gc/incminimark.py:2252`),
    /// which is why RPython reads a GCREF local back out of its shadow-stack
    /// slot rather than from a register it filled before the last operation
    /// that could collect; the raw `ptr` field is that stale register, so
    /// every read of the frame goes through the slot instead.  A
    /// [`Self::new_boxed`] frame holds no slot and is not GC-managed, so its
    /// address is fixed and the field is the only answer.
    #[inline]
    fn frame_ptr(&self) -> *mut PyFrame {
        match &self.owner_root {
            Some(owner_root) => owner_root.get().0 as *mut PyFrame,
            None => self.ptr,
        }
    }

    /// Relinquish ownership, returning the inner-frame pointer. The header
    /// remains at `ptr - GC_HEADER_SIZE`; reclaim via [`FrameBox::from_raw`].
    pub fn into_raw(mut self) -> *mut PyFrame {
        // Read the slot before releasing it — afterwards the handle has no
        // root to answer from and only the stale field remains.
        let ptr = self.frame_ptr();
        drop(self.owner_root.take());
        std::mem::forget(self);
        ptr
    }

    /// Reconstruct ownership from a pointer previously produced by
    /// [`FrameBox::into_raw`].
    ///
    /// # Safety
    /// `ptr` must originate from [`FrameBox::into_raw`] / [`FrameBox::new`]
    /// and must not have been freed.
    pub unsafe fn from_raw(ptr: *mut PyFrame) -> Self {
        // Publish before asking the collector whether the address is managed:
        // the slot is what a walker reads, so it has to name the frame before
        // anything consults the collector about it.
        let owner_root = majit_gc::shadow_stack::OwnerRootGuard::new(majit_ir::GcRef(ptr as usize));
        if pyre_object::gc_hook::try_gc_owns_object(ptr as pyre_object::gc_hook::GCREF) {
            FrameBox {
                ptr,
                owner_root: Some(owner_root),
            }
        } else {
            drop(owner_root);
            FrameBox {
                ptr,
                owner_root: None,
            }
        }
    }

    /// Raw pointer to the inner frame (header at `ptr - GC_HEADER_SIZE`).
    pub fn as_mut_ptr(&mut self) -> *mut PyFrame {
        self.frame_ptr()
    }

    /// Does the collector own this frame, so that [`Drop`] leaves the memory
    /// alive for whatever roots still reach it?
    ///
    /// `false` is the [`FrameBox::new_boxed`] fallback, whose `Drop` frees the
    /// frame at scope end — a caller that publishes `as_mut_ptr` past its own
    /// scope must ask this first.  Answered from the ownership decision
    /// [`FrameBox::new`] already made, so it neither re-enters the collector
    /// for a question already answered nor relies on an address range test.
    pub fn is_gc_owned(&self) -> bool {
        self.owner_root.is_some()
    }

    /// pyframe.py initialize_as_generator — wrap this frame in a generator
    /// object that takes ownership of it. PyPy does `GeneratorIterator(self)`:
    /// the generator references the same frame, no copy. FrameBox already holds
    /// the heap, header-bearing frame, so ownership transfers straight through
    /// `into_raw` without the snapshot copy `PyFrame::initialize_as_generator`
    /// needs for the borrowed-`&mut self` case.
    pub fn into_generator(self) -> crate::PyResult {
        self.into_generator_named(None, None)
    }

    /// `pyframe.py initialize_as_generator(name, qualname)` — function
    /// calls pass the function's current writable metadata so each newly
    /// created generator freezes it independently of the code object.
    /// `__name__` / `__qualname__` are the function's own immutable string
    /// objects, retained by reference exactly as `generator.py:22-23` does.
    pub fn into_generator_named(
        mut self,
        name: Option<PyObjectRef>,
        qualname: Option<PyObjectRef>,
    ) -> crate::PyResult {
        self.fix_array_ptrs();
        let register_final = unsafe {
            crate::pycode::w_code_yields_inside_try(self.pycode as pyre_object::PyObjectRef)
        };
        let is_coroutine = self.code().flags.contains(crate::CodeFlags::COROUTINE);
        let _origin_roots = pyre_object::gc_roots::push_roots();
        let name_slot = name.map(|name| {
            let slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(name);
            slot
        });
        let qualname_slot = qualname.map(|qualname| {
            let slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(qualname);
            slot
        });
        let coroutine_origin_slot = if is_coroutine {
            let origin = capture_coroutine_origin(crate::call::getexecutioncontext());
            let _ = pyre_object::gc_roots::pin_root(origin);
            Some(pyre_object::gc_roots::shadow_stack_len() - 1)
        } else {
            None
        };
        // A suspended generator frame is off the call chain — `f_back` is
        // None until a resume re-links it (`executioncontext.py enter`
        // rebinds `f_backref = topframeref`; pyre does the same at
        // `execute_frame`).  Null it now so the generator's custom trace,
        // which greys the frame block and recurses `f_backref`, never greys
        // the (possibly already-freed) caller frame captured at suspend.
        self.f_backref = std::ptr::null_mut();
        let frame_ptr = self.into_raw();
        // `w_generator_new` allocates and may trigger a collection. Until the
        // generator owns `frame_ptr` (and its custom trace greys the frame
        // block), the frame is an RPython allocation result still live in the
        // translated caller.  Publish both the managed frame itself and the
        // locals slot (the latter also covers the std::alloc fallback frame)
        // across that allocation.
        let _frame_roots = pyre_object::gc_roots::push_roots();
        let frame_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(frame_ptr as pyre_object::PyObjectRef);
        let _root = FrameLocalsRoot::new(frame_ptr);
        // generator.py `self.pycode = frame.pycode`: preserve the exact
        // code object independently of the frame, which is cleared when the
        // generator finishes.
        let pycode = unsafe { (*frame_ptr).pycode as pyre_object::PyObjectRef };
        // generator.py: GeneratorIterator / Coroutine — native `async def`
        // frames have their own object type.  In particular, a Coroutine is
        // awaitable but is not itself an iterator; `__await__` supplies the
        // separate CoroutineWrapper iterator.
        let generator = if unsafe {
            (*frame_ptr)
                .code()
                .flags
                .contains(crate::CodeFlags::ASYNC_GENERATOR)
        } {
            pyre_object::generator::w_async_generator_new(
                frame_ptr as pyre_object::gc_hook::GCREF,
                pycode,
            )
        } else if is_coroutine {
            pyre_object::generator::w_coroutine_new(
                frame_ptr as pyre_object::gc_hook::GCREF,
                pycode,
            )
        } else {
            pyre_object::generator::w_generator_new(
                frame_ptr as pyre_object::gc_hook::GCREF,
                pycode,
            )
        };
        let _generator_roots = pyre_object::gc_roots::push_roots();
        let generator_slot = pyre_object::gc_roots::shadow_stack_len();
        let generator = pyre_object::gc_roots::pin_root(generator);
        if let Some(slot) = coroutine_origin_slot {
            unsafe {
                pyre_object::generator::w_coroutine_set_origin(
                    generator,
                    pyre_object::gc_roots::shadow_stack_get(slot),
                );
            }
        }
        // generator.py:22-23 stores the function's existing `_name` /
        // `_qualname` objects directly on the generator.
        if let Some(slot) = name_slot {
            unsafe {
                pyre_object::generator::w_generator_set_name(
                    generator,
                    pyre_object::gc_roots::shadow_stack_get(slot),
                )
            };
        }
        if let Some(slot) = qualname_slot {
            unsafe {
                pyre_object::generator::w_generator_set_qualname(
                    generator,
                    pyre_object::gc_roots::shadow_stack_get(slot),
                )
            };
        }
        // pyframe.py `initialize_as_generator`: `self.f_generator_wref =
        // rweakref.ref(gen)`.  The WEAKREF allocation can move the generator
        // and collects, so the frame is read back from its pin in
        // `_frame_roots` after it, and the generator from its own pin below.
        // An old frame storing a nursery WEAKREF runs the write barrier.
        unsafe {
            let wref = pyre_object::weakref::w_weakref_new(generator);
            let frame_ptr = pyre_object::gc_roots::shadow_stack_get(frame_slot) as *mut PyFrame;
            (*frame_ptr).f_generator_wref = wref as PyObjectRef;
            if pyre_object::gc_hook::try_gc_owns_object(frame_ptr as pyre_object::gc_hook::GCREF) {
                pyre_object::gc_hook::try_gc_write_barrier(
                    frame_ptr as pyre_object::gc_hook::GCREF,
                );
            }
        }
        let generator = pyre_object::gc_roots::shadow_stack_get(generator_slot);
        // generator.py: every Coroutine needs its `_finalize_` hook for
        // the never-awaited warning. Ordinary generators only need one when
        // collection must unwind a suspended `finally`/`with` body.
        if is_coroutine || register_final {
            crate::executioncontext::register_finalizer(generator);
        }
        Ok(generator)
    }
}

/// Capture `coroutine.cr_origin` from the visible caller chain.
///
/// The coroutine frame has not executed yet, so its `f_backref` is empty.
/// Creation therefore walks the live ExecutionContext from its current top
/// frame, using the PyPy no-hidden-frame traversal, and records summaries in
/// most-recent-first order.
fn capture_coroutine_origin(ec: *const PyExecutionContext) -> PyObjectRef {
    if ec.is_null() {
        return w_none();
    }
    let depth = unsafe { (*ec).coroutine_origin_tracking_depth };
    if depth <= 0 {
        return w_none();
    }

    let _roots = pyre_object::gc_roots::push_roots();
    let mut summary_slots = Vec::new();
    let mut frame = unsafe { (*ec).gettopframe_nohidden() };
    for _ in 0..depth {
        if frame.is_null() {
            break;
        }
        // `fget_f_lineno` below reads `last_instr`, a virtualizable field.
        // The force and the filename object both sit between the reads of this
        // frame, so it is read back out of the anchor at each of them.
        let anchor = unsafe { crate::eval::FrameAnchor::from_raw(frame) };
        crate::executioncontext::force_frame(frame);
        let w_code = unsafe { (*anchor.live()).fget_f_code() };
        let filename_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ =
            pyre_object::gc_roots::pin_root(unsafe { crate::pycode::w_code_filename_obj(w_code) });
        let lineno_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(unsafe { (*anchor.live()).fget_f_lineno() });
        let funcname_slot = pyre_object::gc_roots::shadow_stack_len();
        let w_code = unsafe { (*anchor.live()).fget_f_code() };
        let _ = pyre_object::gc_roots::pin_root(unsafe { crate::pycode::w_code_name_obj(w_code) });
        let summary = w_tuple_new(vec![
            pyre_object::gc_roots::shadow_stack_get(filename_slot),
            pyre_object::gc_roots::shadow_stack_get(lineno_slot),
            pyre_object::gc_roots::shadow_stack_get(funcname_slot),
        ]);
        let _ = pyre_object::gc_roots::pin_root(summary);
        summary_slots.push(pyre_object::gc_roots::shadow_stack_len() - 1);
        frame = crate::executioncontext::ExecutionContext::getnextframe_nohidden(anchor.live());
    }
    w_tuple_new(
        summary_slots
            .into_iter()
            .map(pyre_object::gc_roots::shadow_stack_get)
            .collect(),
    )
}

impl std::ops::Deref for FrameBox {
    type Target = PyFrame;
    #[inline]
    fn deref(&self) -> &PyFrame {
        unsafe { &*self.frame_ptr() }
    }
}

impl std::ops::DerefMut for FrameBox {
    #[inline]
    fn deref_mut(&mut self) -> &mut PyFrame {
        unsafe { &mut *self.frame_ptr() }
    }
}

impl Drop for FrameBox {
    fn drop(&mut self) {
        // GC-managed (old-gen) frames are reclaimed by a major mark-sweep
        // when no root reaches them (`pyframe.py class PyFrame(W_Root)`;
        // `executioncontext.py leave` frees nothing).  Their
        // The collector reclaims their GC-managed locals array, debug data,
        // and block chain with the frame, so no manual cleanup runs here.
        // Only a `std::alloc` snapshot / bootstrap fallback box is freed
        // manually — reconstruct and drop it, which runs `PyFrame::drop`.
        if self.owner_root.take().is_some() {
            return;
        }
        unsafe {
            let prefix = (self.ptr as *mut u8).sub(GC_HEADER_SIZE) as *mut GcFramePrefix;
            drop(Box::from_raw(prefix));
        }
    }
}

/// RAII guard registering a frame's `locals_cells_stack_w` slot as a GC root
/// for the duration of an allocating frame-setup step. Mirrors the callee-locals
/// root the eval path holds in `call.rs`: during setup the freshly-installed
/// locals/cells live only in that array, so an intervening collection would
/// drop or mis-forward them unless the slot is rooted.
///
/// The guard stores the frame, not the field address. `jtransform.py`
/// `rewrite_op_getsubstruct` refuses a GC interior in the Ref bank — the
/// gcmap would treat `frame+offset` as its own object. Computing the slot
/// only inside residual helpers keeps that address out of compiled slots.
pub struct FrameLocalsRoot {
    frame: *mut PyFrame,
    registered: bool,
}

impl FrameLocalsRoot {
    #[majit_macros::dont_look_inside]
    pub fn new(frame_ptr: *mut PyFrame) -> Self {
        let registered = unsafe { register_frame_locals_slot(frame_ptr) };
        Self {
            frame: frame_ptr,
            registered,
        }
    }
}

impl Drop for FrameLocalsRoot {
    fn drop(&mut self) {
        if self.registered {
            unregister_frame_locals_slot(self.frame);
        }
    }
}

/// `addr_of_mut!(locals_cells_stack_w)` is an interior address. Residual so a
/// compiled gcmap cannot mark it as a GCREF (`rewrite_op_getsubstruct`).
#[majit_macros::dont_look_inside]
pub unsafe fn register_frame_locals_slot(frame_ptr: *mut PyFrame) -> bool {
    let slot = unsafe {
        std::ptr::addr_of_mut!((*frame_ptr).locals_cells_stack_w)
            as *mut pyre_object::gc_hook::GCREF
    };
    unsafe { pyre_object::gc_hook::try_gc_add_root(slot) }
}

#[majit_macros::dont_look_inside]
pub fn unregister_frame_locals_slot(frame_ptr: *mut PyFrame) {
    let slot = unsafe {
        std::ptr::addr_of_mut!((*frame_ptr).locals_cells_stack_w)
            as *mut pyre_object::gc_hook::GCREF
    };
    pyre_object::gc_hook::try_gc_remove_root(slot);
}

/// Write `locals_cells_stack_w` through a pointer. A by-value
/// `frame.locals_cells_stack_w =` is a non-deref virtualizable-field
/// projection, which `vable_projection_census` allows only on
/// `FrameBox::new`. `inline(never)` keeps the deref in this function's
/// LLBC so the census does not attribute it to the caller.
#[inline(never)]
unsafe fn store_locals_cells_stack_w(frame_ptr: *mut PyFrame, array: *mut FixedObjectArray) {
    unsafe {
        (*frame_ptr).locals_cells_stack_w = array;
    }
}

#[inline]
fn remember_frame_debug_data(debugdata: *mut FrameDebugData) {
    if pyre_object::gc_hook::try_gc_owns_object(debugdata as pyre_object::gc_hook::GCREF) {
        pyre_object::gc_hook::try_gc_write_barrier(debugdata as pyre_object::gc_hook::GCREF);
    }
}

unsafe fn clone_debugdata_ptr(
    ptr: *mut FrameDebugData,
    allocation: FrameLocalsArrayAllocation,
) -> *mut FrameDebugData {
    unsafe {
        if ptr.is_null() {
            std::ptr::null_mut()
        } else if allocation == FrameLocalsArrayAllocation::OldGenGc {
            let _roots = pyre_object::gc_roots::push_roots();
            let src_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ptr as pyre_object::PyObjectRef);
            let raw = pyre_object::gc_hook::try_gc_alloc_nursery_raw(
                FRAME_DEBUG_DATA_GC_TYPE_ID,
                std::mem::size_of::<FrameDebugData>(),
            );
            let ptr = pyre_object::gc_roots::shadow_stack_get(src_slot) as *mut FrameDebugData;
            if !raw.is_null() {
                std::ptr::write(raw as *mut FrameDebugData, (*ptr).clone());
                // The clone may carry young locals / trace callback refs.
                remember_frame_debug_data(raw as *mut FrameDebugData);
                return raw as *mut FrameDebugData;
            }
            pyre_object::lltype::malloc_raw((*ptr).clone())
        } else {
            pyre_object::lltype::malloc_raw((*ptr).clone())
        }
    }
}

unsafe fn clear_debugdata_ptr(ptr: &mut *mut FrameDebugData) {
    unsafe {
        if !(*ptr).is_null() {
            if !pyre_object::gc_hook::try_gc_owns_object(*ptr as pyre_object::gc_hook::GCREF) {
                drop(Box::from_raw(*ptr));
            }
            *ptr = std::ptr::null_mut();
        }
    }
}

impl Drop for PyFrame {
    fn drop(&mut self) {
        // Reached only for a `std::alloc`-backed frame (the `FrameBox`
        // fallback box, or a bare stack `PyFrame`): it owns `std::alloc`
        // resources, so free them here.  GC-managed frames never run
        // `PyFrame::drop`; the collector reclaims their frame-owned resources.
        unsafe { self.free_owned_contents(true) };
    }
}

impl PyFrame {
    #[inline]
    fn aux_allocation(&self) -> FrameLocalsArrayAllocation {
        if pyre_object::gc_hook::try_gc_owns_object(
            self as *const Self as pyre_object::gc_hook::GCREF,
        ) {
            FrameLocalsArrayAllocation::OldGenGc
        } else {
            FrameLocalsArrayAllocation::StdAlloc
        }
    }

    /// Free the `std::alloc` resources owned by a snapshot, fallback, or bare
    /// stack frame: its `locals_cells_stack_w` array and `FrameDebugData` box.
    /// This is reached only through `Drop for
    /// PyFrame`; GC-managed frames and all of their corresponding resources
    /// are reclaimed by the collector.
    ///
    /// `free_locals_array` gates freeing `locals_cells_stack_w`: it is a
    /// `std::alloc` block for this Drop-only regime (free it), but can remain
    /// false if a future regime-mixup supplies a GC-managed array.  The
    /// `try_gc_owns_object` checks in this cleanup path are retained as a
    /// guard against freeing collector-owned storage.
    ///
    /// # Safety
    /// Runs at most once per frame — the pointers are nulled as they are
    /// freed so a second call is a no-op.
    pub unsafe fn free_owned_contents(&mut self, free_locals_array: bool) {
        if free_locals_array && !self.locals_cells_stack_w.is_null() {
            unsafe { dealloc_array_with_gc_header(self.locals_cells_stack_w) };
            self.locals_cells_stack_w = std::ptr::null_mut();
        }
        unsafe {
            clear_debugdata_ptr(&mut self.debugdata);
        }
    }

    #[inline]
    pub fn set_locals_w(&mut self, index: usize, value: PyObjectRef) {
        unsafe { &mut *self.locals_cells_stack_w }.set_ref(index, value);
    }

    /// Restore the per-trace mutable frame state — instruction pointer, value
    /// stack depth, and `locals_cells_stack_w` contents — from a snapshot
    /// taken before a bridge trace.  `trace_and_compile_from_bridge` uses this
    /// to undo any concrete-execution mutation the full-body walker applied to
    /// the live frame during the walk, so a `BridgeCompiled` resume re-enters
    /// at the exact guard state instead of mid-body / past a stepped loop
    /// counter.  `src` is a `snapshot_for_tracing` of this same frame, so the
    /// arrays share length; the `min` is defensive.
    ///
    /// These three are exactly the mutable virtualizable fields the walk can
    /// touch (may-force residual calls advance `last_instr` and the value
    /// stack).  The remaining `_virtualizable_` fields — `pycode`, `w_globals`,
    /// `debugdata` — are deliberately NOT restored: `pycode`/`w_globals` are
    /// frame-invariant and `debugdata` is debug-only; none are written on the
    /// live frame during tracing (the walk writes only the symbolic `PyreSym`
    /// shadow), so restoring them would be dead.
    ///
    /// The slot copy moves Refs between two frames, so it ends on
    /// [`remember_frame_locals_array`]: `locals_cells_stack_w` is its own GC
    /// object, and an old destination array taking a young value has to join
    /// the remembered set.  One arm covers the whole batch because nothing in
    /// the loop can collect — `to_vec` has already allocated by then.
    pub fn restore_resume_state_from(&mut self, src: &PyFrame) {
        self.last_instr = src.last_instr;
        self.valuestackdepth = src.valuestackdepth;
        let src_vals = locals_w!(src).as_slice().to_vec();
        let dst = locals_w_mut!(self);
        let n = src_vals.len().min(dst.as_slice().len());
        let dst_p = dst.items_mut_ptr();
        for (i, &v) in src_vals.iter().take(n).enumerate() {
            // `ll_setitem_fast` on the locals array.
            unsafe {
                *dst_p.add(i) = v;
            }
        }
        remember_frame_locals_array(self.locals_cells_stack_w);
    }
}

/// Extract raw CodeObject from frame's PyCode.
///
/// PyPy: `frame.pycode` gives `PyCode` which IS the code object.
/// pyre: PyCode wraps a raw CodeObject — this extracts it.
///
/// `@jit.elidable` (`rlib/jit.py`): deterministic, no allocation,
/// no raise — pure pointer cast through `w_code_get_ptr`.
/// Mapped to `EF_ELIDABLE_CANNOT_RAISE` (`call.py:299`) so the
/// metainterp can execute it at trace time and the walker can read
/// the concrete result back into the Ref bank.
#[majit_macros::elidable_cannot_raise]
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn pyframe_get_pycode(frame: &PyFrame) -> *const CodeObject {
    unsafe { crate::w_code_get_ptr(frame.pycode as pyre_object::PyObjectRef) as *const CodeObject }
}

/// Name the frame a stack underflow was detected on. A bare depth
/// assertion says an opcode popped from an empty stack but not which
/// opcode of which code object, which is the whole question when the
/// underflow means the frame — or the code behind it — is no longer the
/// one the loop started with.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`):
/// [`PyFrame::pop`] reaches this on the branch it never takes, so the
/// instruction-window formatting below — a `Vec<String>` built through
/// `enumerate`/`skip`/`take`/`map` over the `CodeUnits` deref — would
/// otherwise be lowered into every graph that pops a value.  The report
/// diverges by panicking; the `()` result is what crosses the residual
/// boundary.
#[majit_macros::dont_look_inside]
#[cold]
#[inline(never)]
pub fn report_stack_underflow(frame: &PyFrame) {
    let code = unsafe { &*pyframe_get_pycode(frame) };
    let window: Vec<String> = code
        .instructions
        .iter()
        .copied()
        .enumerate()
        .skip(frame.next_instr().saturating_sub(4))
        .take(8)
        .map(|(pc, unit)| format!("{pc}:{unit:?}"))
        .collect();
    panic!(
        "value-stack underflow: depth={} base={} (nlocals={} ncells={}) \
         pc={} last_instr={} ninstrs={} code={:?} file={:?} frame={:p} \
         pycode={:p} back={:p} window=[{}]",
        frame.valuestackdepth,
        frame.stack_base(),
        frame.nlocals(),
        frame.ncells(),
        frame.next_instr(),
        frame.last_instr,
        code.instructions.len(),
        code.obj_name,
        code.source_path,
        frame,
        frame.pycode,
        frame.f_backref,
        window.join(" "),
    );
}

/// True when this frame owes trace events no path the JIT takes can emit.
///
/// Only a frame carrying its own `f_trace` does.  Such a frame owes a `line`
/// event at every instruction that can start one, fired by `ec.bytecode_trace`
/// from the dispatch loop, and compiled code has no route to that call.
///
/// Neither hook that installs globally is here, and both are deliberate.
/// `executioncontext.py settrace` keeps the JIT on and widens `trace_limit`
/// rather than turning it off, and what it owes is answered per event
/// elsewhere: this frame's own `call` / `return` by the bracket the portal
/// carries (`eval::eval_with_jit_inner`), a callee's by the walker declining
/// to enter one while a hook is installed
/// (`jitcode_dispatch::ec_hook_installed`), and a frame that arms local
/// tracing by `record_portal_debugdata_guard` keeping it from compiling.
/// `setprofile` owes the same frame-level pair, from the same bracket plus
/// `leave`'s `_trace('leaveframe')`, and owes `c_call` / `c_return` on top —
/// which the residual call helpers fire themselves
/// (`call_jit.rs residual_call_c_profile_frame`), the walker keeping them
/// reachable by folding no call while a profiler is installed
/// (`jitcode_dispatch::residual_call::walker_foldable_runtime_helper`).  A
/// profiled loop therefore compiles and reports, calls and all.
pub fn frame_tracing_active(frame: &PyFrame) -> bool {
    !frame.get_w_f_trace().is_null()
}

#[repr(C)]
#[derive(Clone)]
pub struct FrameDebugData {
    /// `FrameDebugData.__init__` / `PyFrame.get_w_globals`: the code object's
    /// first globals lives on `PyCode`; only a frame running that same code in
    /// another globals object stores the override here.  This is the one-field
    /// frame-size optimization introduced by PyPy commit `bcd8653e5ec`.
    pub w_globals: PyObjectRef,
    /// pyframe.py:44 — the frame's locals mapping (`self.w_locals`).
    /// At module scope it is the `w_globals` dict; in a class body it is
    /// the class namespace; for a function it is the dict lazily
    /// materialised by `fast2locals`; `exec(src, g, mapping)` binds an
    /// arbitrary `__getitem__` mapping here.  `STORE/LOAD/DELETE_NAME`
    /// route through `space.setitem/getitem/delitem(w_locals, ...)`.
    pub w_locals: PyObjectRef,
    /// CPython 3.14 `PyFrameObject.f_extra_locals` — keys written through a
    /// `FrameLocalsProxy` which do not name a writable fast local.  This is
    /// deliberately separate from `w_locals`: for a non-optimized eval/class
    /// frame with a live PEP 709 hidden local, the proxy exposes its fast
    /// locals plus this dict, not the original module/class locals mapping
    /// (`Objects/frameobject.c:187-213,246-317,370-405`).
    pub w_extra_locals: PyObjectRef,
    /// pyframe.py:37
    pub w_f_trace: PyObjectRef,
    /// pyframe.py:40
    pub is_being_profiled: bool,
    /// pyframe.py:41
    pub is_in_line_tracing: bool,
    /// pyframe.py:42
    pub f_trace_lines: bool,
    /// pyframe.py:43
    pub f_trace_opcodes: bool,
    /// pyframe.py:38
    pub instr_prev_plus_one: isize,
    /// pyframe.py:39
    pub f_lineno: isize,
    /// pyframe.py:45
    pub hidden_operationerr: PyObjectRef,
}

impl FrameDebugData {
    /// `pyframe.py FrameDebugData.__init__`: initialize the rare-frame
    /// globals override from the code object's first-seen globals.
    ///
    /// Named `init`, not `new`: Charon spells every inherent impl as
    /// `<Impl>`, so `FrameDebugData::new` and `PyFrame::new` share
    /// `pyframe::<Impl>::new` when `self_ty_root` is missing. The
    /// `__init__` leaf cannot collide with `PyFrame::new`.
    pub fn init(pycode: *const (), init_lineno: isize) -> Self {
        Self {
            w_globals: if pycode.is_null() {
                pyre_object::PY_NULL
            } else {
                unsafe { crate::w_code_get_w_globals(pycode as PyObjectRef) }
            },
            w_locals: pyre_object::PY_NULL,
            w_extra_locals: pyre_object::PY_NULL,
            w_f_trace: pyre_object::PY_NULL,
            is_being_profiled: false,
            is_in_line_tracing: false,
            f_trace_lines: true,
            f_trace_opcodes: false,
            instr_prev_plus_one: 0,
            f_lineno: init_lineno,
            hidden_operationerr: pyre_object::PY_NULL,
        }
    }
}

impl Default for FrameDebugData {
    fn default() -> Self {
        Self::init(std::ptr::null(), -1)
    }
}

/// Byte offset of `w_locals` in `FrameDebugData`.
pub const FRAME_DEBUG_DATA_W_LOCALS_OFFSET: usize = std::mem::offset_of!(FrameDebugData, w_locals);

/// Byte offset of `w_extra_locals` in `FrameDebugData`.
pub const FRAME_DEBUG_DATA_W_EXTRA_LOCALS_OFFSET: usize =
    std::mem::offset_of!(FrameDebugData, w_extra_locals);

/// Byte offset of `w_f_trace` in `FrameDebugData`, for the JIT's
/// GETFIELD_GC_R lowering of `pyframe.py get_w_f_trace`.  A trace recorded on
/// a frame that already carries a debug block — every frame does once a trace
/// function has been called on it — pins the slot NULL with a guard, so arming
/// local tracing leaves compiled code instead of running the tail silent.
pub const FRAME_DEBUG_DATA_W_F_TRACE_OFFSET: usize =
    std::mem::offset_of!(FrameDebugData, w_f_trace);

/// Byte offset of the rare per-frame globals override.
pub const FRAME_DEBUG_DATA_W_GLOBALS_OFFSET: usize =
    std::mem::offset_of!(FrameDebugData, w_globals);

/// Allocated size of a `FrameDebugData`.
pub const FRAME_DEBUG_DATA_SIZE: usize = std::mem::size_of::<FrameDebugData>();

// ── Virtualizable field offsets ───────────────────────────────────────
//
// These constants tell the JIT where each virtualizable field lives
// inside a PyFrame, so it can read/write them via raw pointer arithmetic.
// Equivalent to PyPy's `_virtualizable_` descriptor on pyframe.py.

/// Byte offset of `pycode` in `PyFrame`.
pub const PYFRAME_PYCODE_OFFSET: usize = std::mem::offset_of!(PyFrame, pycode);

/// Byte offset of `vable_token` in `PyFrame`.
pub const PYFRAME_VABLE_TOKEN_OFFSET: usize = std::mem::offset_of!(PyFrame, vable_token);

/// Byte offset of `last_instr` in `PyFrame`.
pub const PYFRAME_LAST_INSTR_OFFSET: usize = std::mem::offset_of!(PyFrame, last_instr);

/// Byte offset of `valuestackdepth` in `PyFrame`.
pub const PYFRAME_VALUESTACKDEPTH_OFFSET: usize = std::mem::offset_of!(PyFrame, valuestackdepth);

/// Byte offset of `locals_cells_stack_w` in `PyFrame`.
pub const PYFRAME_LOCALS_CELLS_STACK_OFFSET: usize =
    std::mem::offset_of!(PyFrame, locals_cells_stack_w);

/// Byte offset of `debugdata` in `PyFrame`.
pub const PYFRAME_DEBUGDATA_OFFSET: usize = std::mem::offset_of!(PyFrame, debugdata);

/// Byte offset of `f_generator_wref` in `PyFrame`.
/// `PyObjectRef` slot — points into the GC heap (possibly nursery).
pub const PYFRAME_F_GENERATOR_WREF_OFFSET: usize = std::mem::offset_of!(PyFrame, f_generator_wref);

/// Byte offset of `w_yielding_from` in `PyFrame`.
/// `PyObjectRef` slot — points into the GC heap (possibly nursery).
pub const PYFRAME_W_YIELDING_FROM_OFFSET: usize = std::mem::offset_of!(PyFrame, w_yielding_from);

/// Byte offset of `f_backref` in `PyFrame`.
/// `*mut PyFrame` — once `NewWithVtable(PyFrame)` lands in trace IR,
/// chained recursive callees may have a nursery-allocated parent frame
/// reachable through this pointer.
pub const PYFRAME_F_BACKREF_OFFSET: usize = std::mem::offset_of!(PyFrame, f_backref);

/// Byte offset of `w_builtin` in `PyFrame` (the picked builtin Module).
/// Read by the descr GC walker so a collection survives across the
/// guard exit / re-entry edge.
pub const PYFRAME_W_BUILTIN_OFFSET: usize = std::mem::offset_of!(PyFrame, w_builtin);

// Backward-compat aliases used by JIT code.
pub const PYFRAME_STACK_DEPTH_OFFSET: usize = PYFRAME_VALUESTACKDEPTH_OFFSET;
pub const PYFRAME_LOCALS_OFFSET: usize = PYFRAME_LOCALS_CELLS_STACK_OFFSET;

/// pytraceback.py offset2lineno(c, stopat) — convert instruction index to line number.
/// Matches RPython: negative `stopat` means "frame not yet started", returns
/// first-line.
///
/// [`crate::pycode::w_code_addr2line`] with the instruction index converted to
/// the byte offset `tb_lasti` carries.  `-1` is its miss, the value upstream
/// returns from the same zero-length walk and the value both readers of a line
/// number render as `None`; it is also `LINENO_NOT_COMPUTED`, so a traceback
/// node stamped with it resolves lazily and answers `None` there too.
///
/// The `PyCode` wrapper is the argument, not the bare `CodeObject`, because
/// the walk is seeded with `co_firstlineno` and only the wrapper carries it
/// as the signed integer app-level set: `CodeObject.first_line_number` is an
/// `Option<OneIndexed>` and cannot hold the zero and negative values
/// `CodeType(...)` accepts.
///
/// # Safety
/// `w_code` must be a live `PyCode`.
#[inline]
pub unsafe fn offset2lineno(w_code: PyObjectRef, stopat: isize) -> isize {
    let addrq = (stopat as i64).saturating_mul(2);
    unsafe { crate::pycode::w_code_addr2line(w_code, addrq) as isize }
}

// ─────────────────────────────────────────────────────────────────────
// `f_lineno` jump validation — a port of CPython 3.14's `mark_stacks`
// (`Objects/frameobject.c`).
//
// PyPy's `fset_f_lineno` (`pyframe.py`) validates a debugger line-jump
// against the 3.11-era block model: it scans for `SETUP_LOOP` /
// `SETUP_FINALLY` / `END_FINALLY` / `POP_BLOCK` and walks `co_lnotab`.
// pyre runs 3.14-structural bytecode where those block-setup opcodes are
// pseudo-ops the compiler lowers into `co_exceptiontable`; there is no
// `co_lnotab`.  A line-by-line PyPy port is therefore impossible — the
// opcodes it inspects don't exist here.  The behaviour-correct 3.14
// equivalent is `mark_stacks`: a fixpoint reachability analysis that
// abstracts each operand-stack slot to a `Kind`, so a jump is admitted
// only when the source and target abstract stacks are compatible.
// ─────────────────────────────────────────────────────────────────────

/// `frameobject.c` `Kind` — the abstract contents of one operand-stack
/// slot.  Packed 3 bits per slot into an `i64` (`BITS_PER_BLOCK = 3`),
/// so the abstract stack of up to `MAX_STACK_ENTRIES = 21` slots is a
/// single integer that can be compared for equality across paths.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
enum StackKind {
    Iterator = 1,
    Except = 2,
    Object = 3,
    Null = 4,
    Lasti = 5,
}

const MARK_BITS_PER_BLOCK: i64 = 3;
const MARK_MASK: i64 = (1 << MARK_BITS_PER_BLOCK) - 1;
/// `1 << (63 - BITS_PER_BLOCK)` — a stack whose unsigned value is at or
/// above this would lose its top slot when shifted left.  The negative
/// sentinels (`OVERFLOWED` / `UNINITIALIZED`) reinterpret as huge
/// unsigned values, so this test also propagates them.
const MARK_WILL_OVERFLOW: u64 = 1u64 << (63 - MARK_BITS_PER_BLOCK);

/// `frameobject.c` sentinels for a per-instruction abstract stack.
const MARK_UNINITIALIZED: i64 = -2;
const MARK_OVERFLOWED: i64 = -1;
const MARK_EMPTY_STACK: i64 = 0;

#[inline]
fn mark_push_value(stack: i64, kind: StackKind) -> i64 {
    mark_push_kind_bits(stack, kind as i64)
}

/// `push_value` with the kind supplied as raw bits — used by `COPY`,
/// which re-pushes a slot peeked from the stack (whose kind is not
/// statically one of the [`StackKind`] variants).
#[inline]
fn mark_push_kind_bits(stack: i64, kind_bits: i64) -> i64 {
    if (stack as u64) >= MARK_WILL_OVERFLOW {
        MARK_OVERFLOWED
    } else {
        (stack << MARK_BITS_PER_BLOCK) | kind_bits
    }
}

#[inline]
fn mark_pop_value(stack: i64) -> i64 {
    // Arithmetic right shift preserves the OVERFLOWED / UNINITIALIZED
    // sentinels (both negative).
    stack >> MARK_BITS_PER_BLOCK
}

#[inline]
fn mark_top_of_stack(stack: i64) -> i64 {
    stack & MARK_MASK
}

#[inline]
fn mark_peek(stack: i64, n: i64) -> i64 {
    (stack >> (MARK_BITS_PER_BLOCK * (n - 1))) & MARK_MASK
}

#[inline]
fn mark_stack_swap(stack: i64, n: i64) -> i64 {
    let top = mark_top_of_stack(stack);
    let nth = mark_peek(stack, n);
    let shift = MARK_BITS_PER_BLOCK * (n - 1);
    let stack = stack & !(MARK_MASK << shift) | (top << shift);
    (stack & !MARK_MASK) | nth
}

/// `frameobject.c pop_to_level` — pop the abstract stack down to `level`
/// slots.
#[inline]
fn mark_pop_to_level(mut stack: i64, level: i32) -> i64 {
    let mut depth = 0i64;
    let mut s = stack;
    while s > MARK_EMPTY_STACK {
        s = mark_pop_value(s);
        depth += 1;
    }
    while depth > level as i64 {
        stack = mark_pop_value(stack);
        depth -= 1;
    }
    stack
}

/// True when a jump from `from` kind to `to` kind is admissible
/// (`frameobject.c compatible_kind`).
#[inline]
fn mark_compatible_kind(from: i64, to: i64) -> bool {
    if to == 0 {
        return false;
    }
    if to == StackKind::Object as i64 {
        return from != StackKind::Null as i64;
    }
    if to == StackKind::Null as i64 {
        return true;
    }
    from == to
}

/// `frameobject.c compatible_stack` — pop `from_stack` to the target
/// depth, then compare each slot's kind.
fn mark_compatible_stack(mut from_stack: i64, target_stack: i64) -> bool {
    if from_stack < 0 || target_stack < 0 {
        return false;
    }
    let mut to = target_stack;
    // Depth of each packed stack.
    let depth = |mut s: i64| -> i64 {
        let mut d = 0;
        while s > MARK_EMPTY_STACK {
            s = mark_pop_value(s);
            d += 1;
        }
        d
    };
    while depth(from_stack) > depth(to) {
        from_stack = mark_pop_value(from_stack);
    }
    while to > MARK_EMPTY_STACK {
        if from_stack <= MARK_EMPTY_STACK {
            return false;
        }
        if !mark_compatible_kind(mark_top_of_stack(from_stack), mark_top_of_stack(to)) {
            return false;
        }
        from_stack = mark_pop_value(from_stack);
        to = mark_pop_value(to);
    }
    from_stack == MARK_EMPTY_STACK
}

/// Diagnostic for an incompatible target stack
/// (`frameobject.c explain_incompatible_stack`).
fn mark_explain_incompatible_stack(target_stack: i64) -> &'static str {
    if target_stack == MARK_OVERFLOWED {
        return "stack too deep to analyze";
    }
    if target_stack == MARK_UNINITIALIZED {
        return "can't jump into an exception handler, or code may be unreachable";
    }
    let top = mark_top_of_stack(target_stack);
    if top == StackKind::Except as i64 {
        "can't jump into an 'except' block as there's no exception"
    } else if top == StackKind::Lasti as i64 {
        "can't jump into a re-raising block as there's no location"
    } else if top == StackKind::Object as i64 || top == StackKind::Null as i64 {
        "incompatible stacks"
    } else if top == StackKind::Iterator as i64 {
        "can't jump into the body of a for loop"
    } else {
        "incompatible stacks"
    }
}

/// `frameobject.c marklines` — the source line that starts at each
/// instruction-unit index (`-1` where no line change begins).
///
/// ```c
/// while (_PyLineTable_NextAddressRange(&bounds)) {
///     assert(bounds.ar_start / (int)sizeof(_Py_CODEUNIT) < len);
///     if (bounds.ar_line != last_line && bounds.ar_line != -1) {
///         linestarts[bounds.ar_start / (int)sizeof(_Py_CODEUNIT)] = bounds.ar_line;
///         last_line = bounds.ar_line;
///     }
/// }
/// ```
///
/// Only the **start** unit of a range is a jump target, and a `NO_LOCATION`
/// range contributes none: `ar_line` is `-1` there and the second test drops
/// it.  `code_locations` cannot express either — it holds one `SourceLocation`
/// per unit whose `line` is a `OneIndexed`, so a range carrying line `0` or no
/// line at all reads back as `1` and turns the module `RESUME` into a legal
/// target for `f_lineno = 1`.
///
/// `firstlineno` is `co->co_firstlineno` as app-level set it; the assert is a
/// bounds check here because a replaced `co_linetable` can describe ranges past
/// `len`, which upstream would write out of bounds in a release build.
fn mark_lines(code: &CodeObject, firstlineno: i32, len: usize) -> Vec<i32> {
    let mut linestarts = vec![-1i32; len];
    let mut bounds = crate::pycode::PyCodeAddressRange::new(&code.linetable, firstlineno);
    let mut last_line = -1i32;
    while bounds.advance() {
        if bounds.ar_line != last_line && bounds.ar_line != -1 {
            if let Some(slot) = usize::try_from(bounds.ar_start / 2)
                .ok()
                .and_then(|index| linestarts.get_mut(index))
            {
                *slot = bounds.ar_line;
            }
            last_line = bounds.ar_line;
        }
    }
    linestarts
}

/// `frameobject.c first_line_not_before` — the smallest line `>= line`
/// present in `lines`, or `-1`.
fn mark_first_line_not_before(lines: &[i32], line: i32) -> i32 {
    let mut result = i32::MAX;
    for &l in lines {
        if l < result && l >= line {
            result = l;
        }
    }
    if result == i32::MAX { -1 } else { result }
}

/// `frameobject.c mark_stacks` — fixpoint abstract-interpretation of the
/// operand stack.  Returns `stacks[i]` = the packed abstract stack on
/// entry to unit `i` (`UNINITIALIZED` where unreachable), length
/// `len + 1`.
fn mark_stacks(code: &CodeObject, len: usize) -> Vec<i64> {
    use crate::bytecode::Instruction;

    let mut stacks = vec![MARK_UNINITIALIZED; len + 1];
    stacks[0] = MARK_EMPTY_STACK;

    let mut todo = true;
    while todo {
        todo = false;
        // ── Scan instructions ──
        let mut i = 0usize;
        while i < len {
            let mut next_stack = stacks[i];
            let Some((opcode, op_arg)) = crate::pyopcode::decode_instruction_at(code, i) else {
                i += 1;
                continue;
            };
            // `decode_instruction_at` already accumulates EXTENDED_ARG
            // prefixes into `op_arg` at the real opcode's index.  Mirror
            // CPython's inner `while opcode == EXTENDED_ARG` loop by
            // propagating this unit's entry stack onto the following unit
            // (the next prefix or the real opcode), so an instruction with
            // an EXTENDED_ARG prefix inherits the prefix's entry stack.
            if matches!(opcode, Instruction::ExtendedArg) {
                if i + 1 < stacks.len() {
                    stacks[i + 1] = next_stack;
                }
                i += 1;
                continue;
            }
            let raw_arg: u32 = op_arg.into();
            let caches = opcode.cache_entries();
            let next_i = i + caches + 1;

            if next_stack == MARK_UNINITIALIZED {
                i = next_i;
                continue;
            }

            // Relative jump targets are measured from `next_i` (past the
            // opcode's cache slots), matching `absolutize_jump_target`.
            match opcode {
                Instruction::PopJumpIfFalse { delta }
                | Instruction::PopJumpIfTrue { delta }
                | Instruction::PopJumpIfNone { delta }
                | Instruction::PopJumpIfNotNone { delta } => {
                    let j = next_i + delta.get(op_arg).as_usize();
                    next_stack = mark_pop_value(next_stack);
                    if j <= len {
                        stacks[j] = next_stack;
                    }
                    stacks[next_i] = next_stack;
                }
                Instruction::Send { delta } => {
                    let j = next_i + delta.get(op_arg).as_usize();
                    if j <= len {
                        stacks[j] = next_stack;
                    }
                    stacks[next_i] = next_stack;
                }
                Instruction::JumpForward { delta } => {
                    let j = next_i + delta.get(op_arg).as_usize();
                    if j <= len {
                        stacks[j] = next_stack;
                    }
                }
                Instruction::JumpBackward { delta }
                | Instruction::JumpBackwardNoInterrupt { delta } => {
                    let j = next_i.saturating_sub(delta.get(op_arg).as_usize());
                    if stacks[j] == MARK_UNINITIALIZED && j < i {
                        todo = true;
                    }
                    stacks[j] = next_stack;
                }
                Instruction::GetIter | Instruction::GetAiter => {
                    next_stack = mark_push_value(mark_pop_value(next_stack), StackKind::Iterator);
                    stacks[next_i] = next_stack;
                }
                Instruction::ForIter { delta } => {
                    // Fall-through: iterator stays on the stack, the loop
                    // variable (Object) is pushed above it.  Exhaustion
                    // branch: the iterator stays on the stack (net 0) — it
                    // is popped later by `POP_ITER`, matching the runtime
                    // stack effect (liveness.rs ForIter `(d+1, d)`).  No
                    // loop variable is pushed on that branch.
                    let target_stack = mark_push_value(next_stack, StackKind::Object);
                    stacks[next_i] = target_stack;
                    let j = next_i + delta.get(op_arg).as_usize();
                    if j <= len {
                        stacks[j] = next_stack;
                    }
                }
                // The for-loop iterator is one slot here (`GET_ITER` above):
                // `end_for` leaves the stack alone and `pop_iter` pops that
                // one slot, where `stack_effect` counts the iterator/index
                // pair of the bytecode's own stack model.
                Instruction::EndFor => {
                    stacks[next_i] = next_stack;
                }
                Instruction::PopIter => {
                    next_stack = mark_pop_value(next_stack);
                    stacks[next_i] = next_stack;
                }
                Instruction::EndAsyncFor => {
                    next_stack = mark_pop_value(mark_pop_value(next_stack));
                    stacks[next_i] = next_stack;
                }
                Instruction::PushExcInfo => {
                    // Runtime (codewriter PushExcInfo): pop the new
                    // exception, push the previous exception (Except slot),
                    // then push the new exception back (Object).  Net +1,
                    // but the shape is `[.., Except, Object]` — model both
                    // slots so `f_lineno` jump validation matches the live
                    // handler stack.
                    let below = mark_pop_value(next_stack);
                    next_stack = mark_push_value(below, StackKind::Except);
                    next_stack = mark_push_value(next_stack, StackKind::Object);
                    stacks[next_i] = next_stack;
                }
                Instruction::PopExcept => {
                    next_stack = mark_pop_value(next_stack);
                    stacks[next_i] = next_stack;
                }
                Instruction::ReturnValue => {
                    // End of a path.
                }
                Instruction::RaiseVarargs { .. } | Instruction::Reraise { .. } => {
                    // End of a path.
                }
                Instruction::PushNull => {
                    next_stack = mark_push_value(next_stack, StackKind::Null);
                    stacks[next_i] = next_stack;
                }
                Instruction::LoadGlobal { .. } => {
                    next_stack = mark_push_value(next_stack, StackKind::Object);
                    if raw_arg & 1 != 0 {
                        next_stack = mark_push_value(next_stack, StackKind::Null);
                    }
                    stacks[next_i] = next_stack;
                }
                Instruction::LoadAttr { .. } => {
                    if raw_arg & 1 != 0 {
                        next_stack = mark_pop_value(next_stack);
                        next_stack = mark_push_value(next_stack, StackKind::Object);
                        next_stack = mark_push_value(next_stack, StackKind::Null);
                    }
                    stacks[next_i] = next_stack;
                }
                Instruction::Swap { .. } => {
                    next_stack = mark_stack_swap(next_stack, raw_arg as i64);
                    stacks[next_i] = next_stack;
                }
                Instruction::Copy { .. } => {
                    next_stack =
                        mark_push_kind_bits(next_stack, mark_peek(next_stack, raw_arg as i64));
                    stacks[next_i] = next_stack;
                }
                _ => {
                    // `PyCompile_OpcodeStackEffect(opcode, oparg)` — the
                    // net slot delta, with every pushed slot abstracted to
                    // Object.
                    let mut delta = opcode.stack_effect(raw_arg);
                    while delta < 0 {
                        next_stack = mark_pop_value(next_stack);
                        delta += 1;
                    }
                    while delta > 0 {
                        next_stack = mark_push_value(next_stack, StackKind::Object);
                        delta -= 1;
                    }
                    stacks[next_i] = next_stack;
                }
            }
            i = next_i;
        }

        // ── Scan the exception table ──
        for entry in crate::pycode::decode_exceptiontable(&code.exceptiontable) {
            let start_offset = (entry.start / 2) as usize;
            let handler = (entry.target / 2) as usize;
            let level = entry.depth as i32;
            let lasti = entry.lasti;
            if start_offset >= stacks.len() || handler >= stacks.len() {
                continue;
            }
            if stacks[start_offset] != MARK_UNINITIALIZED && stacks[handler] == MARK_UNINITIALIZED {
                todo = true;
                let mut target_stack = mark_pop_to_level(stacks[start_offset], level);
                if lasti {
                    target_stack = mark_push_value(target_stack, StackKind::Lasti);
                }
                target_stack = mark_push_value(target_stack, StackKind::Except);
                stacks[handler] = target_stack;
            }
        }
    }
    stacks
}

/// pyframe.py:105-106 — cell + free variable slot count.
///
/// Returns the number of *extra* slots beyond `varnames` needed for
/// cellvars and freevars. CPython 3.11+ unified the slot layout
/// (`co_localsplusnames`): a cellvar that also appears in varnames
/// (i.e. a parameter captured by an inner function) shares its
/// varname slot via `MAKE_CELL`. Only cellvars NOT in varnames need
/// a fresh slot. Without this overlap filtering, freevar indices
/// shift by the overlap count and `LOAD_DEREF`/`LOAD_FAST` for
/// freevars reads the wrong slot — see decorator-with-args test
/// `def repeat(n): def wrap(fn): def inner(): return (n, fn)`
/// where `n` resolved to `fn` because of the off-by-one slot.
#[inline]
#[majit_macros::elidable_cannot_raise]
pub fn npure_cellvars(code: &CodeObject) -> usize {
    let mut count = 0;
    for c in code.cellvars.iter() {
        let cs: &str = c.as_ref();
        if !cellvar_overlaps_varname(code, cs) {
            count += 1;
        }
    }
    count
}

/// True when cellvar `name` also names a varname, so the unified slot layout
/// gave it that varname's slot and it is not part of the pure-cellvar band.
#[inline]
#[majit_macros::elidable_cannot_raise]
fn cellvar_overlaps_varname(code: &CodeObject, name: &str) -> bool {
    for v in code.varnames.iter() {
        let vs: &str = v.as_ref();
        if vs == name {
            return true;
        }
    }
    false
}

/// Position in `code.cellvars` of the `n`-th cellvar that does not also name a
/// varname -- the `n`-th entry of the pure-cellvar band [`npure_cellvars`]
/// counts.  Returns `code.cellvars.len()` when `n` is past the band.
#[inline]
#[majit_macros::elidable_cannot_raise]
fn nth_pure_cellvar_index(code: &CodeObject, n: usize) -> usize {
    let mut seen = 0;
    for ci in 0..code.cellvars.len() {
        let cs: &str = code.cellvars[ci].as_ref();
        if cellvar_overlaps_varname(code, cs) {
            continue;
        }
        if seen == n {
            return ci;
        }
        seen += 1;
    }
    code.cellvars.len()
}

#[inline]
#[majit_macros::elidable_cannot_raise]
pub fn ncells(code: &CodeObject) -> usize {
    npure_cellvars(code) + code.freevars.len()
}

/// True when localsplus slot `idx` is a `varnames` entry that `MAKE_CELL`
/// turned into a cell, so the slot holds the cell and a reader has to
/// dereference it to reach the value.
#[inline]
fn cell_slot(code: &CodeObject, idx: usize) -> bool {
    idx < code.localspluskinds.len()
        && code.localspluskinds[idx] & crate::bytecode::CO_FAST_CELL != 0
}

/// The frame's locals-plus slots in `co_localsplusnames` order — every
/// `varnames` entry, then the `cellvars` that are not already varnames, then
/// the `freevars` — each paired with its slot index and whether the slot holds
/// a cell.
///
/// `framelocalsproxy_getkeyindex` scans this order and
/// `frame_locals_proxy_snapshot` fills a dict in it, so a name resolves to the
/// same slot on both routes only as long as they walk it the same way.
fn locals_plus_names(code: &CodeObject) -> Vec<(usize, &str, bool)> {
    let mut names = Vec::new();
    for (index, name) in code.varnames.iter().enumerate() {
        names.push((index, name.as_str(), cell_slot(code, index)));
    }
    let mut extra = 0usize;
    for name in &code.cellvars {
        if code.varnames.iter().any(|local| local == name) {
            continue;
        }
        names.push((code.varnames.len() + extra, name.as_str(), true));
        extra += 1;
    }
    for name in &code.freevars {
        names.push((code.varnames.len() + extra, name.as_str(), true));
        extra += 1;
    }
    names
}

/// True when localsplus slot `idx` carries `CO_FAST_HIDDEN`, i.e. an inlined
/// comprehension's iteration variable (PEP 709).  The flag records that the
/// surrounding binding is saved and restored by the comprehension bytecode;
/// while the comprehension is running, its current value is still visible to
/// `locals()` like every other fast local.
#[inline]
#[majit_macros::elidable_cannot_raise]
pub fn hidden_local(code: &CodeObject, idx: usize) -> bool {
    // closure-free, Option-pattern-free `localspluskinds.get(idx)` rewrite —
    // keeps the bounds check a plain `lt + getitem` instead of walking into an
    // `Option<&u8>` the walker cannot fold.
    idx < code.localspluskinds.len()
        && code.localspluskinds[idx] & crate::bytecode::CO_FAST_HIDDEN != 0
}

/// `LOAD_DEREF` unbound-variable error for the unified deref slot `idx`,
/// shared by the interpreter (`load_deref`) and the JIT residual
/// (`bh_load_deref_value_fn`).
///
/// pyopcode.py `_load_deref_unbound`: a cell variable (a captured local whose
/// deref slot shares its `varnames` index via `MAKE_CELL`, or a pure cellvar
/// in the `[nvarnames, nvarnames + npure_cellvars)` band) reports "cannot
/// access local variable '{name}' where it is not associated with a value"; a
/// free variable reports "cannot access free variable '{name}' where it is not
/// associated with a value in enclosing scope".
/// `idx` follows the `npure_cellvars` slot layout: `varnames` occupy
/// `[0, nvarnames)`, pure cellvars (those not also varnames) the next
/// `npure_cellvars` slots, then freevars.
/// The name and free/cell kind of the local / cell / free variable at
/// localsplus index `idx`, following the `[varnames, pure cellvars, freevars]`
/// layout.  Used to resolve the name behind a `*_DEREF` oparg; an out-of-range
/// index yields an empty name rather than panicking.
pub fn deref_name_and_kind(code: &CodeObject, idx: usize) -> (&str, bool) {
    let nvarnames = code.varnames.len();
    if idx < nvarnames {
        return (code.varnames[idx].as_ref(), false);
    }
    let cell_slot = idx - nvarnames;
    let npure = npure_cellvars(code);
    if cell_slot < npure {
        let ci = nth_pure_cellvar_index(code, cell_slot);
        let name = if ci < code.cellvars.len() {
            code.cellvars[ci].as_ref()
        } else {
            ""
        };
        (name, false)
    } else {
        let free_idx = cell_slot - npure;
        let name = if free_idx < code.freevars.len() {
            code.freevars[free_idx].as_ref()
        } else {
            ""
        };
        (name, true)
    }
}

/// Whether the pinned RustPython compiler encoded a class-namespace
/// `__class__` assignment as `STORE_DEREF`.  CPython 3.14 emits `STORE_NAME`
/// for the explicit binding while retaining the separate implicit
/// `__class__` cell used by methods.  A class body is non-optimized and the
/// affected slot is a cell variable, never a free variable (`nonlocal
/// __class__` must continue to mutate its enclosing cell).
#[inline]
pub fn class_scope_class_deref_is_name(code: &CodeObject, idx: usize) -> bool {
    if code.flags.contains(CodeFlags::OPTIMIZED) {
        return false;
    }
    let (name, is_free) = deref_name_and_kind(code, idx);
    !is_free && name == "__class__"
}

/// Unified localsplus index of an outer `__class__` free variable, when a
/// class body also owns the distinct implicit `__class__` cell.  CPython 3.14
/// addresses this collision with `LOAD_FROM_DICT_OR_DEREF`.
pub fn class_scope_outer_class_freevar(code: &CodeObject) -> Option<usize> {
    let free_pos = code
        .freevars
        .iter()
        .position(|name| name.as_str() == "__class__")?;
    Some(code.varnames.len() + npure_cellvars(code) + free_pos)
}

pub fn deref_unbound_error(code: &CodeObject, idx: usize) -> crate::PyError {
    let (name, is_free) = deref_name_and_kind(code, idx);
    let message = if is_free {
        format!(
            "cannot access free variable '{name}' where it is not associated with a value in enclosing scope"
        )
    } else {
        format!("cannot access local variable '{name}' where it is not associated with a value")
    };
    if is_free {
        crate::PyError::oefmt_name_error_text(message, name)
    } else {
        crate::PyError::unbound_local_error(message)
    }
}

/// Whether calling a code object with these flags produces a suspended
/// frame object (generator / coroutine / async generator) rather than
/// running the body eagerly.
/// pyframe.py `_is_generator_or_coroutine`: CO_COROUTINE |
/// CO_GENERATOR | CO_ASYNC_GENERATOR.  CO_ITERABLE_COROUTINE is a marker
/// layered on a generator function (it always co-occurs with
/// CO_GENERATOR), so it is not tested here.
#[inline]
pub fn code_flags_make_generator(flags: crate::CodeFlags) -> bool {
    flags.intersects(
        crate::CodeFlags::GENERATOR
            | crate::CodeFlags::COROUTINE
            | crate::CodeFlags::ASYNC_GENERATOR,
    )
}

/// `typedef.py _make_descr_typecheck_wrapper` around `PyFrame.fget_getdictscope`.
///
/// `GetSetProperty(PyFrame.fget_getdictscope)` binds this as `fget`.  The
/// residual `jit_force_virtualizable` is the `replace_force_virtualizable_with_call`
/// half; `jtransform.py rewrite_op_jit_force_virtualizable` deletes it from
/// jitcode.  Publishing the wrapper descriptor seeds the same candidate
/// graph `BuiltinCode.func` would contribute from the generated wrapper.
///
/// The `__majit_wrap_` leaf is what makes that seeding happen, and it is not
/// decoration: `CallControl::compute_builtin_wrapper_indirect_graphs`
/// populates the `BuiltinCode.func` PBC family by that prefix, and the family
/// is what enters `candidate_graphs` and gets a `get_jitcode`.  Written by
/// hand rather than by `#[pyre_methods]`, the wrapper still has to spell it —
/// otherwise it publishes an address, binds at runtime, joins no family, and
/// the deletion above never runs on anything, because the codewriter never
/// looks inside this graph at all.  Every frame getset below carries it for
/// the same reason, and `jit_fnaddr.rs`
/// `every_builtin_wrapper_descriptor_carries_the_family_prefix` is the census
/// that keeps them there.
pub fn __majit_wrap_descr_typecheck_fget_getdictscope(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let f = args.get(1).copied().unwrap_or(pyre_object::PY_NULL) as *mut PyFrame;
    if f.is_null() {
        return Ok(pyre_object::w_none());
    }
    let anchor = unsafe { crate::eval::FrameAnchor::from_raw(f) };
    crate::executioncontext::jit_force_virtualizable(anchor.live());
    unsafe { &mut *anchor.live() }.fget_getdictscope()
}

/// Publish a hand-written frame getset wrapper in the same immutable function
/// family `#[pyre_methods]` generates for interp2app wrappers.
///
/// PyPy's `BuiltinCode.func` is a PBC: every wrapper body is a candidate graph
/// of the indirect call, including the `jit_force_virtualizable` that rtyping
/// injected immediately before a redirected field access.  Rust erases the
/// function identity to `BuiltinCodeFn`; omitting one wrapper here therefore
/// makes that descriptor opaque to the codewriter and leaves the runtime force
/// inside a `CALL_MAY_FORCE`.  Keep every manual frame descriptor in one list
/// so `f_locals` cannot be the only orthodox member by accident.
macro_rules! register_frame_builtin_wrapper {
    ($static_name:ident, $func:ident) => {
        crate::builtin_wrapper_descriptor!($static_name, $func);
    };
}

register_frame_builtin_wrapper!(
    __majit_builtin_wrapper_target_fget_getdictscope,
    __majit_wrap_descr_typecheck_fget_getdictscope
);

/// `typedef.py _make_descr_typecheck_wrapper` around `PyFrame::fget_f_code`.
///
/// `GetSetProperty` binds this as its `fget` for `f_code`.
///
/// No force marker, although `pycode` IS one of the redirected fields
/// `virtualizable_gen.rs` declares — the one exception the sibling gateways'
/// rule needs, and it turns on what the marker is FOR rather than on the field
/// list.  Upstream `rvirtualizable.py hook_access_field` injects at every
/// redirected access in every graph, so it never has to ask; pyre places the
/// marker by hand at the gateway, so each placement owes the question "can the
/// trace's shadow and the live frame disagree about this field?".
///
/// For `pycode` they cannot.  It is written exactly twice, both at frame
/// construction (`PyFrame::__init__` and the allocator's field
/// initialisation), and never again; a trace neither writes it back nor
/// reseeds it.  [`PyFrame::restore_resume_state_from`] states the same thing
/// from the other side — it restores `last_instr`, `valuestackdepth` and the
/// locals array, and calls `pycode` frame-invariant.  So the marker cannot
/// make this read answer differently, and all it does reach is the escape
/// flush, which `virtualizable.py force_now` turns into
/// `TOKEN_TRACING_RESCALL` → `TOKEN_NONE` → `pyjitpl` `ABORT_ESCAPE`.
///
/// Measured, with identical output on every fixture: carrying it costs
/// `synth/exception_raise_caught_same_frame_tb` `guard_failures` 605 -> 803,
/// `synth/exception_reentry_guard_finally_residual`
/// 5/1027/0/6 -> 8/1903/2/5 (bridges/guard_failures/aborted/compiled),
/// `synth/c_call_reports_the_callee_frame_not_the_inliners` its `root:inner`
/// compile, and `synth/frame_chain_survives_a_recursive_call_assembler`
/// `loop:hot` and `root:rec` — each of them a `tb_frame.f_code` or
/// `frame.f_code` read from interpreted code nested inside a residual call,
/// which is where the marker forces the caller's live virtualizable.
pub fn __majit_wrap_descr_typecheck_fget_f_code(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let f = args.get(1).copied().unwrap_or(pyre_object::PY_NULL) as *mut PyFrame;
    if f.is_null() {
        return Ok(pyre_object::w_none());
    }
    let anchor = unsafe { crate::eval::FrameAnchor::from_raw(f) };
    Ok(unsafe { &*anchor.live() }.fget_f_code())
}

register_frame_builtin_wrapper!(
    __majit_builtin_wrapper_target_fget_f_code,
    __majit_wrap_descr_typecheck_fget_f_code
);

/// `typedef.py _make_descr_typecheck_wrapper` around `PyFrame::get_w_globals`.
///
/// `GetSetProperty` binds this as its `fget` for `f_globals`.
pub fn __majit_wrap_descr_typecheck_get_w_globals(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let f = args.get(1).copied().unwrap_or(pyre_object::PY_NULL) as *mut PyFrame;
    if f.is_null() {
        return Ok(pyre_object::w_none());
    }
    let anchor = unsafe { crate::eval::FrameAnchor::from_raw(f) };
    crate::executioncontext::jit_force_virtualizable(anchor.live());
    let w = unsafe { &*anchor.live() }.get_w_globals();
    Ok(if w.is_null() {
        pyre_object::w_none()
    } else {
        w
    })
}

register_frame_builtin_wrapper!(
    __majit_builtin_wrapper_target_get_w_globals,
    __majit_wrap_descr_typecheck_get_w_globals
);

/// `typedef.py _make_descr_typecheck_wrapper` around `PyFrame::fget_f_back`.
///
/// `GetSetProperty` binds this as its `fget` for `f_back`.
///
/// No force marker.  `rvirtualizable.py hook_access_field` emits
/// `jit_force_virtualizable` only under
/// `if self.my_redirected_fields.get(cname.value)`, and the field this reads —
/// `f_backref` — is not one: `interp_jit.py` declares `_virtualizable_` as
/// `last_instr`, `pycode`, `valuestackdepth`, `locals_cells_stack_w[*]`,
/// `debugdata`, `w_globals`.  [`PyFrame::fget_f_back`] states what a marker
/// here costs instead — `f_backref` holds a `jit.virtual_ref`, so reading it
/// IS the force, and it is the foldable one.
///
/// Measured on `synth/getframe_inlined_callee_own_frame`: with the marker,
/// `frame.f_back` raised `VableEscapedDuringResidualCall` 35 times and the
/// fixture compiled no loop; `frame.f_code`, whose field IS redirected, raised
/// none because the walker folds that read instead of residualizing it.
pub fn __majit_wrap_descr_typecheck_fget_f_back(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let f = args.get(1).copied().unwrap_or(pyre_object::PY_NULL) as *mut PyFrame;
    if f.is_null() {
        return Ok(pyre_object::w_none());
    }
    let anchor = unsafe { crate::eval::FrameAnchor::from_raw(f) };
    let back = unsafe { &*anchor.live() }.fget_f_back();
    Ok(if back.is_null() {
        pyre_object::w_none()
    } else {
        // Exposing the frame to app level: mark escaped so the JIT
        // materialises it (pyframe.py mark_as_escaped), mirroring `_getframe`.
        unsafe { (*back).mark_as_escaped() };
        back as pyre_object::PyObjectRef
    })
}

register_frame_builtin_wrapper!(
    __majit_builtin_wrapper_target_fget_f_back,
    __majit_wrap_descr_typecheck_fget_f_back
);

/// `typedef.py _make_descr_typecheck_wrapper` around `PyFrame::get_generator`.
///
/// `GetSetProperty` binds this as its `fget` for `f_generator`.
pub fn __majit_wrap_descr_typecheck_get_generator(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let f = args.get(1).copied().unwrap_or(pyre_object::PY_NULL) as *mut PyFrame;
    if f.is_null() {
        return Ok(pyre_object::w_none());
    }
    let owner = unsafe { &*f }.get_generator();
    Ok(if owner.is_null() {
        pyre_object::w_none()
    } else {
        owner
    })
}

register_frame_builtin_wrapper!(
    __majit_builtin_wrapper_target_get_generator,
    __majit_wrap_descr_typecheck_get_generator
);

/// `typedef.py _make_descr_typecheck_wrapper` around `PyFrame::fget_f_lasti`.
///
/// `GetSetProperty` binds this as its `fget` for `f_lasti`.
///
///
/// pyre stores `last_instr` as an instruction-unit index (increments
/// by 1 per instruction); CPython's `f_lasti` is a byte offset
/// (2 bytes per code unit).  Report the byte-offset form (× 2) so
/// `dis` / `code.co_positions()` consumers that do `f_lasti // 2`
/// recover the right instruction — the same adaptation `tb_lasti`
/// uses (`typedef.rs` tb_lasti getter).
///
/// A negative `last_instr` means the frame has not run an instruction, and
/// that is not a coordinate this getter can hand out: `frame_lasti_get_impl`
/// turns only a negative `_PyInterpreterFrame_LASTI` into `-1`, and the
/// pointer is not negative there.  Frame setup consumes the
/// compiler-inserted prologue, so the pointer rests on the first `RESUME`
/// -- `code->_co_firsttraceable` -- which is what the `call` event reports:
/// 0 for a plain function, 2 past a closure's `COPY_FREE_VARS`, 4 past a
/// generator's `RETURN_GENERATOR` / `POP_TOP`.  Handing out `-1` instead
/// named a coordinate no frame ever executes.  `pyframe.py fget_f_lasti`
/// returns `last_instr` unscaled and stops there, so both adaptations live
/// here rather than in [`PyFrame::fget_f_lasti`].
///
/// A generator is the one shape whose not-yet-started frame rests somewhere
/// else: `RETURN_GENERATOR` is what built the generator, so the frame it
/// left behind sits on the instruction after it until the first resumption
/// walks on to the `RESUME`.  Only the generator's own started flag
/// separates the two, and the frame is reachable from Python in both states
/// through `gi_frame`.
///
/// That flag is read through the back-reference, which is cleared when the
/// generator is collected (`generator_finalize`) and when a generator that
/// never ran finishes.  A frame that outlives its generator therefore
/// cannot say which of the two it rests on, and the sentinel stands: the
/// `RESUME` would be the coordinate for a frame still at its `call` event,
/// which this one is not.
///
/// The flag alone does not separate a first `send` from a first `throw`,
/// which marks the generator started and then injects before the `RESUME`
/// is reached -- that frame is still on the instruction after
/// `RETURN_GENERATOR`.  Nothing reads it there: the trace events that
/// would are not emitted for an injection into a generator that never ran,
/// and the frame is unlinked by the time the call returns.  Whatever
/// starts emitting them owns that distinction.
///
/// `-1` is left for a code object with no `RESUME` too, where that index
/// runs past the end and the frame starts no line either.
pub fn __majit_wrap_descr_typecheck_fget_f_lasti(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let f = args.get(1).copied().unwrap_or(pyre_object::PY_NULL) as *mut PyFrame;
    if f.is_null() {
        return Ok(pyre_object::w_none());
    }
    let anchor = unsafe { crate::eval::FrameAnchor::from_raw(f) };
    crate::executioncontext::jit_force_virtualizable(anchor.live());
    let frame = unsafe { &*anchor.live() };
    let lasti = frame.fget_f_lasti();
    let lasti = if lasti < 0 {
        let code = frame.code();
        let owner = frame.get_generator();
        let index = match crate::pycode::after_return_generator_index(code) {
            Some(after_return_generator) => (!owner.is_null()).then(|| {
                if unsafe { pyre_object::generator::w_generator_is_started(owner) } {
                    crate::pycode::first_traceable_index(code)
                } else {
                    after_return_generator
                }
            }),
            None => Some(crate::pycode::first_traceable_index(code)),
        };
        match index {
            Some(index) if index < code.instructions.len() => index as i64,
            _ => -1,
        }
    } else {
        lasti as i64
    };
    Ok(pyre_object::w_int_new(if lasti < 0 {
        -1
    } else {
        lasti * 2
    }))
}

register_frame_builtin_wrapper!(
    __majit_builtin_wrapper_target_fget_f_lasti,
    __majit_wrap_descr_typecheck_fget_f_lasti
);

/// `typedef.py _make_descr_typecheck_wrapper` around `PyFrame::fget_f_builtins`.
///
/// `GetSetProperty` binds this as its `fget` for `f_builtins`.
///
/// No force marker, for the reason
/// [`__majit_wrap_descr_typecheck_fget_f_back`] gives:
/// [`PyFrame::fget_f_builtins`] reads `w_builtin` and the module's `w_dict`,
/// and `_virtualizable_` names neither.
pub fn __majit_wrap_descr_typecheck_fget_f_builtins(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let f = args.get(1).copied().unwrap_or(pyre_object::PY_NULL) as *mut PyFrame;
    if f.is_null() {
        return Ok(pyre_object::w_none());
    }
    let anchor = unsafe { crate::eval::FrameAnchor::from_raw(f) };
    let w = unsafe { &*anchor.live() }.fget_f_builtins();
    Ok(if w.is_null() {
        pyre_object::w_none()
    } else {
        w
    })
}

register_frame_builtin_wrapper!(
    __majit_builtin_wrapper_target_fget_f_builtins,
    __majit_wrap_descr_typecheck_fget_f_builtins
);

/// `typedef.py _make_descr_typecheck_wrapper` around `PyFrame::fget_f_lineno`.
///
/// `GetSetProperty` binds this as its `fget` for `f_lineno`.
pub fn __majit_wrap_descr_typecheck_fget_f_lineno(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let f = args.get(1).copied().unwrap_or(pyre_object::PY_NULL) as *mut PyFrame;
    if f.is_null() {
        return Ok(pyre_object::w_none());
    }
    let anchor = unsafe { crate::eval::FrameAnchor::from_raw(f) };
    crate::executioncontext::jit_force_virtualizable(anchor.live());
    Ok(unsafe { &*anchor.live() }.fget_f_lineno())
}

register_frame_builtin_wrapper!(
    __majit_builtin_wrapper_target_fget_f_lineno,
    __majit_wrap_descr_typecheck_fget_f_lineno
);

/// `typedef.py _make_descr_typecheck_wrapper` around `PyFrame::fset_f_lineno`.
///
/// `GetSetProperty` binds this as its `fset` for `f_lineno`.
pub fn __majit_wrap_descr_typecheck_fset_f_lineno(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let f = args.get(1).copied().unwrap_or(pyre_object::PY_NULL) as *mut PyFrame;
    if f.is_null() {
        return Ok(pyre_object::w_none());
    }
    // `int_w` reaches `__index__`, which is application-level Python.
    let anchor = unsafe { crate::eval::FrameAnchor::from_raw(f) };
    let new_lineno = crate::baseobjspace::int_w(args[2])
        .map_err(|_| crate::PyError::value_error("lineno must be an integer"))?;
    crate::executioncontext::jit_force_virtualizable(anchor.live());
    unsafe { &mut *anchor.live() }.fset_f_lineno(new_lineno as isize)?;
    Ok(pyre_object::w_none())
}

register_frame_builtin_wrapper!(
    __majit_builtin_wrapper_target_fset_f_lineno,
    __majit_wrap_descr_typecheck_fset_f_lineno
);

/// `typedef.py _make_descr_typecheck_wrapper` around `PyFrame::fget_f_trace`.
///
/// `GetSetProperty` binds this as its `fget` for `f_trace`.
pub fn __majit_wrap_descr_typecheck_fget_f_trace(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let f = args.get(1).copied().unwrap_or(pyre_object::PY_NULL) as *mut PyFrame;
    if f.is_null() {
        return Ok(pyre_object::w_none());
    }
    let anchor = unsafe { crate::eval::FrameAnchor::from_raw(f) };
    crate::executioncontext::jit_force_virtualizable(anchor.live());
    let w = unsafe { &*anchor.live() }.fget_f_trace();
    Ok(if w.is_null() {
        pyre_object::w_none()
    } else {
        w
    })
}

register_frame_builtin_wrapper!(
    __majit_builtin_wrapper_target_fget_f_trace,
    __majit_wrap_descr_typecheck_fget_f_trace
);

/// `typedef.py _make_descr_typecheck_wrapper` around `PyFrame::fset_f_trace`.
///
/// `GetSetProperty` binds this as its `fset` for `f_trace`.
pub fn __majit_wrap_descr_typecheck_fset_f_trace(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let f = args.get(1).copied().unwrap_or(pyre_object::PY_NULL) as *mut PyFrame;
    if !f.is_null() {
        let anchor = unsafe { crate::eval::FrameAnchor::from_raw(f) };
        crate::executioncontext::jit_force_virtualizable(anchor.live());
        unsafe { &mut *anchor.live() }.fset_f_trace(args[2]);
    }
    Ok(pyre_object::w_none())
}

register_frame_builtin_wrapper!(
    __majit_builtin_wrapper_target_fset_f_trace,
    __majit_wrap_descr_typecheck_fset_f_trace
);

/// `typedef.py _make_descr_typecheck_wrapper` around `PyFrame::fdel_f_trace`.
///
/// `GetSetProperty` binds this as its `fdel` for `f_trace`.
pub fn __majit_wrap_descr_typecheck_fdel_f_trace(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let f = args.get(1).copied().unwrap_or(pyre_object::PY_NULL) as *mut PyFrame;
    if !f.is_null() {
        let anchor = unsafe { crate::eval::FrameAnchor::from_raw(f) };
        crate::executioncontext::jit_force_virtualizable(anchor.live());
        unsafe { &mut *anchor.live() }.fdel_f_trace();
    }
    Ok(pyre_object::w_none())
}

register_frame_builtin_wrapper!(
    __majit_builtin_wrapper_target_fdel_f_trace,
    __majit_wrap_descr_typecheck_fdel_f_trace
);

/// `typedef.py _make_descr_typecheck_wrapper` around `PyFrame::fget_f_trace_lines`.
///
/// `GetSetProperty` binds this as its `fget` for `f_trace_lines`.
pub fn __majit_wrap_descr_typecheck_fget_f_trace_lines(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let f = args.get(1).copied().unwrap_or(pyre_object::PY_NULL) as *mut PyFrame;
    if f.is_null() {
        return Ok(pyre_object::w_none());
    }
    let anchor = unsafe { crate::eval::FrameAnchor::from_raw(f) };
    crate::executioncontext::jit_force_virtualizable(anchor.live());
    Ok(pyre_object::w_bool_from(
        unsafe { &*anchor.live() }.fget_f_trace_lines(),
    ))
}

register_frame_builtin_wrapper!(
    __majit_builtin_wrapper_target_fget_f_trace_lines,
    __majit_wrap_descr_typecheck_fget_f_trace_lines
);

/// `typedef.py _make_descr_typecheck_wrapper` around `PyFrame::fset_f_trace_lines`.
///
/// `GetSetProperty` binds this as its `fset` for `f_trace_lines`.
pub fn __majit_wrap_descr_typecheck_fset_f_trace_lines(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let f = args.get(1).copied().unwrap_or(pyre_object::PY_NULL) as *mut PyFrame;
    if !f.is_null() {
        // `is_true` reaches `__bool__` / `__len__`.
        let anchor = unsafe { crate::eval::FrameAnchor::from_raw(f) };
        let v = crate::baseobjspace::is_true(args[2])?;
        crate::executioncontext::jit_force_virtualizable(anchor.live());
        unsafe { &mut *anchor.live() }.fset_f_trace_lines(v);
    }
    Ok(pyre_object::w_none())
}

register_frame_builtin_wrapper!(
    __majit_builtin_wrapper_target_fset_f_trace_lines,
    __majit_wrap_descr_typecheck_fset_f_trace_lines
);

/// `typedef.py _make_descr_typecheck_wrapper` around `PyFrame::fget_f_trace_opcodes`.
///
/// `GetSetProperty` binds this as its `fget` for `f_trace_opcodes`.
pub fn __majit_wrap_descr_typecheck_fget_f_trace_opcodes(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let f = args.get(1).copied().unwrap_or(pyre_object::PY_NULL) as *mut PyFrame;
    if f.is_null() {
        return Ok(pyre_object::w_none());
    }
    let anchor = unsafe { crate::eval::FrameAnchor::from_raw(f) };
    crate::executioncontext::jit_force_virtualizable(anchor.live());
    Ok(pyre_object::w_bool_from(
        unsafe { &*anchor.live() }.fget_f_trace_opcodes(),
    ))
}

register_frame_builtin_wrapper!(
    __majit_builtin_wrapper_target_fget_f_trace_opcodes,
    __majit_wrap_descr_typecheck_fget_f_trace_opcodes
);

/// `typedef.py _make_descr_typecheck_wrapper` around `PyFrame::fset_f_trace_opcodes`.
///
/// `GetSetProperty` binds this as its `fset` for `f_trace_opcodes`.
pub fn __majit_wrap_descr_typecheck_fset_f_trace_opcodes(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let f = args.get(1).copied().unwrap_or(pyre_object::PY_NULL) as *mut PyFrame;
    if !f.is_null() {
        // `is_true` reaches `__bool__` / `__len__`.
        let anchor = unsafe { crate::eval::FrameAnchor::from_raw(f) };
        let v = crate::baseobjspace::is_true(args[2])?;
        crate::executioncontext::jit_force_virtualizable(anchor.live());
        unsafe { &mut *anchor.live() }.fset_f_trace_opcodes(v);
    }
    Ok(pyre_object::w_none())
}

register_frame_builtin_wrapper!(
    __majit_builtin_wrapper_target_fset_f_trace_opcodes,
    __majit_wrap_descr_typecheck_fset_f_trace_opcodes
);

impl PyFrame {
    /// pyframe.py getdebug → self.debugdata
    #[inline]
    fn getdebug_data(&self) -> Option<&FrameDebugData> {
        // Spelled without a closure: a closure aggregate lowers to a
        // synthetic ctor call the walker cannot bind, and upstream's
        // `getdebug` is a plain None-checked field read.
        if self.debugdata.is_null() {
            None
        } else {
            Some(unsafe { &*self.debugdata })
        }
    }

    /// pyframe.py getorcreatedebug
    #[inline]
    fn getorcreate_debug_data(&mut self, init_lineno: isize) -> &mut FrameDebugData {
        if self.debugdata.is_null() {
            // The stable allocation below is still a collection point.  Keep
            // the owning frame rooted and reload it before publishing the new
            // debugdata pointer, mirroring the translated RPython shadow-stack
            // reload around `getorcreatedebug`.
            let frame_anchor = crate::eval::FrameAnchor::new(self);
            let allocation = unsafe { (*frame_anchor.live()).aux_allocation() };
            let raw = if allocation == FrameLocalsArrayAllocation::OldGenGc {
                pyre_object::gc_hook::try_gc_alloc_nursery_raw(
                    FRAME_DEBUG_DATA_GC_TYPE_ID,
                    std::mem::size_of::<FrameDebugData>(),
                ) as *mut FrameDebugData
            } else {
                std::ptr::null_mut()
            };
            // Build the payload only after that allocation.  `FrameDebugData::init`
            // copies the code object's first-seen globals into this stack
            // temporary; a collection inside the allocation forwards the
            // anchored frame and the `PyCode` slot the copy came from, but
            // nothing names the temporary, so constructing it first would
            // publish a pre-collection address into the new block.  Only an
            // atomic field read separates the allocation from the write below,
            // so the fresh block crosses no safepoint unrooted.
            let value = FrameDebugData::init(unsafe { (*frame_anchor.live()).pycode }, init_lineno);
            let debugdata = if raw.is_null() {
                pyre_object::lltype::malloc_raw(value)
            } else {
                unsafe { std::ptr::write(raw, value) };
                raw
            };
            unsafe { (*frame_anchor.live()).debugdata = debugdata };
            // Old-gen frame → nursery `debugdata` is an old-to-young
            // field store (`incminimark.py write_barrier`). Remembering
            // only the payload leaves the frame off
            // `old_objects_pointing_to_young`.
            let live_frame = frame_anchor.live() as *mut Self;
            pyre_object::gc_hook::try_gc_write_barrier(live_frame as pyre_object::gc_hook::GCREF);
            let debugdata = unsafe { (*live_frame).debugdata };
            remember_frame_debug_data(debugdata);
            return unsafe { &mut *debugdata };
        }
        // Callers mutate the returned payload directly. Remember the container
        // immediately before exposing it so old debugdata keeps any young refs
        // written by the caller visible to the next minor collection.
        remember_frame_debug_data(self.debugdata);
        unsafe { &mut *self.debugdata }
    }

    /// PyPy-compatible `getdebug()`.
    #[inline]
    pub fn getdebug(&self) -> Option<&FrameDebugData> {
        self.getdebug_data()
    }

    /// PyPy-compatible `getorcreatedebug()`.
    #[inline]
    pub fn getorcreatedebug(&mut self, init_lineno: isize) -> &mut FrameDebugData {
        self.getorcreate_debug_data(init_lineno)
    }

    /// `pyframe.py PyFrame.getcode`: `hint(self.pycode, promote=True)`.
    /// `pycode` is a virtualizable field, so a trace reads it as a box; the
    /// promote is what makes the code object — and every code-invariant the
    /// callers below read off it — a constant for the compiled loop.  The
    /// undecorated [`PyFrame::code`] is the read for paths upstream reaches
    /// through `self.pycode` directly.
    #[inline]
    pub fn getcode(&self) -> &CodeObject {
        let pycode = majit_metainterp::jit::promote(self.pycode);
        unsafe {
            &*(crate::w_code_get_ptr(pycode as pyre_object::PyObjectRef) as *const CodeObject)
        }
    }

    /// PyPy-compatible `fget_code`.
    #[inline]
    pub fn fget_code(&self) -> &CodeObject {
        self.code()
    }

    /// PyPy `get_w_globals_storage` compatibility alias.
    #[inline]
    pub fn get_w_globals_storage(&self) -> PyObjectRef {
        self.get_w_globals()
    }

    /// `pyframe.py PyFrame.get_w_globals`: the normal case reads the
    /// promoted code object's first-seen globals. A frame executing a shared
    /// code object in another namespace carries the override in debugdata.
    #[inline]
    pub fn get_w_globals(&self) -> PyObjectRef {
        if let Some(data) = self.getdebug_data() {
            return data.w_globals;
        }
        if self.pycode.is_null() {
            return pyre_object::PY_NULL;
        }
        // `pyframe.py get_w_globals` `return jit.promote(self.pycode).w_globals`.  Without
        // the promote the code object stays a varying value on the trace and
        // the globals read is a load the optimizer cannot fold; `w_globals` is
        // not a virtualizable field, so nothing else pins it to a constant.
        let pycode = majit_metainterp::jit::promote(self.pycode);
        unsafe { crate::w_code_get_w_globals(pycode as PyObjectRef) }
    }

    /// `pyframe.py PyFrame.set_w_globals`: force the rare globals override
    /// into lazily-created debugdata. Root the value across that allocation;
    /// RPython's shadow-stack transform does the same for the live argument.
    pub fn set_w_globals(&mut self, w_globals: PyObjectRef) {
        let roots = pyre_object::gc_roots::push_roots();
        let slot = roots.base();
        let _ = roots.pin_root(w_globals);
        self.getorcreate_debug_data(-1).w_globals = roots.get(slot);
        remember_frame_debug_data(self.debugdata);
    }

    /// pyframe.py get_w_f_trace
    #[inline]
    pub fn get_w_f_trace(&self) -> PyObjectRef {
        // Closure-free for the walker; upstream reads `debugdata.w_f_trace`
        // behind two None checks.
        match self.getdebug_data() {
            Some(data) if !data.w_f_trace.is_null() => data.w_f_trace,
            _ => pyre_object::PY_NULL,
        }
    }

    /// pyframe.py get_is_being_profiled
    #[inline]
    pub fn get_is_being_profiled(&self) -> bool {
        // Closure-free for the walker (see `getdebug_data`).
        match self.getdebug_data() {
            Some(data) => data.is_being_profiled,
            None => false,
        }
    }

    /// `getorcreatedebug().w_locals` — the STORE_NAME / DELETE_NAME target
    /// (`pyopcode.py`): the class namespace, or the globals dict at
    /// module scope. Lazily allocates an empty dict if the frame has none.
    /// Unlike `getdictscope` it performs no `fast2locals` materialization, so
    /// it never disturbs `CO_FAST_HIDDEN` slots.
    #[inline]
    pub fn get_or_create_w_locals(&mut self) -> PyObjectRef {
        let existing = self.get_w_locals();
        if !existing.is_null() {
            return existing;
        }
        // Only the allocating arm below needs the frame held as a root.
        let frame_anchor = crate::eval::FrameAnchor::new(self);
        // A frame that reaches STORE_NAME / SETUP_ANNOTATIONS without a
        // bound locals mapping is degenerate (module / class / exec all
        // bind one in `initialize_frame_scopes` / `setdictscope`).
        // Allocate a fresh dict so the write still lands somewhere
        // observable instead of faulting.
        let roots = pyre_object::gc_roots::push_roots();
        let locals_slot = roots.base();
        let _ = roots.pin_root(unsafe { pyre_object::w_dict_new() });
        let frame = unsafe { &mut *frame_anchor.live() };
        // `getorcreate_debug_data` allocates the debug block this frame does not
        // have yet, and an assignment evaluates its value before its place, so
        // combining the two reads the dict's address before that allocation and
        // stores it after — leaving the caller the live word and the frame the
        // dead one.
        let debug = frame.getorcreate_debug_data(-1);
        debug.w_locals = roots.get(locals_slot);
        roots.get(locals_slot)
    }

    /// PyPy-compatible `__init__` hook.
    #[inline]
    pub fn __init__(&mut self, code: *const (), mut outer_func: PyObjectRef) {
        let _ = outer_func;
        let allocation = self.aux_allocation();
        self.pycode = code;
        let raw =
            unsafe { crate::w_code_get_ptr(code as pyre_object::PyObjectRef) as *const CodeObject };
        if !self.locals_cells_stack_w.is_null()
            && !pyre_object::gc_hook::try_gc_owns_object(
                self.locals_cells_stack_w as pyre_object::gc_hook::GCREF,
            )
        {
            unsafe { dealloc_array_with_gc_header(self.locals_cells_stack_w) };
        }
        self.locals_cells_stack_w = unsafe {
            alloc_frame_locals_array(
                (&*raw).varnames.len() + ncells(&*raw) + (&*raw).max_stackdepth as usize,
                PY_NULL,
                allocation,
            )
        };
        self.valuestackdepth = unsafe { (&*raw).varnames.len() + ncells(&*raw) };
        self.last_instr = -1;
        self.flags = 0;
        self.f_generator_wref = PY_NULL;
        self.w_yielding_from = PY_NULL;
        self.f_backref = std::ptr::null_mut();
        unsafe {
            clear_debugdata_ptr(&mut self.debugdata);
        }
        // This storage-only hook carries no globals object.
        let mut w_globals = PY_NULL;
        self.w_builtin = pyre_object::with_roots!(outer_func, w_globals =>
            crate::baseobjspace::frame_builtin_obj(w_globals, crate::call::getexecutioncontext())
        );
        // `pyframe.py PyFrame.__init__`: only a code/globals identity mismatch
        // creates a per-frame debugdata override.
        if unsafe { crate::w_code_frame_stores_global(code as PyObjectRef, w_globals) } {
            self.set_w_globals(w_globals);
        }
        // pyframe.py — final step of __init__.
        self.initialize_frame_scopes(outer_func, code).expect(
            "PyFrame::__init__: initialize_frame_scopes raised — caller should use createframe",
        );
        remember_frame_locals_array(self.locals_cells_stack_w);
        // Old-gen frame → nursery locals array is an old-to-young field
        // store (`incminimark.py write_barrier`). Remembering only the
        // array leaves the frame off `old_objects_pointing_to_young`.
        if pyre_object::gc_hook::try_gc_owns_object(
            self as *mut PyFrame as pyre_object::gc_hook::GCREF,
        ) {
            pyre_object::gc_hook::try_gc_write_barrier(
                self as *mut PyFrame as pyre_object::gc_hook::GCREF,
            );
        }
    }

    /// PyPy-compatible `__repr__`.
    #[inline]
    pub fn __repr__(&self) -> String {
        format!(
            "<{}.{} executing {} at line {}>",
            module_path!(),
            std::any::type_name::<Self>()
                .rsplit("::")
                .next()
                .unwrap_or("PyFrame"),
            self.code().obj_name.as_str(),
            self.get_last_lineno()
        )
    }

    /// PyPy-compatible `fget_w_globals_storage`.
    #[inline]
    pub fn fget_w_globals_storage(&self) -> PyObjectRef {
        self.get_w_globals_storage()
    }

    /// PyPy-compatible `_getcell`.
    #[inline]
    pub fn _getcell(&self, varindex: usize) -> PyObjectRef {
        locals_w!(self)
            .as_slice()
            .get(self.nlocals() + varindex)
            .copied()
            .unwrap_or(PY_NULL)
    }

    /// PyPy-compatible `getclosure`.
    #[inline]
    pub fn getclosure(&self) -> PyObjectRef {
        PY_NULL
    }

    /// pyframe.py:213-218 — bind the locals namespace of code that is not
    /// `OPTIMIZED`.  A class body gets a fresh namespace mapping; anything
    /// else (module-level code, and the same code reached through
    /// `FunctionType(co, globals)`) aliases its globals, so `STORE_NAME` /
    /// `LOAD_NAME` / `DELETE_NAME` and `locals()` land there.  Optimized code
    /// keeps its fast-locals array and binds nothing.
    ///
    /// The binding is the canonical `W_DictObject`, not the raw `DictStorage`
    /// proxy, so those opcodes route through the object.
    fn bind_unoptimized_locals_scope(&mut self) {
        let flags = unsafe { (*pyframe_get_pycode(self)).flags };
        if flags.contains(CodeFlags::OPTIMIZED) {
            return;
        }
        if flags.contains(CodeFlags::NEWLOCALS) {
            // `build_class` replaces this with the `__prepare__` namespace via
            // `setdictscope`; an orphan NEWLOCALS frame still has a usable
            // mapping.  `newdict(module=True)` — the mapping a class body binds
            // names in carries `ModuleDictStrategy.version?`.
            let w_locals = pyre_object::w_module_dict_new();
            self.getorcreate_debug_data(-1).w_locals = w_locals;
        } else {
            let w_globals = self.get_w_globals();
            self.getorcreate_debug_data(-1).w_locals = w_globals;
        }
    }

    /// pyframe.py initialize_frame_scopes.
    ///
    /// Errors mirror pyframe.py (TypeError "directly executed code
    /// object may not contain free variables") and pyframe.py
    /// (ValueError "code object received a closure with an unexpected
    /// number of free variables") so callers can surface them through
    /// PyPy's OperationError-equivalent path instead of panicking.
    /// `pyframe.py initialize_frame_scopes` is `@jit.unroll_safe`.
    /// The cell/freevar loop is the reason: without the hint the
    /// whole function is one residual.
    #[inline]
    #[majit_macros::unroll_safe]
    pub fn initialize_frame_scopes(
        &mut self,
        outer_func: PyObjectRef,
        _code: *const (),
    ) -> Result<(), crate::PyError> {
        self.bind_unoptimized_locals_scope();

        let code = unsafe { &*pyframe_get_pycode(self) };
        let npure = npure_cellvars(code);
        let nfreevars = code.freevars.len();
        if npure == 0 && nfreevars == 0 {
            return Ok(());
        }

        let closure = if !outer_func.is_null() && unsafe { crate::is_function(outer_func) } {
            unsafe { crate::function_get_closure(outer_func) }
        } else {
            PY_NULL
        };
        let closure_size = if closure.is_null() {
            0
        } else {
            unsafe { w_tuple_len(closure) }
        };
        if nfreevars > 0 && outer_func.is_null() {
            return Err(crate::PyError::type_error(
                "directly executed code object may not contain free variables",
            ));
        }
        if closure_size != nfreevars {
            return Err(crate::PyError::value_error(
                "code object received a closure with an unexpected number of free variables"
                    .to_string(),
            ));
        }

        // CPython 3.11+ unified slot layout: only cellvars NOT in
        // varnames take a fresh slot.  See `npure_cellvars` and
        // `new_for_call_with_closure` for the call-site mirror.
        let mut index = code.varnames.len();
        for _ in 0..npure {
            // pyframe.py `PyFrame.initialize_frame_scopes` `Cell(None, self.pycode.cell_families[i])`.
            let family = unsafe {
                crate::pycode::w_code_cell_family(self.pycode as pyre_object::PyObjectRef, index)
            };
            let cell = pyre_object::w_cell_new(PY_NULL, family);
            self.set_locals_w(index, cell);
            index += 1;
        }
        for i in 0..nfreevars {
            let cell = unsafe { w_tuple_getitem(closure, i as i64).unwrap_or(PY_NULL) };
            self.set_locals_w(index, cell);
            index += 1;
        }
        Ok(())
    }

    /// pyframe.py setdictscope(w_locals, skip_free_vars=False) —
    /// install `w_locals` as the frame's locals mapping and reflect
    /// its entries into the fastlocals via `locals2fast`.
    ///
    /// `pypy/interpreter/pyopcode.py ensure_ns` admits any object
    /// exposing `__getitem__` as locals, so the mapping may be a plain dict
    /// (module / class / `exec(src, g, l)`) or an arbitrary mapping; both
    /// share this path and `STORE_NAME` / `LOAD_NAME` / `DELETE_NAME` route
    /// through `space.setitem` / `space.getitem` / `space.delitem` on it.
    #[inline]
    pub fn setdictscope(&mut self, w_locals: PyObjectRef) -> Result<(), crate::PyError> {
        // A frame with no debug block yet allocates one below, which
        // `getorcreate_debug_data` names as a collection point.  It anchors the
        // frame it is called on; `w_locals` is the caller's word and nothing
        // else names it, and an assignment evaluates its value before its
        // place, so writing it directly would store a pre-collection address.
        let roots = pyre_object::gc_roots::push_roots();
        let locals_slot = roots.base();
        let _ = roots.pin_root(w_locals);
        let debug = self.getorcreate_debug_data(-1);
        debug.w_locals = roots.get(locals_slot);
        self.locals2fast(false)
    }

    /// Read the frame's locals mapping registered by `setdictscope`
    /// (or `initialize_frame_scopes`).  Returns `PY_NULL` when the frame has
    /// no locals bound yet (a function before its first `fast2locals`).
    #[inline]
    pub fn get_w_locals(&self) -> PyObjectRef {
        // `pyframe.py get_w_locals`: plain None-check, no closure.
        // `map_or` lowers to a synthetic-transparent-ctor residual the
        // walker cannot bind (`get_w_locals::closure`).
        match self.getdebug_data() {
            None => pyre_object::PY_NULL,
            Some(data) => data.w_locals,
        }
    }

    /// CPython 3.14 `PyFrameObject.f_extra_locals`, allocated by
    /// `framelocalsproxy_setitem` only when a proxy write does not target a
    /// writable fast local.  It is always an exact dict and is not the
    /// module/class/exec mapping kept in [`Self::get_w_locals`].
    fn get_or_create_extra_locals(&mut self) -> PyObjectRef {
        let existing = self
            .getdebug_data()
            .map_or(pyre_object::PY_NULL, |data| data.w_extra_locals);
        if !existing.is_null() {
            return existing;
        }
        // `w_dict_new` returns a movable nursery object, while allocating the
        // debug payload below may collect.  CPython's frame owns both through
        // traced references; mirror that lifetime by pinning the dict until
        // the payload has been installed and then reload its forwarded value.
        let roots = pyre_object::gc_roots::push_roots();
        let slot = roots.base();
        let _ = roots.pin_root(unsafe { pyre_object::w_dict_new() });
        let debug = self.getorcreate_debug_data(-1);
        debug.w_extra_locals = roots.get(slot);
        roots.get(slot)
    }

    /// `f_extra_locals` on the frame debug payload. `PY_NULL` when the
    /// frame has no extra-locals dict.
    ///
    /// Plain None-check, no closure. `map_or` lowers to a
    /// synthetic-transparent-ctor residual the walker cannot bind
    /// (`get_extra_locals::closure`), and `tuple.__new__` reaches that
    /// residual after `proxy_list_new`.
    #[inline]
    pub fn get_extra_locals(&self) -> PyObjectRef {
        // A call through `getdebug_data` residualizes as a symbolic
        // `Option<&FrameDebugData>` ctor the walker cannot bind
        // (`getdebug_data` at the first byte of this body). Read the
        // debug payload field directly, the same None-checked getfield
        // `pyframe.py getdebug` is.
        if self.debugdata.is_null() {
            pyre_object::PY_NULL
        } else {
            unsafe { (*self.debugdata).w_extra_locals }
        }
    }

    /// pyframe.py getdictscope — runs `fast2locals` then returns
    /// `self.debugdata.w_locals` (the locals mapping object).
    ///
    /// `fast2locals` lazily materialises a fresh dict for a function frame
    /// (pyframe.py `space.newdict(instance=True)`) and caches it in
    /// `w_locals`, so repeated calls return the same object —
    /// `frame.f_locals is frame.f_locals` holds.
    #[inline]
    pub fn getdictscope(&mut self) -> Result<PyObjectRef, crate::PyError> {
        self.fast2locals()?;
        Ok(self.get_w_locals())
    }

    /// pyframe.py `fget_getdictscope` — the `f_locals` getset.
    ///
    /// Upstream is `return self.getdictscope()`.  Optimized frames answer
    /// with a write-through `FrameLocalsProxy` instead: `lib-python/3/test/
    /// test_frame.py` `FrameLocalsProxy.__name__`.  The residual force lives
    /// on [`__majit_wrap_descr_typecheck_fget_getdictscope`], the GetSetProperty wrapper,
    /// not here — `jtransform.py rewrite_op_jit_force_virtualizable` deletes
    /// it from jitcode; the interpreter copy still runs.
    pub fn fget_getdictscope(&mut self) -> Result<PyObjectRef, crate::PyError> {
        if self.code().flags.contains(crate::CodeFlags::OPTIMIZED)
            || self.has_active_hidden_locals()
        {
            return Ok(frame_locals_proxy::new(self as *mut PyFrame as PyObjectRef));
        }
        let w = self.getdictscope()?;
        Ok(if w.is_null() {
            pyre_object::w_dict_new()
        } else {
            w
        })
    }

    /// The mapping `locals()` hands back, and the implicit locals `exec` /
    /// `eval` run against (`_PyEval_GetFrameLocals`).
    ///
    /// A frame whose `f_locals` is a `FrameLocalsProxy` — every optimized
    /// frame — yields an INDEPENDENT dict copy, so `locals() is locals()` is
    /// false, a snapshot does not track later stores, and writing into one
    /// reaches neither the fast locals nor the next snapshot (PEP 667).
    /// Module and class frames keep handing back their real namespace, which
    /// is what makes a module-level `locals() is globals()` still hold.
    pub fn frame_locals_snapshot(&mut self) -> Result<PyObjectRef, crate::PyError> {
        if self.code().flags.contains(crate::CodeFlags::OPTIMIZED)
            || self.has_active_hidden_locals()
        {
            return self.frame_locals_proxy_snapshot();
        }
        self.getdictscope()
    }

    /// CPython 3.14 `_PyFrame_HasHiddenLocals`: a non-optimized module/class
    /// frame uses a locals proxy only while a PEP 709 hidden slot is bound.
    pub fn has_active_hidden_locals(&self) -> bool {
        let code = self.code();
        (0..code.varnames.len())
            .any(|index| hidden_local(code, index) && !locals_w!(self)[index].is_null())
    }

    /// `dict(proxy)` — what `PyDict_Update` of a fresh dict from the proxy
    /// builds, and so what `locals()`, `FrameLocalsProxy.copy` and
    /// `framelocalsproxy_repr` all hand back: every bound locals-plus slot in
    /// `co_localsplusnames` order, then the frame object's separate
    /// `f_extra_locals` dict.
    ///
    /// It is the DEDUPLICATED view.  `framelocalsproxy_keys` reports a name
    /// held by both halves twice, but each of those keys reads back through
    /// `framelocalsproxy_getitem`, which answers from the slot — so the pair
    /// collapses to one entry carrying the slot's value at the slot's
    /// position.  `FrameLocalsProxy::pin_entries` is the undeduplicated view
    /// that `keys`, `values`, `items` and `__len__` use.
    ///
    /// The original non-optimized `f_locals` mapping is deliberately absent
    /// while a PEP 709 hidden slot makes the proxy active;
    /// `framelocalsproxy_keys` and `framelocalsproxy_getitem` read only the
    /// fast array and `f_extra_locals`.
    /// `@jit.unroll_safe` on `PyFrame.fast2locals` cancels `contains_loop` in
    /// `look_inside_graph`, and the slot loop below needs the same treatment:
    /// without it the whole function is one residual call, and the `f_locals`
    /// read behind it forces the virtualizable for the length of that call.
    #[majit_macros::unroll_safe]
    pub fn frame_locals_proxy_snapshot(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let roots = pyre_object::gc_roots::push_roots();
        let snapshot_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(unsafe { pyre_object::w_dict_new() });

        let code = self.code();
        // A plain loop rather than a closure over `self`: a capture path is
        // truncated at a raw-pointer deref, so a closure would capture the
        // `locals_cells_stack_w` field itself and the borrow would read as the
        // address of the virtualizable array instead of a load from it, which
        // suppresses the protocol.  `fast2locals` keeps this shape for the same
        // reason.
        for (index, name, cell_slot) in locals_plus_names(code) {
            if index >= locals_w!(self).len() {
                continue;
            }
            let slot = locals_w!(self)[index];
            let value = if cell_slot && !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
                unsafe { pyre_object::w_cell_get(slot) }
            } else {
                slot
            };
            if value.is_null() {
                continue;
            }
            // `setitem_str_object` allocates the string key and can move
            // `value`; keep it in a shadow-stack slot and reload it for
            // the store.
            let value_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(value);
            setitem_str_object(roots.get(snapshot_slot), name, roots.get(value_slot))?;
        }

        // The extras half comes second, and a key a bound slot already
        // supplied is skipped: `framelocalsproxy_getitem` answers such a key
        // from the slot, so the store would only rewrite the entry with the
        // value it already holds while leaving it at its slot position.
        let extra = self.get_extra_locals();
        if !extra.is_null() {
            let extra_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(extra);
            merge_extra_locals(roots.get(snapshot_slot), roots.get(extra_slot))?;
        }
        Ok(roots.get(snapshot_slot))
    }

    /// Test-helper constructor — creates a frame with a fresh execution
    /// context.
    ///
    /// TODO: PyPy has no equivalent — tests there call
    /// `space.appexec` or build the context explicitly before invoking
    /// `space.createframe`.  Pyre keeps `PyFrame::new` for test
    /// ergonomics (~67 callers across `pyre-interpreter`, `pyre-jit`,
    /// `pyre-jit-trace` test modules) but routes the body through
    /// `createframe` (PyPy `baseobjspace.py`) so every heap-allocated
    /// `PyFrame` flows through the canonical entry point.
    pub fn new(code: CodeObject) -> FrameBox {
        let execution_context = Rc::new(PyExecutionContext::default());
        let ctx_ptr = Rc::as_ptr(&execution_context);
        let frame = Self::new_with_context(code, execution_context)
            .expect("PyFrame::new: test entry code must not carry freevars");
        // `threadlocals.py:enter_thread` — the ExecutionContext slot belongs to
        // the OS-thread locals and is installed once at thread entry, not
        // re-stamped per frame.  The other entry points do this themselves
        // (`pyrex/src/lib.rs`, `pyre-wasm/src/lib.rs`,
        // `pyre-wasm-test/src/main.rs`); as the remaining entry point this one
        // must too, or `space.getexecutioncontext()` stays null and every
        // context reached through it — the class-body frame's builtins among
        // them — falls back to the empty default.
        crate::call::set_last_exec_ctx(ctx_ptr);
        frame
    }

    /// Module-entry adapter for `createframe` — leaks owned arguments
    /// into the raw-pointer shape that `createframe` expects, builds
    /// the module-entry `__dict__` (with `__name__ = "__main__"`), and
    /// returns the resulting heap-allocated frame.
    ///
    /// TODO: PyPy's `space.createframe(code, w_globals,
    /// outer_func)` (`baseobjspace.py`) takes already-constructed
    /// `code` and `w_globals` Python objects and reads execution context
    /// from `space.threadlocals`. PyPy callers (`pypy/interpreter/main.py
    /// run_module`, etc.) build `w_globals` and set `__name__` themselves
    /// before calling `createframe`. Pyre's adapter exists because:
    /// (1) Rust has no GC, so `code`/`w_globals`/`ec` lifetimes must be
    /// leaked into raw pointers manually; (2) pyre's `ExecutionContext`
    /// is per-frame (not per-space), so callers pass `Rc<EC>` explicitly.
    /// Three production callers (`pyrex/lib.rs`, `pyre-wasm/lib.rs`,
    /// `pyre-wasm-test/main.rs`) plus `PyFrame::new` (test helper)
    /// invoke this; without the adapter each would duplicate the 9-line
    /// leak setup.
    ///
    /// Returns `Result<Box<Self>, PyError>` so createframe's freevar /
    /// closure errors surface as interpreter errors instead of panics.
    pub fn new_with_context(
        code: CodeObject,
        execution_context: Rc<PyExecutionContext>,
    ) -> Result<FrameBox, crate::PyError> {
        // `fresh_module_globals` seeds `__builtins__ = space.builtin`
        // (PyPy `main.py:45 / Module.__init__` parity) into a proxy-less
        // celldict.  Just set `__name__` on top.
        let w_globals = execution_context.fresh_module_globals();
        // Root the fresh globals across the `__name__` store; the frame
        // construction below roots them again for its own span.
        let _root = pyre_object::gc_roots::push_roots();
        let globals_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_globals);
        let name_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new("__main__"));
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                pyre_object::gc_roots::shadow_stack_get(globals_slot),
                "__name__",
                pyre_object::gc_roots::shadow_stack_get(name_slot),
            );
        }
        Self::new_with_context_and_globals(
            code,
            execution_context,
            pyre_object::gc_roots::shadow_stack_get(globals_slot),
        )
    }

    /// `new_with_context` over a globals dict the caller already owns.
    ///
    /// A run that has to consult `sys.path_hooks` before it knows whether it
    /// is executing a source file at all needs `__main__`'s namespace — and
    /// the importlib bootstrap that runs in it — to exist before any code
    /// object does, so the namespace cannot come from the frame.
    pub fn new_with_context_and_globals(
        code: CodeObject,
        execution_context: Rc<PyExecutionContext>,
        w_globals: PyObjectRef,
    ) -> Result<FrameBox, crate::PyError> {
        // Root the globals across the code/object allocations and the frame
        // construction; `createframe_obj` publishes `w_globals` to the code
        // on its first use, or stores a later distinct dictionary on the
        // frame's debug data, and both locations root it once they return.
        let _root = pyre_object::gc_roots::push_roots();
        let w_globals = pyre_object::gc_roots::pin_root(w_globals);
        let w_code = pyre_object::gc_roots::pin_root(crate::box_code_object(code));
        let ctx_ptr = Rc::into_raw(execution_context);
        crate::createframe_obj(w_code as *const (), w_globals, ctx_ptr, None)
    }

    /// PyPy `space.createframe(code, w_globals, None)` for an already wrapped
    /// code object, notably `app_main.py:execfile`'s `marshal.load(f)` result.
    /// Keeping that wrapper preserves its exact first-line, filename and
    /// decoded-location state; cloning only the compiler `CodeObject` would
    /// silently discard the wrapper-owned parts of marshal's result.
    pub fn new_with_context_and_globals_obj(
        w_code: PyObjectRef,
        execution_context: Rc<PyExecutionContext>,
        w_globals: PyObjectRef,
    ) -> Result<FrameBox, crate::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let root_base = _roots.publish(&[w_code, w_globals]);
        _roots.normalize(root_base, 2);
        let ctx_ptr = Rc::into_raw(execution_context);
        crate::createframe_obj(
            _roots.get(root_base) as *const (),
            _roots.get(root_base + 1),
            ctx_ptr,
            None,
        )
    }

    /// RPython MetaInterp traces against its own MIFrame stack instead of
    /// mutating the live interpreter frame in place. pyre still executes
    /// bytecodes concretely during tracing, so use an owned snapshot when
    /// recording a trace to keep the real frame state unchanged until the
    /// interpreter actually executes the same path.
    pub fn snapshot_for_tracing(&self) -> FrameBox {
        // A tracer holds this snapshot off the `CURRENT_FRAME` chain across
        // the whole `trace_bytecode` walk, during which a major GC cycle can
        // complete.  `FrameBox::new` is what makes that safe: it acquires a
        // translated-livevar root for the frame, held for exactly the handle's
        // lifetime, which is the walk (`trace_bytecode` takes the box by value
        // and hands it back).  Before that root existed the snapshot had to be
        // `new_boxed` — GC-invisible with a deterministic scope-end free —
        // because nothing reached it.
        //
        // Being GC-owned is what lets the trace's synchronization target be a
        // forwardable object: a `new_boxed` snapshot is an address the
        // collector does not own, so `pytraceback_object_custom_trace` has to
        // skip its frame edge and `TraceCtx::virtualizable_heap_ptr` has no
        // slot to read back.  Same allocation shape as
        // `snapshot_for_generator` below, for the same reason.
        let mut frame =
            FrameBox::new(self.build_snapshot_frame(FrameLocalsArrayAllocation::OldGenGc));
        // fix_array_ptrs AFTER Box allocation: inline_buf ptr must
        // point to the heap-allocated frame, not a stale stack address.
        frame.fix_array_ptrs();
        frame
    }

    /// Build the copied `PyFrame` value shared by the tracer snapshot and
    /// the generator snapshot.  Frame-LOCAL state (`locals_cells_stack_w` /
    /// `valuestackdepth` / `last_instr`) is COPIED, so snapshot mutations to
    /// locals/stack are discarded — the abort-safety the snapshot exists
    /// for.  The code/debugdata-derived globals is the SAME dict ptr, so a concrete shared-heap
    /// write during recording would leak to the real heap and double-apply
    /// on the compiled loop's re-run; Gap 10 removed that path (inline-frame
    /// STORE_GLOBAL records as deferred IR, applied exactly once).
    fn build_snapshot_frame(&self, allocation: FrameLocalsArrayAllocation) -> PyFrame {
        PyFrame {
            ob_header: frame_ob_header(),
            pycode: self.pycode,
            locals_cells_stack_w: unsafe {
                let values = locals_w!(self).to_vec();
                let array = alloc_frame_locals_array(values.len(), PY_NULL, allocation);
                for (i, value) in values.into_iter().enumerate() {
                    (*array).items_mut_ptr().add(i).write(value);
                }
                remember_frame_locals_array(array);
                array
            },
            valuestackdepth: self.valuestackdepth,
            last_instr: self.last_instr,
            flags: self.flags,
            failed_attr_cleanup: self.failed_attr_cleanup,
            debugdata: unsafe { clone_debugdata_ptr(self.debugdata, allocation) },
            vable_token: self.vable_token,
            f_generator_wref: self.f_generator_wref,
            w_yielding_from: self.w_yielding_from,
            f_backref: self.f_backref,
            w_builtin: self.w_builtin,
        }
    }

    /// Snapshot a borrowed frame into a GC-managed owned frame for
    /// `initialize_as_generator`.  Unlike `snapshot_for_tracing` this uses
    /// `FrameBox::new` (GC lifetime): a generator's suspended frame lives as
    /// long as the generator object reaches it (`generator.py` holds the
    /// frame), and the generator's custom trace greys the frame block.
    pub fn snapshot_for_generator(&self) -> FrameBox {
        // `build_snapshot_frame` finishes its allocations before returning.
        // `FrameBox::new` then uses the non-collecting `alloc_in_oldgen` path
        // and write-barriers the frame before this returns.
        let mut frame =
            FrameBox::new(self.build_snapshot_frame(FrameLocalsArrayAllocation::OldGenGc));
        frame.fix_array_ptrs();
        frame
    }

    /// Number of local variable slots (from code object).
    ///
    /// `@jit.elidable`: read-only access to `varnames.len()`, no
    /// allocation, no raise.  `EF_ELIDABLE_CANNOT_RAISE` parity.
    #[majit_macros::elidable_cannot_raise]
    #[inline]
    pub fn nlocals(&self) -> usize {
        unsafe { (&*pyframe_get_pycode(self)).varnames.len() }
    }

    /// Number of cell + free variable slots.
    #[majit_macros::elidable_cannot_raise]
    #[inline]
    pub fn ncells(&self) -> usize {
        // `npure_cellvars` (the O(cellvars × varnames) overlap count) is
        // code-invariant and cached on the `PyCode` wrapper; `freevars.len()`
        // is a direct read.  Falls back to the full walk for stub frames
        // whose `pycode` wrapper carries the `u32::MAX` sentinel.
        let code = unsafe { &*pyframe_get_pycode(self) };
        match unsafe {
            crate::pycode::w_code_npure_cellvars(self.pycode as pyre_object::PyObjectRef)
        } {
            Some(npure) => npure + code.freevars.len(),
            None => ncells(code),
        }
    }

    /// First index of the operand stack (after locals and cells).
    #[inline]
    pub fn stack_base(&self) -> usize {
        self.nlocals() + self.ncells()
    }

    // ── Stack operations ──────────────────────────────────────────────

    /// Reload `self` the way RPython's GC transform reloads every livevar from
    /// the shadow stack after a safepoint.
    ///
    /// pyre has no such pass — `gc_current_object_address` calls that gap a
    /// documented TODO — so a `&mut PyFrame` held across an allocating call
    /// names the abandoned nursery copy once a minor collection relocates a
    /// JIT-created frame. Following the forwarding stub here is what lets the
    /// opcode bodies keep the `pop; op; push` shape `pyopcode.py` uses instead
    /// of each call site carrying an anchor of its own.
    ///
    /// Cheap on the common path: a nursery range compare, and the header read
    /// only for an address the nursery owns.
    #[inline]
    fn live_mut(&mut self) -> &mut Self {
        let addr = self as *mut Self as pyre_object::gc_hook::GCREF;
        unsafe { &mut *(pyre_object::gc_hook::try_gc_current_object_address(addr) as *mut Self) }
    }

    #[inline]
    pub fn push(&mut self, value: PyObjectRef) {
        // Both writes below — the stack slot and the depth — have to land on
        // the live frame, so reload once and use it for both.
        let frame = self.live_mut();
        frame.assert_stack_index(frame.valuestackdepth);
        let idx = frame.valuestackdepth;
        frame.set_locals_w(idx, value);
        frame.valuestackdepth = idx + 1;
    }

    /// Reads and writes through the caller's `&mut self`, without the
    /// [`Self::live_mut`] reload [`Self::push`] takes.
    ///
    /// That is sound because the opcode bodies keep `pyopcode.py`'s
    /// `pop; op; push` order: every pop runs before the operation that can
    /// collect, and the push that follows the operation reloads for itself.
    /// A body that allocates between two pops breaks the premise — the second
    /// pop would read the abandoned copy — and owes the reload at its own call
    /// site.
    #[inline]
    pub fn pop(&mut self) -> PyObjectRef {
        // `pyframe.py popvalue_maybe_none`: `assert_stack_index` checks
        // against `_stack_start()` only untranslated, and `assert depth >= 0`
        // is the one check the translated interpreter keeps.
        if self.valuestackdepth == 0 {
            report_stack_underflow(self);
        }
        let depth = self.valuestackdepth - 1;
        self.assert_stack_index(depth);
        let value = locals_w!(self)[depth];
        self.set_locals_w(depth, PY_NULL);
        self.valuestackdepth = depth;
        value
    }

    #[inline]
    pub fn peek(&self) -> PyObjectRef {
        // The index is computed into a local first, as
        // `pyframe.py peekvalue_maybe_none` computes `index` before
        // its subscript.  A subscript evaluates its receiver before its index
        // expression, so spelling the arithmetic inside the brackets emits the
        // `locals_cells_stack_w` read ahead of the subtraction's overflow
        // check, and that check's branch then carries the array out of the
        // block on a link — which `_check_no_vable_array` rejects.
        let index = self.valuestackdepth - 1;
        locals_w!(self)[index]
    }

    #[inline]
    #[allow(dead_code)]
    pub fn peek_at(&self, depth: usize) -> PyObjectRef {
        // `pyframe.py peekvalue_maybe_none`: the same two checks as
        // [`Self::pop`].
        if self.valuestackdepth <= depth {
            report_stack_underflow(self);
        }
        // Hoisted for the reason given on [`Self::peek`].
        let index = self.valuestackdepth - 1 - depth;
        self.assert_stack_index(index);
        locals_w!(self)[index]
    }

    /// Null the locals_cells_stack slots at and above `depth`, the
    /// fresh-frame parity clear (`write_from_resume_data_partial` does not
    /// trim).  Used after a resume-time `valuestackdepth` correction so a GC
    /// scan before the next push does not observe a stale operand pointer
    /// above the live depth.
    pub fn clear_stack_above(&mut self, depth: usize) {
        let arr = locals_w_mut!(self).as_mut_slice();
        // `valuestackdepth` is an absolute index. A resume or residual can
        // publish a depth below `stack_base()`; trimming from there would
        // NULL live locals/cells. Operand slots start at the base.
        let start = if self.pycode.is_null() {
            depth
        } else {
            depth.max(self.stack_base())
        };
        for slot in arr.iter_mut().skip(start) {
            *slot = PY_NULL;
        }
    }

    /// PyPy-compatible stack operation aliases.
    #[inline]
    pub fn pushvalue(&mut self, value: PyObjectRef) {
        self.push(value)
    }

    /// pyframe.py pushvalue_none
    #[inline]
    pub fn pushvalue_none(&mut self) {
        let depth = self.valuestackdepth;
        debug_assert!(locals_w!(self)[depth].is_null());
        self.valuestackdepth = depth + 1;
    }

    /// PyPy-compatible stack index guard.
    #[inline]
    pub fn assert_stack_index(&self, index: usize) {
        debug_assert!(self._check_stack_index(index));
    }

    /// PyPy-compatible stack index validator.
    ///
    /// Asserts both lower and upper bounds: a valid stack write goes to
    /// `stack_base() <= index < locals_cells_stack_w.len()`. Pyre's
    /// `PyObjectArray` is allocated with `nlocals + ncells +
    /// max_stackdepth` slots (`alloc_frame_locals_array`), so writing at or past
    /// `array_len` overruns the heap buffer — catastrophic in release
    /// mode, where `PyObjectArray` indexing is unchecked. This guard
    /// converts the heap overrun into a debug-mode assertion failure
    /// that surfaces tracer/JIT vsd miscalculations at the source
    /// rather than as silent malloc corruption later.
    #[inline]
    pub fn _check_stack_index(&self, index: usize) -> bool {
        index >= self.stack_base() && index < locals_w!(self).len()
    }

    /// pyframe.py popvalue
    #[inline]
    pub fn popvalue(&mut self) -> PyObjectRef {
        let value = self.popvalue_maybe_none();
        assert!(!value.is_null());
        value
    }

    /// pyframe.py popvalue_maybe_none
    #[inline]
    pub fn popvalue_maybe_none(&mut self) -> PyObjectRef {
        let depth = self.valuestackdepth - 1;
        self.assert_stack_index(depth);
        let w_object = locals_w!(self)[depth];
        self.set_locals_w(depth, PY_NULL);
        self.valuestackdepth = depth;
        w_object
    }

    /// PyPy `PyFrame._new_popvalues` factory.
    #[inline]
    pub fn _new_popvalues() -> fn(&mut Self, usize) -> Vec<PyObjectRef> {
        Self::popvalues
    }

    /// PyPy-compatible pop-values helper.
    /// `pyframe.py popvalues` is `@jit.unroll_safe`. The return is a
    /// `Vec` (two residual words); until that ABI exists, looking
    /// inside residualizes a one-slot pointer and the length is
    /// garbage. Keep the loop residual.
    #[inline]
    pub fn popvalues(&mut self, n: usize) -> Vec<PyObjectRef> {
        let mut out = vec![PY_NULL; n];
        let mut idx = n;
        while idx > 0 {
            idx -= 1;
            out[idx] = self.popvalue();
        }
        out
    }

    /// PyPy-compatible `popvalues_mutable`.
    /// Same residual-ABI constraint as `popvalues`.
    #[inline]
    pub fn popvalues_mutable(&mut self, n: usize) -> Vec<PyObjectRef> {
        self.popvalues(n)
    }

    /// pyframe.py peekvalues. Same residual-ABI constraint as `popvalues`.
    #[inline]
    pub fn peekvalues(&self, n: usize) -> Vec<PyObjectRef> {
        let base = self.valuestackdepth - n;
        // Reads cover `[base, valuestackdepth)`; the highest index is
        // `valuestackdepth - 1 < len`, so only the lower `base` bound needs
        // guarding. When `n == 0` nothing is read and `base ==
        // valuestackdepth` may equal `len` at peak depth, so skip the
        // upper-bound `assert_stack_index` for the empty peek.
        if n > 0 {
            self.assert_stack_index(base);
        } else {
            debug_assert!(base >= self.stack_base());
        }
        let mut values_w = vec![PY_NULL; n];
        let mut idx = n;
        while idx > 0 {
            idx -= 1;
            let slot = base + idx;
            values_w[idx] = locals_w!(self)[slot];
        }
        values_w
    }

    /// pyframe.py dropvalues — `@jit.unroll_safe`, and `n` is promoted.
    #[inline]
    #[majit_macros::unroll_safe]
    pub fn dropvalues(&mut self, n: usize) {
        let n = majit_metainterp::jit::promote(n);
        let finaldepth = self.valuestackdepth - n;
        self.assert_stack_index(finaldepth);
        while self.valuestackdepth > finaldepth {
            let idx = self.valuestackdepth - 1;
            self.set_locals_w(idx, PY_NULL);
            self.valuestackdepth -= 1;
        }
    }

    /// PyPy-compatible `pushrevvalues`.
    /// `pyframe.py pushrevvalues` is `@jit.unroll_safe`.
    #[inline]
    #[majit_macros::unroll_safe]
    pub fn pushrevvalues(&mut self, _n: usize, values_w: &[PyObjectRef]) {
        let n = if _n == 0 { values_w.len() } else { _n };
        assert!(n <= values_w.len());
        let mut idx = n;
        while idx > 0 {
            idx -= 1;
            self.push(values_w[idx]);
        }
    }

    /// PyPy-compatible `dupvalues`.
    /// `pyframe.py dupvalues` is `@jit.unroll_safe` and peeks one
    /// slot at a time (`peekvalue(delta)`), not `peekvalues`.
    #[inline]
    #[majit_macros::unroll_safe]
    pub fn dupvalues(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        let delta = n - 1;
        let mut remaining = n;
        while remaining > 0 {
            remaining -= 1;
            let w_value = self.peekvalue(delta);
            self.push(w_value);
        }
    }

    /// PyPy-compatible `peekvalue()`.
    #[inline]
    pub fn peekvalue(&self, index_from_top: usize) -> PyObjectRef {
        self.peek_at(index_from_top)
    }

    /// PyPy-compatible `peekvalue_maybe_none()`.
    #[inline]
    pub fn peekvalue_maybe_none(&self, index_from_top: usize) -> PyObjectRef {
        let index = self
            .valuestackdepth
            .checked_sub(index_from_top + 1)
            .unwrap_or(usize::MAX);
        if index == usize::MAX || index < self.stack_base() {
            return PY_NULL;
        }
        locals_w!(self)[index]
    }

    /// PyPy-compatible `settopvalue()`.
    #[inline]
    pub fn settopvalue(&mut self, value: PyObjectRef, index_from_top: usize) {
        let depth = self.valuestackdepth;
        let index = depth.saturating_sub(index_from_top + 1);
        self.assert_stack_index(index);
        assert!(index < depth);
        self.set_locals_w(index, value);
    }

    /// `pyframe.py dropvaluesuntil` — `@jit.unroll_safe`, and
    /// `finaldepth` is promoted so the drop loop unrolls at the
    /// handler's known stack depth.
    #[inline]
    #[majit_macros::unroll_safe]
    pub fn dropvaluesuntil(&mut self, finaldepth: usize) {
        let finaldepth = majit_metainterp::jit::promote(finaldepth);
        self.assert_stack_index(finaldepth);
        while self.valuestackdepth > finaldepth {
            let idx = self.valuestackdepth - 1;
            self.set_locals_w(idx, PY_NULL);
            self.valuestackdepth -= 1;
        }
    }

    /// PyPy-compatible exception-info unwind helper.
    #[inline]
    pub fn _exc_info_unroll(&self, _for_hidden: bool) -> PyObjectRef {
        let _ = _for_hidden;
        pyre_object::w_none()
    }

    /// PyPy-compatible unexpected-exception converter.
    #[inline]
    pub fn _convert_unexpected_exception(&self, _e: PyObjectRef) -> PyObjectRef {
        let _ = _e;
        pyre_object::w_none()
    }

    /// PyPy-compatible pickle state helper.
    #[inline]
    pub fn _reduce_state(&self) -> PyObjectRef {
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(pyre_object::w_none());
        fields.push(pyre_object::w_none());
        fields.push(pyre_object::w_none());
        fields.push(pyre_object::w_int_new(self.last_instr as i64));
        fields.push(pyre_object::w_int_new(self.valuestackdepth as i64));
        pyre_object::w_tuple_new(fields.take())
    }

    /// PyPy-compatible `descr__reduce__`.
    #[inline]
    pub fn descr__reduce__(&self) -> PyObjectRef {
        // `_reduce_state` opens its own set; close it before this one
        // pins the returned nursery tuple.
        let state = self._reduce_state();
        let _roots = pyre_object::gc_roots::push_roots();
        let state_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(state);
        pyre_object::w_tuple_new(vec![
            pyre_object::w_none(),
            pyre_object::w_none(),
            pyre_object::gc_roots::shadow_stack_get(state_slot),
        ])
    }

    /// PyPy-compatible `descr__setstate__`.
    #[inline]
    pub fn descr__setstate__(&mut self, _state: PyObjectRef) {
        let _ = _state;
    }

    /// PyPy-compatible execution entrypoint.
    #[inline]
    pub fn run(&mut self) -> crate::PyResult {
        if self._is_generator_or_coroutine() {
            self.initialize_as_generator()
        } else {
            self.execute_frame(None, None)
        }
    }

    /// `run`, but dispatching the non-generator body through the registered
    /// eval function (`call::get_eval_fn`) instead of the plain interpreter.
    ///
    /// `interp_jit.py dispatch` applies the jitdriver to every frame uniformly,
    /// so module / class / exec'd code reaches `jit_merge_point` exactly like
    /// a called function does.  `run` hardcodes the plain interpreter, which
    /// keeps these entry frames off the portal; this variant restores the
    /// uniform routing for the run-sites that want it (exec / eval / import /
    /// class body), matching how `call_user_function` reaches the portal via
    /// `get_eval_fn`.
    ///
    /// `settrace` is honored without forcing plain eval: installing a tracefunc
    /// does not set `FORCE_PLAIN_EVAL` (that flag is a blackhole / `force_fn`
    /// re-entry guard, unrelated to tracing).  The JIT eval override
    /// (`eval_with_jit`) instead reads `ec.w_tracefunc` inline each bytecode
    /// for line events and routes non-JIT-eligible frames through
    /// `execute_frame` (so `call_trace` / `return_trace` frame events still
    /// fire), exactly as a normal call reaches the portal via `get_eval_fn`;
    /// and when the portal declines a frame it falls back to `execute_frame`
    /// (plain), so this never re-enters the portal.
    #[inline]
    pub fn run_with_jit(&mut self) -> crate::PyResult {
        if self._is_generator_or_coroutine() {
            self.initialize_as_generator()
        } else {
            crate::call::get_eval_fn()(self, None)
        }
    }

    /// pyframe.py resume_execute_frame, after any suspended delegate has
    /// either yielded or completed in `eval_frame_plain_with_resume`.
    #[inline]
    pub fn resume_execute_frame(
        &mut self,
        w_arg_or_err: PyObjectRef,
    ) -> Result<usize, crate::PyError> {
        if self.last_instr != -1 {
            self.pushvalue(w_arg_or_err);
            Ok(self.last_instr as usize + 1)
        } else {
            Ok(0)
        }
    }

    /// PyPy-compatible execution entrypoint with optional inbound values.
    #[inline]
    #[allow(unused_variables)]
    pub fn execute_frame(
        &mut self,
        w_inputvalue: Option<PyObjectRef>,
        operr: Option<crate::PyError>,
    ) -> crate::PyResult {
        // A frame can be executed by a thread that has not run pyre code
        // before — the launcher's interpreter thread, and the Rust tests that
        // drive a frame directly — so this is one of the entry points that
        // takes the GIL (rgil.py:186-193).
        crate::module::thread::ensure_runtime_thread();
        // pyframe.py `execute_frame.insert_stack_check_here = True`.
        // RPython's transform inserts the check at this graph entry, so
        // builtin/object-space calls remain usable while an existing frame is
        // handling RecursionError and every newly entered Python frame is
        // still guarded.
        let mut resume = crate::call::FrameResumeArgs {
            w_inputvalue,
            operr,
            throw_args: None,
        };
        // `w_arg_or_err` is a GC local from this entry through `call_trace`
        // and `resume_execute_frame` (`pyframe.py execute_frame`).
        // `stack_check`'s overflow arm allocates `RecursionError` while a
        // thrown `OperationError` (`error.py`) and the sent value are still
        // the argument. One shadow-stack bracket covers the whole span.
        let pins = crate::pin_resume_args!(&mut resume);
        crate::stack_check::drain_jit_pending_exception()?;
        crate::reload_resume_args!(pins.as_ref(), &mut resume);
        crate::stack_check::stack_check()?;
        crate::reload_resume_args!(pins.as_ref(), &mut resume);
        crate::eval::eval_frame_plain_with_resume(self, &mut resume, pins.as_ref())
    }

    /// Generator/coroutine execution entry carrying the original throw
    /// arguments for non-generator delegates.  This is the Rust representation
    /// of PyPy's `SApplicationException` payload passed through
    /// `PyFrame.execute_frame(w_arg_or_err)`.
    ///
    /// Dispatches through the registered eval function for the same reason
    /// `run_with_jit` does: `interp_jit.py dispatch` applies the jitdriver to
    /// every frame uniformly, and a resumed frame is a frame.  Routing this
    /// straight to the plain evaluator instead kept every generator,
    /// coroutine and async-generator body off the portal — 15.9% of the
    /// standard library's loop-bearing code objects, none of which the tracer
    /// could ever see.
    #[inline]
    pub fn execute_generator_frame(
        &mut self,
        w_inputvalue: Option<PyObjectRef>,
        operr: Option<crate::PyError>,
        throw_args: Option<([PyObjectRef; 3], usize)>,
    ) -> crate::PyResult {
        let mut resume = crate::call::FrameResumeArgs {
            w_inputvalue,
            operr,
            throw_args,
        };
        crate::call::get_eval_fn()(self, Some(&mut resume))
    }

    /// Plain-interpreter entry for a frame that may or may not be a resumption.
    ///
    /// This is what the default eval function runs, and what every JIT decline
    /// path falls back to.  A declining path that dropped `resume` and called
    /// `execute_frame(None, None)` instead would silently discard the value a
    /// `send()` delivered or the exception a `throw()` did, so the payload
    /// travels with the frame through the fallback.
    #[inline]
    pub fn execute_frame_plain(
        &mut self,
        resume: Option<&mut crate::call::FrameResumeArgs>,
    ) -> crate::PyResult {
        match resume {
            Some(resume) => self.execute_frame_resumed(resume),
            None => self.execute_frame(None, None),
        }
    }

    /// `execute_generator_frame`'s plain-interpreter half: the resume without
    /// the portal.  The stack check that used to sit in
    /// `execute_generator_frame` lives here — the JIT entry performs its own.
    #[inline]
    pub fn execute_frame_resumed(
        &mut self,
        resume: &mut crate::call::FrameResumeArgs,
    ) -> crate::PyResult {
        // Same `w_arg_or_err` bracket as `execute_frame`: the overflow arm
        // allocates before `call_trace` / `resume_execute_frame` consume the
        // sent value and the thrown `OperationError`.
        let pins = crate::pin_resume_args!(resume);
        crate::stack_check::drain_jit_pending_exception()?;
        crate::reload_resume_args!(pins.as_ref(), resume);
        crate::stack_check::stack_check()?;
        crate::reload_resume_args!(pins.as_ref(), resume);
        crate::eval::eval_frame_plain_with_resume(self, resume, pins.as_ref())
    }

    /// pyframe.py `hide(self): return self.pycode.hidden_applevel`.
    ///
    /// PyPy creates a `PyFrame` for every callable that has a Code
    /// object — including gateway builtins (`BuiltinCode`,
    /// `gateway.py hidden_applevel = True`) and the
    /// `app_main.py`-style internal frames
    /// (`pycompiler.compile(..., hidden_applevel=True)`).  The
    /// `hide()` flag lets `_trace` skip those internal frames so
    /// user-visible callbacks never see the gateway machinery.
    ///
    /// Pyre does not allocate a `PyFrame` for builtin calls — the
    /// `CallableKind::Builtin` arm (`classify_callable`, `runtime_ops.rs`)
    /// invokes the BuiltinCode function pointer directly, with no
    /// frame attached, so the trace path never observes builtin
    /// frames.  Hidden-applevel user frames (the `app_main.py`
    /// case) are not reachable in pyre yet either — pyre has no
    /// `pycompiler.compile(hidden_applevel=True)` call site.  The
    /// field still lives on PyCode, and this accessor mirrors
    /// PyPy's object-field read.
    #[inline]
    pub fn hide(&self) -> bool {
        unsafe { crate::w_code_hidden_applevel(self.pycode as PyObjectRef) }
    }

    /// `escaped` status bit (pyframe.py:80).
    pub const FLAG_ESCAPED: u8 = 0b01;
    /// `frame_finished_execution` status bit.
    pub const FLAG_FRAME_FINISHED: u8 = 0b10;
    /// A failed LOAD_ATTR is waiting for this frame's POP_EXCEPT cleanup.
    const FAILED_ATTR_AFTER_POP_EXCEPT: u8 = 4;
    /// The next dispatch boundary must collect the failed LOAD_ATTR receiver.
    const FAILED_ATTR_BEFORE_OPCODE: u8 = u8::MAX;

    /// pyframe.py:80 `escaped`.
    #[inline]
    pub fn escaped(&self) -> bool {
        self.flags & Self::FLAG_ESCAPED != 0
    }

    #[inline]
    pub fn set_escaped(&mut self, value: bool) {
        if value {
            self.flags |= Self::FLAG_ESCAPED;
        } else {
            self.flags &= !Self::FLAG_ESCAPED;
        }
    }

    /// `frame_finished_execution`.
    #[inline]
    pub fn frame_finished_execution(&self) -> bool {
        self.flags & Self::FLAG_FRAME_FINISHED != 0
    }

    #[inline]
    pub fn set_frame_finished_execution(&mut self, value: bool) {
        if value {
            self.flags |= Self::FLAG_FRAME_FINISHED;
        } else {
            self.flags &= !Self::FLAG_FRAME_FINISHED;
        }
    }

    /// Keep CPython-refcount compatibility cleanup on the exact red frame
    /// that owns the exception handler.  A process-global frame slot loses a
    /// request when another inlined/nested frame fails an attribute lookup.
    #[inline]
    pub fn defer_failed_attr_until_pop_except(&mut self) {
        self.failed_attr_cleanup = Self::FAILED_ATTR_AFTER_POP_EXCEPT;
    }

    #[inline]
    pub fn failed_attr_after_pop_except(&mut self) {
        if self.failed_attr_cleanup != Self::FAILED_ATTR_AFTER_POP_EXCEPT {
            return;
        }
        self.failed_attr_cleanup = 3;
    }

    #[inline]
    pub fn failed_attr_after_stack_pop(&mut self) {
        if !(1..=3).contains(&self.failed_attr_cleanup) {
            return;
        }
        self.failed_attr_cleanup -= 1;
        if self.failed_attr_cleanup == 0 {
            self.failed_attr_cleanup = Self::FAILED_ATTR_BEFORE_OPCODE;
        }
    }

    #[inline]
    pub fn take_failed_attr_before_opcode(&mut self) -> bool {
        // This state is normally zero for the lifetime of a hot loop.  Promote
        // it just like PyPy promotes frame dispatch state: a failed LOAD_ATTR
        // changes the word and invalidates the compiled path, so exception
        // cleanup resumes on the concrete per-frame red state.
        let cleanup = majit_metainterp::jit::promote(self.failed_attr_cleanup);
        if cleanup != Self::FAILED_ATTR_BEFORE_OPCODE {
            return false;
        }
        self.failed_attr_cleanup = 0;
        true
    }

    /// pyframe.py mark_as_escaped
    #[inline]
    pub fn mark_as_escaped(&mut self) {
        self.set_escaped(true);
    }

    /// pyframe.py `get_builtin` — returns `self.builtin` (the
    /// per-frame picked builtin Module, set at frame creation by
    /// `pick_builtin(w_globals)`).  Falls back to the EC's default
    /// builtin when the frame was constructed without globals.
    #[inline]
    pub fn get_builtin(&self) -> PyObjectRef {
        if !self.w_builtin.is_null() {
            return self.w_builtin;
        }
        let execution_context = crate::call::getexecutioncontext();
        if execution_context.is_null() {
            return pyre_object::PY_NULL;
        }
        unsafe { (*execution_context).get_builtin() }
    }

    /// `frame.f_backref()` — force the caller vref.  `f_backref` holds a
    /// `jit.virtual_ref`; calling it materializes the caller frame (identity
    /// at interp level, `force_virtual` once the JIT virtualized it).  Callers
    /// that need the raw unforced vref (e.g. `ExecutionContext::leave`'s
    /// `topframeref = frame.f_backref` restore, no parens) read the field
    /// directly instead.
    #[inline]
    pub fn get_f_back(&self) -> *mut PyFrame {
        crate::executioncontext::force_vref(self.f_backref)
    }

    /// pyframe.py `fget_f_builtins` — `self.get_builtin()
    /// .getdict(space)`, which `module.py` defines as `self.w_dict`.
    /// Pyre's `w_module_new` constructs `w_dict` at allocation time so
    /// every Module surfaces a stable, non-null identity here.
    #[inline]
    pub fn fget_f_builtins(&self) -> PyObjectRef {
        let w_builtin = self.get_builtin();
        if w_builtin.is_null() {
            return pyre_object::PY_NULL;
        }
        if unsafe { pyre_object::is_module(w_builtin) } {
            return unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
        }
        w_builtin
    }

    /// pyframe.py `fget_f_back` — the next non-hidden frame, i.e.
    /// `ExecutionContext.getnextframe_nohidden(self)`, skipping
    /// `hidden_applevel` gateway / bridge frames.  The plain
    /// `get_f_back()` accessor (used internally, including by the
    /// nohidden walker itself) returns the raw `f_backref` link.
    #[inline]
    pub fn fget_f_back(&self) -> *mut PyFrame {
        // pyframe.py `fget_f_back` → `get_f_back` →
        // `ExecutionContext.getnextframe_nohidden`, with no force of either
        // end.  Upstream needs none: `f_backref` is a `jit.virtual_ref`
        // (executioncontext.py), so the read at :80 `frame.f_backref()`
        // IS the force, and it is the foldable vref one rather than an
        // unconditional materialisation —
        // executioncontext.py `force_all_frames` says so outright
        // ("We get this effect simply by reading the f_back field of all
        // frames").  Forcing both ends concretely here instead escapes the
        // virtualizable during tracing and loses the loop.
        let this = self as *const PyFrame as *mut PyFrame;
        crate::executioncontext::ExecutionContext::getnextframe_nohidden(this)
    }

    /// pyframe.py fget_code → self.getcode().  Returns the `PyCode`
    /// wrapper object (`self.pycode`) itself, which is what `frame.f_code`
    /// yields to Python — not the inner `CodeObject`.
    #[inline]
    pub fn fget_f_code(&self) -> PyObjectRef {
        self.pycode as PyObjectRef
    }

    /// pyframe.py descr_repr — `<frame at 0x…, file '…', line …,
    /// code …>` via `getrepr(space, "frame", moreinfo)`.
    pub fn descr_repr(&self) -> rustpython_wtf8::Wtf8Buf {
        let filename = crate::gateway::fsdecode_filename_wtf8(&unsafe {
            crate::pycode::code_filename_bytes(self.pycode as pyre_object::PyObjectRef)
        });
        // [3.14-spec] `frame_repr` builds the filename with
        // `"<frame at %p, file %R, line %d, code %S>"`, so `str.__repr__`
        // both quotes it and escapes it — a Windows path comes out with its
        // backslashes doubled.  pyframe.py `descr_repr` writes
        // `", file '%s', line %s, code %s"` instead, whose hand-written quotes
        // pass the name through unescaped; the rendered text is observable, so
        // the 3.14 form governs and the literal `'` cannot stay (`%R` picks
        // the quote character too).  `code_repr` is a separate decision and
        // keeps its own quotes around a `%U`, which is why the two disagree on
        // the same path.
        let w_name =
            unsafe { crate::pycode::w_code_name_obj(self.pycode as pyre_object::PyObjectRef) };
        let mut repr = rustpython_wtf8::Wtf8Buf::new();
        repr.push_str("<frame at ");
        repr.push_str(&crate::display::repr_addr(self as *const PyFrame as usize));
        repr.push_str(", file ");
        repr.push_str(&crate::display::format_wtf8_repr(&filename));
        repr.push_str(", line ");
        repr.push_str(&self.get_last_lineno().to_string());
        repr.push_str(", code ");
        if !w_name.is_null() {
            repr.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(w_name) });
        }
        repr
    }

    /// `frame_clear` (`frame.clear()`): clear most references held by the
    /// frame.  Refuses on an executing (non-generator) frame or a running
    /// generator (`"cannot clear an executing frame"`) and on a generator
    /// frame suspended at a `yield` (`"cannot clear a suspended frame"`);
    /// a not-yet-started or already-exhausted generator is finalized.
    /// Otherwise clears `w_f_trace`, resets
    /// `w_locals` to a fresh dict, and replaces every local / cell / free
    /// var / stack slot (cells are rebound to fresh empty cells so a
    /// shared inner/outer cell is not mutated).
    pub fn descr_clear(&mut self) -> Result<(), crate::PyError> {
        // CPython 3.14 `frameobject.c:frame_clear_impl`: a frame owned by a
        // generator/coroutine is finalized and then returned from directly.
        // In particular, clearing a never-started coroutine frame must run
        // the unawaited-coroutine warning while the owner is still reachable;
        // merely marking it exhausted loses that finalizer event.
        if self._is_generator_or_coroutine() {
            let w_gen = self.get_generator();
            if !w_gen.is_null() {
                if unsafe { pyre_object::generator::w_generator_is_running(w_gen) } {
                    return Err(crate::PyError::runtime_error(
                        "cannot clear an executing frame",
                    ));
                }
                // A generator started but not exhausted is suspended at a
                // `yield`; clearing it would drop the live operand stack
                // out from under a later `resume`, so refuse.
                let suspended = unsafe {
                    pyre_object::generator::w_generator_is_started(w_gen)
                        && !pyre_object::generator::w_generator_is_exhausted(w_gen)
                };
                if suspended {
                    return Err(crate::PyError::runtime_error(
                        "cannot clear a suspended frame",
                    ));
                }
                // `_PyGen_Finalize` can execute Python and collect.  Keep both
                // owner and frame as explicit roots, then reload both before
                // severing the association; neither raw pointer may survive
                // the call unchanged across a nursery move.
                let frame_anchor = crate::eval::FrameAnchor::new(self);
                let roots = pyre_object::gc_roots::push_roots();
                let gen_slot = roots.base();
                let _ = roots.pin_root(w_gen);
                crate::baseobjspace::generator_finalize(roots.get(gen_slot))?;
                // CPython 3.14 `frame_clear_impl` finishes a never-started
                // generator after `_PyGen_Finalize`, then clears the owned
                // interpreter frame.  The suspended case was rejected above,
                // so this is the same completion path used by close() and
                // normal generator return, including severing `gi_frame`.
                unsafe {
                    crate::baseobjspace::generator_frame_is_finished(
                        roots.get(gen_slot),
                        &mut *frame_anchor.live(),
                        true,
                    )
                };
                crate::baseobjspace::generator_close_finalizer_boundary(
                    crate::baseobjspace::take_pending_close_finalizer(),
                );
                return Ok(());
            }
            // pyframe.py:815-820: a dead `f_generator_wref` simply skips the
            // generator finalizer and proceeds to clear the escaped frame.
            // It is not an executing non-generator frame.
            self.clear_references();
            return Ok(());
        }
        if !self.frame_finished_execution() {
            return Err(crate::PyError::runtime_error(
                "cannot clear an executing frame",
            ));
        }

        self.clear_references();
        Ok(())
    }

    /// Whether the graph behind an edge `_PyFrame_ClearExceptCode` is about to
    /// release contains an app-level finalizer.  A false answer proves that a
    /// prompt-finalization collection cannot run Python code.
    #[majit_macros::dont_look_inside]
    pub(crate) fn cleared_references_have_pending_finalizer(&self) -> bool {
        if !majit_gc::gc_has_pending_finalizers() {
            return false;
        }

        let mut released = Vec::with_capacity(locals_w!(self).len() + 4);
        for index in 0..locals_w!(self).len() {
            let value = locals_w!(self)[index];
            if !value.is_null() {
                released.push(majit_ir::GcRef(value as usize));
            }
        }
        if !self.w_yielding_from.is_null() {
            released.push(majit_ir::GcRef(self.w_yielding_from as usize));
        }
        // `f_backref` is the caller frame, still on the stack.  CPython's
        // refcount drop is only the generator frame's own locals;
        // walking the caller here treats live test-method objects as
        // released and forces a heap collect on every `close()`.
        if let Some(debug) = self.getdebug_data() {
            released.extend(
                [debug.w_locals, debug.w_extra_locals, debug.w_f_trace]
                    .into_iter()
                    .filter(|value| !value.is_null())
                    .map(|value| majit_ir::GcRef(value as usize)),
            );
        }
        majit_gc::subgraph_has_pending_finalizer(&released)
    }

    /// Clear the frame-owned reference region after its owner has already
    /// completed.  This is the internal `_PyFrame_ClearExceptCode` half used
    /// by generator completion; unlike public `frame.clear()`, it must not
    /// re-enter `_PyGen_Finalize` on the generator that is doing the clearing.
    ///
    /// Allocates a fresh locals dict and empty cells. The walk residualizes
    /// the call, so the address has to be a registered fnaddr.
    #[majit_macros::dont_look_inside]
    pub(crate) fn clear_references(&mut self) {
        if let Some(debug) = self.getdebug() {
            let had_locals = !debug.w_locals.is_null();
            // Allocate before remembering debugdata: a minor can happen here.
            let w_locals = had_locals.then(|| unsafe { pyre_object::w_dict_new() });
            let d = self.getorcreate_debug_data(-1);
            d.w_f_trace = pyre_object::PY_NULL;
            // CPython `frame_tp_clear` drops `f_extra_locals`; proxy-only
            // writes must not survive an explicit frame clear.
            d.w_extra_locals = pyre_object::PY_NULL;
            if let Some(w_locals) = w_locals {
                d.w_locals = w_locals;
            }
        }

        // Clear locals, cell/free vars, and the stack.  A cell slot is
        // rebound to a fresh empty cell (not mutated in place, since it may
        // still be shared by an inner/outer function).
        let len = locals_w!(self).len();
        for i in 0..len {
            let w_oldvalue = locals_w!(self)[i];
            let w_newvalue = if !w_oldvalue.is_null() && unsafe { pyre_object::is_cell(w_oldvalue) }
            {
                // pyframe.py `PyFrame.descr_clear` `Cell(None, w_oldvalue.family)` — the
                // replacement stands for the same binding, so it keeps the
                // family the cleared cell shared with its siblings.
                pyre_object::w_cell_new(pyre_object::PY_NULL, unsafe {
                    pyre_object::w_cell_family(w_oldvalue)
                })
            } else {
                pyre_object::PY_NULL
            };
            self.set_locals_w(i, w_newvalue);
        }
        self.valuestackdepth = 0;
    }

    /// pyframe.py fget_f_lasti → space.newint(self.last_instr)
    #[inline]
    pub fn fget_f_lasti(&self) -> isize {
        self.executing_instr()
    }

    /// The instruction this frame is suspended at, as Python names it.
    ///
    /// [`Self::last_instr`] is a dispatch coordinate, not an instruction
    /// index: the interpreter stores the unit it is about to run -- cache
    /// words included, pyre dispatches them as no-ops -- and every resume
    /// writer stores `resume_pc - 1` so that `next_instr()` recovers the
    /// resume point.  After a `CALL` those agree on the call's LAST CACHE
    /// WORD, which is not an instruction at all.
    ///
    /// `owning_instruction` maps a cache word back to the opcode that owns
    /// it, and is the identity on every value that already names one.  Only
    /// the Python-visible readers take it; `next_instr()` and the resume
    /// consumers keep reading the field raw, so the coordinate they round-trip
    /// through is untouched.
    #[inline]
    pub fn executing_instr(&self) -> isize {
        if self.last_instr < 0 {
            return self.last_instr;
        }
        crate::pyopcode::owning_instruction(&self.code().instructions, self.last_instr as usize)
            as isize
    }

    /// PyPy-compatible `fget_f_trace`.
    #[inline]
    pub fn fget_f_trace(&self) -> PyObjectRef {
        self.get_w_f_trace()
    }

    /// pyframe.py fset_f_trace
    #[inline]
    pub fn fset_f_trace(&mut self, w_trace: PyObjectRef) {
        if w_trace.is_null() || w_trace == pyre_object::w_none() {
            self.getorcreate_debug_data(-1).w_f_trace = pyre_object::PY_NULL;
        } else {
            let lineno = self.get_last_lineno();
            let d = self.getorcreate_debug_data(-1);
            d.w_f_trace = w_trace;
            d.f_lineno = lineno;
        }
    }

    /// pyframe.py fdel_f_trace
    #[inline]
    pub fn fdel_f_trace(&mut self) {
        self.getorcreate_debug_data(-1).w_f_trace = pyre_object::PY_NULL;
    }

    /// pyframe.py get_f_trace_lines
    #[inline]
    pub fn get_f_trace_lines(&self) -> bool {
        // `pyframe.py get_f_trace_lines`: None → True. Avoid `is_none_or`
        // so the walker does not residual a synthetic closure ctor.
        match self.getdebug_data() {
            None => true,
            Some(d) => d.f_trace_lines,
        }
    }

    /// pyframe.py get_f_trace_opcodes
    #[inline]
    pub fn get_f_trace_opcodes(&self) -> bool {
        // `pyframe.py get_f_trace_opcodes`: None → False. Avoid `is_some_and`
        // so the walker does not residual a synthetic closure ctor.
        match self.getdebug_data() {
            None => false,
            Some(d) => d.f_trace_opcodes,
        }
    }

    /// pyframe.py fget_f_trace_lines
    #[inline]
    pub fn fget_f_trace_lines(&self) -> bool {
        self.get_f_trace_lines()
    }

    /// pyframe.py fset_f_trace_lines
    #[inline]
    pub fn fset_f_trace_lines(&mut self, value: bool) {
        self.getorcreate_debug_data(-1).f_trace_lines = value;
    }

    /// pyframe.py fget_f_trace_opcodes
    #[inline]
    pub fn fget_f_trace_opcodes(&self) -> bool {
        self.get_f_trace_opcodes()
    }

    /// pyframe.py fset_f_trace_opcodes
    #[inline]
    pub fn fset_f_trace_opcodes(&mut self, value: bool) {
        self.getorcreate_debug_data(-1).f_trace_opcodes = value;
    }

    /// PyPy-compatible `fget_f_exc_type`.
    #[inline]
    pub fn fget_f_exc_type(&self) -> PyObjectRef {
        pyre_object::PY_NULL
    }

    /// PyPy-compatible `fget_f_exc_value`.
    #[inline]
    pub fn fget_f_exc_value(&self) -> PyObjectRef {
        pyre_object::PY_NULL
    }

    /// PyPy-compatible `fget_f_exc_traceback`.
    #[inline]
    pub fn fget_f_exc_traceback(&self) -> PyObjectRef {
        pyre_object::PY_NULL
    }

    /// PyPy-compatible `fget_f_restricted`.
    #[inline]
    pub fn fget_f_restricted(&self) -> bool {
        false
    }

    /// pyframe.py get_last_lineno → pytraceback.offset2lineno(pycode, last_instr)
    ///
    /// `-1` is `PyFrame_GetLineNumber`'s no-line answer, which
    /// [`PyFrame::fget_f_lineno`] renders as `None`.  It reaches here for a
    /// replacement `co_linetable` that describes no range covering
    /// `last_instr` — including an empty one — and for the `NO_LOCATION`
    /// ranges a compiler-generated cleanup sits in.
    ///
    /// A frame that has not run an instruction resolves against its first
    /// `RESUME`, not against `PyCode_Addr2Line`'s `addrq < 0` arm.  Upstream
    /// has no "not started" offset to resolve: `_PyInterpreterFrame_LASTI` is
    /// `_co_firsttraceable` while the frame sits on that `RESUME`, so
    /// `frame_getlineno` answers the instruction's own line.  That differs from
    /// `co_firstlineno` for a module body, whose `RESUME` carries line **0**
    /// while `co_firstlineno` is 1 — and the gap is observable, because
    /// `_trace` seeds the last-traced line from here on the `call` event, so
    /// answering `co_firstlineno` made the `RESUME`'s own line look like a
    /// change and fired a `line` event upstream does not have.
    ///
    /// It differs from instruction **0** too, and only for a code object with a
    /// prologue: `COPY_FREE_VARS` and `MAKE_CELL` are stamped with no line at
    /// all, so a closure resolved at 0 answers `-1` and reports `f_lineno` as
    /// `None` where the `RESUME` two units later carries the `def` line.
    #[inline]
    pub fn get_last_lineno(&self) -> isize {
        self.get_lineno_at(self.last_instr)
    }

    /// [`Self::get_last_lineno`] for an instruction the caller names instead of
    /// the one the frame's own field records.
    ///
    /// `last_instr` is a virtualizable field, so a compiled loop keeps the
    /// executing coordinate in its own trace-time state and the frame's copy is
    /// only made current by a force.  A caller that already holds that
    /// coordinate answers this read without one.
    #[inline]
    pub fn get_lineno_at(&self, last_instr: isize) -> isize {
        let stopat = if last_instr < 0 {
            crate::pycode::first_traceable_index(self.code()) as isize
        } else {
            last_instr
        };
        unsafe { offset2lineno(self.pycode as PyObjectRef, stopat) }
    }

    /// pycode.py `_get_lineno_for_pc_tracing(frame.last_instr)` — the line a
    /// trace function attributes to the instruction the frame is executing, or
    /// `-1` when that instruction cannot start a line.
    ///
    /// Upstream decodes the line table into one entry per code unit and leaves
    /// `-1` in every entry the decode never reached, so `-1` is the "no line
    /// here" answer `run_trace_func` guards on twice.  pyre's table covers
    /// every instruction, so the sentinel comes from the opcode instead:
    /// [`crate::pycode::instruction_can_start_a_line`].
    ///
    /// Separate from [`Self::get_last_lineno`], which answers the *frame's*
    /// line for `f_lineno` and reports `co_firstlineno` for a frame that has
    /// not started.  That is the right answer for the getset — a not-yet-run
    /// generator's `f_lineno` is its `def` line — and the wrong one for a line
    /// event, which is why this is a second entry point rather than an edit to
    /// that one.
    /// `pycode.py _get_lineno_for_pc_tracing` is `@jit.elidable`.
    #[inline]
    #[majit_macros::elidable]
    pub fn get_lineno_for_pc_tracing(&self) -> isize {
        if self.last_instr < 0
            || !crate::pycode::instruction_can_start_a_line(self.code(), self.last_instr as usize)
        {
            return -1;
        }
        self.get_last_lineno()
    }

    /// pyframe.py fget_f_lineno — the line currently executing.
    ///
    /// ```c
    /// int lineno = PyFrame_GetLineNumber(f);
    /// if (lineno < 0) {
    ///     Py_RETURN_NONE;
    /// }
    /// return PyLong_FromLong(lineno);
    /// ```
    ///
    /// `frame_getlineno` reads the same number whether or not the frame is
    /// traced, so the `f_trace` test that used to substitute the code
    /// object's first line for a missing one is gone: it stood in for a
    /// resolver that answered `-1` only for a line table it could not decode,
    /// and the resolver now answers `-1` wherever `PyCode_Addr2Line` does.
    #[inline]
    pub fn fget_f_lineno(&self) -> PyObjectRef {
        self.f_lineno_at(self.last_instr)
    }

    /// [`Self::fget_f_lineno`] for an instruction the caller names, on the same
    /// terms as [`Self::get_lineno_at`].
    ///
    /// The whole getter body stays here rather than being split across the
    /// caller: the `-1` sentinel and the `None` it renders as are one
    /// decision, and a caller reproducing part of it would answer a
    /// differently-shaped read.
    #[inline]
    pub fn f_lineno_at(&self, last_instr: isize) -> PyObjectRef {
        let lineno = self.get_lineno_at(last_instr);
        if lineno < 0 {
            return pyre_object::w_none();
        }
        pyre_object::w_int_new(lineno as i64)
    }

    /// `frameobject.c frame_lineno_set` — set the line the frame will
    /// resume at, validating the jump against [`mark_stacks`].
    ///
    /// Only a trace function may jump (`get_w_f_trace()` non-null); the
    /// jump target must land on a real source line, must not enter an
    /// exception handler / for-loop body / re-raise block, and the source
    /// and target abstract operand stacks must be compatible.  On success
    /// the operand stack is unwound to the target depth (closing dropped
    /// values, restoring `exc_info` for popped `Except` slots) and
    /// `last_instr` is repointed.
    ///
    /// Deviation: `frame_lineno_set` also binds `None` to any local the
    /// compiler proved live-but-unbound at the target (the `PyStackRef_None`
    /// pass over `co_localspluskinds`).  pyre keeps unbound locals as
    /// `PY_NULL` guarded by `LOAD_FAST_CHECK`, so that pass is omitted; the
    /// only observable difference is an `UnboundLocalError` where the
    /// jumped-to code reads such a local before assigning it, versus
    /// silently seeing `None`.
    pub fn fset_f_lineno(&mut self, new_f_lineno: isize) -> Result<(), crate::PyError> {
        // frame_lineno_set: you can only jump from within a trace
        // function, not via `_getframe` / similar hackery.
        if self.get_w_f_trace().is_null() {
            return Err(crate::PyError::value_error(
                "f_lineno can only be set by a trace function.",
            ));
        }
        // A newly-entered frame (call event) has no dispatched
        // instruction yet.
        if self.last_instr == -1 {
            return Err(crate::PyError::value_error(
                "can't jump from the 'call' trace event of a new frame",
            ));
        }
        // Jumps are allowed from a `line` trace event, and — outside a
        // line event — only when the frame is suspended at `YIELD_VALUE`
        // (`frame_lineno_set_impl` lists `PY_MONITORING_EVENT_PY_YIELD`
        // in the same allowed group as line/jump events).  The pending
        // yield value is accounted for by the `is_suspended` unwind
        // below.  pyframe.py:685-689:
        //     if not d.is_in_line_tracing:
        //         if ord(code[self.last_instr]) != YIELD_VALUE:
        //             raise "can only jump from a 'line' trace event"
        if !self.getdebug().is_some_and(|d| d.is_in_line_tracing) {
            let at_yield = matches!(
                crate::pyopcode::decode_instruction_at(self.code(), self.last_instr as usize),
                Some((crate::bytecode::Instruction::YieldValue { .. }, _))
            );
            if !at_yield {
                return Err(crate::PyError::value_error(
                    "can only jump from a 'line' trace event",
                ));
            }
        }
        // `frame_is_suspended` — a generator/coroutine frame that has
        // started, is not currently running, and is not exhausted has a
        // pending `yield` value on the stack that the resume will pop, so
        // the unwind must account for it.
        let is_suspended = if self._is_generator_or_coroutine() {
            let w_gen = self.get_generator();
            !w_gen.is_null()
                && unsafe {
                    pyre_object::generator::w_generator_is_started(w_gen)
                        && !pyre_object::generator::w_generator_is_running(w_gen)
                        && !pyre_object::generator::w_generator_is_exhausted(w_gen)
                }
        } else {
            false
        };

        let code = self.code();
        let len = code.instructions.len();
        // `if (new_lineno < f->f_frame->f_code->co_firstlineno)` — the bound
        // is the code object's own first line, which app-level can set to
        // zero or below, not a `OneIndexed` floor of 1.
        let first_line =
            unsafe { crate::pycode::w_code_firstlineno_raw(self.pycode as PyObjectRef) };

        let mut new_lineno = new_f_lineno as i32;
        if new_lineno < first_line {
            return Err(crate::PyError::value_error(format!(
                "line {new_lineno} comes before the current code block"
            )));
        }

        let lines = mark_lines(code, first_line, len);
        new_lineno = mark_first_line_not_before(&lines, new_lineno);
        if new_lineno < 0 {
            return Err(crate::PyError::value_error(format!(
                "line {new_f_lineno} comes after the current code block"
            )));
        }

        let stacks = mark_stacks(code, len);
        // `mark_stacks` steps `i + cache_entries + 1`, so every cache index
        // keeps `MARK_UNINITIALIZED` and a frame suspended inside a call's
        // cache region would read "can't jump from unreachable code".
        let last_instr = self.executing_instr() as usize;
        let start_stack = *stacks.get(last_instr).unwrap_or(&MARK_UNINITIALIZED);

        let mut best_stack = MARK_OVERFLOWED;
        let mut best_addr: isize = -1;
        let mut err: i32 = -1;
        let mut msg: String = "cannot find bytecode for specified line".to_string();
        for i in 0..len {
            if lines[i] != new_lineno {
                continue;
            }
            let target_stack = stacks[i];
            if mark_compatible_stack(start_stack, target_stack) {
                err = 0;
                if target_stack > best_stack {
                    best_stack = target_stack;
                    best_addr = i as isize;
                }
            } else if err < 0 {
                if start_stack == MARK_OVERFLOWED {
                    msg = "stack too deep to analyze".to_string();
                } else if start_stack == MARK_UNINITIALIZED {
                    msg = "can't jump from unreachable code".to_string();
                } else {
                    msg = mark_explain_incompatible_stack(target_stack).to_string();
                    err = 1;
                }
            }
        }
        if err != 0 {
            return Err(crate::PyError::value_error(msg));
        }

        // Unwind the operand stack from `start_stack` down to
        // `best_stack`, closing dropped values.  A dropped `Except` slot
        // restores the previous `exc_info` (pyre keeps the active
        // exception in the TLS current-exception slot, saved beneath the
        // handler on the value stack — see `push_exc_info`).
        let mut cur_stack = start_stack;
        if is_suspended {
            // Account for the value popped by yield.
            cur_stack = mark_pop_value(cur_stack);
        }
        while cur_stack > best_stack {
            if mark_top_of_stack(cur_stack) == StackKind::Except as i64 {
                // The popped value is the saved previous exception; make
                // it current again.
                crate::eval::restore_exc_info(self.popvalue());
            } else {
                // `PyStackRef_XCLOSE(_PyFrame_StackPop(f->f_frame))` — the
                // `X` is load-bearing.  A slot below the jump target can be
                // unbound, which is what a jump out of the body of a `with`
                // pops: the block's `__exit__` slot is NULL until the
                // cleanup runs.
                self.popvalue_maybe_none();
            }
            cur_stack = mark_pop_value(cur_stack);
        }

        self.getorcreate_debug_data(-1).f_lineno = new_lineno as isize;
        self.last_instr = best_addr;
        Ok(())
    }

    /// PyPy-compatible `setfastscope`.
    #[inline]
    pub fn setfastscope(&mut self, scope_w: &[PyObjectRef]) {
        assert!(scope_w.len() <= self.nlocals());
        for (index, value) in scope_w.iter().copied().enumerate() {
            self.set_locals_w(index, value);
        }
        // In this port, cell initialization is performed as part of scope load.
        self.init_cells();
    }

    /// pyframe.py locals2fast(skip_free_vars=False) — reflect the
    /// locals mapping back into the fastlocals.  Reads each varname / cellvar
    /// / freevar from the mapping via `space.finditem_str` (KeyError →
    /// missing); a frame with no locals bound has nothing to copy.
    /// `pyframe.py locals2fast` is `@jit.unroll_safe`.
    #[majit_macros::unroll_safe]
    pub fn locals2fast(&mut self, skip_free_vars: bool) -> Result<(), crate::PyError> {
        // `pyframe.py:589` binds `w_locals` once, ahead of both loops, and
        // every `space.finditem_str` below reads that one binding.  It
        // allocates a key per slot and dispatches a `__getitem__` that may run
        // Python, so it can collect, and the GC transform reloads both the
        // binding and `self` from their root slots after each such call.
        // Reproduce that rather than retaining either pre-move Rust pointer
        // between iterations, exactly as the `fast2locals` twin does.
        let frame_anchor = crate::eval::FrameAnchor::new(self);
        let w_locals = unsafe { (*frame_anchor.live()).get_w_locals() };
        if w_locals.is_null() {
            return Ok(());
        }
        let locals_roots = pyre_object::gc_roots::push_roots();
        let locals_slot = locals_roots.base();
        let _ = locals_roots.pin_root(w_locals);
        let code = unsafe { (*frame_anchor.live()).getcode() };
        let numlocals = code.varnames.len();

        for i in 0..numlocals {
            // FrameLocalsProxy writes do not target hidden comprehension
            // locals; they fall through to the underlying module/class
            // namespace instead (frameobject.c framelocalsproxy_getkeyindex).
            if hidden_local(code, i) {
                continue;
            }
            let name = &code.varnames[i];
            let w_value =
                finditem_str_object(locals_roots.get(locals_slot), name)?.unwrap_or(PY_NULL);
            let frame = unsafe { &mut *frame_anchor.live() };
            let slot = locals_w!(frame)[i];
            let is_cell_slot = i < code.localspluskinds.len()
                && code.localspluskinds[i] & crate::bytecode::CO_FAST_CELL != 0;
            if is_cell_slot && !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
                unsafe { pyre_object::w_cell_set(slot, w_value) };
            } else {
                frame.set_locals_w(i, w_value);
            }
        }

        // Same positional band as `fast2locals`: a filtered `Vec` would
        // residualize iterator adapters under `unroll_safe`.
        let npure = npure_cellvars(code);
        let include_freevars = code.flags.contains(CodeFlags::OPTIMIZED) && !skip_free_vars;
        let freevarnames_len = if include_freevars {
            npure + code.freevars.len()
        } else {
            npure
        };
        for i in 0..freevarnames_len {
            let name: &str = if i < npure {
                code.cellvars[nth_pure_cellvar_index(code, i)].as_ref()
            } else {
                code.freevars[i - npure].as_ref()
            };
            let idx = numlocals + i;
            let frame = unsafe { &*frame_anchor.live() };
            if idx < locals_w!(frame).len() {
                let w_value =
                    finditem_str_object(locals_roots.get(locals_slot), name)?.unwrap_or(PY_NULL);
                let frame = unsafe { &mut *frame_anchor.live() };
                let slot = locals_w!(frame)[idx];
                if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
                    unsafe { pyre_object::w_cell_set(slot, w_value) };
                } else {
                    frame.set_locals_w(idx, w_value);
                }
            }
        }
        Ok(())
    }

    /// pyframe.py init_cells.
    ///
    /// In the CPython 3.11+ unified slot layout that pyre adopted via
    /// `npure_cellvars`, every cellvar that also appears in varnames
    /// shares its varname slot — and the bytecode walker emits
    /// `MAKE_CELL i` to wrap that local into a cell at function entry
    /// (see `RustPython` 0.5 compiler `bytecode/instruction.rs:233`).
    /// That removes the need for the legacy `_compute_args_as_cellvars`
    /// arg-to-cell copy: there is no separate cellvar slot to copy
    /// into.  Pure cellvars (cellvars not in varnames) are pre-
    /// initialised as empty cells in `new_for_call_with_closure`.  This
    /// function therefore has nothing to do; keep the entry point so
    /// other call sites that already invoke it remain valid.
    #[inline]
    pub fn init_cells(&mut self) {}

    /// pyframe.py fast2locals — copy the fastlocals into the locals
    /// mapping via `space.setitem_str` (`pyframe.py`), using `space.delitem`
    /// for missing slots (`pyframe.py`; `delitem`'s `KeyError` is
    /// silently dropped).  A function frame with no locals bound yet lazily
    /// allocates a fresh dict (pyframe.py `self.space.newdict(instance=True)`)
    /// and caches it, so `locals() is locals()` holds.  Errors propagate.
    ///
    /// `@jit.unroll_safe` (`pyframe.py`) cancels `contains_loop` in the
    /// policy (`codewriter/policy.rs look_inside_graph`), so the slot loop
    /// below does not keep the codewriter out.  Without it the whole function
    /// is one residual call, and the `f_locals` read behind it forces the
    /// virtualizable for the length of that call.
    #[majit_macros::unroll_safe]
    pub fn fast2locals(&mut self) -> Result<(), crate::PyError> {
        // `space.setitem_str` / `space.delitem` allocate one key per slot and
        // can collect.  RPython's GC transform keeps both the frame and its
        // `w_locals` mapping live across the whole loop; do the same rather
        // than retaining either pre-move Rust pointer between iterations.
        let frame_anchor = crate::eval::FrameAnchor::new(self);
        let w_locals = unsafe { (&mut *frame_anchor.live()).get_or_create_w_locals() };
        let locals_roots = pyre_object::gc_roots::push_roots();
        let locals_slot = locals_roots.base();
        let _ = locals_roots.pin_root(w_locals);
        let code = unsafe { (*frame_anchor.live()).getcode() };
        let varnames = &code.varnames;
        let numlocals = varnames.len();

        for i in 0..numlocals {
            // Non-optimized frames expose a bound hidden slot through a
            // FrameLocalsProxy snapshot instead.  Copying it into the actual
            // module/class namespace would leak the comprehension variable.
            if hidden_local(code, i) && !code.flags.contains(CodeFlags::OPTIMIZED) {
                continue;
            }
            let name = &varnames[i];
            let frame = unsafe { &*frame_anchor.live() };
            let slot = locals_w!(frame)[i];
            let is_cell_slot = i < code.localspluskinds.len()
                && code.localspluskinds[i] & crate::bytecode::CO_FAST_CELL != 0;
            let w_value =
                if is_cell_slot && !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
                    unsafe { pyre_object::w_cell_get(slot) }
                } else {
                    slot
                };
            if !w_value.is_null() {
                setitem_str_object(locals_roots.get(locals_slot), name, w_value)?;
            } else {
                delitem_str_object(locals_roots.get(locals_slot), name)?;
            }
        }

        // `_PyFrame_GetLocals`: a frame that is not optimized and has no hidden
        // locals hands back its mapping as it stands.  Its namespace is
        // authoritative -- every binding that belongs there arrived by
        // `STORE_NAME` -- so a cell holds nothing the mapping is missing, and
        // copying one in publishes a name the body never bound.  A class body
        // carries `__classdict__` as a cell whose value is the namespace
        // itself, and writing that back leaves the namespace holding itself.
        if !code.flags.contains(CodeFlags::OPTIMIZED) {
            return Ok(());
        }
        // `freevarnames = co_cellvars` and, under `CO_OPTIMIZED`,
        // `+ co_freevars` -- one loop over the concatenation.  The
        // concatenation is not materialized here: the pure-cellvar band is
        // addressed positionally through [`nth_pure_cellvar_index`], the same
        // overlap test [`npure_cellvars`] counts with.  A cellvar that also
        // names a varname shares that varname's slot in the unified layout and
        // was written by the loop above, so it holds no band position.
        //
        // Positional addressing rather than a filtered `Vec` because this
        // graph is `unroll_safe`: the walker descends into the body, and a
        // collected `Vec` puts the closure environment, the iterator adapters
        // and the allocation on the descent's blocker frontier as calls with
        // no lowering -- `fast2locals::closure`, `Filter::collect`,
        // `Iter::filter`, `Range::any`, `Box::new`.  None of them has an
        // upstream counterpart; the upstream loop indexes two name arrays.
        let npure = npure_cellvars(code);
        let freevarnames_len = npure + code.freevars.len();
        for i in 0..freevarnames_len {
            let name: &str = if i < npure {
                code.cellvars[nth_pure_cellvar_index(code, i)].as_ref()
            } else {
                code.freevars[i - npure].as_ref()
            };
            let idx = numlocals + i;
            let frame = unsafe { &*frame_anchor.live() };
            if idx < locals_w!(frame).len() {
                let slot = locals_w!(frame)[idx];
                let w_value = if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
                    unsafe { pyre_object::w_cell_get(slot) }
                } else {
                    slot
                };
                if !w_value.is_null() {
                    setitem_str_object(locals_roots.get(locals_slot), name, w_value)?;
                } else {
                    // A function frame owns its cellvars in the cell, so an
                    // empty cell means the local is unbound and its entry goes.
                    delitem_str_object(locals_roots.get(locals_slot), name)?;
                }
            }
        }
        Ok(())
    }

    /// Whether a modelled `fast2locals` can reproduce this code object's
    /// locals mapping from the fastlocals alone.
    ///
    /// The cell / freevar half of [`PyFrame::fast2locals`] (pyframe.py)
    /// is one extra read per slot — `w_cell_get`, i.e. `Cell.contents` — over a
    /// slot the caller already holds, so a model that can read that field
    /// reproduces it, and both `locals()` expansion arms do.
    ///
    /// What stays refused: a non-OPTIMIZED (module / class / exec) frame,
    /// which hands back its LIVE namespace rather than the independent copy
    /// [`PyFrame::frame_locals_snapshot`] builds for a function frame, and
    /// `CO_FAST_HIDDEN` slots, which `fast2locals` skips only on non-OPTIMIZED
    /// frames and whose comprehension names must not leak into one.
    pub fn code_locals_are_modelled_fastlocals(code: &CodeObject) -> bool {
        code.flags.contains(CodeFlags::OPTIMIZED)
            && !(0..code.varnames.len()).any(|i| hidden_local(code, i))
    }

    /// The `fast2locals` names for the slots at and above
    /// `code.varnames.len()`, in slot order: the PURE cellvars (those not
    /// sharing a varname slot) followed by the freevars, exactly as
    /// [`PyFrame::fast2locals`] walks them.
    ///
    /// A cellvar that also appears in `varnames` shares that slot in the
    /// 3.11+ unified layout, so it is named by `varnames` and must not appear
    /// here a second time.
    ///
    /// An iterator rather than a list because the callers want opposite ends
    /// of it: the walker takes the count once while modelling the frame, and
    /// [`jit_locals_dict_setitem_cell`] takes one name per bound slot, on a
    /// compiled path.
    pub fn cell_slot_names(code: &CodeObject) -> impl Iterator<Item = &str> {
        let varnames = &code.varnames;
        code.cellvars
            .iter()
            .map(AsRef::as_ref)
            .filter(move |cs: &&str| !varnames.iter().any(|v| <_ as AsRef<str>>::as_ref(v) == *cs))
            .chain(code.freevars.iter().map(AsRef::as_ref))
    }

    /// pyframe.py `make_arguments` — build Arguments from the value
    /// stack. `methodcall` and `w_function` are stored on the resulting
    /// Arguments for diagnostic use (better error messages on argument
    /// mismatch); pyre's call path currently passes only the positional
    /// vector, so the parameters are recorded as part of the parity contract
    /// without downstream consumers yet.
    #[inline]
    pub fn make_arguments(
        &self,
        nargs: usize,
        _methodcall: bool,
        _w_function: PyObjectRef,
    ) -> Vec<PyObjectRef> {
        self.peekvalues(nargs)
    }

    /// PyPy-compatible argument list builder.
    #[inline]
    #[allow(unused_variables)]
    pub fn argument_factory(
        &self,
        _arguments: &[PyObjectRef],
        _keywords: &[PyObjectRef],
        _keywords_w: &[PyObjectRef],
        _w_star: PyObjectRef,
        _w_starstar: PyObjectRef,
        _methodcall: bool,
    ) -> Vec<PyObjectRef> {
        let mut args = Vec::new();
        args.extend_from_slice(_arguments);
        args.extend_from_slice(_keywords);
        args.extend_from_slice(_keywords_w);
        if !_w_star.is_null() {
            args.push(_w_star);
        }
        if !_w_starstar.is_null() {
            args.push(_w_starstar);
        }
        args
    }

    /// Create a new frame for a function call.
    ///
    /// The `code` pointer is shared from the function object -- no clone.
    /// `closure` is a tuple of cell objects from the enclosing scope,
    /// or PY_NULL if the function has no free variables.
    pub fn new_for_call(
        code: *const (),
        args: &[PyObjectRef],
        execution_context: *const PyExecutionContext,
    ) -> Self {
        Self::new_for_call_with_closure(code, args, execution_context, PY_NULL)
    }

    /// JIT warm-entry frame builder. The only callers are the `call_jit.rs`
    /// residual-frame helpers (`create_callee_frame_impl*`,
    /// `jit_create_self_recursive_callee_frame_1`, `jit_force_callee_frame`),
    /// which run from compiled code through an `i64`-pointer ABI with no error
    /// channel, so this stays infallible. PRE-EXISTING-ADAPTATION: a compiled
    /// trace sets the virtualizable frame fields directly (pyjitpl frame
    /// setup) instead of re-running `__init__` as a residual call, so there is
    /// no fallible `createframe` residual to mirror and `pick_builtin` cannot
    /// newly raise here — `w_globals` carries an `__builtins__` already validated
    /// when the interpreter first built the frame.
    pub fn new_for_call_with_globals_obj(
        code: *const (),
        args: &[PyObjectRef],
        w_globals: PyObjectRef,
        execution_context: *const PyExecutionContext,
    ) -> Self {
        Self::new_for_call_with_closure_and_globals_obj(
            code,
            args,
            w_globals,
            execution_context,
            PY_NULL,
            FrameLocalsArrayAllocation::StdAlloc,
        )
    }

    /// Create a new frame for a function call with a closure.
    pub fn new_for_call_with_closure(
        code: *const (),
        args: &[PyObjectRef],
        execution_context: *const PyExecutionContext,
        closure: PyObjectRef,
    ) -> Self {
        let w_globals = PY_NULL;
        Self::new_for_call_with_closure_and_globals_obj(
            code,
            args,
            w_globals,
            execution_context,
            closure,
            FrameLocalsArrayAllocation::StdAlloc,
        )
    }

    /// pyframe.py `Frame.__init__` resolves `self.builtin` through
    /// `pick_builtin(w_globals)`, which raises a non-KeyError
    /// `OperationError` straight out of `__init__`.  Fallible frame builder
    /// mirroring that path.
    pub fn try_new_for_call_with_closure_and_globals_obj(
        code: *const (),
        args: &[PyObjectRef],
        w_globals: PyObjectRef,
        execution_context: *const PyExecutionContext,
        closure: PyObjectRef,
        allocation: FrameLocalsArrayAllocation,
    ) -> Result<Self, crate::PyError> {
        // The RPython shadowstack transform roots every live call argument
        // around `pick_builtin`, which may allocate or execute a mapping
        // lookup.  A JIT residual call passes Rust copies of gcmap-backed
        // values here; after a collection only the gcmap/shadow slots are
        // forwarded, so never carry the original slice across that call.
        let _roots = pyre_object::gc_roots::push_roots();
        let mut live = Vec::with_capacity(3 + args.len());
        live.push(code as PyObjectRef);
        live.push(w_globals);
        live.push(closure);
        live.extend_from_slice(args);
        let root_base = _roots.pin_roots(&live);
        let w_builtin = crate::baseobjspace::frame_builtin_obj_checked(
            _roots.get(root_base + 1),
            execution_context,
        )?;
        let mut current_args: Vec<PyObjectRef> = Vec::with_capacity(args.len());
        for i in 0..args.len() {
            current_args.push(_roots.get(root_base + 3 + i));
        }
        Ok(Self::finish_for_call_with_globals_obj(
            _roots.get(root_base) as *const (),
            &current_args,
            _roots.get(root_base + 1),
            execution_context,
            _roots.get(root_base + 2),
            w_builtin,
            allocation,
        ))
    }

    /// PyPy `space.createframe(code, function.w_func_globals, function)`.
    /// The semantic globals carrier is the dict object.
    /// Infallible sibling of `try_new_for_call_with_closure_and_globals_obj`.
    /// All interpreter and tracer entry points use the `try_` variant; this
    /// one survives only for the JIT warm-entry path via
    /// `new_for_call_with_globals_obj` (see that method for why fallibility has
    /// no upstream counterpart there). `pick_builtin_obj` drops a non-KeyError
    /// `__builtins__` lookup, which is unreachable on that path.
    pub fn new_for_call_with_closure_and_globals_obj(
        code: *const (),
        args: &[PyObjectRef],
        w_globals: PyObjectRef,
        execution_context: *const PyExecutionContext,
        closure: PyObjectRef,
        allocation: FrameLocalsArrayAllocation,
    ) -> Self {
        let _roots = pyre_object::gc_roots::push_roots();
        let mut live = Vec::with_capacity(3 + args.len());
        live.push(code as PyObjectRef);
        live.push(w_globals);
        live.push(closure);
        live.extend_from_slice(args);
        let root_base = _roots.pin_roots(&live);
        let w_builtin =
            crate::baseobjspace::frame_builtin_obj(_roots.get(root_base + 1), execution_context);
        let mut current_args: Vec<PyObjectRef> = Vec::with_capacity(args.len());
        for i in 0..args.len() {
            current_args.push(_roots.get(root_base + 3 + i));
        }
        Self::finish_for_call_with_globals_obj(
            _roots.get(root_base) as *const (),
            &current_args,
            _roots.get(root_base + 1),
            execution_context,
            _roots.get(root_base + 2),
            w_builtin,
            allocation,
        )
    }

    /// Common tail of the frame builders: everything in `Frame.__init__`
    /// except the `pick_builtin` resolution, which is lifted to the callers
    /// so the fallible variant can propagate its error.
    fn finish_for_call_with_globals_obj(
        code: *const (),
        args: &[PyObjectRef],
        w_globals: PyObjectRef,
        execution_context: *const PyExecutionContext,
        closure: PyObjectRef,
        w_builtin: PyObjectRef,
        allocation: FrameLocalsArrayAllocation,
    ) -> Self {
        // `alloc_frame_locals_array` is a collection point.  Keep the call
        // inputs in real shadow-stack slots and populate the new locals array
        // from those forwarded slots, matching gctransform/framework.py's
        // push_roots/pop_roots around a GC allocation.
        let _roots = pyre_object::gc_roots::push_roots();
        // The scalars and `args` are two slices of one livevar set, so both are
        // written before either is queried: a forwarding query is a safepoint,
        // and `args` would still be off the root stack while the scalars ran
        // theirs.
        //
        // Through the bracket's own cell rather than the free functions, which
        // re-resolve the thread local on every call.
        let root_base = _roots.publish(&[code as PyObjectRef, w_globals, closure, w_builtin]);
        let args_base = _roots.publish(args);
        _roots.normalize(root_base, 4 + args.len());
        let code_ref =
            unsafe { &*(crate::w_code_get_ptr(_roots.get(root_base)) as *const CodeObject) };
        let num_locals = code_ref.varnames.len();
        let num_cells = ncells(code_ref);
        let max_stack = code_ref.max_stackdepth as usize;

        let locals_cells_stack_w = unsafe {
            alloc_frame_locals_array(num_locals + num_cells + max_stack, PY_NULL, allocation)
        };
        // The allocation result is a translated livevar before it is stored in
        // the eventual PyFrame. Publish it before `w_cell_new`, the barrier or
        // anything else below: until a heap edge names it, the shadow-stack
        // slot is the only thing that keeps it off the next sweep. It joins the
        // bracket the call inputs already opened: its lifetime is the rest of
        // this function body, so it needs no owner of its own.
        let locals_idx = _roots.publish(&[locals_cells_stack_w as PyObjectRef]);
        _roots.normalize(locals_idx, 1);

        {
            // Bind positional arguments directly -- no intermediate Vec.
            // Reload the array after each collecting call (`w_cell_new`);
            // `set_ref` is `setarrayitem_gc` so an old spill remembers young
            // children.
            let nargs = args.len().min(num_locals);
            for i in 0..nargs {
                let arr = unsafe { &mut *(_roots.get(locals_idx) as *mut FixedObjectArray) };
                arr.set_ref(i, _roots.get(args_base + i));
            }

            // Each cellvar that also appears in varnames shares its
            // varname slot (MAKE_CELL wraps the local). Only cellvars
            // not in varnames take a fresh slot in the cell region.
            // Allocating cells for the overlap would shift freevar
            // indices and break LOAD_DEREF on `def repeat(n): def
            // wrap(fn): def inner(): return (n, fn)` style closures.
            let npure = npure_cellvars(code_ref);
            for i in 0..npure {
                // pyframe.py `PyFrame.initialize_frame_scopes` `Cell(None, self.pycode.cell_families[i])`.
                let family = unsafe {
                    crate::pycode::w_code_cell_family(_roots.get(root_base), num_locals + i)
                };
                let cell = pyre_object::w_cell_new(PY_NULL, family);
                let arr = unsafe { &mut *(_roots.get(locals_idx) as *mut FixedObjectArray) };
                arr.set_ref(num_locals + i, cell);
            }
            let closure = _roots.get(root_base + 2);
            if !closure.is_null() {
                let nfreevars = code_ref.freevars.len();
                for i in 0..nfreevars {
                    let cell = unsafe { w_tuple_getitem(closure, i as i64).unwrap() };
                    let arr = unsafe { &mut *(_roots.get(locals_idx) as *mut FixedObjectArray) };
                    arr.set_ref(num_locals + npure + i, cell);
                }
            }
        }

        // Reload after the cell allocations: the pin slot is the only root
        // until the frame stores the array. A nursery-full spill is old-gen
        // and may hold young cells, so it joins the remembered set.
        let locals_cells_stack_w = _roots.get(locals_idx) as *mut FixedObjectArray;
        remember_frame_locals_array(locals_cells_stack_w);

        let frame_stores_global = unsafe {
            crate::w_code_frame_stores_global(_roots.get(root_base), _roots.get(root_base + 1))
        };

        let frame = PyFrame {
            ob_header: frame_ob_header(),
            pycode: _roots.get(root_base) as *const (),
            locals_cells_stack_w,
            valuestackdepth: num_locals + num_cells,
            last_instr: -1,
            flags: 0,
            failed_attr_cleanup: 0,
            debugdata: std::ptr::null_mut(),
            vable_token: 0,
            f_generator_wref: PY_NULL,
            w_yielding_from: PY_NULL,
            f_backref: std::ptr::null_mut(),
            w_builtin: _roots.get(root_base + 3),
        };
        // pyframe.py `__init__` opens with `self = hint(self,
        // access_directly=True, fresh_virtualizable=True)`, and `pycode.py
        // funcrun` re-establishes the same pair on the frame it just built
        // before binding the arguments and calling `init_cells`.  This
        // constructor is both halves fused, so one pair covers them.  The
        // hint precedes the globals store, as `__init__` does.
        let mut frame = majit_metainterp::jit::hint_fresh_virtualizable(
            majit_metainterp::jit::hint_access_directly(frame),
        );
        {
            // The globals store and the scope binding allocate. A collection
            // forwards the pin slot, not the field of this unboxed frame, so
            // register the field's slot for the span, as `createframe_obj` does
            // around `initialize_frame_scopes`.
            let _locals_root = FrameLocalsRoot::new(&mut frame);
            if frame_stores_global {
                frame.set_w_globals(_roots.get(root_base + 1));
            }
            // This constructor bypasses `initialize_frame_scopes`, so apply the
            // scope binding it would have done.  `FunctionType(co, globals)` over
            // a module-level code object arrives here, and without the binding its
            // `STORE_NAME`s land in a throwaway mapping instead of `globals`.
            frame.bind_unoptimized_locals_scope();
            frame.init_cells();
        }
        // `bind_unoptimized_locals_scope` allocates the class-body mapping, and
        // this aggregate's fields are native copies no collection rewrites.
        // Store through a pointer so the write is not a non-deref
        // virtualizable-field projection (`vable_projection_census` allows
        // those only on `FrameBox::new`).
        unsafe {
            store_locals_cells_stack_w(&mut frame, _roots.get(locals_idx) as *mut FixedObjectArray);
        }
        frame
    }

    /// Borrow the raw CodeObject.
    #[inline]
    pub fn code(&self) -> &CodeObject {
        unsafe { &*pyframe_get_pycode(self) }
    }

    #[inline]
    pub fn _is_generator_or_coroutine(&self) -> bool {
        code_flags_make_generator(self.code().flags)
    }

    /// pyframe.py initialize_as_generator
    ///
    /// Adaptation: pyre builds the caller's PyFrame on the interpreter stack,
    /// so we snapshot it onto the heap before handing ownership to the
    /// generator object. The backref (`f_generator_wref`) is set on that
    /// heap-owned snapshot — not on the temporary caller frame — so later
    /// `get_generator()` calls through the surviving frame pointer return
    /// the right object.
    #[inline]
    pub fn initialize_as_generator(&mut self) -> crate::PyResult {
        // pyframe.py initialize_as_generator wraps `self` directly. A borrowed `&mut self` cannot
        // hand ownership to the generator, so snapshot into an owned FrameBox
        // first. Callers that already own a FrameBox should use
        // `FrameBox::into_generator` to skip this copy.  Use the GC-managed
        // snapshot: the generator owns the frame for its whole life, off the
        // `CURRENT_FRAME` chain, so its lifetime is GC reachability.
        self.snapshot_for_generator().into_generator()
    }

    /// pyframe.py `get_generator`: `self.f_generator_wref()`.
    #[inline]
    pub fn get_generator(&self) -> PyObjectRef {
        unsafe {
            pyre_object::weakref::w_weakref_deref(
                self.f_generator_wref as *const pyre_object::weakref::Weakref,
            )
        }
    }

    /// Returns RPython's `last_instr` (= `next_instr`) value, derived from
    /// pyre's CPython-aligned storage. See the field comment on `last_instr`
    /// for the convention difference. PyPy `pyopcode.py:172` would read this
    /// field directly; pyre derives it because storage matches `f_lasti`.
    #[inline]
    pub fn next_instr(&self) -> usize {
        let last = self.last_instr;
        let base: isize = if last < 0 { 0 } else { last };
        let step: usize = if last < 0 { 0 } else { 1 };
        (base as usize).wrapping_add(step)
    }

    /// Inverse of `next_instr()`. Stores `next_instr - 1` so the field
    /// continues to satisfy `f_lasti = last_instr` (CPython convention).
    /// RPython would write `next_instr` directly here.
    #[inline]
    pub fn set_last_instr_from_next_instr(&mut self, next_instr: usize) {
        self.last_instr = next_instr as isize - 1;
    }

    /// Repoint internal array pointers after a struct move.
    ///
    /// `PyObjectArray` with small-buffer optimization stores an inline
    /// buffer whose address changes on move. Call this once after the
    /// frame is at its final stack location.
    #[inline]
    pub fn fix_array_ptrs(&mut self) {
        // locals_cells_stack_w is now a heap-allocated pointer;
        // fix_ptr on PyObjectArray is a no-op.
    }

    /// Load a constant from the code object by raw index.
    /// Used by the blackhole interpreter's bh_load_const_fn.
    pub fn load_const_pyobj(&self, idx: usize) -> PyObjectRef {
        let value =
            unsafe { crate::pycode::w_code_const(self.pycode as pyre_object::PyObjectRef, idx) };
        if value.is_null() {
            pyre_object::w_none()
        } else {
            value
        }
    }
}

/// Load a constant from a CodeObject without a PyFrame.
/// Used by the blackhole's bh_load_const_fn when the code pointer
/// comes from a virtualizable field read.
pub fn load_const_from_code(code: &CodeObject, idx: usize) -> PyObjectRef {
    let constants = code_constants(code);
    if idx >= constants.len() {
        return pyre_object::w_none();
    }
    pyobject_from_constant(&constants[idx])
}

/// `co_names[idx]` — the global/attribute name at `idx`, used by the JIT
/// full-body walker to resolve a `LOAD_GLOBAL` namei to the name string for
/// the module-dict cell-cache lookup.  `None` when `idx` is out of range.
pub fn load_name_from_code(code: &CodeObject, idx: usize) -> Option<&str> {
    // closure-free, Option-pattern-free `names.get(idx)` rewrite — keeps the
    // bounds check a plain `lt + getitem` instead of walking into an
    // `Option<&String>` the walker cannot fold.
    if idx < code.names.len() {
        Some(code.names[idx].as_str())
    } else {
        None
    }
}

pub(crate) fn code_constants(code: &CodeObject) -> &[crate::bytecode::ConstantData] {
    unsafe {
        std::slice::from_raw_parts(
            code.constants.as_ptr() as *const crate::bytecode::ConstantData,
            code.constants.len(),
        )
    }
}

/// Materialise a single `ConstantData` into a `PyObjectRef`.
///
/// Line-by-line port of the `ConstantOpcodeHandler for PyFrame` impl
/// (`eval.rs`) routed through `load_const_value`
/// (`pyopcode.rs`). The blackhole's `bh_load_const_fn` lacks a
/// `&mut PyFrame` to dispatch through the trait, so this free function
/// mirrors each `*_constant` body directly. Variant order matches
/// `pyopcode.rs::load_const_value` so future additions stay in sync.
pub fn pyobject_from_constant(constant: &crate::bytecode::ConstantData) -> PyObjectRef {
    use crate::bytecode::ConstantData;
    match constant {
        // `pyopcode.rs` — promote bigints to W_LongObject just
        // like `load_const_value` does before invoking the trait.
        ConstantData::Integer { value } => {
            let value = crate::compiler_bigint_to_rbigint(value);
            match value.toint() {
                Ok(value) => pyre_object::intobject::w_int_new(value),
                Err(_) => pyre_object::longobject::w_long_new(value),
            }
        }
        // `eval.rs` `float_constant`.
        ConstantData::Float { value } => pyre_object::floatobject::w_float_new(*value),
        // `eval.rs` `bool_constant` — bools must surface as
        // W_BoolObject (`is space.w_True/w_False`), not W_IntObject.
        ConstantData::Boolean { value } => pyre_object::w_bool_from(*value),
        // `eval.rs` `str_constant` — `box_str_constant` interns
        // matching `space.newtext` per `unicodeobject.py wrapunicode`.
        ConstantData::Str { value } => pyre_object::unicodeobject::box_str_constant(value),
        // `eval.rs` `bytes_constant`.
        ConstantData::Bytes { value } => pyre_object::bytesobject::w_bytes_from_bytes(value),
        // Reached only for a code constant nested inside a container constant;
        // top-level `LOAD_CONST` routes through `co_consts_w` in `bh_load_const_fn`
        // so the blackhole shares the interpreter's wrapper.  The exceptions are
        // this function's own fallbacks and `load_const_from_code`, which never
        // consults `co_consts_w`, so a top-level code constant can arrive here
        // too.  Either way the compiler does not emit a code object inside a
        // container, so the copy is defensive; `box_code_constant` records why
        // it is kept rather than removed.
        ConstantData::Code { code } => crate::pycode::box_code_constant(code),
        // `eval.rs` `none_constant`.
        ConstantData::None => pyre_object::w_none(),
        // `eval.rs` `ellipsis_constant`.
        ConstantData::Ellipsis => pyre_object::special::w_ellipsis(),
        // `load_const_value`'s `Tuple` arm (`pyopcode.rs`) — recurse + delegate
        // to the default `build_tuple` body (`eval.rs`, which calls
        // `build_tuple_from_refs`).
        ConstantData::Tuple { elements } => {
            // The recursion mints an object per element, so the elements
            // already built are pinned across it -- a plain `Vec` is not a
            // root area (`build_list_storage`).
            let mut items = pyre_object::gc_roots::RootedItems::new();
            for element in elements {
                items.push(pyobject_from_constant(element));
            }
            crate::runtime_ops::build_tuple_from_refs(&items.take())
        }
        // `load_const_value`'s `Slice` arm (`pyopcode.rs`) — recurse over
        // `[start, stop, step]` before invoking `slice_constant` (`eval.rs`).
        ConstantData::Slice { elements } => {
            // Same hazard as the `Tuple` arm with three named operands: each
            // recursion allocates over the ones already built, and
            // `w_slice_new` allocates over all three.
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.base();
            let _ = roots.pin_root(pyobject_from_constant(&elements[0]));
            let _ = roots.pin_root(pyobject_from_constant(&elements[1]));
            let _ = roots.pin_root(pyobject_from_constant(&elements[2]));
            pyre_object::w_slice_new(roots.get(base), roots.get(base + 1), roots.get(base + 2))
        }
        // `load_const_value`'s `Frozenset` arm (`pyopcode.rs`) — recurse +
        // delegate to `frozenset_constant` (`eval.rs`).
        ConstantData::Frozenset { elements } => {
            // As in the `Tuple` arm.
            let mut items = pyre_object::gc_roots::RootedItems::new();
            for element in elements {
                items.push(pyobject_from_constant(element));
            }
            pyre_object::w_frozenset_from_items(&items.take())
        }
        ConstantData::Complex { value } => {
            pyre_object::complexobject::w_complex_new(value.re, value.im)
        }
    }
}

// Virtualizable configuration is in jit/frame_layout.rs

/// pypy/interpreter/baseobjspace.py `createframe`.
///
/// ```python
/// def createframe(self, code, w_globals, outer_func=None):
///     "Create an empty PyFrame suitable for this code object."
///     return self.FrameClass(self, code, w_globals, outer_func)
/// ```
///
/// Returns `Box<PyFrame>` matching PyPy's heap-allocated PyFrame (RPython
/// class instance — `pyframe.py class PyFrame(W_Root)`).  The Box keeps
/// every `PyFrame` heap-resident.
///
/// The body inlines `pyframe.py PyFrame.__init__` line-by-line:
/// allocate `locals_cells_stack_w` of size `nlocals + ncellvars + nfreevars
/// + stacksize`, set `valuestackdepth`, optionally bind debug `w_globals`
/// when `code.frame_stores_global(w_globals)`, then call
/// `self.initialize_frame_scopes(outer_func, code)` (`pyframe.py`)
/// which performs cell init, freevar copy from `outer_func.closure`, and
/// raises on freevar/closure-size mismatch.  No constructor switch — both
/// branches share the same allocation + scope-init path so the cell/freevar
/// invariants of `pyframe.py` hold uniformly.
///
/// `outer_func` carries the closure-providing function reference per PyPy
/// (`function.py:126-127, 208-209, 219-220`).  `None` for module / exec /
/// REPL frames where freevars must be empty (PyPy raises TypeError
/// "directly executed code object may not contain free variables" in
/// `pyframe.py`).  `Some(func)` for function calls AND class-body
/// execution (`pypy/module/__builtin__/compiling.py:208`), where the
/// function's closure tuple seeds the freevar slots via
/// `function_get_closure`.  Pyre adds `execution_context` and `w_globals`
/// as explicit parameters because pyre lacks PyPy's `space` implicit
/// carrier.
///
/// **Args binding** (positional argument values into
/// `locals_cells_stack_w[0..nargs]`) is **caller-side** per PyPy
/// `pycode.py funcrun`: caller invokes
/// `space.createframe(...) → args.parse_into_scope(...) → frame.init_cells()
/// → frame.run(...)`.  createframe itself never binds args and never calls
/// `init_cells()`.
///
/// # Safety
/// `code` and `execution_context` must be valid pointers
/// for the duration the returned `Box<PyFrame>` is alive.  `outer_func`,
/// when `Some`, must be a valid Function `PyObjectRef`.
pub fn createframe(
    code: *const (),
    execution_context: *const PyExecutionContext,
    outer_func: Option<PyObjectRef>,
) -> Result<FrameBox, crate::PyError> {
    // Storage-only entry: there is no globals object to forward.
    let w_globals = PY_NULL;
    createframe_obj(code, w_globals, execution_context, outer_func)
}

/// `baseobjspace.py createframe` with the globals passed as the dict
/// object; `PyCode.frame_stores_global` chooses code storage or debugdata.
pub fn createframe_obj(
    code: *const (),
    w_globals: PyObjectRef,
    execution_context: *const PyExecutionContext,
    outer_func: Option<PyObjectRef>,
) -> Result<FrameBox, crate::PyError> {
    // pyframe.py PyFrame.__init__ — line-by-line.
    //   self.space = space               (pyre: implicit, no field)
    //   self.pycode = code               (pycode field below)
    //   if code.frame_stores_global(w_globals):
    //       self.getorcreatedebug().w_globals = w_globals
    //   ncellvars = len(code.co_cellvars)
    //   nfreevars = len(code.co_freevars)
    //   size = code.co_nlocals + ncellvars + nfreevars + code.co_stacksize
    //   self.locals_cells_stack_w = [None] * size
    //   self.valuestackdepth = code.co_nlocals + ncellvars + nfreevars
    //   ...
    //   self.initialize_frame_scopes(outer_func, code)
    //
    // Preserve the exact object either on `PyCode.w_globals` (normal case) or
    // the rare `FrameDebugData.w_globals` override.
    let raw = unsafe { crate::w_code_get_ptr(code as PyObjectRef) as *const CodeObject };
    let code_ref = unsafe { &*raw };
    let num_locals = code_ref.varnames.len();
    let num_cells = ncells(code_ref);
    let max_stack = code_ref.max_stackdepth as usize;
    let size = num_locals + num_cells + max_stack;
    // `frame_builtin_obj` and `FrameBox::new` are both collection points.
    // `FrameBox::new` republishes the `PyFrame`'s own GCREF fields, but the
    // globals override is not one of them any more and the code object retains
    // only the FIRST globals dictionary, so nothing else names this one: a
    // collection between here and `set_w_globals` would leave `f_globals`
    // reading a relocated pointer. Carry the whole set on the root stack and
    // read each back after the last allocation before its use, as
    // `PyFrame::new` does for its own constructor.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = _roots.base();
    let _ = _roots.pin_root(code as PyObjectRef);
    let _ = _roots.pin_root(w_globals);
    let _ = _roots.pin_root(outer_func.unwrap_or(PY_NULL));
    let frame_stores_global = unsafe {
        crate::w_code_frame_stores_global(_roots.get(root_base), _roots.get(root_base + 1))
    };
    let _ = _roots.pin_root(crate::baseobjspace::frame_builtin_obj(
        _roots.get(root_base + 1),
        execution_context,
    ));
    let locals_cells_stack_w =
        unsafe { alloc_frame_locals_array(size, PY_NULL, FrameLocalsArrayAllocation::NurseryGc) };
    let frame = PyFrame {
        ob_header: frame_ob_header(),
        pycode: _roots.get(root_base) as *const (),
        locals_cells_stack_w,
        valuestackdepth: num_locals + num_cells,
        last_instr: -1,
        flags: 0,
        failed_attr_cleanup: 0,
        debugdata: std::ptr::null_mut(),
        vable_token: 0,
        f_generator_wref: PY_NULL,
        w_yielding_from: PY_NULL,
        f_backref: std::ptr::null_mut(),
        w_builtin: _roots.get(root_base + 3),
    };
    // pyframe.py `__init__` — `self = hint(self, access_directly=True,
    // fresh_virtualizable=True)`.  Upstream spells the two kwargs on one
    // `hint()`; pyre dispatches one helper per kwarg.  On the frame itself,
    // before the box: `self` upstream is the frame, and the propagation reads
    // the hinted value's class root.  Both precede the globals store, as
    // `__init__` does.
    let frame = majit_metainterp::jit::hint_fresh_virtualizable(
        majit_metainterp::jit::hint_access_directly(frame),
    );
    let mut frame = FrameBox::new(frame);
    if frame_stores_global {
        frame.set_w_globals(_roots.get(root_base + 1));
    }
    // pyframe.py `PyFrame.__init__` — final step. PY_NULL plays the role of
    // Python `None` per the existing `initialize_frame_scopes` convention.
    // Top-level module / interactive / expression code
    // arrives here without CO_NEWLOCALS — RustPython codegen emits empty
    // flags for the seed CodeInfo (`crates/codegen/src/compile.rs Compiler::new`)
    // so initialize_frame_scopes selects the `!OPTIMIZED && !NEWLOCALS`
    // arm and binds `w_locals = w_globals` per pyframe.py.
    let outer_ref = _roots.get(root_base + 2);
    // initialize_frame_scopes allocates (the locals dict and `w_cell_new`
    // cells) and stores the cells into `locals_cells_stack_w`; root the slot so
    // a collection mid-init can't drop the cells already written there.
    {
        // The owned Rust `FrameBox` is not itself visible to the collector.
        // RPython keeps the freshly allocated frame as a livevar across these
        // initialization allocations, so expose the frame object alongside
        // the separately registered locals-array field.
        let _frame_roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(frame.as_mut_ptr() as pyre_object::PyObjectRef);
        let _root = FrameLocalsRoot::new(frame.as_mut_ptr());
        frame.initialize_frame_scopes(outer_ref, _roots.get(root_base) as *const ())?;
    }
    remember_frame_locals_array(frame.locals_cells_stack_w);

    Ok(frame)
}

/// `space.finditem_str(w_obj, key)` — `space.getitem` with KeyError
/// remapped to `None`.  Non-KeyError errors propagate unchanged so
/// `fast2locals`/`locals2fast` raise as PyPy does at `pyframe.py` /
/// `pyframe.py` (`pypy/objspace/std/objspace.py finditem_str` re-
/// raises everything except `KeyError`).
fn finditem_str_object(
    w_obj: PyObjectRef,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    // Minting the key allocates, so the mapping moves between entry and the
    // lookup that uses it as the receiver.  A stale receiver reaches
    // `dict.__getitem__` as a descriptor mismatch against its own type name,
    // not as a crash, so publish it across the mint the way its `setitem` and
    // `delitem` siblings do.
    let roots = pyre_object::gc_roots::push_roots();
    let object_slot = roots.base();
    let _ = roots.pin_root(w_obj);
    let key_slot = object_slot + 1;
    let _ = roots.pin_root(unsafe { pyre_object::w_str_new_managed(name) });
    match crate::baseobjspace::getitem(roots.get(object_slot), roots.get(key_slot)) {
        Ok(v) if !v.is_null() => Ok(Some(v)),
        Ok(_) => Ok(None),
        Err(e) if e.kind == crate::PyErrorKind::KeyError => Ok(None),
        Err(e) => Err(e),
    }
}

/// `space.setitem(w_obj, w_str_new(name), value)` — every error
/// propagates (PyPy `pyframe.py:568` `space.setitem_str` does not
/// swallow exceptions).
fn setitem_str_object(
    w_obj: PyObjectRef,
    name: &str,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let object_slot = roots.base();
    let _ = roots.pin_root(w_obj);
    let _ = roots.pin_root(value);
    let key_slot = object_slot + 2;
    let _ = roots.pin_root(unsafe { pyre_object::w_str_new_managed(name) });
    crate::baseobjspace::setitem(
        roots.get(object_slot),
        roots.get(key_slot),
        roots.get(object_slot + 1),
    )
    .map(|_| ())
}

/// `space.delitem(w_obj, w_str_new(name))` — `pyframe.py:571-574 /
/// 589-593` ignores `KeyError` only; other errors propagate.
fn delitem_str_object(w_obj: PyObjectRef, name: &str) -> Result<(), crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let object_slot = roots.base();
    let _ = roots.pin_root(w_obj);
    let key_slot = object_slot + 1;
    let _ = roots.pin_root(unsafe { pyre_object::w_str_new_managed(name) });
    match crate::baseobjspace::delitem(roots.get(object_slot), roots.get(key_slot)) {
        Ok(_) => Ok(()),
        Err(e) if e.kind == crate::PyErrorKind::KeyError => Ok(()),
        Err(e) => Err(e),
    }
}

/// `pyframe.py self.space.newdict(instance=True)` — the mapping
/// `fast2locals` materialises for a frame that has none yet, for a trace that
/// models the fastlocals reads instead of residualizing
/// `interp_inspect.py locals`.
///
/// Takes no `PyFrame`, so nothing reachable from here can call
/// [`crate::executioncontext::force_frame`] — a helper that touched the frame
/// would re-arm the escape this modelling exists to remove.  The store back
/// into `debugdata.w_locals` is therefore NOT modelled either: the caller
/// pins the frame's absent mapping with a guard and hands this dict out
/// directly, which for an OPTIMIZED frame is already the independent copy
/// `frame_locals_snapshot` returns.
pub extern "C" fn jit_locals_dict_new() -> PyObjectRef {
    unsafe { pyre_object::w_dict_new() }
}

/// `pyframe.py fast2locals` for ONE visible fastlocal slot: bind
/// `code.varnames[index]` to `value` in `dict`.
///
/// `dict` is the frame's own locals mapping — `getorcreatedebug().w_locals`,
/// which the modelled expansion reads through the `debugdata` virtualizable
/// field and hands in as an ordinary Ref operand.  No `PyFrame` reaches this
/// helper, so nothing under it can call
/// [`crate::executioncontext::force_frame`]; that is the whole point of the
/// split, since a helper that touched the frame would re-arm the escape the
/// modelling exists to remove.
///
/// Returns `dict` so the unrolled slot chain threads the (possibly forwarded)
/// mapping from one store to the next instead of holding a raw address across
/// an allocating call.
///
/// # Safety
/// `dict` must be a live dict, `code` a live `CodeObject` with
/// `index < varnames.len()`, and `value` a live non-null `W_Root`.
pub extern "C" fn jit_locals_dict_setitem_local(
    dict: PyObjectRef,
    code: *const CodeObject,
    index: i64,
    value: PyObjectRef,
) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(dict);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(value);
    let code = unsafe { &*(code) };
    let name: &str = &code.varnames[index as usize];
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            name,
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        );
    }
    pyre_object::gc_roots::shadow_stack_get(dict_slot)
}

/// `pyframe.py fast2locals` for ONE cell or freevar slot: bind
/// `PyFrame::cell_slot_names(code)[index]` to `value` in `dict`.
///
/// `index` counts from the first slot ABOVE `code.varnames.len()`, which is
/// where the pure cellvars and the freevars live; `value` is the cell's
/// CONTENTS, already read by the caller, not the cell.  Splitting the read out
/// is what keeps this helper off the frame: same reason
/// [`jit_locals_dict_setitem_local`] takes a value rather than a slot.
///
/// Returns `dict` for the same threading reason as
/// [`jit_locals_dict_setitem_local`].
///
/// # Safety
/// `dict` must be a live dict, `code` a live `CodeObject` with `index` inside
/// [`PyFrame::cell_slot_names`], and `value` a live non-null `W_Root`.
pub extern "C" fn jit_locals_dict_setitem_cell(
    dict: PyObjectRef,
    code: *const CodeObject,
    index: i64,
    value: PyObjectRef,
) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(dict);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(value);
    let code = unsafe { &*(code) };
    let name = PyFrame::cell_slot_names(code)
        .nth(index as usize)
        .expect("cell slot index is inside cell_slot_names");
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            name,
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        );
    }
    pyre_object::gc_roots::shadow_stack_get(dict_slot)
}

/// `pyframe.py fast2locals` for ONE visible fastlocal slot that is
/// unbound: remove `code.varnames[index]` from `dict`.
///
/// The delete is fallible for the same reason
/// [`crate::baseobjspace::delitem`] is — a stored key whose hash collides
/// with the varname can reach a user `__eq__` that raises — so it routes
/// through the checked spelling.  A missing key is not an error here
/// (`w_dict_delitem_checked` reports it as `Ok(false)`), which is the
/// `KeyError` arm `delitem_str_object` swallows.  Anything else is reported
/// as `PY_NULL` and the pending slot is drained, so the guarded side exit
/// re-runs the residual and raises from the eval loop.
///
/// Returns `dict` on success, for the same threading reason as
/// [`jit_locals_dict_setitem_local`].
///
/// # Safety
/// `dict` must be a live dict and `code` a live `CodeObject` with
/// `index < varnames.len()`.
pub extern "C" fn jit_locals_dict_delitem_local(
    dict: PyObjectRef,
    code: *const CodeObject,
    index: i64,
) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(dict);
    let code = unsafe { &*(code) };
    let name: &str = &code.varnames[index as usize];
    let key_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(unsafe { pyre_object::w_str_new_managed(name) });
    let deleted = unsafe {
        pyre_object::dictmultiobject::w_dict_delitem_checked(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            pyre_object::gc_roots::shadow_stack_get(key_slot),
        )
    };
    match deleted {
        Ok(_) => pyre_object::gc_roots::shadow_stack_get(dict_slot),
        Err(_) => {
            let _ = crate::baseobjspace::take_pending_dict_key_error(
                pyre_object::gc_roots::shadow_stack_get(key_slot),
            );
            pyre_object::PY_NULL
        }
    }
}

/// The copy half of [`PyFrame::frame_locals_snapshot`]: an INDEPENDENT dict
/// holding what the frame's own locals mapping holds (PEP 667), which is what
/// `locals()` / `vars()` hand back for an OPTIMIZED frame.
///
/// Reports a failing copy as `PY_NULL` rather than publishing it, so the
/// guarded side exit re-runs the residual and raises from the eval loop.
///
/// # Safety
/// `w_locals` must be a live mapping.
pub extern "C" fn jit_locals_dict_snapshot(w_locals: PyObjectRef) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let locals_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_locals);
    let snapshot_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(unsafe { pyre_object::w_dict_new() });
    // `dict_update` walks a mapping's `keys()`, so both sides are reloaded
    // across it as well as across the `w_dict_new` above.  It is the
    // `PyDict_Update(dct, self)` copy `frame_getlocals` makes, so the update
    // percolates rather than reporting a non-mapping.
    match crate::opcode_ops::dict_update(
        pyre_object::gc_roots::shadow_stack_get(snapshot_slot),
        pyre_object::gc_roots::shadow_stack_get(locals_slot),
    ) {
        Ok(()) => pyre_object::gc_roots::shadow_stack_get(snapshot_slot),
        Err(_) => pyre_object::PY_NULL,
    }
}

/// The modelled `locals()` rebuild's extras half: append `src` to `dst` once
/// the fastlocals are in.  A key that names no writable fast local lives only
/// in `f_extra_locals`, so the rebuild has to take that dict or it drops the
/// key; a key both halves carry keeps the fastlocal, because that is the one
/// `framelocalsproxy_getitem` answers with.
///
/// It runs [`merge_extra_locals`], the half [`PyFrame::frame_locals_proxy_snapshot`]
/// runs, so the trace and the eval loop cannot order or resolve the two halves
/// differently.
///
/// Reports a failing merge as `PY_NULL` so the guarded side exit re-runs
/// the residual.  Returns `dst` on success so the slot chain can thread it.
///
/// # Safety
/// `dst` and `src` must be live mappings.
pub extern "C" fn jit_locals_dict_update(dst: PyObjectRef, src: PyObjectRef) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let dst_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(dst);
    let src_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(src);
    match merge_extra_locals(
        pyre_object::gc_roots::shadow_stack_get(dst_slot),
        pyre_object::gc_roots::shadow_stack_get(src_slot),
    ) {
        Ok(()) => pyre_object::gc_roots::shadow_stack_get(dst_slot),
        Err(_) => pyre_object::PY_NULL,
    }
}

/// Name whichever frame on the current call chain still points at a
/// `locals_cells_stack_w` array a collection has already moved.
///
/// Installed as `pyre_object::gc_hook`'s stale-array holder reporter: the
/// object-space store site sees only the array, while the holder's generation
/// is the datum that separates the two causes.  An old-generation holder means
/// the minor collection never scanned it — a remembered-set miss.  A holder
/// that is itself forwarded means the frame was never traced, so every field
/// read out of it names pre-collection memory.  Walks `f_backref` because a
/// JIT-built inline callee frame sits on that chain and nowhere else.
///
/// This runs on an already-fatal path, so it stops rather than chase an
/// unvalidated `f_backref`: losing the walk costs one line, while faulting
/// inside it would cost the whole report.
pub fn report_stale_locals_array_holder(stale_addr: usize) -> bool {
    let mut frame = crate::eval::current_frame();
    let mut depth = 0usize;
    let mut found = false;
    while !frame.is_null() && depth < 64 {
        let frame_addr = frame as usize;
        let owned =
            pyre_object::gc_hook::try_gc_owns_object(frame_addr as pyre_object::gc_hook::GCREF);
        if depth > 0 && !owned {
            crate::host_seam::emit_stderr(
                format!(
                    "STALE ARRAY holder scan: depth={depth} frame={frame_addr:#x} is not \
GC-owned; stopping rather than dereferencing an unvalidated f_backref\n"
                )
                .as_bytes(),
            );
            break;
        }
        let array = unsafe { (*frame).locals_cells_stack_w } as usize;
        let nursery = majit_gc::gc_is_nursery_object(frame_addr);
        let (type_id, forwarded, tracks_young) = if owned {
            let header = unsafe { *majit_gc::header::header_of(frame_addr) };
            (
                header.type_id(),
                header.is_forwarded(),
                header.has_flag(majit_gc::GcFlags::GCFLAG_TRACK_YOUNG_PTRS),
            )
        } else {
            (u32::MAX, false, false)
        };
        let holds = array == stale_addr;
        found |= holds;
        crate::host_seam::emit_stderr(
            format!(
                "STALE ARRAY holder scan: depth={depth} frame={frame_addr:#x} \
locals={array:#x} holds_stale={holds} gc_owned={owned} nursery={nursery} \
type_id={type_id} frame_forwarded={forwarded} track_young_ptrs={tracks_young}\n"
            )
            .as_bytes(),
        );
        frame = unsafe { (*frame).f_backref };
        depth += 1;
    }
    found
}

#[cfg(test)]
mod tests {
    use super::load_const_from_code;

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn pyframe_common_layout_has_no_eager_globals_word() {
        assert_eq!(std::mem::size_of::<super::PyFrame>(), 104);
    }

    #[test]
    fn shared_code_stores_only_the_different_globals_on_debugdata() {
        // Module code creates debugdata for `w_locals` even in PyPy.  Use an
        // optimized function code object so this test isolates the uncommon
        // globals override that `PyCode.frame_stores_global` controls.
        let outer = crate::compile_exec("def f():\n    return 1\n").expect("compile");
        let code = super::code_constants(&outer)
            .iter()
            .find_map(|constant| match constant {
                crate::bytecode::ConstantData::Code { code } => Some(code.as_ref()),
                _ => None,
            })
            .expect("nested function code");
        let w_code = crate::pycode::box_code_constant(code);
        let first_globals = pyre_object::w_dict_new();
        let other_globals = pyre_object::w_dict_new();

        let first =
            super::createframe_obj(w_code as *const (), first_globals, std::ptr::null(), None)
                .expect("first frame");
        assert!(first.debugdata.is_null());
        assert_eq!(first.get_w_globals(), first_globals);

        let other =
            super::createframe_obj(w_code as *const (), other_globals, std::ptr::null(), None)
                .expect("shared-code frame");
        assert!(!other.debugdata.is_null());
        assert_eq!(other.get_w_globals(), other_globals);
        assert_eq!(
            unsafe { crate::w_code_get_w_globals(w_code) },
            first_globals
        );
    }

    fn nested_code_yields_inside_try(source: &str) -> bool {
        let outer = crate::compile_exec(source).expect("compile");
        let code = super::code_constants(&outer)
            .iter()
            .find_map(|constant| match constant {
                crate::bytecode::ConstantData::Code { code } => Some(code.as_ref()),
                _ => None,
            })
            .expect("nested function code");
        let w_code = crate::pycode::box_code_constant(code);
        unsafe { crate::pycode::w_code_yields_inside_try(w_code) }
    }

    #[test]
    fn generator_finalizer_flag_ignores_python314_stopiteration_wrapper() {
        assert!(!nested_code_yields_inside_try(
            "def items():\n    yield 1\n"
        ));
    }

    #[test]
    fn generator_finalizer_flag_finds_yield_inside_try_or_with() {
        assert!(nested_code_yields_inside_try(
            "def items(cm):\n    try:\n        yield 1\n    finally:\n        pass\n",
        ));

        assert!(nested_code_yields_inside_try(
            "def items(cm):\n    with cm:\n        yield 1\n"
        ));
    }

    #[test]
    fn generator_finalizer_flag_ignores_yield_after_try() {
        assert!(!nested_code_yields_inside_try(
            "def items():\n    try:\n        pass\n    finally:\n        pass\n    yield 1\n",
        ));
    }

    #[test]
    fn load_const_from_code_returns_ellipsis_singleton() {
        let code = crate::compile_eval("...");
        let code = code.expect("compile should succeed");
        let ellipsis_index = super::code_constants(&code)
            .iter()
            .position(|constant| matches!(constant, crate::bytecode::ConstantData::Ellipsis))
            .expect("compiled code should contain an Ellipsis constant");

        let loaded = load_const_from_code(&code, ellipsis_index);
        assert_eq!(loaded, pyre_object::special::w_ellipsis());
    }

    // ── mark_stacks (f_lineno jump validation) ──

    use super::{
        MARK_EMPTY_STACK, MARK_UNINITIALIZED, StackKind, mark_compatible_stack,
        mark_first_line_not_before, mark_lines, mark_push_value, mark_stacks,
    };

    #[test]
    fn mark_stacks_entry_is_empty_and_reachable() {
        let code = crate::compile_exec("x = 1\ny = x + 1\n").expect("compile");
        let len = code.instructions.len();
        let stacks = mark_stacks(&code, len);
        // Entry to the first instruction is the empty stack.
        assert_eq!(stacks[0], MARK_EMPTY_STACK);
        // Every real (decodable, non-cache) instruction is reachable in
        // straight-line code — none stay UNINITIALIZED.
        for pc in 0..len {
            if let Some((instr, _)) = crate::pyopcode::decode_instruction_at(&code, pc) {
                if matches!(instr, crate::bytecode::Instruction::Cache) {
                    continue;
                }
                assert_ne!(
                    stacks[pc], MARK_UNINITIALIZED,
                    "pc {pc} ({instr:?}) should be reachable"
                );
            }
        }
    }

    #[test]
    fn mark_stacks_for_iter_target_carries_iterator() {
        // The FOR_ITER body sees the iterator underneath the loop var, so
        // the abstract stack at the body has an Iterator slot at bottom.
        let code = crate::compile_exec("t = 0\nfor i in range(3):\n    t += i\n").expect("compile");
        let len = code.instructions.len();
        let stacks = mark_stacks(&code, len);
        // Find a GET_ITER; the slot it produces is Iterator and stays on
        // the stack through the loop body.
        let mut saw_iterator_slot = false;
        for pc in 0..len {
            if stacks[pc] > MARK_EMPTY_STACK
                && (stacks[pc] & 0b111 == StackKind::Iterator as i64
                    || (stacks[pc] >> 3) & 0b111 == StackKind::Iterator as i64)
            {
                saw_iterator_slot = true;
                break;
            }
        }
        assert!(
            saw_iterator_slot,
            "a for-loop's abstract stacks should contain an Iterator slot"
        );
    }

    #[test]
    fn mark_stacks_after_inner_loop_keeps_outer_iterator() {
        // The statement after an inner loop still runs inside the outer one,
        // so its entry stack is exactly the outer loop's iterator.
        let code = crate::compile_exec(
            "for i in [1]:
    for j in [2]:
        pass
    x = 6
x = 7
",
        )
        .expect("compile");
        let len = code.instructions.len();
        let stacks = mark_stacks(&code, len);
        let first = code.first_line_number.map(|n| n.get() as i32).unwrap_or(1);
        let lines = mark_lines(&code, first, len);
        let entry = |line: i32| {
            (0..len)
                .find(|&pc| lines[pc] == line)
                .map(|pc| stacks[pc])
                .expect("line has a start")
        };
        let outer = mark_push_value(MARK_EMPTY_STACK, StackKind::Iterator);
        assert_eq!(entry(4), outer);
        assert_eq!(entry(5), MARK_EMPTY_STACK);
    }

    #[test]
    fn mark_lines_and_first_line_not_before() {
        let code = crate::compile_exec("a = 1\nb = 2\nc = 3\n").expect("compile");
        let len = code.instructions.len();
        let first = code.first_line_number.map(|n| n.get() as i32).unwrap_or(1);
        let lines = mark_lines(&code, first, len);
        // A module body opens with a `RESUME` whose range carries line 0,
        // below `co_firstlineno`; it is a range start, so it is marked.
        assert_eq!(lines[0], 0);
        // The earliest recorded statement line is the first source line.
        let min_line = lines.iter().copied().filter(|&l| l > 0).min().unwrap();
        assert_eq!(min_line, first);
        // A request before the code resolves to the first line.
        assert_eq!(mark_first_line_not_before(&lines, first), first);
        // A request past the end resolves to -1.
        assert_eq!(mark_first_line_not_before(&lines, first + 1000), -1);
    }

    #[test]
    fn compatible_stack_same_and_incompatible() {
        // Identical stacks are compatible.
        assert!(mark_compatible_stack(MARK_EMPTY_STACK, MARK_EMPTY_STACK));
        // An Object target accepts an Iterator source popped to depth, but
        // an Iterator target rejects an Object source (can't fabricate a
        // loop iterator).
        let obj = super::mark_push_value(MARK_EMPTY_STACK, StackKind::Object);
        let iter = super::mark_push_value(MARK_EMPTY_STACK, StackKind::Iterator);
        assert!(
            mark_compatible_stack(iter, obj),
            "Object target accepts any non-Null"
        );
        assert!(
            !mark_compatible_stack(obj, iter),
            "Iterator target rejects Object source"
        );
    }

    fn nested_function_frame(source: &str) -> super::FrameBox {
        crate::test_hooks::install_hash_hook();
        crate::typedef::init_typeobjects();
        let outer = crate::compile_exec(source).expect("compile");
        let code = super::code_constants(&outer)
            .iter()
            .find_map(|constant| match constant {
                crate::bytecode::ConstantData::Code { code } => Some(code.as_ref()),
                _ => None,
            })
            .expect("nested function code");
        assert!(
            code.flags.contains(crate::CodeFlags::OPTIMIZED),
            "function frames expose FrameLocalsProxy"
        );
        let w_code = crate::pycode::box_code_constant(code);
        let w_globals = pyre_object::w_dict_new();
        super::createframe_obj(w_code as *const (), w_globals, std::ptr::null(), None)
            .expect("frame")
    }

    #[test]
    fn frame_locals_proxy_contains_len_keys_getitem() {
        // Bound slots answer membership, length, keys and subscript from the
        // fast array; unbound slots are skipped; extras follow without
        // replacing a bound name.
        let mut frame = nested_function_frame("def f(a):\n    b = 1\n");
        let bound = pyre_object::w_int_new(7);
        frame.set_locals_w(0, bound);
        let proxy = frame.fget_getdictscope().expect("proxy");
        assert!(
            super::frame_locals_proxy::viewed_frame(proxy).is_some(),
            "optimized frame.f_locals is a FrameLocalsProxy"
        );

        let roots = pyre_object::gc_roots::push_roots();
        let proxy_slot = roots.base();
        let _ = roots.pin_root(proxy);
        let _ = roots.pin_root(bound);
        let a_slot = proxy_slot + 2;
        let _ = roots.pin_root(pyre_object::w_str_new_managed("a"));
        let b_slot = a_slot + 1;
        let _ = roots.pin_root(pyre_object::w_str_new_managed("b"));
        let missing_slot = b_slot + 1;
        let _ = roots.pin_root(pyre_object::w_str_new_managed("missing"));

        assert!(crate::baseobjspace::contains(roots.get(proxy_slot), roots.get(a_slot)).unwrap());
        assert!(!crate::baseobjspace::contains(roots.get(proxy_slot), roots.get(b_slot)).unwrap());
        assert!(
            !crate::baseobjspace::contains(roots.get(proxy_slot), roots.get(missing_slot)).unwrap()
        );

        let got = crate::baseobjspace::getitem(roots.get(proxy_slot), roots.get(a_slot)).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(got) }, 7);

        let len = crate::baseobjspace::len(roots.get(proxy_slot)).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(len) }, 1);

        let extra_key_slot = missing_slot + 1;
        let _ = roots.pin_root(pyre_object::w_str_new_managed("extra"));
        let extra_val_slot = extra_key_slot + 1;
        let _ = roots.pin_root(pyre_object::w_int_new(9));
        crate::baseobjspace::setitem(
            roots.get(proxy_slot),
            roots.get(extra_key_slot),
            roots.get(extra_val_slot),
        )
        .unwrap();

        assert!(
            crate::baseobjspace::contains(roots.get(proxy_slot), roots.get(extra_key_slot))
                .unwrap()
        );
        let len = crate::baseobjspace::len(roots.get(proxy_slot)).unwrap();
        assert_eq!(unsafe { pyre_object::w_int_get_value(len) }, 2);

        let keys = crate::baseobjspace::call_method(roots.get(proxy_slot), "keys", &[]);
        assert!(
            !keys.is_null(),
            "keys() failed: {:?}",
            crate::call::take_call_error()
        );
        unsafe {
            assert_eq!(pyre_object::w_list_len(keys), 2);
            let k0 = pyre_object::w_list_getitem(keys, 0).unwrap();
            let k1 = pyre_object::w_list_getitem(keys, 1).unwrap();
            assert_eq!(pyre_object::w_str_get_wtf8(k0).as_bytes(), b"a");
            assert_eq!(pyre_object::w_str_get_wtf8(k1).as_bytes(), b"extra");
        }
    }

    #[test]
    fn frame_locals_proxy_contains_rejects_unhashable() {
        let mut frame = nested_function_frame("def f(a):\n    return a\n");
        let proxy = frame.fget_getdictscope().expect("proxy");
        let roots = pyre_object::gc_roots::push_roots();
        let proxy_slot = roots.base();
        let _ = roots.pin_root(proxy);
        let key_slot = proxy_slot + 1;
        let _ = roots.pin_root(pyre_object::w_list_new(Vec::new()));
        let err = crate::baseobjspace::contains(roots.get(proxy_slot), roots.get(key_slot))
            .expect_err("unhashable key");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
    }
}
