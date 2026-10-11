//! itertools implementation — PyPy: pypy/module/itertools/interp_itertools.py
//!
//! Verbatim move of the inline block previously in importing.rs.

pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), crate::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // `W_Count.typedef` and the sibling classes are module TypeDefs.  Empty
    // startup does not import itertools, so `init_typeobjects` does not
    // build them. Pin `ns` first: this build collects and can drop the GIL.
    crate::typedef::ensure_itertools_typeobjects();
    // PyPy exports W_Chain.typedef itself. Its __new__ and classmethod
    // from_iterable both preserve lazy traversal of the outer iterable.
    crate::__pyre_put_new!(
        ns_slot,
        "chain",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::CHAIN_TYPE)
            .expect("itertools.chain TypeDef initialized")
            .as_ptr()
    );
    // PyPy exports W_StarMap.typedef itself; its __new__ stores a live source
    // iterator and next_w performs one expanded call at a time.
    crate::__pyre_put_new!(
        ns_slot,
        "starmap",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::STARMAP_TYPE)
            .expect("itertools.starmap TypeDef initialized")
            .as_ptr()
    );
    // PyPy exposes W_Count.typedef / W_Repeat.typedef themselves from the
    // module, not function-shaped constructor shims.  Their `__new__` slots
    // perform allocation and argument parsing.
    crate::__pyre_put_new!(
        ns_slot,
        "count",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::COUNT_TYPE)
            .expect("itertools.count TypeDef initialized")
            .as_ptr()
    );
    crate::__pyre_put_new!(
        ns_slot,
        "repeat",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::REPEAT_TYPE)
            .expect("itertools.repeat TypeDef initialized")
            .as_ptr()
    );
    // PyPy exports W_ISlice.typedef itself.  The native object retains its
    // source iterator plus count/next/stop/step cursor state and therefore
    // skips and yields incrementally rather than materializing the result.
    crate::__pyre_put_new!(
        ns_slot,
        "islice",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::ISLICE_TYPE)
            .expect("itertools.islice TypeDef initialized")
            .as_ptr()
    );
    // PyPy W_GroupBy and W_GroupByIterator share the live source cursor.
    crate::__pyre_put_new!(
        ns_slot,
        "groupby",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::GROUPBY_TYPE)
            .expect("itertools.groupby TypeDef initialized")
            .as_ptr()
    );
    crate::__pyre_put_new!(
        ns_slot,
        "_grouper",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::GROUPBY_ITERATOR_TYPE)
            .expect("itertools._grouper TypeDef initialized")
            .as_ptr()
    );
    // PyPy W_TeeIterable copies hold independent cursors into one shared
    // W_TeeChainedListNode chain.
    crate::__pyre_put_new!(
        ns_slot,
        "tee",
        crate::make_builtin_function("tee", crate::typedef::itertools_tee)
    );
    crate::__pyre_put_new!(
        ns_slot,
        "_tee",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::TEE_ITERABLE_TYPE)
            .expect("itertools._tee TypeDef initialized")
            .as_ptr()
    );
    crate::__pyre_put_new!(
        ns_slot,
        "_tee_dataobject",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::TEE_DATAOBJECT_TYPE)
            .expect("itertools._tee_dataobject TypeDef initialized")
            .as_ptr()
    );
    // PyPy W_Permutations: retain the pool, indices, and rollover cycles,
    // yielding one permutation at a time.
    crate::__pyre_put_new!(
        ns_slot,
        "permutations",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::PERMUTATIONS_TYPE)
            .expect("itertools.permutations TypeDef initialized")
            .as_ptr()
    );
    // PyPy W_Combinations: retain pool/index/result state and advance one
    // lexicographic combination at a time.
    crate::__pyre_put_new!(
        ns_slot,
        "combinations",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::COMBINATIONS_TYPE)
            .expect("itertools.combinations TypeDef initialized")
            .as_ptr()
    );
    // PyPy W_CombinationsWithReplacement: retain pool/index/result state and
    // advance one repeated combination at a time.
    crate::__pyre_put_new!(
        ns_slot,
        "combinations_with_replacement",
        crate::typedef::gettypefor(
            &pyre_object::interp_itertools::COMBINATIONS_WITH_REPLACEMENT_TYPE,
        )
        .expect("itertools.combinations_with_replacement TypeDef initialized")
        .as_ptr()
    );
    // PyPy W_Product: retain only the pool snapshots and odometer state rather
    // than eagerly materializing the Cartesian result.
    crate::__pyre_put_new!(
        ns_slot,
        "product",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::PRODUCT_TYPE)
            .expect("itertools.product TypeDef initialized")
            .as_ptr()
    );
    // PyPy exports W_ZipLongest.typedef.  Construction keeps each source as a
    // live iterator, so unbounded inputs remain lazy.
    crate::__pyre_put_new!(
        ns_slot,
        "zip_longest",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::ZIP_LONGEST_TYPE)
            .expect("itertools.zip_longest TypeDef initialized")
            .as_ptr()
    );
    // PyPy exports the live W_Accumulate iterator TypeDef.  Its running total,
    // optional function, and initial value stay on the object and next_w
    // advances the source lazily.
    crate::__pyre_put_new!(
        ns_slot,
        "accumulate",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::ACCUMULATE_TYPE)
            .expect("itertools.accumulate TypeDef initialized")
            .as_ptr()
    );
    // W_Compress.typedef is exported directly, matching PyPy's dedicated
    // live iterator rather than materializing both inputs into a list.
    crate::__pyre_put_new!(
        ns_slot,
        "compress",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::COMPRESS_TYPE)
            .expect("itertools.compress TypeDef initialized")
            .as_ptr()
    );
    // PyPy exposes these W_Root subclasses through their TypeDefs.  Their
    // `__new__` slots retain the two-argument/subclass-init gateway behavior.
    crate::__pyre_put_new!(
        ns_slot,
        "takewhile",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::TAKEWHILE_TYPE)
            .expect("itertools.takewhile TypeDef initialized")
            .as_ptr()
    );
    crate::__pyre_put_new!(
        ns_slot,
        "dropwhile",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::DROPWHILE_TYPE)
            .expect("itertools.dropwhile TypeDef initialized")
            .as_ptr()
    );
    crate::__pyre_put_new!(
        ns_slot,
        "filterfalse",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::FILTERFALSE_TYPE)
            .expect("itertools.filterfalse TypeDef initialized")
            .as_ptr()
    );
    // PyPy exports the native W_Pairwise / W_Cycle TypeDefs. Their __new__
    // slots allocate the requested subtype and retain a live source iterator.
    crate::__pyre_put_new!(
        ns_slot,
        "pairwise",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::PAIRWISE_TYPE)
            .expect("itertools.pairwise TypeDef initialized")
            .as_ptr()
    );
    crate::__pyre_put_new!(
        ns_slot,
        "cycle",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::CYCLE_TYPE)
            .expect("itertools.cycle TypeDef initialized")
            .as_ptr()
    );
    // CPython 3.14 exports the live `batched` iterator TypeDef.  The source is
    // consumed only when `__next__` requests one batch.
    crate::__pyre_put_new!(
        ns_slot,
        "batched",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::BATCHED_TYPE)
            .expect("itertools.batched TypeDef initialized")
            .as_ptr()
    );
    Ok(())
}
