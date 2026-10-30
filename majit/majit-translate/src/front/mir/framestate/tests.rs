use crate::front::mir::tests::fixture_llbc;

#[test]
fn liveness_marks_loop_carried_index_local_as_live_in() {
    use majit_charon_reader::ullbc::Unstructured;
    // A self-loop whose only read of the loop-carried local `_2` is
    // as an `Index` offset (`_3 = _1[_2]`); `_2` is redefined on the
    // back edge from `_3`, so it is never read as a plain operand.
    // The index offset lives in the projection element, not in the
    // projected place's `inner`, so a liveness pass that descends
    // only into `inner` misses it: `_2` is not marked live-in at the
    // loop block, the monotonic lowering mints no block inputarg for
    // it, and the loop-header read fails loud as an uninitialised
    // local (#176).  `mark_projection_index_offset_use` closes the
    // gap — assert `_2` is live-in at bb1.
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 0, "col": 0},
                "end": {"line": 0, "col": 0}
            },
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place_local = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let copy_local = |i: u64| serde_json::json!({"Copy": place_local(i)});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    // bb0:  _2 = const 0;  goto bb1
    let bb0 = serde_json::json!({
        "statements": [stmt(serde_json::json!({
            "Assign": [place_local(2), {"Use": [{"Const": null}, "No"]}]
        }))],
        "kind": "Regular",
        "terminator": {"kind": {"Goto": {"target": 1}}}
    });
    // bb1:  _3 = _1[_2];  _2 = _3;  goto bb1   (self-loop)
    let read_index = stmt(serde_json::json!({
        "Assign": [
            place_local(3),
            {"Use": [{"Copy": {
                "kind": {"Projection": [
                    place_local(1),
                    {"Index": {"offset": copy_local(2), "from_end": false}}
                ]},
                "ty": ty()
            }}, "No"]}
        ]
    }));
    let carry = stmt(serde_json::json!({
        "Assign": [place_local(2), {"Use": [copy_local(3), "No"]}]
    }));
    let bb1 = serde_json::json!({
        "statements": [read_index, carry],
        "kind": "Regular",
        "terminator": {"kind": {"Goto": {"target": 1}}}
    });
    let body_json = serde_json::json!({
        "span": span(),
        "locals": {
            "arg_count": 1,
            "locals": [local(0), local(1), local(2), local(3)]
        },
        "body": [bb0, bb1]
    });
    let body: Unstructured =
        serde_json::from_value(body_json).expect("fixture Unstructured parses");
    // Base dataflow only: this case is closed by
    // `mark_projection_index_offset_use` in the statement scan, not by
    // the index-write `extra_live` set, so pass an empty extra_live.
    let no_cells = bit_set::BitSet::new();
    let live = super::compute_mir_liveness(
        &fixture_llbc(),
        &body,
        &[],
        &bit_set::BitSet::new(),
        &no_cells,
    );
    assert!(
        live[1].contains(2),
        "loop-carried index local _2 must be live-in at the loop block bb1"
    );
    // Sanity: the array base _1 (read as the projection inner) is
    // also live-in, and the throwaway temp _3 (defined before its
    // only use within bb1) is not.
    assert!(live[1].contains(1), "array base _1 must be live-in at bb1");
    assert!(
        !live[1].contains(3),
        "temp _3 is block-local, not live-in at bb1"
    );
}

#[test]
fn liveness_marks_offset_bytes_local_as_live_in() {
    use majit_charon_reader::ullbc::Unstructured;
    // `_2` is defined in bb0 and read only as `Offset.bytes` in bb1
    // (`_3 = (*_1).offset(_2)`).  The byte count lives in the
    // projection element, not in `inner`, so a liveness pass that
    // only walks `Index.offset` misses it: `_2` is not a live-in of
    // bb1 and lowering mints no block inputarg.  Upstream
    // `_compute_liveness_must_continue` marks every register
    // argument alive.  `mark_projection_index_offset_use` must
    // treat `Offset.bytes` like `Index.offset`.
    let span = || {
        serde_json::json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 0, "col": 0},
                "end": {"line": 0, "col": 0}
            },
            "generated_from_span": null
        })
    };
    let ty = || serde_json::json!({"Deduplicated": 0});
    let place_local = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
    let copy_local = |i: u64| serde_json::json!({"Copy": place_local(i)});
    let local = |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
    let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
    let bb0 = serde_json::json!({
        "statements": [stmt(serde_json::json!({
            "Assign": [place_local(2), {"Use": [{"Const": null}, "No"]}]
        }))],
        "kind": "Regular",
        "terminator": {"kind": {"Goto": {"target": 1}}}
    });
    let read_offset = stmt(serde_json::json!({
        "Assign": [
            place_local(3),
            {"Use": [{"Copy": {
                "kind": {"Projection": [
                    place_local(1),
                    {"Offset": {"bytes": copy_local(2), "in_bounds": true, "ty_is_sized": true}}
                ]},
                "ty": ty()
            }}, "No"]}
        ]
    }));
    let bb1 = serde_json::json!({
        "statements": [read_offset],
        "kind": "Regular",
        "terminator": {"kind": "Return"}
    });
    let body_json = serde_json::json!({
        "span": span(),
        "locals": {
            "arg_count": 1,
            "locals": [local(0), local(1), local(2), local(3)]
        },
        "body": [bb0, bb1]
    });
    let body: Unstructured =
        serde_json::from_value(body_json).expect("fixture Unstructured parses");
    let no_cells = bit_set::BitSet::new();
    let live = super::compute_mir_liveness(
        &fixture_llbc(),
        &body,
        &[],
        &bit_set::BitSet::new(),
        &no_cells,
    );
    assert!(
        live[1].contains(2),
        "Offset.bytes local _2 must be live-in at bb1"
    );
    assert!(
        live[1].contains(1),
        "pointer base _1 must be live-in at bb1"
    );
    assert!(
        !live[1].contains(3),
        "temp _3 is block-local, not live-in at bb1"
    );
}
