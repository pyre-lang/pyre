/// Widening integer storage keeps every address bit (`rbuiltin.py` `gen_cast`).
#[test]
fn int_scalar_width_covers_word_accepts_widening_integer() {
    assert!(super::int_scalar_width_covers_word("U64", 8));
    assert!(super::int_scalar_width_covers_word("I64", 8));
    assert!(super::int_scalar_width_covers_word("U128", 8));
    assert!(!super::int_scalar_width_covers_word("U32", 8));
    assert!(super::int_scalar_width_covers_word("U64", 4));
    assert!(super::int_scalar_width_covers_word("U32", 4));
    assert!(!super::int_scalar_width_covers_word("U16", 4));
}

/// `a` calls `b` and `c`, both call `d`. `e` calls `f`, and `f` calls `e` and `d`.
#[test]
fn address_escape_walk_memo_matches_an_unmemoized_walk() {
    use super::{AddressEscapeWalk, function_address_escape, plain_local};
    use majit_charon_reader::Llbc;
    use serde_json::{Value, json};

    const A: u64 = 0;
    const B: u64 = 1;
    const C: u64 = 2;
    const D: u64 = 3;
    const E: u64 = 4;
    const F: u64 = 5;

    fn span() -> Value {
        json!({"Deduplicated": 0})
    }
    fn local_place(index: u64) -> Value {
        json!({"kind": {"Local": index}, "ty": {"Deduplicated": 0}})
    }
    fn bb(kind: Value) -> Value {
        json!({
            "statements": [],
            "kind": "Regular",
            "terminator": {"span": span(), "kind": kind}
        })
    }
    fn call_term(fun: u64, target: u64, on_unwind: u64) -> Value {
        json!({
            "Call": {
                "call": {
                    "func": {"Regular": {"kind": {"Fun": fun}, "generics": {}}},
                    "args": [{"Move": local_place(1)}]
                },
                "dest": local_place(0),
                "target": target,
                "on_unwind": on_unwind
            }
        })
    }
    fn call_then_return(fun: u64) -> Vec<Value> {
        vec![
            bb(call_term(fun, 1, 2)),
            bb(json!("Return")),
            bb(json!("UnwindResume")),
        ]
    }
    fn fun_decl(def_id: u64, name: &str, body: Vec<Value>) -> Value {
        json!({
            "def_id": def_id,
            "item_meta": {
                "name": [{"Ident": [name, 0]}],
                "span": span(),
                "source_text": null,
                "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                "is_local": true
            },
            "signature": {
                "is_unsafe": false,
                "inputs": [{"Deduplicated": 0}],
                "output": {"Deduplicated": 0}
            },
            "body": {"Unstructured": {
                "span": span(),
                "locals": {
                    "arg_count": 1,
                    "locals": [
                        {"index": 0, "name": null, "span": span(), "ty": {"Deduplicated": 0}},
                        {"index": 1, "name": "p", "span": span(), "ty": {"Deduplicated": 0}}
                    ]
                },
                "body": body
            }}
        })
    }

    let file = json!({
        "charon_version": "t",
        "has_errors": false,
        "translated": {
            "crate_name": "c",
            "fun_decls": [
                fun_decl(A, "a", vec![
                    bb(call_term(B, 1, 3)),
                    bb(call_term(C, 2, 3)),
                    bb(json!("Return")),
                    bb(json!("UnwindResume")),
                ]),
                fun_decl(B, "b", call_then_return(D)),
                fun_decl(C, "c", call_then_return(D)),
                fun_decl(D, "d", vec![bb(json!("Return"))]),
                fun_decl(E, "e", call_then_return(F)),
                fun_decl(F, "f", vec![
                    bb(call_term(E, 1, 3)),
                    bb(call_term(D, 2, 3)),
                    bb(json!("Return")),
                    bb(json!("UnwindResume")),
                ]),
            ],
            "files": []
        }
    });
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture parses");
    let entry = [plain_local(1, 1, 0)];

    let mut memo = AddressEscapeWalk::new();
    let a_memo = function_address_escape(&llbc, A, &entry, &mut memo);
    let e_memo = function_address_escape(&llbc, E, &entry, &mut memo);
    assert_eq!(memo.walk_count(D), 1);

    let mut raw_a = AddressEscapeWalk::without_memo();
    let a_raw = function_address_escape(&llbc, A, &entry, &mut raw_a);
    let mut raw_e = AddressEscapeWalk::without_memo();
    let e_raw = function_address_escape(&llbc, E, &entry, &mut raw_e);
    assert_eq!(a_memo, a_raw);
    assert_eq!(e_memo, e_raw);
    assert!(raw_a.walk_count(D) > 1);
}
