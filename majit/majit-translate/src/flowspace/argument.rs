//! Arguments objects.
//!
//! RPython basis: `rpython/flowspace/argument.py` (127 LOC).
//!
//! `Signature` landed first because `bytecode.HostCode::signature`
//! required it. This extension adds `CallSpec`, `_rawshape`, `flatten`,
//! `as_list`, and `fromshape`. Order matches upstream `argument.py`:
//! `Signature` first, then `CallSpec`.

/// Descriptor for a function's formal parameter list.
///
/// RPython basis: `rpython/flowspace/argument.py:Signature`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    pub argnames: Vec<String>,
    pub varargname: Option<String>,
    pub kwargname: Option<String>,
}

impl Signature {
    /// RPython: `Signature.__init__`.
    pub fn new(
        argnames: Vec<String>,
        varargname: Option<String>,
        kwargname: Option<String>,
    ) -> Self {
        Self {
            argnames,
            varargname,
            kwargname,
        }
    }

    /// RPython: `Signature.find_argname`.
    pub fn find_argname(&self, name: &str) -> i32 {
        match self.argnames.iter().position(|n| n == name) {
            Some(idx) => idx as i32,
            None => -1,
        }
    }

    /// RPython: `Signature.num_argnames`.
    pub fn num_argnames(&self) -> usize {
        self.argnames.len()
    }

    /// RPython: `Signature.has_vararg`.
    pub fn has_vararg(&self) -> bool {
        self.varargname.is_some()
    }

    /// RPython: `Signature.has_kwarg`.
    pub fn has_kwarg(&self) -> bool {
        self.kwargname.is_some()
    }

    /// RPython: `Signature.scope_length`.
    pub fn scope_length(&self) -> usize {
        self.argnames.len() + usize::from(self.has_vararg()) + usize::from(self.has_kwarg())
    }

    /// RPython: `Signature.getallvarnames`.
    pub fn getallvarnames(&self) -> Vec<String> {
        let mut out = self.argnames.clone();
        if let Some(ref v) = self.varargname {
            out.push(v.clone());
        }
        if let Some(ref k) = self.kwargname {
            out.push(k.clone());
        }
        out
    }
    /// Tuple-shaped accessor matching upstream's `(argnames,
    /// varargname, kwargname)` protocol consumed by the annotator.
    pub fn tuple_view(&self) -> (&[String], Option<&str>, Option<&str>) {
        (
            &self.argnames,
            self.varargname.as_deref(),
            self.kwargname.as_deref(),
        )
    }

    /// RPython `Signature.__len__` (argument.py).
    pub fn len_tuple(&self) -> usize {
        3
    }

    /// RPython `Signature.__getitem__` (argument.py).
    #[expect(dead_code)] // argument.py Signature.__getitem__; Rust callers use tuple_view today
    pub(crate) fn getitem(&self, i: usize) -> SignatureItem<'_> {
        match i {
            0 => SignatureItem::Argnames(&self.argnames),
            1 => SignatureItem::Varargname(self.varargname.as_deref()),
            2 => SignatureItem::Kwargname(self.kwargname.as_deref()),
            _ => panic!("IndexError"),
        }
    }
}

/// Item returned by [`Signature::getitem`].
///
/// Kept for RPython `Signature.__getitem__` parity even though current Rust
/// callers prefer [`Signature::tuple_view`].
#[expect(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SignatureItem<'a> {
    Argnames(&'a [String]),
    Varargname(Option<&'a str>),
    Kwargname(Option<&'a str>),
}

use super::model::{Constant, Hlvalue};
use std::collections::HashMap;

/// Shape key of a call site: `(shape_cnt, shape_keys, shape_star)`.
///
/// RPython basis: `CallSpec._rawshape()` returns `(int, tuple-of-str,
/// bool)`; we mirror the three fields explicitly. `shape_keys` is
/// kept sorted (upstream uses `tuple(sorted(self.keywords))`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CallShape {
    /// Number of positional arguments — RPython `shape_cnt`.
    pub shape_cnt: usize,
    /// Sorted keyword names — RPython `shape_keys` (`tuple(sorted(...))`).
    pub shape_keys: Vec<String>,
    /// Whether a `*args` slot is present — RPython `shape_star`.
    pub shape_star: bool,
}

/// Arguments passed into a function call site: the `a, b, *c, **d`
/// part in `return func(a, b, *c, **d)`.
///
/// RPython basis: `argument.py` — `class CallSpec`.
///
/// Upstream stores `keywords` as a plain `dict`; its iteration order
/// is unspecified. `_rawshape()` compensates with
/// `tuple(sorted(self.keywords))`. Rust mirrors that storage model
/// with `HashMap<String, Hlvalue>` and performs the same explicit
/// sort in `rawshape()` so `shape_keys` is always the sorted key
/// tuple. The storage itself carries no order, matching upstream
/// semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallSpec {
    /// RPython `CallSpec.arguments_w`.
    pub arguments_w: Vec<Hlvalue>,
    /// RPython `CallSpec.keywords` — plain `HashMap` mirroring
    /// upstream's unordered `dict`.
    pub keywords: HashMap<String, Hlvalue>,
    /// RPython `CallSpec.w_stararg`.
    pub w_stararg: Option<Hlvalue>,
}

impl CallSpec {
    /// RPython `CallSpec.__init__(args_w, keywords=None, w_stararg=None)`.
    pub fn new(
        arguments_w: Vec<Hlvalue>,
        keywords: Option<HashMap<String, Hlvalue>>,
        w_stararg: Option<Hlvalue>,
    ) -> Self {
        CallSpec {
            arguments_w,
            keywords: keywords.unwrap_or_default(),
            w_stararg,
        }
    }

    /// RPython `CallSpec._rawshape()`.
    ///
    /// Upstream: `shape_keys = tuple(sorted(self.keywords))`. We sort
    /// explicitly since `HashMap::keys()` has no guaranteed order.
    pub fn rawshape(&self) -> CallShape {
        let mut shape_keys: Vec<String> = self.keywords.keys().cloned().collect();
        shape_keys.sort();
        CallShape {
            shape_cnt: self.arguments_w.len(),
            shape_keys,
            shape_star: self.w_stararg.is_some(),
        }
    }

    /// RPython `CallSpec.flatten()` — "Argument <-> list of w_objects
    /// together with 'shape' information".
    pub fn flatten(&self) -> (CallShape, Vec<Hlvalue>) {
        let shape = self.rawshape();
        let mut data_w: Vec<Hlvalue> = self.arguments_w.clone();
        for key in &shape.shape_keys {
            // `shape.shape_keys` came from `self.keywords.keys()`;
            // `unwrap` matches upstream's `self.keywords[key]` direct
            // indexing.
            data_w.push(self.keywords.get(key).unwrap().clone());
        }
        if shape.shape_star {
            data_w.push(self.w_stararg.clone().unwrap());
        }
        (shape, data_w)
    }

    /// RPython `argument.py` — `CallSpec.as_list`.
    ///
    /// ```python
    /// def as_list(self):
    ///     assert not self.keywords
    ///     if self.w_stararg is None:
    ///         return self.arguments_w
    ///     else:
    ///         return self.arguments_w + [const(x) for x in self.w_stararg.value]
    /// ```
    ///
    /// The `for x in self.w_stararg.value` iteration uses Python's
    /// general iterator protocol. Rust port mirrors that contract via
    /// `ConstValue::iter_items`, which covers tuple/list/str/dict for
    /// the variants flow-space currently constructs.
    pub fn as_list(&self) -> Vec<Hlvalue> {
        assert!(
            self.keywords.is_empty(),
            "as_list() called with keywords present"
        );
        match &self.w_stararg {
            None => self.arguments_w.clone(),
            Some(Hlvalue::Constant(stararg)) => {
                let items = stararg.value.iter_items().unwrap_or_else(|| {
                    panic!(
                        "CallSpec.as_list expected iterable Constant for w_stararg, got {:?}",
                        stararg.value
                    )
                });
                let mut out = self.arguments_w.clone();
                out.extend(
                    items
                        .into_iter()
                        .map(|value| Hlvalue::Constant(Constant::new(value))),
                );
                out
            }
            Some(other) => panic!("CallSpec.as_list expected Constant stararg, got {other:?}"),
        }
    }

    /// RPython `CallSpec.fromshape(cls, (shape_cnt, shape_keys,
    /// shape_star), data_w)`.
    pub fn fromshape(shape: &CallShape, data_w: Vec<Hlvalue>) -> Self {
        let shape_cnt = shape.shape_cnt;
        let end_keys = shape_cnt + shape.shape_keys.len();
        let mut p = end_keys;
        let args_w: Vec<Hlvalue> = data_w[..shape_cnt].to_vec();
        let keyword_slice = &data_w[shape_cnt..end_keys];
        let mut keywords: HashMap<String, Hlvalue> = HashMap::new();
        for (name, value) in shape.shape_keys.iter().zip(keyword_slice.iter()) {
            keywords.insert(name.clone(), value.clone());
        }
        let w_stararg = if shape.shape_star {
            let v = data_w[p].clone();
            p += 1;
            Some(v)
        } else {
            None
        };
        let _ = p; // `p` is post-increment upstream; assign-read-drop matches.
        CallSpec {
            arguments_w: args_w,
            keywords,
            w_stararg,
        }
    }
}

#[cfg(test)]
mod test {
    use super::super::model::{ConstValue, Constant, Hlvalue};
    use super::*;

    // RPython basis: no upstream test for Signature alone — `test_argument.py`
    // exercises `CallSpec`. These smoke tests pin the behaviours that
    // `bytecode::HostCode::signature` consumes.

    #[test]
    fn signature_basic_arity() {
        let sig = Signature::new(vec!["x".into(), "y".into()], None, None);
        assert_eq!(sig.num_argnames(), 2);
        assert!(!sig.has_vararg());
        assert!(!sig.has_kwarg());
        assert_eq!(sig.scope_length(), 2);
        assert_eq!(sig.find_argname("y"), 1);
        assert_eq!(sig.find_argname("z"), -1);
    }

    #[test]
    fn signature_with_varargs_and_kwargs() {
        let sig = Signature::new(vec!["a".into()], Some("args".into()), Some("kw".into()));
        assert_eq!(sig.scope_length(), 3);
        assert_eq!(
            sig.getallvarnames(),
            vec!["a".to_string(), "args".into(), "kw".into()]
        );
    }

    #[test]
    fn signature_tuple_protocol_indexes_three_fields() {
        let sig = Signature::new(vec!["a".into()], Some("args".into()), Some("kw".into()));
        assert_eq!(sig.len_tuple(), 3);
        assert_eq!(sig.tuple_view().0, ["a"]);
        assert_eq!(sig.tuple_view().1, Some("args"));
        assert_eq!(sig.tuple_view().2, Some("kw"));
    }

    // CallSpec

    // Line-by-line port of `test_argument.py:test_flatten_CallSpec`.
    // Upstream uses plain Python ints (e.g. `CallSpec([1, 2, 3])`).
    // Rust wraps each int in `Hlvalue::Constant(Constant(Int(_)))` —
    // `CallSpec` is type-neutral otherwise.

    fn hi(n: i64) -> Hlvalue {
        Hlvalue::Constant(Constant::new(ConstValue::Int(n)))
    }

    fn kw(pairs: &[(&str, i64)]) -> HashMap<String, Hlvalue> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), hi(*v)))
            .collect()
    }

    #[test]
    fn flatten_positional_only() {
        let args = CallSpec::new(vec![hi(1), hi(2), hi(3)], None, None);
        let (shape, data) = args.flatten();
        assert_eq!(shape.shape_cnt, 3);
        assert!(shape.shape_keys.is_empty());
        assert!(!shape.shape_star);
        assert_eq!(data, vec![hi(1), hi(2), hi(3)]);
    }

    #[test]
    fn flatten_positional_and_keyword() {
        let args = CallSpec::new(vec![hi(1)], Some(kw(&[("b", 2), ("c", 3)])), None);
        let (shape, data) = args.flatten();
        assert_eq!(shape.shape_cnt, 1);
        assert_eq!(shape.shape_keys, vec!["b".to_string(), "c".into()]);
        assert!(!shape.shape_star);
        assert_eq!(data, vec![hi(1), hi(2), hi(3)]);
    }

    #[test]
    fn flatten_sorts_keyword_order() {
        // Upstream: `CallSpec([1, 2, 3, 4, 5], {'e': 5, 'd': 7})`
        // flattens to `((5, ('d', 'e'), False), [1, 2, 3, 4, 5, 7, 5])`.
        let args = CallSpec::new(
            vec![hi(1), hi(2), hi(3), hi(4), hi(5)],
            Some(kw(&[("e", 5), ("d", 7)])),
            None,
        );
        let (shape, data) = args.flatten();
        assert_eq!(shape.shape_cnt, 5);
        assert_eq!(shape.shape_keys, vec!["d".to_string(), "e".into()]);
        assert!(!shape.shape_star);
        assert_eq!(data, vec![hi(1), hi(2), hi(3), hi(4), hi(5), hi(7), hi(5)]);
    }

    #[test]
    fn fromshape_is_flatten_inverse() {
        let args = CallSpec::new(
            vec![hi(1), hi(2)],
            Some(kw(&[("a", 10), ("b", 20)])),
            Some(hi(99)),
        );
        let (shape, data) = args.flatten();
        let round_trip = CallSpec::fromshape(&shape, data);
        assert_eq!(round_trip, args);
    }

    #[test]
    fn as_list_without_keywords_or_star_returns_positionals() {
        let args = CallSpec::new(vec![hi(1), hi(2)], None, None);
        assert_eq!(args.as_list(), vec![hi(1), hi(2)]);
    }

    #[test]
    fn as_list_unpacks_star_tuple() {
        let star = Hlvalue::Constant(Constant::new(ConstValue::Tuple(vec![
            ConstValue::Int(2),
            ConstValue::Int(3),
        ])));
        let args = CallSpec::new(vec![hi(1)], None, Some(star));
        assert_eq!(args.as_list(), vec![hi(1), hi(2), hi(3)]);
    }

    #[test]
    #[should_panic(expected = "keywords present")]
    fn as_list_rejects_keywords() {
        let args = CallSpec::new(vec![hi(1)], Some(kw(&[("x", 1)])), None);
        let _ = args.as_list();
    }
}
