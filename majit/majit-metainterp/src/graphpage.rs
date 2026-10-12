//! RPython parity module for `rpython/jit/metainterp/graphpage.py`.
//!
//! The upstream module is a debugging/viewer helper: it turns optimized
//! `ResOperation` procedures into Graphviz DOT and opens `dotviewer`.
//! Pyre does not embed `dotviewer`, so this port preserves the module and
//! symbol surface while returning DOT source for callers/tests to inspect.

use majit_ir::{DescrRef, Op, OpCode, OpRef, Type, Value, descr_identity};

const BOX_COLOR: (u8, u8, u8) = (128, 0, 96);

#[derive(Clone, Debug, PartialEq, Eq)]
struct LinkInfo {
    label: String,
    color: (u8, u8, u8),
}

#[derive(Default)]
struct ResOpMemo {
    names: Vec<(OpRef, String)>,
}

impl ResOpMemo {
    // graphpage.py:69 keys `memo` by box object (resoperation.py repr_short names  allow-line-citation
    // by object identity). This viewer runs post-optimization on owned `Op`
    // values, where args are `Operand`s but a result is only its `op.pos`
    // `OpRef` (no producer `Rc` to key on). `OpRef` is the identity both sides
    // share: a producer position is unique to one op, and bound inputargs /
    // constants shed to their canonical `OpRef`, so a result named `iN` and its
    // downstream arg uses all resolve to the same key — reproducing upstream's
    // one-name-per-box sharing in pyre's position-threaded representation.
    // Two distinct producers can never share a position, so the only inputs
    // `OpRef`-keying collapses (distinct position-only boxes at one position)
    // cannot arise in a real trace — the deviation is unobservable.
    pub fn get(&self, key: OpRef) -> Option<&str> {
        self.names
            .iter()
            .find_map(|(k, v)| (*k == key).then_some(v.as_str()))
    }

    pub fn set(&mut self, key: OpRef, value: String) {
        if let Some((_, old)) = self.names.iter_mut().find(|(k, _)| *k == key) {
            *old = value;
        } else {
            self.names.push((key, value));
        }
    }
}

trait ResOpProcedure {
    fn get_operations(&self) -> &[Op];

    fn get_display_text(&self, _memo: &mut ResOpMemo) -> Option<String> {
        None
    }
}

impl ResOpProcedure for Vec<Op> {
    fn get_operations(&self) -> &[Op] {
        self.as_slice()
    }
}

impl ResOpProcedure for &[Op] {
    fn get_operations(&self) -> &[Op] {
        self
    }
}

pub struct SubGraph {
    failargs: Vec<OpRef>,
    subinputargs: Vec<OpRef>,
    suboperations: Vec<Op>,
}

impl SubGraph {
    pub fn new(failargs: Vec<OpRef>, subinputargs: Vec<OpRef>, suboperations: Vec<Op>) -> Self {
        Self {
            failargs,
            subinputargs,
            suboperations,
        }
    }
}

impl ResOpProcedure for SubGraph {
    fn get_operations(&self) -> &[Op] {
        &self.suboperations
    }

    fn get_display_text(&self, memo: &mut ResOpMemo) -> Option<String> {
        for (failarg, inputarg) in self.failargs.iter().zip(self.subinputargs.iter()) {
            if let Some(name) = memo.get(*failarg).map(str::to_string) {
                memo.set(*inputarg, name);
            }
        }
        None
    }
}

pub fn display_procedures(
    procedures: &[&[Op]],
    errmsg: Option<&str>,
    highlight_procedures: &[u8],
) -> String {
    let graphs = procedures
        .iter()
        .enumerate()
        .map(|(i, procedure)| {
            (
                procedure as &dyn ResOpProcedure,
                *highlight_procedures.get(i).unwrap_or(&0),
            )
        })
        .collect::<Vec<_>>();
    let mut graphpage = ResOpGraphPage::default();
    graphpage.compute(&graphs, errmsg);
    graphpage.source
}

pub fn is_interesting_guard(_op: &Op) -> bool {
    // PyPy checks for `op.getdescr()._debug_suboperations`. Pyre does not
    // currently attach that viewer-only bridge payload to fail descrs.
    false
}

pub fn getdescr(op: &Op) -> Option<DescrRef> {
    op.getdescr()
}

#[derive(Default)]
pub struct ResOpGraphPage {
    pub source: String,
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    pub links: Vec<(String, (String, (u8, u8, u8)))>,
}

impl ResOpGraphPage {
    fn compute(&mut self, graphs: &[(&dyn ResOpProcedure, u8)], errmsg: Option<&str>) {
        let mut resopgen = ResOpGen::new(None);
        for (graph, highlight) in graphs {
            resopgen.add_graph(*graph, *highlight);
        }
        if let Some(errmsg) = errmsg {
            resopgen.set_errmsg(errmsg.to_string());
        }
        self.source = resopgen.getsource();
        self.links = resopgen.getlinks();
    }
}

struct GraphEntry<'a> {
    procedure: &'a dyn ResOpProcedure,
    highlight: u8,
}

pub struct ResOpGen<'a> {
    graphs: Vec<GraphEntry<'a>>,
    block_starters: Vec<Vec<usize>>,
    all_operations: Vec<(usize, (usize, usize))>,
    errmsg: Option<String>,
    target_tokens: Vec<(usize, (usize, usize))>,
    metainterp_sd: Option<()>,
    memo: ResOpMemo,
    pendingedges: Vec<(String, String, EdgeAttrs)>,
    dotgen: Option<DotGen>,
}

impl<'a> ResOpGen<'a> {
    pub const CLUSTERING: bool = true;
    pub const BOX_COLOR: (u8, u8, u8) = BOX_COLOR;

    pub fn new(metainterp_sd: Option<()>) -> Self {
        Self {
            graphs: Vec::new(),
            block_starters: Vec::new(),
            all_operations: Vec::new(),
            errmsg: None,
            target_tokens: Vec::new(),
            metainterp_sd,
            memo: ResOpMemo::default(),
            pendingedges: Vec::new(),
            dotgen: None,
        }
    }

    pub fn op_name(&self, graphindex: usize, opindex: usize) -> String {
        format!("g{graphindex}op{opindex}")
    }

    pub fn mark_starter(&mut self, graphindex: usize, opindex: usize) {
        let starters = &mut self.block_starters[graphindex];
        if !starters.contains(&opindex) {
            starters.push(opindex);
        }
    }

    fn add_graph(&mut self, graph: &'a dyn ResOpProcedure, highlight: u8) {
        let graphindex = self.graphs.len();
        self.graphs.push(GraphEntry {
            procedure: graph,
            highlight,
        });
        for (i, op) in graph.get_operations().iter().enumerate() {
            self.all_operations
                .push((op as *const Op as usize, (graphindex, i)));
        }
    }

    pub fn find_starters(&mut self) {
        self.block_starters = (0..self.graphs.len()).map(|_| vec![0]).collect();
        for graphindex in 0..self.graphs.len() {
            let mut mergepointblock = None;
            let operations = self.graphs[graphindex].procedure.get_operations();
            for (i, op) in operations.iter().enumerate() {
                if is_interesting_guard(op) {
                    self.mark_starter(graphindex, i + 1);
                }
                if op.opcode == OpCode::DebugMergePoint {
                    if mergepointblock.is_none() {
                        mergepointblock = Some(i);
                    }
                } else if op.opcode == OpCode::Label {
                    self.mark_starter(graphindex, i);
                    if let Some(descr) = getdescr(op) {
                        self.target_tokens
                            .push((descr_identity(&descr), (graphindex, i)));
                    }
                    mergepointblock = Some(i);
                } else if let Some(block) = mergepointblock.take() {
                    self.mark_starter(graphindex, block);
                }
            }
        }
        for starters in &mut self.block_starters {
            starters.sort_unstable();
        }
    }

    pub fn set_errmsg(&mut self, errmsg: String) {
        self.errmsg = Some(errmsg);
    }

    pub fn getsource(&mut self) -> String {
        self.find_starters();
        self.pendingedges.clear();
        self.dotgen = Some(DotGen::new("resop"));
        self.emit(r#"clusterrank="local""#);
        self.generrmsg();
        for graphindex in 0..self.graphs.len() {
            self.gengraph(graphindex);
        }
        for (from, to, attrs) in std::mem::take(&mut self.pendingedges) {
            self.dotgen
                .as_mut()
                .expect("dotgen initialized")
                .emit_edge(&from, &to, attrs);
        }
        self.dotgen.take().expect("dotgen initialized").generate()
    }

    fn emit(&mut self, line: impl Into<String>) {
        self.dotgen.as_mut().expect("dotgen initialized").emit(line);
    }

    fn generrmsg(&mut self) {
        let Some(errmsg) = self.errmsg.clone() else {
            return;
        };
        self.dotgen
            .as_mut()
            .expect("dotgen initialized")
            .emit_node("errmsg", NodeAttrs::new("box", &errmsg).fillcolor("red"));
        if !self.graphs.is_empty() && !self.block_starters[0].is_empty() {
            let opindex = *self.block_starters[0].iter().max().unwrap();
            let blockname = self.op_name(0, opindex);
            self.pendingedges
                .push((blockname, "errmsg".to_string(), EdgeAttrs::default()));
        }
    }

    pub fn getgraphname(&self, graphindex: usize) -> String {
        format!("graph{graphindex}")
    }

    fn gengraph(&mut self, graphindex: usize) {
        let graphname = self.getgraphname(graphindex);
        if Self::CLUSTERING {
            self.emit(format!("subgraph cluster{graphindex} {{"));
        }
        let label = self.graphs[graphindex]
            .procedure
            .get_display_text(&mut self.memo);
        if let Some(label) = label {
            let fillcolor = match self.graphs[graphindex].highlight {
                1 => "#f084c2",
                2 => "#808080",
                _ => "#84f0c2",
            };
            self.dotgen.as_mut().expect("dotgen initialized").emit_node(
                &graphname,
                NodeAttrs::new("octagon", &label).fillcolor(fillcolor),
            );
            self.pendingedges
                .push((graphname, self.op_name(graphindex, 0), EdgeAttrs::default()));
        }
        let starters = self.block_starters[graphindex].clone();
        for opindex in starters {
            self.genblock(graphindex, opindex);
        }
        if Self::CLUSTERING {
            self.emit("}");
        }
    }

    fn genedge(&mut self, from: (usize, usize), to: (usize, usize), attrs: EdgeAttrs) {
        self.pendingedges.push((
            self.op_name(from.0, from.1),
            self.op_name(to.0, to.1),
            attrs,
        ));
    }

    fn genblock(&mut self, graphindex: usize, opstartindex: usize) {
        let operations = self.graphs[graphindex].procedure.get_operations();
        if opstartindex >= operations.len() {
            return;
        }
        let blockname = self.op_name(graphindex, opstartindex);
        let block_starters = self.block_starters[graphindex].clone();
        let mut lines = Vec::new();
        let mut opindex = opstartindex;
        loop {
            let op = &operations[opindex];
            lines.push(self.repr_op(op));
            opindex += 1;
            if opindex >= operations.len() {
                break;
            }
            if block_starters.contains(&opindex) {
                self.genedge(
                    (graphindex, opstartindex),
                    (graphindex, opindex),
                    EdgeAttrs::default(),
                );
                break;
            }
        }
        let op = &operations[opindex.saturating_sub(1)];
        if op.opcode == OpCode::Jump
            && let Some(descr) = getdescr(op)
        {
            let key = descr_identity(&descr);
            if let Some((_, target)) = self.target_tokens.iter().find(|(k, _)| *k == key) {
                self.genedge(
                    (graphindex, opstartindex),
                    *target,
                    EdgeAttrs::default().weight("0"),
                );
            }
        }
        lines.push(String::new());
        let label = lines.join(r"\l");
        self.dotgen
            .as_mut()
            .expect("dotgen initialized")
            .emit_node(&blockname, NodeAttrs::new("box", &label));
    }

    fn repr_op(&self, op: &Op) -> String {
        if op.opcode == OpCode::DebugMergePoint && self.metainterp_sd.is_some() {
            // PyPy asks warmstate for a printable location here. Pyre does not
            // expose the equivalent metainterp_sd viewer hook yet, so keep the
            // operation's regular display until that hook exists.
        }
        format!("{op}")
    }

    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    fn getlinks(&mut self) -> Vec<(String, (String, (u8, u8, u8)))> {
        let mut links = Vec::new();
        for graphindex in 0..self.graphs.len() {
            let operations = self.graphs[graphindex].procedure.get_operations();
            for op in operations {
                for arg in op.getarglist() {
                    self.add_link_for_box(&mut links, arg.to_opref());
                }
                if op.result_type() != Type::Void && !op.pos().get().is_none() {
                    self.add_link_for_box(&mut links, op.pos().get());
                }
            }
        }
        links
    }

    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    fn add_link_for_box(&mut self, links: &mut Vec<(String, (String, (u8, u8, u8)))>, box_: OpRef) {
        let short = self.repr_short(box_);
        if short.len() > 1
            && matches!(short.as_bytes()[0], b'i' | b'r' | b'f')
            && short[1..].chars().all(|c| c.is_ascii_digit())
            && !links.iter().any(|(key, _)| key == &short)
        {
            let info = LinkInfo {
                label: self.repr_box(box_),
                color: BOX_COLOR,
            };
            links.push((short.clone(), (info.label, info.color)));
        }
    }

    fn repr_short(&mut self, box_: OpRef) -> String {
        if let Some(name) = self.memo.get(box_) {
            return name.to_string();
        }
        let name = match const_value_of(box_) {
            Some(value) => format_value(value),
            None if box_.is_none() => "_".to_string(),
            None => {
                let prefix = match box_.ty() {
                    Some(Type::Int) => "i",
                    Some(Type::Ref) => "r",
                    Some(Type::Float) => "f",
                    _ => "v",
                };
                format!("{prefix}{}", box_.raw())
            }
        };
        self.memo.set(box_, name.clone());
        name
    }

    fn repr_box(&mut self, box_: OpRef) -> String {
        match const_value_of(box_) {
            Some(value) => format_value(value),
            None => self.repr_short(box_),
        }
    }
}

#[derive(Clone, Default)]
struct EdgeAttrs {
    label: String,
    style: String,
    color: String,
    dir: String,
    weight: String,
}

impl EdgeAttrs {
    fn weight(mut self, weight: &str) -> Self {
        self.weight = weight.to_string();
        self
    }
}

#[derive(Clone)]
struct NodeAttrs {
    shape: String,
    label: String,
    color: String,
    fillcolor: String,
    style: String,
    width: String,
}

impl NodeAttrs {
    fn new(shape: &str, label: &str) -> Self {
        Self {
            shape: shape.to_string(),
            label: label.to_string(),
            color: "black".to_string(),
            fillcolor: "white".to_string(),
            style: "filled".to_string(),
            width: "0.75".to_string(),
        }
    }

    fn fillcolor(mut self, fillcolor: &str) -> Self {
        self.fillcolor = fillcolor.to_string();
        self
    }
}

struct DotGen {
    graphname: String,
    lines: Vec<String>,
}

impl DotGen {
    fn new(graphname: &str) -> Self {
        let graphname = safename(graphname);
        let mut dotgen = Self {
            graphname,
            lines: Vec::new(),
        };
        dotgen.emit(format!("digraph {} {{", dotgen.graphname));
        dotgen
    }

    fn emit(&mut self, line: impl Into<String>) {
        self.lines.push(line.into());
    }

    fn emit_edge(&mut self, name1: &str, name2: &str, attrs: EdgeAttrs) {
        let attrs = EdgeAttrs {
            label: attrs.label,
            style: if attrs.style.is_empty() {
                "dashed".to_string()
            } else {
                attrs.style
            },
            color: if attrs.color.is_empty() {
                "black".to_string()
            } else {
                attrs.color
            },
            dir: if attrs.dir.is_empty() {
                "forward".to_string()
            } else {
                attrs.dir
            },
            weight: if attrs.weight.is_empty() {
                "5".to_string()
            } else {
                attrs.weight
            },
        };
        self.emit(format!(
            r#"edge [label="{}", style="{}", color="{}", dir="{}", weight="{}"];"#,
            escape_attr(&attrs.label),
            escape_attr(&attrs.style),
            escape_attr(&attrs.color),
            escape_attr(&attrs.dir),
            escape_attr(&attrs.weight)
        ));
        self.emit(format!("{} -> {}", safename(name1), safename(name2)));
    }

    fn emit_node(&mut self, name: &str, attrs: NodeAttrs) {
        self.emit(format!(
            r#"{} [shape="{}", label="{}", color="{}", fillcolor="{}", style="{}", width="{}"];"#,
            safename(name),
            escape_attr(&attrs.shape),
            escape_attr(&attrs.label),
            escape_attr(&attrs.color),
            escape_attr(&attrs.fillcolor),
            escape_attr(&attrs.style),
            escape_attr(&attrs.width)
        ));
    }

    fn generate(mut self) -> String {
        self.emit("}");
        self.lines.join("\n")
    }
}

fn escape_attr(value: &str) -> String {
    value
        .replace('\\', r"\\")
        .replace('"', "\\\"")
        .replace('\n', r"\n")
}

/// Extract a constant `Value` from a const-namespace `OpRef`, else `None`.
fn const_value_of(r: OpRef) -> Option<Value> {
    match r {
        OpRef::ConstInt(v) => Some(Value::Int(v)),
        OpRef::ConstFloat(v) => Some(Value::Float(v)),
        OpRef::ConstPtr(v) => Some(Value::Ref(majit_ir::const_ptr_table::resolve(v))),
        _ => None,
    }
}

fn format_value(value: Value) -> String {
    match value {
        Value::Int(v) => v.to_string(),
        Value::Float(v) => v.to_string(),
        Value::Ref(v) => format!("ptr({:#x})", v.0),
        Value::Void => "void".to_string(),
    }
}

fn safename(name: &str) -> String {
    let mut result = String::from("_");
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            result.push(ch);
        } else if ch == '_' {
            result.push_str("__");
        } else {
            result.push_str(&format!("_{:02X}", ch as u32));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::test_support::rooted_resop_operand;
    use majit_ir::{Op, OpCode, OpRef, Type, operand::Operand, value::InputArg};

    #[test]
    fn display_procedures_returns_dot_source() {
        let i0 = InputArg::new_int_rc(0);
        let i1 = InputArg::new_int_rc(1);
        let op = Op::new(
            OpCode::IntAdd,
            &[
                Operand::from_bound_inputarg(&i0),
                Operand::from_bound_inputarg(&i1),
            ],
        );
        op.pos().set(OpRef::int_op(2));
        let jump = Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 2)]);
        let procedure = vec![op, jump];

        let source = display_procedures(&[procedure.as_slice()], None, &[]);

        assert!(source.contains("digraph _resop {"));
        assert!(source.contains("IntAdd"));
        assert!(source.contains("Jump"));
    }

    #[test]
    fn graphpage_exposes_box_links() {
        let i0 = InputArg::new_int_rc(0);
        let i1 = InputArg::new_int_rc(1);
        let op = Op::new(
            OpCode::IntAdd,
            &[
                Operand::from_bound_inputarg(&i0),
                Operand::from_bound_inputarg(&i1),
            ],
        );
        op.pos().set(OpRef::int_op(1));
        let procedure = vec![op];
        let mut page = ResOpGraphPage::default();

        page.compute(&[(&procedure as &dyn ResOpProcedure, 0)], None);

        // Both links, with the colour each carries. The two membership checks
        // this replaces named only the keys: nothing pinned the count, and
        // nothing looked at the value side at all, so the colour the page
        // assigns was free to change silently.
        assert_eq!(
            page.links
                .iter()
                .map(|(key, (name, colour))| (key.as_str(), name.as_str(), *colour))
                .collect::<Vec<_>>(),
            [("i0", "i0", (128, 0, 96)), ("i1", "i1", (128, 0, 96))]
        );
    }

    #[test]
    fn subgraph_copies_parent_failarg_names() {
        let failarg = OpRef::input_arg_int(7);
        let subinputarg = OpRef::input_arg_int(0);
        let subgraph = SubGraph::new(vec![failarg], vec![subinputarg], vec![]);
        let mut memo = ResOpMemo::default();
        memo.set(failarg, "i7".to_string());

        subgraph.get_display_text(&mut memo);

        assert_eq!(memo.get(subinputarg), Some("i7"));
    }

    #[test]
    fn memo_shares_boxes_for_the_same_opref() {
        // A bound inputarg and its position-only `OpRef` view share the same
        // canonical `OpRef` key, so the viewer names them identically.
        let producer = InputArg::new_int_rc(4);
        let first = Operand::from_bound_inputarg(&producer).to_opref();
        let second = OpRef::input_arg_int(4);
        let mut memo = ResOpMemo::default();

        memo.set(first, "i4".to_string());

        assert_eq!(memo.get(second), Some("i4"));
    }
}
