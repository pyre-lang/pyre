//! Code generator for the integer optimization rule DSL.
//!
//! RPython counterpart: `rpython/jit/metainterp/ruleopt/codegen.py`.
//!
//! This module keeps PyPy's rule expansion, matcher tree, and generated
//! Python-mixin text shape.  The generated text is still consumed as an audit
//! artifact in Rust; `optimizeopt/autogenintrules.rs` is the hand-ported
//! runtime counterpart of the same generated methods.

use std::collections::{BTreeMap, HashMap};

use crate::ruleopt::parse::{
    Check, Compute, Element, Expression, File, Pattern, PatternVar, Rule, RuleType,
};

const COMMUTATIVE_OPS: &[&str] = &[
    "int_add", "int_mul", "int_and", "int_mul", "int_or", "int_xor", "int_eq", "int_ne",
];

pub fn generate_commutative_patterns_args(args: &[Pattern]) -> Vec<Vec<Pattern>> {
    if args.is_empty() {
        return vec![Vec::new()];
    }
    let arg0 = &args[0];
    let args1 = &args[1..];
    let mut out = Vec::new();
    for subarg0 in generate_commutative_patterns(arg0) {
        for mut subargs1 in generate_commutative_patterns_args(args1) {
            let mut row = vec![subarg0.clone()];
            row.append(&mut subargs1);
            out.push(row);
        }
    }
    out
}

pub fn generate_commutative_patterns(pattern: &Pattern) -> Vec<Pattern> {
    let Pattern::PatternOp(pattern_op) = pattern else {
        return vec![pattern.clone()];
    };
    let mut out = Vec::new();
    for subargs in generate_commutative_patterns_args(&pattern_op.args) {
        if !COMMUTATIVE_OPS.contains(&pattern_op.opname.as_str())
            || subargs.len() < 2
            || subargs[0].to_string() == subargs[1].to_string()
        {
            out.push(Pattern::PatternOp(pattern_op.newargs(subargs)));
        } else {
            let mut reversed = subargs.clone();
            reversed.reverse();
            out.push(Pattern::PatternOp(pattern_op.newargs(subargs)));
            out.push(Pattern::PatternOp(pattern_op.newargs(reversed)));
        }
    }
    out
}

pub fn generate_commutative_rules(rule: &Rule) -> Vec<Rule> {
    generate_commutative_patterns(&rule.pattern)
        .into_iter()
        .map(|pattern| rule.newpattern(pattern))
        .collect()
}

pub fn sort_rules(rules: &[Rule]) -> Vec<Rule> {
    let mut out = rules.to_vec();
    out.sort_by_key(|rule| rule.pattern.sort_key());
    out
}

pub fn split_by_result_type(rules: &[Rule]) -> (Vec<Rule>, Vec<Rule>, Vec<Rule>) {
    let mut constant_results = Vec::new();
    let mut box_results = Vec::new();
    let mut op_results = Vec::new();
    for rule in rules {
        match &rule.target {
            Pattern::PatternConst(_) => constant_results.push(rule.clone()),
            Pattern::PatternVar(target) if target.name.starts_with('C') => {
                constant_results.push(rule.clone())
            }
            Pattern::PatternVar(_) => box_results.push(rule.clone()),
            Pattern::PatternOp(_) => op_results.push(rule.clone()),
        }
    }
    (constant_results, box_results, op_results)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BaseMatcher {
    IsConstMatcher(IsConstMatcher),
    OpMatcher(OpMatcher),
    Terminal(Terminal),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Matcher {
    pub ifyes: Option<Box<BaseMatcher>>,
    pub ifno: Option<Box<BaseMatcher>>,
    pub nextmatcher: Option<Box<BaseMatcher>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IsConstMatcher {
    pub name: String,
    pub ifyes: Option<Box<BaseMatcher>>,
    pub ifno: Option<Box<BaseMatcher>>,
    pub nextmatcher: Option<Box<BaseMatcher>>,
    pub constname: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpMatcher {
    pub name: String,
    pub opname: String,
    pub ifyes: Option<Box<BaseMatcher>>,
    pub ifno: Option<Box<BaseMatcher>>,
    pub nextmatcher: Option<Box<BaseMatcher>>,
    pub argnames: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Terminal {
    pub rules: Vec<Rule>,
    pub bindings: BTreeMap<Path, String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Path(pub Vec<PathElement>);

impl Path {
    fn child_op(&self, opname: &str, index: usize) -> Self {
        let mut path = self.0.clone();
        path.push(PathElement::Op(opname.to_string(), index));
        Path(path)
    }

    fn child_const(&self) -> Self {
        let mut path = self.0.clone();
        path.push(PathElement::Const);
        Path(path)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum PathElement {
    Op(String, usize),
    Const,
}

pub fn create_matcher(rules: &[Rule]) -> Option<BaseMatcher> {
    let mut opnames = rules.iter().filter_map(|rule| match &rule.pattern {
        Pattern::PatternOp(pattern) => Some(pattern.opname.as_str()),
        _ => None,
    });
    let first = opnames.next()?;
    assert!(opnames.all(|opname| opname == first));
    let patterns = rules
        .iter()
        .map(|rule| match &rule.pattern {
            Pattern::PatternOp(pattern) => pattern.args.clone(),
            _ => unreachable!("top-level rule pattern must be PatternOp"),
        })
        .collect::<Vec<_>>();
    let names = (0..patterns[0].len())
        .map(|i| format!("arg_{i}"))
        .collect::<Vec<_>>();
    let name_paths = (0..patterns[0].len())
        .map(|i| Path(vec![PathElement::Op(first.to_string(), i)]))
        .collect::<Vec<_>>();
    let bindings = name_paths
        .iter()
        .cloned()
        .zip(names.iter().cloned())
        .collect::<BTreeMap<_, _>>();
    _create_matcher(rules.to_vec(), patterns, names, name_paths, bindings)
}

fn _create_matcher(
    rules: Vec<Rule>,
    mut patterns: Vec<Vec<Pattern>>,
    mut names: Vec<String>,
    mut name_paths: Vec<Path>,
    bindings: BTreeMap<Path, String>,
) -> Option<BaseMatcher> {
    if rules.is_empty() {
        return None;
    }
    while !patterns.is_empty() {
        assert_eq!(names.len(), name_paths.len());
        assert_eq!(names.len(), patterns[0].len());
        if patterns[0].is_empty() {
            return Some(BaseMatcher::Terminal(Terminal { rules, bindings }));
        }
        let mut matchpatterns = Vec::new();
        let mut matchrules = Vec::new();
        let mut cantmatchpatterns = Vec::new();
        let mut cantmatchrules = Vec::new();
        let mut restpatterns = Vec::new();
        let mut restrules = Vec::new();

        if patterns.iter().any(|pattern| pattern[0].matches_constant()) {
            for (rule, pattern) in rules.iter().cloned().zip(patterns.iter().cloned()) {
                if pattern[0].matches_constant() {
                    matchrules.push(rule);
                    matchpatterns.push(pattern[1..].to_vec());
                } else if matches!(pattern[0], Pattern::PatternOp(_)) {
                    cantmatchrules.push(rule);
                    cantmatchpatterns.push(pattern);
                } else {
                    restrules.push(rule);
                    restpatterns.push(pattern);
                }
            }
            let constname = format!("C_{}", names[0]);
            let yes_name_paths = name_paths[1..].to_vec();
            let mut yes_bindings = bindings.clone();
            yes_bindings.insert(name_paths[0].child_const(), constname.clone());
            let ifyes = _create_matcher(
                matchrules,
                matchpatterns,
                names[1..].to_vec(),
                yes_name_paths,
                yes_bindings,
            )
            .map(Box::new);
            let ifno = _create_matcher(
                cantmatchrules,
                cantmatchpatterns,
                names.clone(),
                name_paths.clone(),
                bindings.clone(),
            )
            .map(Box::new);
            let nextmatcher =
                _create_matcher(restrules, restpatterns, names, name_paths, bindings).map(Box::new);
            return Some(BaseMatcher::IsConstMatcher(IsConstMatcher {
                name: constname
                    .strip_prefix("C_")
                    .expect("constname prefix")
                    .to_string(),
                ifyes,
                ifno,
                nextmatcher,
                constname,
            }));
        } else if patterns
            .iter()
            .any(|pattern| matches!(pattern[0], Pattern::PatternOp(_)))
        {
            let mut opname: Option<String> = None;
            let mut argnames = Vec::new();
            let mut arg_paths = Vec::new();
            for (rule, pattern) in rules.iter().cloned().zip(patterns.iter().cloned()) {
                if let Pattern::PatternOp(op) = &pattern[0] {
                    let can_match = opname.as_ref().is_none_or(|current| current == &op.opname);
                    if can_match {
                        opname = Some(op.opname.clone());
                        matchrules.push(rule);
                        argnames = (0..op.args.len())
                            .map(|i| format!("{}_{}", names[0], i))
                            .collect();
                        arg_paths = (0..op.args.len())
                            .map(|i| name_paths[0].child_op(&op.opname, i))
                            .collect::<Vec<_>>();
                        let mut row = op.args.clone();
                        row.extend_from_slice(&pattern[1..]);
                        matchpatterns.push(row);
                    } else {
                        cantmatchrules.push(rule);
                        cantmatchpatterns.push(pattern);
                    }
                    continue;
                }
                if pattern[0].matches_constant() {
                    cantmatchrules.push(rule);
                    cantmatchpatterns.push(pattern);
                } else {
                    restrules.push(rule);
                    restpatterns.push(pattern);
                }
            }
            let opname = opname.expect("op matcher has opname");
            let mut yes_name_paths = arg_paths.clone();
            yes_name_paths.extend_from_slice(&name_paths[1..]);
            let mut yes_bindings = bindings.clone();
            for (path, name) in arg_paths.into_iter().zip(argnames.iter().cloned()) {
                yes_bindings.insert(path, name);
            }
            let ifyes = _create_matcher(
                matchrules,
                matchpatterns,
                argnames
                    .iter()
                    .cloned()
                    .chain(names[1..].iter().cloned())
                    .collect(),
                yes_name_paths,
                yes_bindings,
            )
            .map(Box::new);
            let ifno = _create_matcher(
                cantmatchrules,
                cantmatchpatterns,
                names.clone(),
                name_paths.clone(),
                bindings.clone(),
            )
            .map(Box::new);
            let nextmatcher = _create_matcher(
                restrules,
                restpatterns,
                names.clone(),
                name_paths.clone(),
                bindings,
            )
            .map(Box::new);
            return Some(BaseMatcher::OpMatcher(OpMatcher {
                name: names[0].clone(),
                opname,
                ifyes,
                ifno,
                nextmatcher,
                argnames,
            }));
        } else {
            for pattern in &mut patterns {
                pattern.remove(0);
            }
            names.remove(0);
            name_paths.remove(0);
            continue;
        }
    }
    Some(BaseMatcher::Terminal(Terminal { rules, bindings }))
}

#[derive(Default)]
pub struct Codegen {
    code: Vec<String>,
    level: usize,
    bindings: HashMap<String, String>,
    intbound_bindings: HashMap<String, String>,
    method_opname: String,
    name_positions: HashMap<String, usize>,
    pattern_binding_by_string: HashMap<String, String>,
}

impl Codegen {
    pub fn new() -> Self {
        Self::default()
    }

    fn emit(&mut self, line: impl AsRef<str>) {
        let line = line.as_ref();
        if self.level == 0 && (line.starts_with("def ") || line.starts_with("class ")) {
            self.code.push(String::new());
        }
        if line.trim().is_empty() {
            self.code.push(String::new());
        } else {
            self.code
                .push(format!("{}{}", "    ".repeat(self.level), line));
        }
    }

    fn with_indent(&mut self, line: impl AsRef<str>, f: impl FnOnce(&mut Self)) {
        self.emit(line);
        self.level += 1;
        f(self);
        self.level -= 1;
    }

    fn emit_stacking_condition(&mut self, cond: impl AsRef<str>) {
        self.emit(format!("if {}:", cond.as_ref()));
        self.level += 1;
    }

    fn visit_pattern(&mut self, pattern: &Pattern) {
        match pattern {
            Pattern::PatternVar(p) => self.visit_pattern_var(p),
            Pattern::PatternConst(p) => {
                self.emit_stacking_condition(format!(
                    "{} == {}",
                    self.pattern_binding(pattern),
                    p.const_value
                ));
            }
            Pattern::PatternOp(p) => {
                for arg in &p.args {
                    self.visit_pattern(arg);
                }
            }
        }
    }

    fn visit_pattern_var(&mut self, p: &PatternVar) {
        let varname = self.pattern_binding(&Pattern::PatternVar(p.clone()));
        if p.name.starts_with('C') {
            assert!(varname.starts_with("C_"));
            if !self.bindings.contains_key(&p.name) {
                self.bindings.insert(p.name.clone(), varname);
                return;
            }
            if self.bindings[&p.name] == varname {
                return;
            }
            self.emit_stacking_condition(format!("{} == {}", varname, self.bindings[&p.name]));
        } else {
            let intbound_name = format!("b_{varname}");
            if !self.bindings.contains_key(&p.name) {
                self.bindings.insert(p.name.clone(), varname);
                self.intbound_bindings.insert(p.name.clone(), intbound_name);
                return;
            }
            if self.bindings[&p.name] == varname {
                return;
            }
            let varname2 = self.bindings[&p.name].clone();
            self.emit_stacking_condition(format!(
                "self._eq({varname}, b_{varname}, {varname2}, b_{varname2})"
            ));
        }
    }

    fn pattern_binding(&self, pattern: &Pattern) -> String {
        self.pattern_binding_by_string
            .get(&pattern.to_string())
            .cloned()
            .unwrap_or_else(|| pattern.to_string())
    }

    fn generate_target(&mut self, target: &Pattern) {
        match target {
            Pattern::PatternVar(target) => {
                let value = if target.typ == Some(RuleType::Int) {
                    format!("ConstInt({})", self.bindings[&target.name])
                } else {
                    self.bindings[&target.name].clone()
                };
                self.emit(format!("self.make_equal_to(op, {value})"));
            }
            Pattern::PatternConst(target) => {
                self.emit(format!(
                    "self.make_constant_int(op, {})",
                    target.const_value
                ));
            }
            Pattern::PatternOp(target) => {
                let mut args = Vec::new();
                for arg in &target.args {
                    match arg {
                        Pattern::PatternVar(v)
                            if v.name.starts_with('C') || v.typ == Some(RuleType::Int) =>
                        {
                            args.push(format!("ConstInt({})", self.bindings[&v.name]));
                        }
                        Pattern::PatternVar(v) => args.push(self.bindings[&v.name].clone()),
                        Pattern::PatternConst(v) => {
                            args.push(format!("ConstInt({})", v.const_value))
                        }
                        Pattern::PatternOp(_) => unreachable!("generated targets use flat args"),
                    }
                }
                self.emit(format!(
                    "newop = self.replace_op_with(op, rop.{}, args=[{}])",
                    target.opname.to_uppercase(),
                    args.join(", ")
                ));
                self.emit("self.optimizer.send_extra_operation(newop)");
            }
        }
    }

    fn emit_arg_reads(
        &mut self,
        prefix: &str,
        opname: &str,
        numargs: usize,
    ) -> (Vec<String>, Vec<String>) {
        let mut boxnames = Vec::new();
        let mut boundnames = Vec::new();
        for i in 0..numargs {
            let boxname = format!("{prefix}_{i}");
            let boundname = format!("b_{boxname}");
            boxnames.push(boxname.clone());
            boundnames.push(boundname.clone());
            self.emit(format!(
                "{boxname} = get_box_replacement({opname}.getarg({i}))"
            ));
            self.emit(format!("{boundname} = self.getintbound({boxname})"));
        }
        (boxnames, boundnames)
    }

    fn visit_matcher(&mut self, matcher: &BaseMatcher) {
        match matcher {
            BaseMatcher::Terminal(ast) => self.visit_terminal(ast),
            BaseMatcher::IsConstMatcher(ast) => self.visit_is_const_matcher(ast),
            BaseMatcher::OpMatcher(ast) => self.visit_op_matcher(ast),
        }
    }

    fn visit_terminal(&mut self, ast: &Terminal) {
        for rule in &ast.rules {
            self.bindings.clear();
            self.intbound_bindings.clear();
            self.pattern_binding_by_string.clear();
            for (path, name) in &ast.bindings {
                self.add_binding(&rule.pattern, path, name);
            }
            let position = self.name_positions[&rule.name];
            self.generate_rule(rule, position);
        }
    }

    fn visit_is_const_matcher(&mut self, ast: &IsConstMatcher) {
        let currlevel = self.level;
        self.emit_stacking_condition(format!("b_{}.is_constant()", ast.name));
        self.emit(format!(
            "{} = b_{}.get_constant_int()",
            ast.constname, ast.name
        ));
        if let Some(ifyes) = &ast.ifyes {
            self.visit_matcher(ifyes);
        }
        self.level = currlevel;
        if let Some(ifno) = &ast.ifno {
            self.with_indent("else:", |this| this.visit_matcher(ifno));
        }
        if let Some(nextmatcher) = &ast.nextmatcher {
            self.visit_matcher(nextmatcher);
        }
    }

    fn visit_op_matcher(&mut self, ast: &OpMatcher) {
        let currlevel = self.level;
        let boxname = format!("{}_{}", ast.name, ast.opname);
        self.emit(format!(
            "{boxname} = self.optimizer.as_operation({}, rop.{})",
            ast.name,
            ast.opname.to_uppercase()
        ));
        self.emit_stacking_condition(format!("{boxname} is not None"));
        let (boxnames, _) = self.emit_arg_reads(&ast.name, &boxname, ast.argnames.len());
        assert_eq!(boxnames, ast.argnames);
        if let Some(ifyes) = &ast.ifyes {
            self.visit_matcher(ifyes);
        }
        self.level = currlevel;
        if let Some(ifno) = &ast.ifno {
            self.with_indent("else:", |this| this.visit_matcher(ifno));
        }
        if let Some(nextmatcher) = &ast.nextmatcher {
            self.visit_matcher(nextmatcher);
        }
    }

    fn add_binding(&mut self, pattern: &Pattern, path: &Path, name: &str) {
        let mut pattern = pattern;
        for element in &path.0 {
            match element {
                PathElement::Const => {
                    assert!(pattern.matches_constant());
                }
                PathElement::Op(opname, index) => {
                    let Pattern::PatternOp(op) = pattern else {
                        unreachable!("path expects op pattern");
                    };
                    assert_eq!(&op.opname, opname);
                    pattern = &op.args[*index];
                }
            }
        }
        self.pattern_binding_by_string
            .insert(pattern.to_string(), name.to_string());
    }

    fn generate_method(&mut self, opname: &str, rules: &[Rule]) {
        let mut all_rules = Vec::new();
        for rule in rules {
            all_rules.extend(generate_commutative_rules(rule));
        }
        let mut name_positions = HashMap::new();
        let mut names = Vec::new();
        for rule in &all_rules {
            if !name_positions.contains_key(&rule.name) {
                name_positions.insert(rule.name.clone(), name_positions.len());
                names.push(rule.name.clone());
            }
        }
        self.name_positions = name_positions;
        self.method_opname = opname.to_string();
        self.emit(format!("_rule_names_{opname} = {:?}", names));
        self.emit(format!("_rule_fired_{opname} = [0] * {}", names.len()));
        self.emit(format!(
            "_all_rules_fired.append(({opname:?}, _rule_names_{opname}, _rule_fired_{opname}))"
        ));
        self.with_indent(
            format!("def optimize_{}(self, op):", opname.to_uppercase()),
            |this| {
                let numargs = match &rules[0].pattern {
                    Pattern::PatternOp(pattern) => pattern.args.len(),
                    _ => unreachable!("top-level rule pattern must be PatternOp"),
                };
                this.emit_arg_reads("arg", "op", numargs);
                let subsets = split_by_result_type(&all_rules);
                for subset_rules in [subsets.0, subsets.1, subsets.2] {
                    let subset_rules = sort_rules(&subset_rules);
                    if subset_rules.is_empty() {
                        continue;
                    }
                    if let Some(matcher) = create_matcher(&subset_rules) {
                        this.visit_matcher(&matcher);
                    }
                }
                this.emit("return self.emit(op)");
            },
        );
    }

    fn generate_rule(&mut self, rule: &Rule, position: usize) {
        let opname = self.method_opname.clone();
        self.emit(format!(
            "# {}: {} => {}",
            rule.name, rule.pattern, rule.target
        ));
        let currlevel = self.level;
        if let Pattern::PatternOp(pattern) = &rule.pattern {
            for arg in &pattern.args {
                self.visit_pattern(arg);
            }
        }
        for el in &rule.elements {
            match el {
                Element::Compute(el) => self.visit_compute(el),
                Element::Check(el) => self.visit_check(el),
            }
        }
        self.generate_target(&rule.target);
        self.emit(format!("self._rule_fired_{opname}[{position}] += 1"));
        self.emit("return");
        self.level = currlevel;
    }

    fn visit_compute(&mut self, el: &Compute) {
        if el.expr.typ() == RuleType::IntBound {
            self.intbound_bindings
                .insert(el.name.clone(), el.name.clone());
        } else {
            self.bindings.insert(el.name.clone(), el.name.clone());
        }
        let res = self.visit_expression(&el.expr, 0);
        self.emit(format!("{} = {res}", el.name));
    }

    fn visit_check(&mut self, el: &Check) {
        let res = self.visit_expression(&el.expr, 0);
        self.emit_stacking_condition(res);
    }

    fn visit_expression(&self, expr: &Expression, prec: u8) -> String {
        match expr {
            Expression::Name(expr) => {
                if expr.name == "LONG_BIT" {
                    expr.name.clone()
                } else if expr.typ == Some(RuleType::IntBound) {
                    self.intbound_bindings[&expr.name].clone()
                } else {
                    self.bindings[&expr.name].clone()
                }
            }
            Expression::Number(expr) => expr.value.to_string(),
            Expression::Add(expr) => self.visit_int_binop(
                &expr.left,
                &expr.right,
                expr.pysymbol,
                expr.need_ruint,
                8,
                prec,
            ),
            Expression::Sub(expr) => self.visit_int_binop(
                &expr.left,
                &expr.right,
                expr.pysymbol,
                expr.need_ruint,
                8,
                prec,
            ),
            Expression::Mul(expr) => self.visit_int_binop(
                &expr.left,
                &expr.right,
                expr.pysymbol,
                expr.need_ruint,
                9,
                prec,
            ),
            Expression::Div(expr) => {
                self.visit_binop(&expr.left, &expr.right, expr.pysymbol, 9, prec)
            }
            Expression::LShift(expr) => self.visit_int_binop(
                &expr.left,
                &expr.right,
                expr.pysymbol,
                expr.need_ruint,
                7,
                prec,
            ),
            Expression::URShift(expr) => self.visit_int_binop(
                &expr.left,
                &expr.right,
                expr.pysymbol,
                expr.need_ruint,
                7,
                prec,
            ),
            Expression::ARShift(expr) => self.visit_int_binop(
                &expr.left,
                &expr.right,
                expr.pysymbol,
                expr.need_ruint,
                7,
                prec,
            ),
            Expression::OpAnd(expr) => self.visit_int_binop(
                &expr.left,
                &expr.right,
                expr.pysymbol,
                expr.need_ruint,
                6,
                prec,
            ),
            Expression::OpOr(expr) => self.visit_int_binop(
                &expr.left,
                &expr.right,
                expr.pysymbol,
                expr.need_ruint,
                4,
                prec,
            ),
            Expression::OpXor(expr) => self.visit_int_binop(
                &expr.left,
                &expr.right,
                expr.pysymbol,
                expr.need_ruint,
                5,
                prec,
            ),
            Expression::Eq(expr) => {
                self.visit_binop(&expr.left, &expr.right, expr.pysymbol, 3, prec)
            }
            Expression::Ge(expr) => {
                self.visit_binop(&expr.left, &expr.right, expr.pysymbol, 3, prec)
            }
            Expression::Gt(expr) => {
                self.visit_binop(&expr.left, &expr.right, expr.pysymbol, 3, prec)
            }
            Expression::Le(expr) => {
                self.visit_binop(&expr.left, &expr.right, expr.pysymbol, 3, prec)
            }
            Expression::Lt(expr) => {
                self.visit_binop(&expr.left, &expr.right, expr.pysymbol, 3, prec)
            }
            Expression::Ne(expr) => {
                self.visit_binop(&expr.left, &expr.right, expr.pysymbol, 3, prec)
            }
            Expression::ShortcutAnd(expr) => {
                self.visit_binop(&expr.left, &expr.right, expr.pysymbol, 2, prec)
            }
            Expression::ShortcutOr(expr) => {
                self.visit_binop(&expr.left, &expr.right, expr.pysymbol, 1, prec)
            }
            Expression::Invert(expr) => {
                self.visit_unary_op(&expr.left, expr.pysymbol, false, 10, prec)
            }
            Expression::Neg(expr) => {
                self.visit_unary_op(&expr.left, expr.pysymbol, expr.need_ruint, 10, prec)
            }
            Expression::Neg(expr) => {
                // `Neg.need_ruint` (`parse.py`): `intmask(-r_uint(...))`.
                let sub = self.visit_expression(&expr.left, 11);
                format!("intmask({}r_uint({sub}))", expr.pysymbol)
            }
            Expression::Attribute(expr) => {
                let varname = &self.intbound_bindings[&expr.varname];
                if expr.attrname == "ones" {
                    format!("intmask({varname}.tvalue)")
                } else if expr.attrname == "zeros" {
                    format!("intmask(~({varname}.tvalue | {varname}.tmask))")
                } else {
                    format!("intmask({varname}.{})", expr.attrname)
                }
            }
            Expression::MethodCall(expr) => {
                let receiver = self.visit_expression(&expr.value, 0);
                let args = expr
                    .args
                    .iter()
                    .map(|arg| self.visit_expression(arg, 0))
                    .collect::<Vec<_>>();
                format!("{}.{}({})", receiver, expr.methname, args.join(", "))
            }
            Expression::FuncCall(expr) => {
                let args = expr
                    .args
                    .iter()
                    .map(|arg| self.visit_expression(arg, 0))
                    .collect::<Vec<_>>();
                format!("{}({})", expr.funcname, args.join(", "))
            }
        }
    }

    fn visit_binop(
        &self,
        left: &Expression,
        right: &Expression,
        symbol: &str,
        expr_prec: u8,
        prec: u8,
    ) -> String {
        let left = self.visit_expression(left, expr_prec);
        let right = self.visit_expression(right, expr_prec + 1);
        let res = format!("{left} {symbol} {right}");
        if prec > expr_prec {
            format!("({res})")
        } else {
            res
        }
    }

    fn visit_int_binop(
        &self,
        left: &Expression,
        right: &Expression,
        symbol: &str,
        need_ruint: bool,
        expr_prec: u8,
        prec: u8,
    ) -> String {
        if need_ruint {
            let left = self.visit_expression(left, 0);
            let right = self.visit_expression(right, 0);
            format!("intmask(r_uint({left}) {symbol} r_uint({right}))")
        } else {
            self.visit_binop(left, right, symbol, expr_prec, prec)
        }
    }

    fn visit_unary_op(
        &self,
        left: &Expression,
        symbol: &str,
        need_ruint: bool,
        expr_prec: u8,
        prec: u8,
    ) -> String {
        let sub = self.visit_expression(left, expr_prec + 1);
        if need_ruint {
            format!("intmask({symbol}r_uint({sub}))")
        } else {
            let res = format!("{symbol}{sub}");
            if prec > expr_prec {
                format!("({res})")
            } else {
                res
            }
        }
    }

    pub fn generate_code(&mut self, ast: &File) -> String {
        let mut per_op: BTreeMap<String, Vec<Rule>> = BTreeMap::new();
        for rule in &ast.rules {
            let Pattern::PatternOp(pattern) = &rule.pattern else {
                continue;
            };
            per_op
                .entry(pattern.opname.clone())
                .or_default()
                .push(rule.clone());
        }
        for (opname, rules) in per_op {
            self.generate_method(&opname, &rules);
        }
        self.emit("");
        self.code.join("\n")
    }

    pub fn generate_mixin(&mut self, ast: &File) -> String {
        self.with_indent("class OptIntAutoGenerated(object):", |this| {
            this.with_indent("def _eq(self, box1, bound1, box2, bound2):", |this| {
                this.emit("if box1 is box2: return True");
                this.emit("if bound1.is_constant() and bound2.is_constant() and bound1.lower == bound2.lower: return True");
                this.emit("return False");
            });
            this.emit("_all_rules_fired = []");
            this.generate_code(ast);
        });
        self.code.join("\n")
    }
}

trait ExpressionType {
    fn typ(&self) -> RuleType;
}

impl ExpressionType for Expression {
    fn typ(&self) -> RuleType {
        match self {
            Expression::Name(expr) => expr.typ.unwrap_or(RuleType::IntBound),
            Expression::Number(expr) => expr.typ,
            Expression::Add(expr) => expr.typ,
            Expression::Sub(expr) => expr.typ,
            Expression::Mul(expr) => expr.typ,
            Expression::Div(expr) => expr.typ,
            Expression::LShift(expr) => expr.typ,
            Expression::URShift(expr) => expr.typ,
            Expression::ARShift(expr) => expr.typ,
            Expression::OpAnd(expr) => expr.typ,
            Expression::OpOr(expr) => expr.typ,
            Expression::OpXor(expr) => expr.typ,
            Expression::Eq(expr) => expr.typ,
            Expression::Ge(expr) => expr.typ,
            Expression::Gt(expr) => expr.typ,
            Expression::Le(expr) => expr.typ,
            Expression::Lt(expr) => expr.typ,
            Expression::Ne(expr) => expr.typ,
            Expression::ShortcutAnd(expr) => expr.typ,
            Expression::ShortcutOr(expr) => expr.typ,
            Expression::Invert(expr) => expr.typ,
            Expression::Neg(expr) => expr.typ,
            Expression::Attribute(expr) => expr.typ,
            Expression::MethodCall(expr) => expr.typ.unwrap_or(RuleType::IntBound),
            Expression::FuncCall(expr) => expr.typ.unwrap_or(RuleType::Int),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ruleopt;
    use crate::ruleopt::parse;

    #[test]
    fn test_generate_commutative_rules() {
        let s = "add_zero: int_add(x, 0)\n    => x\n";
        let ast = parse::parse(s).unwrap();
        let patterns = generate_commutative_patterns(&ast.rules[0].pattern);
        assert_eq!(patterns[0].to_string(), "int_add(x, 0)");
        assert_eq!(patterns[1].to_string(), "int_add(0, x)");
        assert_eq!(patterns.len(), 2);

        let s = "add_reassoc_consts: int_add(int_add(x, C1), C2)\n    C = C1 + C2\n    => int_add(x, C)\n";
        let ast = parse::parse(s).unwrap();
        let patterns = generate_commutative_patterns(&ast.rules[0].pattern);
        assert_eq!(
            patterns.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                "int_add(int_add(x, C1), C2)",
                "int_add(C2, int_add(x, C1))",
                "int_add(int_add(C1, x), C2)",
                "int_add(C2, int_add(C1, x))",
            ]
        );
    }

    #[test]
    fn test_generate_commutative_rules_only_when_necessary() {
        let s = "or_x_x: int_or(x, x)\n    => x\n";
        let ast = parse::parse(s).unwrap();
        let patterns = generate_commutative_patterns(&ast.rules[0].pattern);
        assert_eq!(patterns[0].to_string(), "int_or(x, x)");
        assert_eq!(patterns.len(), 1);
    }

    #[test]
    fn test_sort_patterns() {
        let s = "int_sub_zero: int_sub(x, 0)\n    => x\nint_sub_x_x: int_sub(x, x)\n    => 0\nint_sub_add: int_sub(int_add(x, y), y)\n    => x\nint_sub_zero_neg: int_sub(0, x)\n    => int_neg(x)\n";
        let ast = parse::parse(s).unwrap();
        let rules = sort_rules(&ast.rules);
        assert_eq!(
            rules.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
            vec![
                "int_sub_zero",
                "int_sub_zero_neg",
                "int_sub_add",
                "int_sub_x_x"
            ]
        );
    }

    #[test]
    fn test_create_matcher() {
        let s = "sub_from_zero: int_sub(0, x)\n    => int_neg(x)\n\nsub_add_consts: int_sub(int_add(x, C1), C2)\n    C = C2 - C1\n    => int_sub(x, C)\n\nsub_add_consts: int_sub(int_add(C1, x), C2)\n    C = C2 - C1\n    => int_sub(x, C)\n";
        let ast = parse::parse(s).unwrap();
        let matcher = create_matcher(&ast.rules).unwrap();
        let BaseMatcher::IsConstMatcher(matcher) = matcher else {
            panic!("expected IsConstMatcher");
        };
        assert_eq!(matcher.name, "arg_0");
        let Some(ifyes) = matcher.ifyes.as_deref() else {
            panic!("expected ifyes");
        };
        let BaseMatcher::Terminal(terminal) = ifyes else {
            panic!("expected terminal");
        };
        assert_eq!(terminal.rules[0].name, "sub_from_zero");
        assert!(matcher.nextmatcher.is_none());
        let Some(ifno) = matcher.ifno.as_deref() else {
            panic!("expected ifno");
        };
        let BaseMatcher::OpMatcher(ifno) = ifno else {
            panic!("expected OpMatcher");
        };
        assert_eq!(ifno.name, "arg_0");
    }

    #[test]
    fn test_generate_code_many() {
        let ast = parse::parse(ruleopt::REAL_RULES).unwrap();
        let mut codegen = Codegen::new();
        let res = codegen.generate_code(&ast);
        assert!(res.contains("def optimize_INT_ADD(self, op):"));
        assert!(res.contains("# add_zero: int_add(x, 0) => x"));
    }

    #[test]
    fn test_generate_unary_minus_safe_for_minint() {
        let s = "sub_const_canonicalize: int_sub(x, C1)\n    C = -C1\n    => int_add(x, C)\n";
        let ast = parse::parse(s).unwrap();
        let mut codegen = Codegen::new();
        let res = codegen.generate_code(&ast);
        assert!(res.contains("intmask(-r_uint("));
        assert!(!res.contains("-C_arg_1"));
    }
}
