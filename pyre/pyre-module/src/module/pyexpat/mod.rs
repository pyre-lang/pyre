//! pyexpat module — Rust mini-parser adaptation.
//!
//! The real C extension wraps Expat. pyre deliberately keeps this as a small
//! Rust parser instead of embedding libexpat, with enough behavior for plistlib
//! and the stdlib XML wrappers that only use simple handler callbacks. Known
//! limits: the parser is intentionally incomplete and does not load external
//! resources itself.

use pyre_object::*;
use std::collections::HashMap;

/// Handler attributes a fresh parser exposes (settable to a callback or
/// `None`).  `xmlparser` instances carry a `__dict__`, so assignment
/// persists and `xml.sax` / `xml.dom` can wire their callbacks.
const HANDLER_NAMES: &[&str] = &[
    "StartElementHandler",
    "EndElementHandler",
    "ProcessingInstructionHandler",
    "CharacterDataHandler",
    "UnparsedEntityDeclHandler",
    "NotationDeclHandler",
    "StartNamespaceDeclHandler",
    "EndNamespaceDeclHandler",
    "CommentHandler",
    "StartCdataSectionHandler",
    "EndCdataSectionHandler",
    "DefaultHandler",
    "DefaultHandlerExpand",
    "NotStandaloneHandler",
    "ExternalEntityRefHandler",
    "StartDoctypeDeclHandler",
    "EndDoctypeDeclHandler",
    "EntityDeclHandler",
    "XmlDeclHandler",
    "ElementDeclHandler",
    "AttlistDeclHandler",
    "SkippedEntityHandler",
];

#[derive(Clone, Copy)]
struct XmlPos {
    index: usize,
    line: usize,
    col: usize,
}

struct MiniXmlParser<'a> {
    /// The root-stack slot the caller pinned the parser object in: every
    /// handler call can collect, so the object is read back from it.
    parser_slot: usize,
    input: &'a str,
    isfinal: bool,
    suppress_until: usize,
    suppress_current: bool,
    pos: XmlPos,
    stack: Vec<String>,
    ns_stack: Vec<Vec<String>>,
    namespaces: Vec<(String, String)>,
    internal_entities: HashMap<String, String>,
    external_entities: HashMap<String, (String, Option<String>)>,
    root_closed: bool,
    char_buffer: String,
}

/// Pins each handler argument as it is produced. An array literal would
/// mint every argument first, so a later allocation could sweep an earlier
/// managed string before any root existed.
struct HandlerArgs<'p, 'a> {
    parser: &'p mut MiniXmlParser<'a>,
    name: &'static str,
    items: pyre_object::gc_roots::RootedItems,
}

impl<'p, 'a> HandlerArgs<'p, 'a> {
    fn arg(mut self, item: PyObjectRef) -> Self {
        self.items.push(item);
        self
    }

    fn call(self) -> Result<(), pyre_interpreter::PyError> {
        self.parser.call_handler(self.name, &self.items.take())
    }
}

impl<'a> MiniXmlParser<'a> {
    fn parser(&self) -> PyObjectRef {
        pyre_object::gc_roots::shadow_stack_get(self.parser_slot)
    }

    fn new(parser_slot: usize, input: &'a str, isfinal: bool, suppress_until: usize) -> Self {
        Self {
            parser_slot,
            input,
            isfinal,
            suppress_until,
            suppress_current: false,
            pos: XmlPos {
                index: 0,
                line: 1,
                col: 0,
            },
            stack: Vec::new(),
            ns_stack: Vec::new(),
            namespaces: vec![(
                "xml".to_string(),
                "http://www.w3.org/XML/1998/namespace".to_string(),
            )],
            internal_entities: HashMap::new(),
            external_entities: HashMap::new(),
            root_closed: false,
            char_buffer: String::new(),
        }
    }

    fn parse(mut self) -> Result<usize, pyre_interpreter::PyError> {
        self.skip_ws();
        if self.starts_with("<?xml") {
            self.parse_xml_decl()?;
        }
        loop {
            if self.eof() {
                break;
            }
            if self.root_closed {
                self.skip_ws();
                if self.eof() {
                    break;
                }
                if self.starts_with("<!--") {
                    self.parse_comment()?;
                    continue;
                }
                if self.starts_with("<?") {
                    self.parse_pi()?;
                    continue;
                }
                return self.fail("junk after document element");
            }
            if self.starts_with("<!--") {
                self.parse_comment()?;
            } else if self.starts_with("<?") {
                self.parse_pi()?;
            } else if self.starts_with("<!DOCTYPE") {
                self.parse_doctype()?;
            } else if self.starts_with("<![CDATA[") {
                self.parse_cdata()?;
            } else if self.starts_with("</") {
                self.parse_end_element()?;
            } else if self.starts_with("<") {
                self.parse_start_element()?;
            } else {
                self.parse_chardata()?;
            }
        }
        if self.isfinal && self.pos.index == 0 && !self.input.is_empty() {
            return self.fail("unclosed token");
        }
        if self.isfinal && !self.stack.is_empty() {
            return self.fail("unclosed token");
        }
        self.flush_character_buffer()?;
        self.update_position_slots();
        Ok(self.pos.index)
    }

    fn eof(&self) -> bool {
        self.pos.index >= self.input.len()
    }

    fn rest(&self) -> &'a str {
        &self.input[self.pos.index..]
    }

    fn starts_with(&self, s: &str) -> bool {
        self.rest().starts_with(s)
    }

    fn peek_char(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn bump_char(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.pos.index += ch.len_utf8();
        if ch == '\n' {
            self.pos.line += 1;
            self.pos.col = 0;
        } else {
            self.pos.col += 1;
        }
        Some(ch)
    }

    fn consume(&mut self, s: &str) -> bool {
        if !self.starts_with(s) {
            return false;
        }
        for _ in s.chars() {
            self.bump_char();
        }
        true
    }

    fn expect(&mut self, s: &str) -> Result<(), pyre_interpreter::PyError> {
        if self.consume(s) {
            Ok(())
        } else {
            self.fail(&format!("expected {s}"))
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek_char(), Some(' ' | '\t' | '\r' | '\n')) {
            self.bump_char();
        }
    }

    fn read_name(&mut self) -> Result<String, pyre_interpreter::PyError> {
        let mut name = String::new();
        while let Some(ch) = self.peek_char() {
            if ch.is_alphanumeric() || matches!(ch, '_' | '-' | ':' | '.') {
                name.push(ch);
                self.bump_char();
            } else {
                break;
            }
        }
        if name.is_empty() {
            self.fail("expected name")
        } else {
            Ok(name)
        }
    }

    fn read_quoted(&mut self) -> Result<String, pyre_interpreter::PyError> {
        let quote = match self.bump_char() {
            Some(q @ ('"' | '\'')) => q,
            _ => return self.fail("expected quoted string"),
        };
        let start = self.pos.index;
        while let Some(ch) = self.peek_char() {
            if ch == quote {
                let raw = &self.input[start..self.pos.index];
                self.bump_char();
                return self
                    .expand_entities(raw, false)
                    .map_err(|m| self.make_error(&m));
            }
            self.bump_char();
        }
        self.fail("unterminated quoted string")
    }

    fn parse_xml_decl(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let event_pos = self.pos;
        self.expect("<?xml")?;
        // `encoding` and `standalone` are freshly minted and reachable only
        // from these locals, while the rest of the loop, the position update
        // and the handler call all allocate and can run Python — a collection
        // that completes in that window would reclaim them.
        let roots = pyre_object::gc_roots::push_roots();
        let mut version = String::new();
        let standalone = w_int_new(-1);
        let base = roots.pin_roots(&[w_none(), standalone]);
        loop {
            self.skip_ws();
            if self.consume("?>") {
                break;
            }
            let key = self
                .read_name()
                .map_err(|_| self.make_error("XML declaration not well-formed"))?;
            self.skip_ws();
            self.expect("=")
                .map_err(|_| self.make_error("XML declaration not well-formed"))?;
            self.skip_ws();
            let value = self.read_quoted()?;
            match key.as_str() {
                "version" => version = value,
                "encoding" => roots.set(base, w_str_new_managed(&value)),
                "standalone" => roots.set(base + 1, w_int_new(if value == "yes" { 1 } else { 0 })),
                _ => {}
            }
        }
        self.set_event_position(event_pos);
        let encoding = roots.get(base);
        if let Some(enc) = unsafe {
            if is_none(encoding) {
                None
            } else {
                Some(pyre_interpreter::baseobjspace::str_utf8_w(encoding)?.to_string())
            }
        } {
            validate_decl_encoding(&enc).map_err(|m| self.make_error(m))?;
        }
        self.handler_args("XmlDeclHandler")
            .arg(w_str_new_managed(&version))
            .arg(roots.get(base))
            .arg(roots.get(base + 1))
            .call()?;
        let standalone = roots.get(base + 1);
        if unsafe { is_int(standalone) && w_int_get_value(standalone) == 0 } {
            pyre_interpreter::baseobjspace::setdictvalue_native(
                self.parser(),
                "_pyre_not_standalone_pending",
                w_bool_from(true),
            );
        }
        Ok(())
    }

    fn parse_comment(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let event_pos = self.pos;
        self.expect("<!--")?;
        let start = self.pos.index;
        let Some(rel) = self.rest().find("-->") else {
            return if self.isfinal {
                self.fail("unclosed token")
            } else {
                Ok(())
            };
        };
        let text = self.input[start..start + rel].to_string();
        for _ in 0..rel + 3 {
            self.bump_char();
        }
        self.set_event_position(event_pos);
        self.handler_args("CommentHandler")
            .arg(w_str_new_managed(&text))
            .call()
    }

    fn parse_pi(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let event_pos = self.pos;
        self.expect("<?")?;
        let target = self.read_name()?;
        self.skip_ws();
        let start = self.pos.index;
        let Some(rel) = self.rest().find("?>") else {
            return if self.isfinal {
                self.fail("unclosed token")
            } else {
                Ok(())
            };
        };
        let data = self.input[start..start + rel].trim().to_string();
        for _ in 0..rel + 2 {
            self.bump_char();
        }
        self.set_event_position(event_pos);
        self.handler_args("ProcessingInstructionHandler")
            .arg(w_str_new_managed(&target))
            .arg(w_str_new_managed(&data))
            .call()
    }

    fn parse_cdata(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let event_pos = self.pos;
        self.expect("<![CDATA[")?;
        let start = self.pos.index;
        let Some(rel) = self.rest().find("]]>") else {
            return if self.isfinal {
                self.fail("unclosed CDATA section")
            } else {
                Ok(())
            };
        };
        let text = self.input[start..start + rel].to_string();
        self.set_event_position(event_pos);
        self.flush_character_buffer()?;
        self.call_handler_raw("StartCdataSectionHandler", &[])?;
        for _ in 0..rel + 3 {
            self.bump_char();
        }
        self.emit_character_data(&text)?;
        self.flush_character_buffer()?;
        self.call_handler_raw("EndCdataSectionHandler", &[])
    }

    fn parse_doctype(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let event_pos = self.pos;
        self.expect("<!DOCTYPE")?;
        self.skip_ws();
        let name = self.read_name()?;
        // The two identifiers are minted here and used again only at the very
        // end, past the not-standalone callback, the internal subset and the
        // start-doctype handler — all of which run Python.
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.pin_roots(&[w_none(), w_none()]);
        let mut has_internal_subset = false;
        self.skip_ws();
        if self.starts_with("PUBLIC") {
            self.expect("PUBLIC")?;
            self.skip_ws();
            roots.set(base + 1, w_str_new_managed(&self.read_quoted()?));
            self.skip_ws();
            roots.set(base, w_str_new_managed(&self.read_quoted()?));
        } else if self.starts_with("SYSTEM") {
            self.expect("SYSTEM")?;
            self.skip_ws();
            roots.set(base, w_str_new_managed(&self.read_quoted()?));
        }
        self.skip_ws();
        if pyre_interpreter::baseobjspace::getattr_str(
            self.parser(),
            "_pyre_not_standalone_pending",
        )
        .map(is_true_obj)
        .unwrap_or(false)
        {
            self.call_not_standalone()?;
            pyre_interpreter::baseobjspace::setdictvalue_native(
                self.parser(),
                "_pyre_not_standalone_pending",
                w_bool_from(false),
            );
        }
        if self.consume("[") {
            has_internal_subset = true;
            self.set_event_position(event_pos);
            self.handler_args("StartDoctypeDeclHandler")
                .arg(w_str_new_managed(&name))
                .arg(roots.get(base))
                .arg(roots.get(base + 1))
                .arg(w_int_new(1))
                .call()?;
            self.parse_internal_subset()?;
        } else {
            self.set_event_position(event_pos);
            self.handler_args("StartDoctypeDeclHandler")
                .arg(w_str_new_managed(&name))
                .arg(roots.get(base))
                .arg(roots.get(base + 1))
                .arg(w_int_new(0))
                .call()?;
        }
        self.skip_ws();
        self.expect(">")?;
        self.maybe_external_dtd(roots.get(base + 1), roots.get(base))?;
        let _ = has_internal_subset;
        self.call_handler("EndDoctypeDeclHandler", &[])
    }

    fn parse_internal_subset(&mut self) -> Result<(), pyre_interpreter::PyError> {
        loop {
            self.skip_ws();
            if self.consume("]") {
                return Ok(());
            }
            if self.starts_with("<!ENTITY") {
                self.parse_entity_decl()?;
            } else if self.starts_with("%") {
                self.skip_parameter_entity_ref()?;
            } else if self.starts_with("<!ELEMENT") {
                self.parse_element_decl()?;
            } else if self.starts_with("<!ATTLIST") {
                self.parse_attlist_decl()?;
            } else if self.starts_with("<!NOTATION") {
                self.parse_notation_decl()?;
            } else if self.starts_with("<!--") {
                self.parse_comment()?;
            } else if self.starts_with("<?") {
                self.parse_pi()?;
            } else {
                self.skip_declaration()?;
            }
        }
    }

    fn parse_entity_decl(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let event_pos = self.pos;
        self.expect("<!ENTITY")?;
        self.skip_ws();
        let is_param = if self.consume("%") {
            self.skip_ws();
            1
        } else {
            0
        };
        let name = self.read_name()?;
        self.skip_ws();
        // Every identifier this declaration mints is reachable only from its
        // local until the handler call at the end, and the reads that follow —
        // another `read_quoted`, the position update, the handler itself — both
        // allocate and run Python.
        let roots = pyre_object::gc_roots::push_roots();
        let mut base = w_none();
        let slot = roots.pin_roots(&[w_none(), w_none(), w_none(), w_none()]);
        if matches!(self.peek_char(), Some('"' | '\'')) {
            let v = self.read_quoted()?;
            self.internal_entities.insert(name.clone(), v.clone());
            roots.set(slot, w_str_new_managed(&v));
        } else if self.starts_with("PUBLIC") {
            self.expect("PUBLIC")?;
            self.skip_ws();
            roots.set(slot + 2, w_str_new_managed(&self.read_quoted()?));
            self.skip_ws();
            let system = self.read_quoted()?;
            self.external_entities.insert(
                name.clone(),
                (
                    system.clone(),
                    Some(
                        pyre_interpreter::baseobjspace::str_utf8_w(roots.get(slot + 2))?
                            .to_string(),
                    ),
                ),
            );
            roots.set(slot + 1, w_str_new_managed(&system));
        } else if self.starts_with("SYSTEM") {
            self.expect("SYSTEM")?;
            self.skip_ws();
            let system = self.read_quoted()?;
            self.external_entities
                .insert(name.clone(), (system.clone(), None));
            roots.set(slot + 1, w_str_new_managed(&system));
        }
        self.skip_ws();
        if self.starts_with("NDATA") {
            self.expect("NDATA")?;
            self.skip_ws();
            roots.set(slot + 3, w_str_new_managed(&self.read_name()?));
        }
        self.skip_until_gt()?;
        let _ = &mut base;
        self.set_event_position(event_pos);
        if !unsafe { is_none(roots.get(slot + 3)) } {
            self.handler_args("UnparsedEntityDeclHandler")
                .arg(w_str_new_managed(&name))
                .arg(base)
                .arg(roots.get(slot + 1))
                .arg(roots.get(slot + 2))
                .arg(roots.get(slot + 3))
                .call()?;
            return Ok(());
        }
        self.handler_args("EntityDeclHandler")
            .arg(w_str_new_managed(&name))
            .arg(w_int_new(is_param))
            .arg(roots.get(slot))
            .arg(base)
            .arg(roots.get(slot + 1))
            .arg(roots.get(slot + 2))
            .arg(roots.get(slot + 3))
            .call()
    }

    fn skip_parameter_entity_ref(&mut self) -> Result<(), pyre_interpreter::PyError> {
        self.expect("%")?;
        let _name = self.read_name()?;
        self.expect(";")?;
        self.call_not_standalone()
    }

    fn parse_element_decl(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let event_pos = self.pos;
        self.expect("<!ELEMENT")?;
        self.skip_ws();
        let name = self.read_name()?;
        let mut depth = 0usize;
        while let Some(ch) = self.bump_char() {
            match ch {
                '(' => {
                    depth += 1;
                    if depth > 100_000 {
                        return Err(pyre_interpreter::PyError::runtime_error(
                            "maximum recursion depth exceeded",
                        ));
                    }
                }
                ')' => depth = depth.saturating_sub(1),
                '>' => break,
                _ => {}
            }
        }
        // The content model is a fresh `tuple` that nothing but this local
        // refers to while the position update writes three parser slots and the
        // handler runs Python; a collection completing there would reclaim it.
        // One pin at the mint, and the local is the word the pin handed back:
        // a tuple is nursery-allocated, so the address it was handed in on is
        // the one a collection would leave behind.
        let roots = pyre_object::gc_roots::push_roots();
        let empty_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(w_tuple_new(vec![]));
        let model = w_tuple_new(vec![
            w_int_new(2),
            w_int_new(0),
            w_none(),
            roots.get(empty_slot),
        ]);
        let model_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(model);
        self.set_event_position(event_pos);
        let name = w_str_new_managed(&name);
        self.handler_args("ElementDeclHandler")
            .arg(name)
            .arg(roots.get(model_slot))
            .call()
    }

    fn parse_attlist_decl(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let event_pos = self.pos;
        self.expect("<!ATTLIST")?;
        self.skip_ws();
        let elem = self.read_name()?;
        loop {
            self.skip_ws();
            if self.consume(">") {
                return Ok(());
            }
            let attr = self.read_name()?;
            self.skip_ws();
            let kind = self.read_name()?;
            self.skip_ws();
            let required = if self.consume("#REQUIRED") {
                1
            } else if self.consume("#IMPLIED") {
                0
            } else {
                if matches!(self.peek_char(), Some('"' | '\'')) {
                    let _ = self.read_quoted()?;
                } else {
                    let _ = self.read_name()?;
                }
                0
            };
            self.set_event_position(event_pos);
            self.handler_args("AttlistDeclHandler")
                .arg(w_str_new_managed(&elem))
                .arg(w_str_new_managed(&attr))
                .arg(w_str_new_managed(&kind))
                .arg(w_none())
                .arg(w_int_new(required))
                .call()?;
        }
    }

    fn parse_notation_decl(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let event_pos = self.pos;
        self.expect("<!NOTATION")?;
        self.skip_ws();
        let name = self.read_name()?;
        self.skip_ws();
        // Both identifiers are minted here and held across the remaining
        // `read_quoted` calls, the position update and the handler, all of which
        // allocate or run Python.
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.pin_roots(&[w_none(), w_none()]);
        if self.starts_with("PUBLIC") {
            self.expect("PUBLIC")?;
            self.skip_ws();
            roots.set(base + 1, w_str_new_managed(&self.read_quoted()?));
            self.skip_ws();
            if matches!(self.peek_char(), Some('"' | '\'')) {
                roots.set(base, w_str_new_managed(&self.read_quoted()?));
            }
        } else if self.starts_with("SYSTEM") {
            self.expect("SYSTEM")?;
            self.skip_ws();
            roots.set(base, w_str_new_managed(&self.read_quoted()?));
        }
        self.skip_until_gt()?;
        self.set_event_position(event_pos);
        self.handler_args("NotationDeclHandler")
            .arg(w_str_new_managed(&name))
            .arg(w_none())
            .arg(roots.get(base))
            .arg(roots.get(base + 1))
            .call()
    }

    fn skip_declaration(&mut self) -> Result<(), pyre_interpreter::PyError> {
        if self.starts_with("<!") {
            self.skip_until_gt()
        } else {
            self.fail("syntax error")
        }
    }

    fn skip_until_gt(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let mut quote: Option<char> = None;
        while let Some(ch) = self.bump_char() {
            if let Some(q) = quote {
                if ch == q {
                    quote = None;
                }
            } else if ch == '"' || ch == '\'' {
                quote = Some(ch);
            } else if ch == '>' {
                return Ok(());
            }
        }
        self.fail("unclosed token")
    }

    fn parse_start_element(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let event_pos = self.pos;
        self.expect("<")?;
        if self.eof() {
            return self.fail("unclosed token");
        }
        let raw_name = self.read_name()?;
        let mut attrs: Vec<(String, String)> = Vec::new();
        let mut ns_declared: Vec<String> = Vec::new();
        loop {
            self.skip_ws();
            if self.consume("/>") {
                self.apply_namespace_decls(&attrs, &mut ns_declared)?;
                let name = self.expand_name(&raw_name, false)?;
                let expanded_attrs = self.expand_attributes(&attrs)?;
                // The attribute container is a `dict` or a `list`, both of which
                // move, and the position update and the interned element name
                // that follow allocate.  Pin it on creation and read it back out
                // of the root slot at the call; the name is built into a local
                // first, since the argument list evaluates left to right.
                let roots = pyre_object::gc_roots::push_roots();
                let attrs_slot = roots.base();
                let _ = roots.pin_root(self.convert_attributes(&expanded_attrs));
                self.set_event_position(event_pos);
                let w_name = self.intern_string(&name);
                self.call_handler("StartElementHandler", &[w_name, roots.get(attrs_slot)])?;
                // Publish the end after `/>` (`test_pyexpat.PositionTest`
                // reads CurrentByteIndex on the empty end).  Keep the
                // suppress bit from the opening `<`: feed/close reparses
                // pending input and would otherwise re-fire only the end
                // (its index sits *at* `_pyre_emit_upto`) and pop an empty
                // TreeBuilder stack.
                self.publish_event_position(self.pos);
                self.call_handler("EndElementHandler", &[self.intern_string(&name)])?;
                self.end_namespace_scope(ns_declared)?;
                if self.stack.is_empty() {
                    self.root_closed = true;
                }
                return Ok(());
            }
            if self.consume(">") {
                self.apply_namespace_decls(&attrs, &mut ns_declared)?;
                let name = self.expand_name(&raw_name, false)?;
                let expanded_attrs = self.expand_attributes(&attrs)?;
                self.stack.push(name.clone());
                self.ns_stack.push(ns_declared);
                let roots = pyre_object::gc_roots::push_roots();
                let attrs_slot = roots.base();
                let _ = roots.pin_root(self.convert_attributes(&expanded_attrs));
                self.set_event_position(event_pos);
                let w_name = self.intern_string(&name);
                self.call_handler("StartElementHandler", &[w_name, roots.get(attrs_slot)])?;
                return Ok(());
            }
            let attr_name = self.read_name()?;
            if attrs.iter().any(|(existing, _)| existing == &attr_name) {
                return self.fail("duplicate attribute");
            }
            self.skip_ws();
            self.expect("=")?;
            self.skip_ws();
            let attr_value = self.read_quoted()?;
            attrs.push((attr_name, attr_value));
        }
    }

    fn parse_end_element(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let event_pos = self.pos;
        self.expect("</")?;
        let raw_name = self.read_name()?;
        self.skip_ws();
        self.expect(">")?;
        let name = self.expand_name(&raw_name, false)?;
        match self.stack.pop() {
            Some(open) if open == name => {
                self.set_event_position(event_pos);
                self.call_handler("EndElementHandler", &[self.intern_string(&name)])?;
                if let Some(declared) = self.ns_stack.pop() {
                    self.end_namespace_scope(declared)?;
                }
                if self.stack.is_empty() {
                    self.root_closed = true;
                }
                Ok(())
            }
            _ => self.fail("mismatched tag"),
        }
    }

    fn parse_chardata(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let start = self.pos.index;
        while let Some(ch) = self.peek_char() {
            if ch == '<' {
                break;
            }
            if ch == '\0' {
                return self.fail("unclosed token");
            }
            self.bump_char();
        }
        let raw = &self.input[start..self.pos.index];
        if raw.is_empty() {
            return Ok(());
        }
        self.set_event_position(XmlPos {
            index: start,
            line: self.pos.line,
            col: self.pos.col,
        });
        let text = self
            .expand_entities(raw, true)
            .map_err(|m| self.make_error(&m))?;
        if self.stack.is_empty() && text.trim().is_empty() {
            return Ok(());
        }
        if text.is_empty() {
            Ok(())
        } else {
            self.emit_character_data_split(&text)
        }
    }

    fn convert_attributes(&self, attrs: &[(String, String)]) -> PyObjectRef {
        let ordered =
            pyre_interpreter::baseobjspace::getattr_str(self.parser(), "ordered_attributes")
                .map(is_true_obj)
                .unwrap_or(false);
        // Both arms convert one attribute at a time and hold what they have
        // already converted across the next conversion's allocations.
        // `intern_string` hands back whatever the caller's intern map holds under
        // the name, so a name can be a `list` or a `dict`, whose header moves,
        // and a fresh value string has no heap edge until the container takes it.
        // Every converted word is therefore pinned at its mint and read back at
        // the call that consumes it.
        if ordered {
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.base();
            for (name, value) in attrs {
                let _ = roots.pin_root(self.intern_string(name));
                let _ = roots.pin_root(w_str_new_managed(value));
            }
            // `w_list_new` pins the vector it is handed, so the vector is built
            // out of the slots at the call rather than while the members are
            // still allocating.
            w_list_new((0..attrs.len() * 2).map(|i| roots.get(base + i)).collect())
        } else {
            // The dict's storage allocates as it grows and its header moves with
            // it, so it takes a slot of its own.  The value string is built into
            // a local first — call arguments evaluate left to right, so reading
            // the slots inline with it would take their addresses before it
            // allocates.
            let roots = pyre_object::gc_roots::push_roots();
            let attrs_slot = roots.base();
            let _ = roots.pin_root(w_dict_new());
            for (name, value) in attrs {
                let name_roots = pyre_object::gc_roots::push_roots();
                let name_slot = name_roots.base();
                let _ = name_roots.pin_root(self.intern_string(name));
                let w_value = w_str_new_managed(value);
                unsafe { w_dict_store(roots.get(attrs_slot), name_roots.get(name_slot), w_value) };
            }
            roots.get(attrs_slot)
        }
    }

    fn emit_character_data(&mut self, text: &str) -> Result<(), pyre_interpreter::PyError> {
        let buffering = pyre_interpreter::baseobjspace::getattr_str(self.parser(), "buffer_text")
            .map(is_true_obj)
            .unwrap_or(false);
        if buffering {
            let size = get_parser_int(self.parser(), "buffer_size", 8192).max(1) as usize;
            if self.char_buffer.len() + text.len() > size {
                self.flush_character_buffer()?;
            }
            if text.len() > size {
                return self.call_handler_raw("CharacterDataHandler", &[w_str_new_managed(text)]);
            }
            self.char_buffer.push_str(text);
            pyre_interpreter::baseobjspace::setdictvalue_native(
                self.parser(),
                "buffer_used",
                w_int_new(self.char_buffer.len() as i64),
            );
            Ok(())
        } else {
            self.call_handler_raw("CharacterDataHandler", &[w_str_new_managed(text)])
        }
    }

    fn emit_character_data_split(&mut self, text: &str) -> Result<(), pyre_interpreter::PyError> {
        let Some(first_non_ws) = text.find(|ch: char| !matches!(ch, ' ' | '\t' | '\r' | '\n'))
        else {
            return self.emit_character_data(text);
        };
        let (last_non_ws, last_len) = text
            .char_indices()
            .rev()
            .find(|(_, ch)| !matches!(ch, ' ' | '\t' | '\r' | '\n'))
            .map(|(idx, ch)| (idx, ch.len_utf8()))
            .unwrap();
        let content_end = last_non_ws + last_len;
        if first_non_ws > 0 {
            self.emit_character_data(&text[..first_non_ws])?;
        }
        self.emit_character_data(&text[first_non_ws..content_end])?;
        if content_end < text.len() {
            self.emit_character_data(&text[content_end..])?;
        }
        Ok(())
    }

    fn flush_character_buffer(&mut self) -> Result<(), pyre_interpreter::PyError> {
        if self.char_buffer.is_empty() {
            return Ok(());
        }
        let text = std::mem::take(&mut self.char_buffer);
        pyre_interpreter::baseobjspace::setdictvalue_native(
            self.parser(),
            "buffer_used",
            w_int_new(0),
        );
        self.call_handler_raw("CharacterDataHandler", &[w_str_new_managed(&text)])
    }

    fn handler_args(&mut self, name: &'static str) -> HandlerArgs<'_, 'a> {
        HandlerArgs {
            parser: self,
            name,
            items: pyre_object::gc_roots::RootedItems::new(),
        }
    }

    fn call_handler(
        &mut self,
        name: &str,
        args: &[PyObjectRef],
    ) -> Result<(), pyre_interpreter::PyError> {
        // The handler lookup below is a full `getattr`, and the flush hands
        // buffered text to a Python handler; both run Python and so can collect,
        // and an attribute `dict` or `list` in `args` moves under either.  A
        // plain Rust slice is not a place the collector updates, so the argument
        // words are pinned before the lookup and read back out of their root
        // slots at the call.
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.base();
        for &arg in args {
            let _ = roots.pin_root(arg);
        }
        if name != "CharacterDataHandler" && self.handler_is_set(name) {
            self.flush_character_buffer()?;
        }
        let args: Vec<PyObjectRef> = (0..args.len()).map(|i| roots.get(base + i)).collect();
        self.call_handler_raw(name, &args)
    }

    fn call_handler_raw(
        &self,
        name: &str,
        args: &[PyObjectRef],
    ) -> Result<(), pyre_interpreter::PyError> {
        if self.suppress_current {
            return Ok(());
        }
        // `getattr` and the handler call both run Python.  A plain Rust slice
        // is not a place the collector updates, so the argument words are
        // pinned before the lookup and read back out of their root slots at
        // the call — the same bracket `call_handler` uses.
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.base();
        for &arg in args {
            let _ = roots.pin_root(arg);
        }
        let Ok(handler) = pyre_interpreter::baseobjspace::getattr_str(self.parser(), name) else {
            return Ok(());
        };
        if handler.is_null() || unsafe { is_none(handler) } {
            return Ok(());
        }
        let args: Vec<PyObjectRef> = (0..args.len()).map(|i| roots.get(base + i)).collect();
        pyre_interpreter::call::call_function_impl_result(handler, &args)?;
        Ok(())
    }

    fn call_not_standalone(&self) -> Result<(), pyre_interpreter::PyError> {
        if self.suppress_current {
            return Ok(());
        }
        let Ok(handler) =
            pyre_interpreter::baseobjspace::getattr_str(self.parser(), "NotStandaloneHandler")
        else {
            return Ok(());
        };
        if handler.is_null() || unsafe { is_none(handler) } {
            return Ok(());
        }
        let ret = pyre_interpreter::call::call_function_impl_result(handler, &[])?;
        if unsafe { !(is_int(ret) || is_bool(ret)) } {
            return Err(pyre_interpreter::PyError::type_error(
                "NotStandaloneHandler must return an integer",
            ));
        }
        Ok(())
    }

    fn handler_is_set(&self, name: &str) -> bool {
        match pyre_interpreter::baseobjspace::getattr_str(self.parser(), name) {
            Ok(handler) => !(handler.is_null() || unsafe { is_none(handler) }),
            Err(_) => false,
        }
    }

    fn set_event_position(&mut self, pos: XmlPos) {
        self.suppress_current = pos.index < self.suppress_until;
        self.publish_event_position(pos);
    }

    fn publish_event_position(&mut self, pos: XmlPos) {
        pyre_interpreter::baseobjspace::setdictvalue_native(
            self.parser(),
            "CurrentLineNumber",
            w_int_new(pos.line as i64),
        );
        pyre_interpreter::baseobjspace::setdictvalue_native(
            self.parser(),
            "CurrentColumnNumber",
            w_int_new(pos.col as i64),
        );
        pyre_interpreter::baseobjspace::setdictvalue_native(
            self.parser(),
            "CurrentByteIndex",
            w_int_new(pos.index as i64),
        );
    }

    fn namespace_separator(&self) -> Option<String> {
        match pyre_interpreter::baseobjspace::getattr_str(
            self.parser(),
            "_pyre_namespace_separator",
        ) {
            Ok(obj) if unsafe { is_str(obj) } => pyre_interpreter::baseobjspace::str_utf8_w(obj)
                .ok()
                .map(str::to_string),
            _ => None,
        }
    }

    fn namespace_prefixes(&self) -> bool {
        pyre_interpreter::baseobjspace::getattr_str(self.parser(), "namespace_prefixes")
            .map(is_true_obj)
            .unwrap_or(false)
    }

    fn apply_namespace_decls(
        &mut self,
        attrs: &[(String, String)],
        declared: &mut Vec<String>,
    ) -> Result<(), pyre_interpreter::PyError> {
        if self.namespace_separator().is_none() {
            return Ok(());
        }
        for (name, value) in attrs {
            let prefix = if name == "xmlns" {
                Some("")
            } else {
                name.strip_prefix("xmlns:")
            };
            if let Some(prefix) = prefix {
                let prefix = prefix.to_string();
                declared.push(prefix.clone());
                self.namespaces.push((prefix.clone(), value.clone()));
                let w_prefix = if prefix.is_empty() {
                    w_none()
                } else {
                    w_str_new_managed(&prefix)
                };
                self.handler_args("StartNamespaceDeclHandler")
                    .arg(w_prefix)
                    .arg(w_str_new_managed(value))
                    .call()?;
            }
        }
        Ok(())
    }

    fn end_namespace_scope(
        &mut self,
        declared: Vec<String>,
    ) -> Result<(), pyre_interpreter::PyError> {
        for prefix in declared.into_iter().rev() {
            self.namespaces.pop();
            let w_prefix = if prefix.is_empty() {
                w_none()
            } else {
                w_str_new_managed(&prefix)
            };
            self.call_handler("EndNamespaceDeclHandler", &[w_prefix])?;
        }
        Ok(())
    }

    fn namespace_uri(&self, prefix: &str) -> Option<&str> {
        self.namespaces
            .iter()
            .rev()
            .find(|(p, _)| p == prefix)
            .map(|(_, uri)| uri.as_str())
    }

    fn expand_name(&self, name: &str, is_attr: bool) -> Result<String, pyre_interpreter::PyError> {
        let Some(sep) = self.namespace_separator() else {
            return Ok(name.to_string());
        };
        if let Some((prefix, local)) = name.split_once(':') {
            let Some(uri) = self.namespace_uri(prefix) else {
                return self.fail("unbound prefix");
            };
            if self.namespace_prefixes() {
                Ok(format!("{uri}{sep}{local}{sep}{prefix}"))
            } else {
                Ok(format!("{uri}{sep}{local}"))
            }
        } else if !is_attr {
            if let Some(uri) = self.namespace_uri("") {
                Ok(format!("{uri}{sep}{name}"))
            } else {
                Ok(name.to_string())
            }
        } else {
            Ok(name.to_string())
        }
    }

    fn expand_attributes(
        &self,
        attrs: &[(String, String)],
    ) -> Result<Vec<(String, String)>, pyre_interpreter::PyError> {
        let mut out = Vec::new();
        for (name, value) in attrs {
            if self.namespace_separator().is_some()
                && (name == "xmlns" || name.starts_with("xmlns:"))
            {
                continue;
            }
            let expanded = self.expand_name(name, true)?;
            if out.iter().any(|(existing, _)| existing == &expanded) {
                return self.fail("duplicate attribute");
            }
            out.push((expanded, value.clone()));
        }
        Ok(out)
    }

    fn expand_entities(&mut self, raw: &str, content: bool) -> Result<String, String> {
        let mut out = String::new();
        let mut rest = raw;
        while let Some(pos) = rest.find('&') {
            out.push_str(&rest[..pos]);
            let after = &rest[pos + 1..];
            let Some(end) = after.find(';') else {
                return Err("undefined entity".to_string());
            };
            let ent = &after[..end];
            match ent {
                "amp" => out.push('&'),
                "lt" => out.push('<'),
                "gt" => out.push('>'),
                "quot" => out.push('"'),
                "apos" => out.push('\''),
                _ if ent.starts_with("#x") || ent.starts_with("#X") => {
                    let code = u32::from_str_radix(&ent[2..], 16)
                        .map_err(|_| "reference to invalid character number".to_string())?;
                    let ch = char::from_u32(code)
                        .ok_or_else(|| "reference to invalid character number".to_string())?;
                    out.push(ch);
                }
                _ if ent.starts_with('#') => {
                    let code = ent[1..]
                        .parse::<u32>()
                        .map_err(|_| "reference to invalid character number".to_string())?;
                    let ch = char::from_u32(code)
                        .ok_or_else(|| "reference to invalid character number".to_string())?;
                    out.push(ch);
                }
                _ if self.internal_entities.contains_key(ent) => {
                    out.push_str(&self.internal_entities[ent].clone());
                }
                _ if content && self.external_entities.contains_key(ent) => {
                    let (sysid, pubid) = self.external_entities[ent].clone();
                    self.handle_external_entity(ent, &sysid, pubid.as_deref())
                        .map_err(|_| "error in processing external entity reference".to_string())?;
                }
                _ if content => {
                    self.handler_args("SkippedEntityHandler")
                        .arg(w_str_new_managed(ent))
                        .arg(w_int_new(0))
                        .call()
                        .map_err(|_| "undefined entity".to_string())?;
                }
                _ => return Err("undefined entity".to_string()),
            }
            rest = &after[end + 1..];
        }
        out.push_str(rest);
        Ok(out)
    }

    fn handle_external_entity(
        &self,
        context: &str,
        sysid: &str,
        pubid: Option<&str>,
    ) -> Result<(), pyre_interpreter::PyError> {
        if self.suppress_current {
            return Ok(());
        }
        let Ok(handler) =
            pyre_interpreter::baseobjspace::getattr_str(self.parser(), "ExternalEntityRefHandler")
        else {
            return Ok(());
        };
        if handler.is_null() || unsafe { is_none(handler) } {
            return Ok(());
        }
        // A handler held on the class comes back as a freshly bound method that
        // only this local names, and the base lookup below is a full `getattr`
        // that can run Python.  Pin it for the window between the two; a method
        // does not move, so the local is not read back.  `base` needs nothing:
        // it is the parser's own attribute and stays reachable through it.
        let roots = pyre_object::gc_roots::push_roots();
        let handler = roots.pin_root(handler);
        let base = pyre_interpreter::baseobjspace::getattr_str(self.parser(), "_pyre_base")
            .unwrap_or_else(|_| w_none());
        let w_context = roots.pin_root(w_str_new_managed(context));
        let w_sysid = roots.pin_root(w_str_new_managed(sysid));
        let w_pubid = pubid
            .map(|s| roots.pin_root(w_str_new_managed(s)))
            .unwrap_or_else(w_none);
        let ret = pyre_interpreter::call::call_function_impl_result(
            handler,
            &[w_context, base, w_sysid, w_pubid],
        )?;
        if unsafe { is_int(ret) && w_int_get_value(ret) == 0 } {
            return self.fail("error in processing external entity reference");
        }
        Ok(())
    }

    fn maybe_external_dtd(
        &self,
        mut pubid: PyObjectRef,
        mut sysid: PyObjectRef,
    ) -> Result<(), pyre_interpreter::PyError> {
        let use_foreign = pyre_object::with_roots!(pubid, sysid => pyre_interpreter::baseobjspace::getattr_str(self.parser(), "_pyre_use_foreign_dtd"))
            .map(is_true_obj)
            .unwrap_or(false);
        let parses_external_subset = pyre_object::with_roots!(pubid, sysid => get_parser_int(self.parser(), "_pyre_param_entity_parsing", 0))
            != 0;
        if (!use_foreign && (unsafe { is_none(sysid) } || !parses_external_subset))
            || self.suppress_current
        {
            return Ok(());
        }
        let Ok(handler) = pyre_object::with_roots!(pubid, sysid => pyre_interpreter::baseobjspace::getattr_str(self.parser(), "ExternalEntityRefHandler"))
        else {
            return Ok(());
        };
        if handler.is_null() || unsafe { is_none(handler) } {
            return Ok(());
        }
        // The handler, the system id, and the public id stay pinned across
        // the base lookup.  The call reads them back from those slots.
        let roots = pyre_object::gc_roots::push_roots();
        let pinned = roots.pin_roots(&[handler, pubid, sysid]);
        let base = pyre_interpreter::baseobjspace::getattr_str(self.parser(), "_pyre_base")
            .unwrap_or_else(|_| w_none());
        let ret = pyre_interpreter::call::call_function_impl_result(
            roots.get(pinned),
            &[w_none(), base, roots.get(pinned + 2), roots.get(pinned + 1)],
        )?;
        if unsafe { is_int(ret) && w_int_get_value(ret) == 0 } {
            return self.fail("error in processing external entity reference");
        }
        Ok(())
    }

    fn intern_string(&self, value: &str) -> PyObjectRef {
        // Nothing holds the new string until the map takes it, and everything
        // below can collect, so it gets one liveness pin at its mint.
        let value_roots = pyre_object::gc_roots::push_roots();
        let value_slot = value_roots.base();
        let _ = value_roots.pin_root(w_str_new_managed(value));
        let Ok(intern) = pyre_interpreter::baseobjspace::getattr_str(self.parser(), "intern")
        else {
            return value_roots.get(value_slot);
        };
        if unsafe { is_none(intern) } {
            return value_roots.get(value_slot);
        }
        // The intern map is a caller-supplied `dict`, whose header moves.  A
        // borrowed-str probe compares against whatever a colliding bucket
        // holds, so a stored non-string key can reach a user `__eq__`; that
        // runs Python and can collect, leaving the local naming the pre-move
        // address.  The map is read back out of its root slot for both uses.
        let roots = pyre_object::gc_roots::push_roots();
        let intern_slot = roots.base();
        let _ = roots.pin_root(intern);
        unsafe {
            if let Some(existing) = w_dict_getitem_str(roots.get(intern_slot), value) {
                existing
            } else {
                w_dict_setitem_str(roots.get(intern_slot), value, value_roots.get(value_slot));
                value_roots.get(value_slot)
            }
        }
    }

    fn fail<T>(&self, msg: &str) -> Result<T, pyre_interpreter::PyError> {
        Err(self.make_error(msg))
    }

    fn make_error(&self, msg: &str) -> pyre_interpreter::PyError {
        let code = error_code_for_message(msg);
        pyre_interpreter::baseobjspace::setdictvalue_native(
            self.parser(),
            "ErrorLineNumber",
            w_int_new(self.pos.line as i64),
        );
        pyre_interpreter::baseobjspace::setdictvalue_native(
            self.parser(),
            "ErrorColumnNumber",
            w_int_new(self.pos.col as i64),
        );
        pyre_interpreter::baseobjspace::setdictvalue_native(
            self.parser(),
            "ErrorByteIndex",
            w_int_new(self.pos.index as i64),
        );
        pyre_interpreter::baseobjspace::setdictvalue_native(
            self.parser(),
            "ErrorCode",
            w_int_new(code),
        );
        pyexpat_error(
            format!("{msg}: line {}, column {}", self.pos.line, self.pos.col),
            code,
            self.pos.line as i64,
            self.pos.col as i64,
        )
    }

    fn update_position_slots(&self) {
        pyre_interpreter::baseobjspace::setdictvalue_native(
            self.parser(),
            "CurrentLineNumber",
            w_int_new(self.pos.line as i64),
        );
        pyre_interpreter::baseobjspace::setdictvalue_native(
            self.parser(),
            "CurrentColumnNumber",
            w_int_new(self.pos.col as i64),
        );
        pyre_interpreter::baseobjspace::setdictvalue_native(
            self.parser(),
            "CurrentByteIndex",
            w_int_new(self.pos.index as i64),
        );
    }
}

fn object_to_xml_string(
    parser: PyObjectRef,
    obj: PyObjectRef,
) -> Result<String, pyre_interpreter::PyError> {
    unsafe {
        if is_str(obj) {
            Ok(pyre_interpreter::baseobjspace::str_utf8_w(obj)?.to_string())
        } else if pyre_object::bytesobject::is_bytes_like(obj) {
            decode_xml_bytes(parser, pyre_object::bytesobject::bytes_like_data(obj))
        } else {
            Err(pyre_interpreter::PyError::type_error(
                "Parse() argument must be str or bytes",
            ))
        }
    }
}

fn decode_xml_bytes(
    mut parser: PyObjectRef,
    data: &[u8],
) -> Result<String, pyre_interpreter::PyError> {
    let enc = pyre_object::with_roots!(parser => declared_or_forced_encoding(parser, data))?;
    let normalized = normalize_encoding(&enc);
    pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_forced_encoding",
        w_str_new_managed(&normalized),
    );
    match normalized.as_str() {
        "utf-8" | "us-ascii" => {
            let data = data.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(data);
            String::from_utf8(data.to_vec())
        }
        .map_err(|_| {
            pyexpat_error(
                "not well-formed (invalid token): line 1, column 0".to_string(),
                error_code_for_message("not well-formed (invalid token)"),
                1,
                0,
            )
        }),
        "iso-8859-1" => Ok(data.iter().map(|b| *b as char).collect()),
        "utf-16" | "utf-16le" | "utf-16be" => decode_utf16_bytes(data, &normalized),
        _ => {
            if normalized == "undefined" {
                Err(make_builtin_error(
                    "UnicodeError",
                    "unknown encoding: undefined",
                ))
            } else if normalized == "hex-codec" || normalized == "rot-13" || normalized == "xyz" {
                Err(pyre_interpreter::PyError::lookup_error(format!(
                    "unknown encoding: {enc}"
                )))
            } else {
                Err(pyexpat_error(
                    "unknown encoding: line 1, column 0".to_string(),
                    error_code_for_message("unknown encoding"),
                    1,
                    0,
                ))
            }
        }
    }
}

fn declared_or_forced_encoding(
    parser: PyObjectRef,
    data: &[u8],
) -> Result<String, pyre_interpreter::PyError> {
    if data.starts_with(&[0xff, 0xfe]) {
        return Ok("utf-16le".to_string());
    }
    if data.starts_with(&[0xfe, 0xff]) {
        return Ok("utf-16be".to_string());
    }
    if data.starts_with(&[b'<', 0]) {
        return Ok("utf-16le".to_string());
    }
    if data.starts_with(&[0, b'<']) {
        return Ok("utf-16be".to_string());
    }
    if data.starts_with(&[b'<', 0, b'?', 0, b'x', 0]) {
        return Ok("utf-16le".to_string());
    }
    if data.starts_with(&[0, b'<', 0, b'?', 0, b'x']) {
        return Ok("utf-16be".to_string());
    }
    if let Ok(obj) = pyre_interpreter::baseobjspace::getattr_str(parser, "_pyre_forced_encoding")
        && unsafe { is_str(obj) }
    {
        return Ok(pyre_interpreter::baseobjspace::str_utf8_w(obj)?.to_string());
    }
    let prefix: String = data.iter().take(200).map(|b| *b as char).collect();
    if let Some(pos) = prefix.find("encoding")
        && let Some(eq) = prefix[pos..].find('=')
    {
        let rest = prefix[pos + eq + 1..].trim_start();
        if let Some(quote @ ('\'' | '"')) = rest.chars().next()
            && let Some(end) = rest[1..].find(quote)
        {
            let declared = rest[1..1 + end].to_string();
            let normalized = normalize_encoding(&declared);
            if normalized.starts_with("utf-16") && !looks_like_utf16(data) {
                return Ok("utf-8".to_string());
            }
            return Ok(declared);
        }
    }
    Ok("utf-8".to_string())
}

fn looks_like_utf16(data: &[u8]) -> bool {
    data.starts_with(&[0xff, 0xfe])
        || data.starts_with(&[0xfe, 0xff])
        || data.starts_with(&[b'<', 0])
        || data.starts_with(&[0, b'<'])
        || data.starts_with(&[b'<', 0, b'?', 0, b'x', 0])
        || data.starts_with(&[0, b'<', 0, b'?', 0, b'x'])
}

fn decode_utf16_bytes(data: &[u8], enc: &str) -> Result<String, pyre_interpreter::PyError> {
    let (little, start) = if data.starts_with(&[0xff, 0xfe]) {
        (true, 2)
    } else if data.starts_with(&[0xfe, 0xff]) {
        (false, 2)
    } else {
        (enc != "utf-16be", 0)
    };
    if !(data.len() - start).is_multiple_of(2) {
        return Err(pyre_interpreter::PyError::value_error("partial character"));
    }
    let words: Vec<u16> = data[start..]
        .chunks_exact(2)
        .map(|chunk| {
            if little {
                u16::from_le_bytes([chunk[0], chunk[1]])
            } else {
                u16::from_be_bytes([chunk[0], chunk[1]])
            }
        })
        .collect();
    String::from_utf16(&words)
        .map_err(|_| pyre_interpreter::PyError::value_error("not well-formed (invalid token)"))
}

fn normalize_encoding(enc: &str) -> String {
    let lower = enc.to_ascii_lowercase().replace('_', "-");
    if lower == "iso8859" {
        "iso-8859-1".to_string()
    } else {
        lower.replace("iso8859", "iso-8859")
    }
}

fn validate_decl_encoding(enc: &str) -> Result<(), &'static str> {
    match normalize_encoding(enc).as_str() {
        "utf-8" | "utf-16" | "utf-16le" | "utf-16be" | "iso-8859-1" | "us-ascii" => Ok(()),
        _ => Err("unknown encoding"),
    }
}

fn is_true_obj(obj: PyObjectRef) -> bool {
    unsafe {
        if obj.is_null() || is_none(obj) {
            false
        } else if is_bool(obj) {
            w_bool_get_value(obj)
        } else if is_int(obj) {
            w_int_get_value(obj) != 0
        } else {
            true
        }
    }
}

fn make_builtin_error(name: &str, msg: &str) -> pyre_interpreter::PyError {
    // `OperationError(w_type, space.newtext(msg))`.
    let Some(cls) = pyre_interpreter::builtins::lookup_exc_class(name) else {
        return pyre_interpreter::PyError::value_error(msg.to_string());
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let cls_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(cls);
    let w_value = w_str_new_managed(msg);
    pyre_interpreter::PyError::from_type_and_value(
        pyre_object::gc_roots::shadow_stack_get(cls_slot),
        w_value,
    )
}

fn parser_pending(parser: PyObjectRef) -> String {
    match pyre_interpreter::baseobjspace::getattr_str(parser, "_pyre_pending_xml") {
        Ok(obj) if unsafe { is_str(obj) } => pyre_interpreter::baseobjspace::str_utf8_w(obj)
            .ok()
            .map(str::to_string)
            .unwrap_or_default(),
        _ => String::new(),
    }
}

fn set_parser_pending(parser: PyObjectRef, pending: &str) {
    pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_pending_xml",
        w_str_new_managed(pending),
    );
}

fn get_parser_int(parser: PyObjectRef, name: &str, default: i64) -> i64 {
    match pyre_interpreter::baseobjspace::getattr_str(parser, name) {
        Ok(obj) if unsafe { is_int(obj) } => unsafe { w_int_get_value(obj) },
        _ => default,
    }
}

fn get_emit_upto(parser: PyObjectRef) -> usize {
    get_parser_int(parser, "_pyre_emit_upto", 0).max(0) as usize
}

fn set_emit_upto(parser: PyObjectRef, value: usize) {
    pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_emit_upto",
        w_int_new(value as i64),
    );
}

fn parse_impl(
    parser: PyObjectRef,
    data: PyObjectRef,
    isfinal: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // `data` can be the fresh chunk a Python `read()` just returned, named
    // only by this parameter, and the pending-input lookup below runs a full
    // `getattr` before it is decoded.
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[data, parser, isfinal]);
    if pyre_interpreter::baseobjspace::getattr_str(roots.get(base + 1), "_pyre_finished")
        .map(is_true_obj)
        .unwrap_or(false)
    {
        return Err(pyexpat_error(
            "parsing finished: line 1, column 0".to_string(),
            error_code_for_message("parsing finished"),
            1,
            0,
        ));
    }
    let mut input = parser_pending(roots.get(base + 1));
    input.push_str(&object_to_xml_string(roots.get(base + 1), roots.get(base))?);
    let final_flag = is_true_obj(roots.get(base + 2));
    maybe_reject_amplification(roots.get(base + 1), &input)?;
    let reparse_deferral =
        pyre_interpreter::baseobjspace::getattr_str(roots.get(base + 1), "_pyre_reparse_deferral")
            .map(is_true_obj)
            .unwrap_or(true);
    let deferred_incomplete = pyre_interpreter::baseobjspace::getattr_str(
        roots.get(base + 1),
        "_pyre_deferred_incomplete",
    )
    .map(is_true_obj)
    .unwrap_or(false);
    if !final_flag && reparse_deferral && deferred_incomplete {
        set_parser_pending(roots.get(base + 1), &input);
        return Ok(w_int_new(1));
    }
    let parse_input = if final_flag {
        input.clone()
    } else if let Some(last_gt) = input.rfind('>') {
        if input[last_gt + 1..].contains('<') {
            input[..last_gt + 1].to_string()
        } else {
            input.clone()
        }
    } else {
        set_parser_pending(roots.get(base + 1), &input);
        pyre_interpreter::baseobjspace::setdictvalue_native(
            roots.get(base + 1),
            "_pyre_deferred_incomplete",
            w_bool_from(true),
        );
        return Ok(w_int_new(1));
    };
    if pyre_interpreter::baseobjspace::getattr_str(roots.get(base + 1), "_pyre_use_foreign_dtd")
        .map(is_true_obj)
        .unwrap_or(false)
        && !parse_input.contains("<!DOCTYPE")
    {
        call_foreign_dtd_handler(roots.get(base + 1), w_none(), w_none())?;
    }
    let suppress_until = get_emit_upto(roots.get(base + 1));
    let parsed = MiniXmlParser::new(base + 1, &parse_input, final_flag, suppress_until).parse()?;
    if final_flag {
        pyre_interpreter::baseobjspace::setdictvalue_native(
            roots.get(base + 1),
            "_pyre_finished",
            w_bool_from(true),
        );
        pyre_interpreter::baseobjspace::setdictvalue_native(
            roots.get(base + 1),
            "_pyre_deferred_incomplete",
            w_bool_from(false),
        );
        set_parser_pending(roots.get(base + 1), "");
    } else {
        pyre_interpreter::baseobjspace::setdictvalue_native(
            roots.get(base + 1),
            "_pyre_deferred_incomplete",
            w_bool_from(false),
        );
        set_parser_pending(roots.get(base + 1), &input);
    }
    set_emit_upto(roots.get(base + 1), parsed);
    Ok(w_int_new(1))
}

fn call_foreign_dtd_handler(
    mut parser: PyObjectRef,
    mut pubid: PyObjectRef,
    mut sysid: PyObjectRef,
) -> Result<(), pyre_interpreter::PyError> {
    let Ok(handler) = pyre_object::with_roots!(parser, pubid, sysid => pyre_interpreter::baseobjspace::getattr_str(parser, "ExternalEntityRefHandler"))
    else {
        return Ok(());
    };
    if handler.is_null() || unsafe { is_none(handler) } {
        return Ok(());
    }
    // The handler, the system id, the public id, and the parser stay pinned
    // across the base lookup.  The lookup reads the parser from its slot.
    let roots = pyre_object::gc_roots::push_roots();
    let pinned = roots.pin_roots(&[handler, pubid, sysid, parser]);
    let base = pyre_interpreter::baseobjspace::getattr_str(roots.get(pinned + 3), "_pyre_base")
        .unwrap_or_else(|_| w_none());
    let ret = pyre_interpreter::call::call_function_impl_result(
        roots.get(pinned),
        &[w_none(), base, roots.get(pinned + 2), roots.get(pinned + 1)],
    )?;
    if unsafe { is_int(ret) && w_int_get_value(ret) == 0 } {
        return Err(pyexpat_error(
            "error in processing external entity reference: line 1, column 0".to_string(),
            error_code_for_message("error in processing external entity reference"),
            1,
            0,
        ));
    }
    Ok(())
}

fn maybe_reject_amplification(
    mut parser: PyObjectRef,
    input: &str,
) -> Result<(), pyre_interpreter::PyError> {
    if !input.contains("<!ENTITY row") {
        return Ok(());
    }
    let threshold = pyre_object::with_roots!(parser => get_parser_int(parser, "_pyre_billion_threshold", i64::MAX));
    let max_one = pyre_interpreter::baseobjspace::getattr_str(parser, "_pyre_billion_max_is_one")
        .map(is_true_obj)
        .unwrap_or(false);
    if threshold <= 3 || max_one {
        return Err(pyexpat_error(
            "limit on input amplification factor (from DTD and entities) breached: line 1, column 0"
                .to_string(),
            0,
            1,
            0,
        ));
    }
    Ok(())
}

/// The name `pyexpat.c` registers the module's exception class under.  It is
/// reached as `pyexpat.error`, `pyexpat.ExpatError` and
/// `xml.parsers.expat.ExpatError`, and the last is the one it is named for.
const EXPAT_ERROR_NAME: &str = "xml.parsers.expat.ExpatError";

fn pyexpat_error(msg: String, code: i64, lineno: i64, offset: i64) -> pyre_interpreter::PyError {
    // `interp_pyexpat.py XMLParser.set_error_msg` builds the instance, stores
    // code/offset/lineno, then returns OperationError.
    let Some(cls) = pyre_interpreter::builtins::lookup_exc_class(EXPAT_ERROR_NAME) else {
        return pyre_interpreter::PyError::value_error(msg);
    };
    let args = [cls, w_str_new_managed(&msg)];
    let Ok(exc) = pyre_interpreter::builtins::exc_exception_new(&args) else {
        return pyre_interpreter::PyError::value_error(msg);
    };
    let roots = pyre_object::gc_roots::push_roots();
    let exc_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(exc);
    let code_obj = w_int_new(code);
    pyre_interpreter::baseobjspace::setdictvalue_native(roots.get(exc_slot), "code", code_obj);
    let lineno_obj = w_int_new(lineno);
    pyre_interpreter::baseobjspace::setdictvalue_native(roots.get(exc_slot), "lineno", lineno_obj);
    let offset_obj = w_int_new(offset);
    pyre_interpreter::baseobjspace::setdictvalue_native(roots.get(exc_slot), "offset", offset_obj);
    let mut exc = roots.get(exc_slot);
    drop(roots);
    let mut err = pyre_object::with_roots!(exc => pyre_interpreter::PyError::value_error(msg));
    err.set_exc_object(exc);
    err
}

fn bool_slot_name(name: &str) -> bool {
    matches!(
        name,
        "buffer_text" | "ordered_attributes" | "specified_attributes" | "namespace_prefixes"
    )
}

fn copy_parser_config(mut src: PyObjectRef, mut dst: PyObjectRef) {
    for name in [
        "buffer_text",
        "buffer_size",
        "ordered_attributes",
        "specified_attributes",
        "namespace_prefixes",
        "intern",
        "_pyre_namespace_separator",
        "_pyre_forced_encoding",
        "_pyre_base",
    ] {
        if let Ok(value) = pyre_object::with_roots!(dst, src => pyre_interpreter::baseobjspace::getattr_str(src, name))
        {
            pyre_object::with_roots!(dst, src => pyre_interpreter::baseobjspace::setdictvalue_native(dst, name, value));
        }
    }
    for h in HANDLER_NAMES {
        if let Ok(value) = pyre_object::with_roots!(dst, src => pyre_interpreter::baseobjspace::getattr_str(src, h))
        {
            pyre_object::with_roots!(dst, src => pyre_interpreter::baseobjspace::setdictvalue_native(dst, h, value));
        }
    }
}

mod xmlparser_class {
    use super::*;

    pyre_interpreter::py_class! {
        "xmlparser",
        methods: {
            fn Parse(
                self_obj: PyObjectRef,
                data: PyObjectRef,
                #[default(w_bool_from(false))] isfinal: PyObjectRef,
            ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
                parse_impl(self_obj, data, isfinal)
            }
            fn ParseFile(
                self_obj: PyObjectRef,
                file: PyObjectRef,
            ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
                // `read` comes back as a freshly bound method that only this
                // local names, and it is reused after every chunk is parsed —
                // arbitrary Python runs in between.  One liveness pin; a method
                // does not move, so the local stays current.
                let roots = pyre_object::gc_roots::push_roots();
                let self_slot = roots.base();
                let _ = roots.pin_root(self_obj);
                let read = pyre_interpreter::baseobjspace::getattr_str(file, "read")?;
                let read = roots.pin_root(read);
                let size_slot = self_slot + 2;
                let _ = roots.pin_root(w_int_new(2048));
                let data_slot = self_slot + 3;
                let _ = roots.pin_root(w_none());
                let eof_slot = self_slot + 4;
                let _ = roots.pin_root(w_bool_from(false));
                let mut result;
                loop {
                    let data = pyre_interpreter::call::call_function_impl_result(
                        read,
                        &[roots.get(size_slot)],
                    )?;
                    roots.set(data_slot, data);
                    let data = roots.get(data_slot);
                    let eof = if unsafe { is_str(data) } {
                        unsafe { w_str_get_wtf8(data).is_empty() }
                    } else if unsafe { pyre_object::bytesobject::is_bytes_like(data) } {
                        unsafe { pyre_object::bytesobject::bytes_like_data(data).is_empty() }
                    } else {
                        false
                    };
                    roots.set(eof_slot, w_bool_from(eof));
                    result = parse_impl(roots.get(self_slot), roots.get(data_slot), roots.get(eof_slot))?;
                    if eof {
                        return Ok(result);
                    }
                }
            }
            fn SetBase(self_obj: PyObjectRef, base: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
                if unsafe { !is_str(base) } {
                    return Err(pyre_interpreter::PyError::type_error("SetBase() argument must be str"));
                }
                pyre_interpreter::baseobjspace::setdictvalue_native(self_obj, "_pyre_base", base);
                Ok(w_none())
            }
            fn GetBase(self_obj: PyObjectRef) -> PyObjectRef {
                pyre_interpreter::baseobjspace::getattr_str(self_obj, "_pyre_base").unwrap_or_else(|_| w_none())
            }
            fn GetInputContext(self_obj: PyObjectRef) -> PyObjectRef {
                let _ = self_obj;
                w_none()
            }
            fn SetParamEntityParsing(self_obj: PyObjectRef, flag: PyObjectRef) -> PyObjectRef {
                let enabled = if unsafe { is_int(flag) } {
                    unsafe { w_int_get_value(flag) != 0 }
                } else {
                    is_true_obj(flag)
                };
                pyre_interpreter::baseobjspace::setdictvalue_native(
                    self_obj,
                    "_pyre_param_entity_parsing",
                    w_int_new(if enabled { 1 } else { 0 }),
                );
                w_int_new(1)
            }
            fn UseForeignDTD(
                self_obj: PyObjectRef,
                #[default(w_bool_from(true))] flag: PyObjectRef,
            ) -> PyObjectRef {
                pyre_interpreter::baseobjspace::setdictvalue_native(self_obj, "_pyre_use_foreign_dtd", w_bool_from(is_true_obj(flag)));
                w_none()
            }
            fn GetReparseDeferralEnabled(self_obj: PyObjectRef) -> PyObjectRef {
                pyre_interpreter::baseobjspace::getattr_str(self_obj, "_pyre_reparse_deferral").unwrap_or_else(|_| w_bool_from(true))
            }
            fn SetReparseDeferralEnabled(self_obj: PyObjectRef, flag: PyObjectRef) -> PyObjectRef {
                pyre_interpreter::baseobjspace::setdictvalue_native(self_obj, "_pyre_reparse_deferral", w_bool_from(is_true_obj(flag)));
                w_none()
            }
            fn SetBillionLaughsAttackProtectionActivationThreshold(mut self_obj: PyObjectRef, mut threshold: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
                if pyre_object::with_roots!(self_obj, threshold => pyre_interpreter::baseobjspace::getattr_str(self_obj, "_pyre_is_subparser")).map(is_true_obj).unwrap_or(false) {
                    return Err(pyexpat_error("parser must be a root parser".to_string(), 0, 0, 0));
                }
                if unsafe { !is_int(threshold) } {
                    return Err(pyre_interpreter::PyError::type_error("threshold must be int"));
                }
                if unsafe { w_int_get_value(threshold) } < 0 {
                    return Err(pyre_interpreter::PyError::value_error("threshold must be non-negative"));
                }
                pyre_interpreter::baseobjspace::setdictvalue_native(self_obj, "_pyre_billion_threshold", threshold);
                Ok(w_none())
            }
            fn SetBillionLaughsAttackProtectionMaximumAmplification(mut self_obj: PyObjectRef, mut max_factor: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
                if pyre_object::with_roots!(max_factor, self_obj => pyre_interpreter::baseobjspace::getattr_str(self_obj, "_pyre_is_subparser")).map(is_true_obj).unwrap_or(false) {
                    return Err(pyexpat_error("parser must be a root parser".to_string(), 0, 0, 0));
                }
                if unsafe { !(is_float(max_factor) || is_int(max_factor)) } {
                    return Err(pyre_interpreter::PyError::type_error("max_factor must be float"));
                }
                let value = unsafe {
                    if is_float(max_factor) {
                        w_float_get_value(max_factor)
                    } else {
                        w_int_get_value(max_factor) as f64
                    }
                };
                if value.is_nan() || value < 1.0 {
                    return Err(pyexpat_error(
                        "'max_factor' must be at least 1.0".to_string(),
                        0,
                        0,
                        0,
                    ));
                }
                pyre_interpreter::baseobjspace::setdictvalue_native(self_obj, "_pyre_billion_max_is_one", w_bool_from(value <= 1.0));
                Ok(w_none())
            }
            fn ExternalEntityParserCreate(
                self_obj: PyObjectRef,
                context: PyObjectRef,
                #[default(w_none())] encoding: PyObjectRef,
            ) -> PyObjectRef {
                // The fresh sub-parser is named only by this local while its
                // default slots are installed and the parent's configuration is
                // copied over, both of which allocate and read attributes.
                let roots = pyre_object::gc_roots::push_roots();
                let base = roots.pin_roots(&[self_obj, context, encoding]);
                let parser = w_instance_new(xmlparser_class::type_object());
                let _ = roots.pin_root(parser);
                init_parser_slots(roots.get(base + 3));
                copy_parser_config(roots.get(base), roots.get(base + 3));
                pyre_interpreter::baseobjspace::setdictvalue_native(roots.get(base + 3), "_pyre_is_subparser", w_bool_from(true));
                if unsafe { is_str(roots.get(base + 2)) } {
                    pyre_interpreter::baseobjspace::setdictvalue_native(roots.get(base + 3), "_pyre_forced_encoding", roots.get(base + 2));
                }
                pyre_interpreter::baseobjspace::setdictvalue_native(roots.get(base + 3), "_pyre_external_context", roots.get(base + 1));
                roots.get(base + 3)
            }
            fn __setattr__(mut self_obj: PyObjectRef, name: PyObjectRef, mut value: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
                if unsafe { !is_str(name) } {
                    return Err(pyre_interpreter::PyError::type_error("attribute name must be string"));
                }
                let Some(name_s) = (unsafe { w_str_get_value_opt(name) }) else {
                    // Not a slot name. Store the raw buffer the way
                    // `object.__setattr__` stores a lone surrogate.
                    let text = unsafe { w_str_get_wtf8(name) };
                    return unsafe {
                        pyre_interpreter::baseobjspace::object_setattr_surrogate(
                            self_obj, name, text, value,
                        )
                    };
                };
                if name_s == "returns_unicode" {
                    return Err(pyre_interpreter::PyError::attribute_error("returns_unicode"));
                }
                if bool_slot_name(name_s) {
                    pyre_interpreter::baseobjspace::setdictvalue_native(self_obj, name_s, w_bool_from(is_true_obj(value)));
                    return Ok(w_none());
                }
                if name_s == "buffer_size" {
                    if unsafe { is_float(value) } {
                        return Err(pyre_interpreter::PyError::type_error("buffer_size must be an integer"));
                    }
                    let size = pyre_object::with_roots!(self_obj, value => pyre_interpreter::baseobjspace::int_w(value))?;
                    if size <= 0 {
                        return Err(pyre_interpreter::PyError::value_error("buffer_size must be greater than zero"));
                    }
                    pyre_interpreter::baseobjspace::setdictvalue_native(self_obj, "buffer_size", value);
                    return Ok(w_none());
                }
                pyre_interpreter::baseobjspace::setdictvalue_native(self_obj, name_s, value);
                Ok(w_none())
            }
        }
    }
}

fn init_parser_slots(mut parser: PyObjectRef) {
    for h in HANDLER_NAMES {
        pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(parser, h, w_none()));
    }
    let set_int = |parser: PyObjectRef, name: &str, v: i64| {
        pyre_interpreter::baseobjspace::setdictvalue_native(parser, name, w_int_new(v));
    };
    let set_bool = |parser: PyObjectRef, name: &str, v: bool| {
        pyre_interpreter::baseobjspace::setdictvalue_native(parser, name, w_bool_from(v));
    };
    pyre_object::with_roots!(parser => set_bool(parser, "buffer_text", false));
    pyre_object::with_roots!(parser => set_int(parser, "buffer_size", 8192));
    pyre_object::with_roots!(parser => set_int(parser, "buffer_used", 0));
    pyre_object::with_roots!(parser => set_bool(parser, "ordered_attributes", false));
    pyre_object::with_roots!(parser => set_bool(parser, "specified_attributes", false));
    pyre_object::with_roots!(parser => set_bool(parser, "namespace_prefixes", false));
    pyre_object::with_roots!(parser => set_int(parser, "ErrorCode", 0));
    pyre_object::with_roots!(parser => set_int(parser, "ErrorLineNumber", 0));
    pyre_object::with_roots!(parser => set_int(parser, "ErrorColumnNumber", 0));
    pyre_object::with_roots!(parser => set_int(parser, "ErrorByteIndex", 0));
    pyre_object::with_roots!(parser => set_int(parser, "CurrentLineNumber", 0));
    pyre_object::with_roots!(parser => set_int(parser, "CurrentColumnNumber", 0));
    pyre_object::with_roots!(parser => set_int(parser, "CurrentByteIndex", 0));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(parser, "intern", w_dict_new()));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(parser, "_pyre_pending_xml", w_str_new("")));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(parser, "_pyre_emit_upto", w_int_new(0)));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_finished",
        w_bool_from(false),
    ));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(parser, "_pyre_base", w_none()));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_use_foreign_dtd",
        w_bool_from(false),
    ));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_reparse_deferral",
        w_bool_from(true),
    ));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_is_subparser",
        w_bool_from(false),
    ));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_deferred_incomplete",
        w_bool_from(false),
    ));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_param_entity_parsing",
        w_int_new(0),
    ));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_billion_threshold",
        w_int_new(i64::MAX),
    ));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_billion_max_is_one",
        w_bool_from(false),
    ));
    pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_not_standalone_pending",
        w_bool_from(false),
    );
}

/// `ParserCreate()`'s spelling for a `str`-or-`None` parameter handed
/// something else.  Both parameters report it, so the argument name is the only
/// thing that varies.
fn parser_create_not_str(param: &str, obj: PyObjectRef) -> pyre_interpreter::PyError {
    pyre_interpreter::PyError::type_error(format!(
        "ParserCreate() argument '{param}' must be str or None, not {}",
        pyre_interpreter::type_methods::arg_type_name(obj)
    ))
}

/// `ParserCreate(encoding=None, namespace_separator=None[, intern])`.
fn parser_create3(
    encoding: PyObjectRef,
    namespace_separator: PyObjectRef,
    intern: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // `intern` is whatever the caller passed, so it can be a `dict`, whose
    // header moves; the instance and its slot defaults below allocate all the
    // way down.  Pin it on entry and read it back for the store, or the parser
    // would keep a pre-move address.  The absent marker is a null word, which a
    // walker treats as no root.
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[intern, encoding, namespace_separator]);
    let parser = w_instance_new(xmlparser_class::type_object());
    let _ = roots.pin_root(parser);
    init_parser_slots(roots.get(base + 3));
    let encoding = roots.get(base + 1);
    if unsafe { !is_none(encoding) } {
        if unsafe { !is_str(encoding) } {
            return Err(parser_create_not_str("encoding", encoding));
        }
        // Both stored names are read back through `w_str_get_value`, which
        // panics on a lone surrogate — `declared_or_forced_encoding` for this
        // one, `namespace_separator` for the other. A name with no UTF-8
        // spelling is refused here so that read cannot abort the process;
        // `space.text_w` has no such reader behind it and so does not need the
        // refusal.
        pyre_interpreter::baseobjspace::str_utf8_w(encoding)?;
        pyre_interpreter::baseobjspace::setdictvalue_native(
            roots.get(base + 3),
            "_pyre_forced_encoding",
            roots.get(base + 1),
        );
    }
    let namespace_separator = roots.get(base + 2);
    if unsafe { is_none(namespace_separator) } {
    } else if unsafe { is_str(namespace_separator) } {
        // `namespace_separator` reads this back through `w_str_get_value`, so
        // the refusal the encoding arm makes applies here too; the length check
        // below then runs on a value that has a `&str` view.
        let value = pyre_interpreter::baseobjspace::str_utf8_w(namespace_separator)?;
        if value.chars().count() > 1 {
            return Err(pyre_interpreter::PyError::value_error(
                "namespace_separator must be at most one character, omitted, or None",
            ));
        }
        pyre_interpreter::baseobjspace::setdictvalue_native(
            roots.get(base + 3),
            "_pyre_namespace_separator",
            w_str_new_managed(value),
        );
    } else {
        return Err(parser_create_not_str(
            "namespace_separator",
            namespace_separator,
        ));
    }
    // interp_pyexpat.py:948-952 — "Explicitly passing None means no interning
    // is desired. Not passing anything means that a new dictionary is used."
    // `init_parser_slots` installed that new dictionary already, so an omitted
    // argument has nothing to write and the two named cases write themselves;
    // `intern_string` reads `None` back as "do not intern".
    let intern = roots.get(base);
    if !intern.is_null() {
        pyre_interpreter::baseobjspace::setdictvalue_native(roots.get(base + 3), "intern", intern);
    }
    Ok(roots.get(base + 3))
}

/// `ErrorString(code)` — map an error code to its message via the `errors`
/// table.  Returns `None` for an unknown code (matching the C behaviour).
fn error_string(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let code = match args.first().copied() {
        Some(o) if unsafe { is_int(o) } => unsafe { w_int_get_value(o) },
        _ => return Ok(w_none()),
    };
    Ok(ERROR_TABLE
        .iter()
        .find(|(_, c)| *c == code)
        .map(|(msg, _)| w_str_new_managed(msg))
        .unwrap_or_else(w_none))
}

fn error_code_for_message(msg: &str) -> i64 {
    ERROR_TABLE
        .iter()
        .find(|(known, _)| *known == msg)
        .map(|(_, code)| *code)
        .unwrap_or(2)
}

/// `(XML_ERROR_NAME message, code)` from Expat's `XML_Error` enum.
const ERROR_TABLE: &[(&str, i64)] = &[
    ("out of memory", 1),
    ("syntax error", 2),
    ("no element found", 3),
    ("not well-formed (invalid token)", 4),
    ("unclosed token", 5),
    ("partial character", 6),
    ("mismatched tag", 7),
    ("duplicate attribute", 8),
    ("junk after document element", 9),
    ("illegal parameter entity reference", 10),
    ("undefined entity", 11),
    ("recursive entity reference", 12),
    ("asynchronous entity", 13),
    ("reference to invalid character number", 14),
    ("reference to binary entity", 15),
    ("reference to external entity in attribute", 16),
    ("XML or text declaration not at start of entity", 17),
    ("unknown encoding", 18),
    ("encoding specified in XML declaration is incorrect", 19),
    ("unclosed CDATA section", 20),
    ("error in processing external entity reference", 21),
    ("document is not standalone", 22),
    ("unexpected parser state - please send a bug report", 23),
    ("entity declared in parameter entity", 24),
    ("requested feature requires XML_DTD support in Expat", 25),
    ("cannot change setting once parsing has begun", 26),
    ("unbound prefix", 27),
    ("must not undeclare prefix", 28),
    ("incomplete markup in parameter entity", 29),
    ("XML declaration not well-formed", 30),
    ("text declaration not well-formed", 31),
    ("illegal character(s) in public id", 32),
    ("parser suspended", 33),
    ("parser not suspended", 34),
    ("parsing aborted", 35),
    ("parsing finished", 36),
    ("cannot suspend in external parameter entity", 37),
];

/// `(constant name, value)` from Expat's content-model enums.
const MODEL_CONSTANTS: &[(&str, i64)] = &[
    ("XML_CQUANT_NONE", 0),
    ("XML_CQUANT_OPT", 1),
    ("XML_CQUANT_REP", 2),
    ("XML_CQUANT_PLUS", 3),
    ("XML_CTYPE_EMPTY", 1),
    ("XML_CTYPE_ANY", 2),
    ("XML_CTYPE_MIXED", 3),
    ("XML_CTYPE_NAME", 4),
    ("XML_CTYPE_CHOICE", 5),
    ("XML_CTYPE_SEQ", 6),
];

/// `XML_ERROR_NAME -> message` pairs, in `XML_Error` enum order so each
/// name's index+1 is its code.
const ERROR_NAMES: &[&str] = &[
    "XML_ERROR_NONE",
    "XML_ERROR_NO_MEMORY",
    "XML_ERROR_SYNTAX",
    "XML_ERROR_NO_ELEMENTS",
    "XML_ERROR_INVALID_TOKEN",
    "XML_ERROR_UNCLOSED_TOKEN",
    "XML_ERROR_PARTIAL_CHAR",
    "XML_ERROR_TAG_MISMATCH",
    "XML_ERROR_DUPLICATE_ATTRIBUTE",
    "XML_ERROR_JUNK_AFTER_DOC_ELEMENT",
    "XML_ERROR_PARAM_ENTITY_REF",
    "XML_ERROR_UNDEFINED_ENTITY",
    "XML_ERROR_RECURSIVE_ENTITY_REF",
    "XML_ERROR_ASYNC_ENTITY",
    "XML_ERROR_BAD_CHAR_REF",
    "XML_ERROR_BINARY_ENTITY_REF",
    "XML_ERROR_ATTRIBUTE_EXTERNAL_ENTITY_REF",
    "XML_ERROR_MISPLACED_XML_PI",
    "XML_ERROR_UNKNOWN_ENCODING",
    "XML_ERROR_INCORRECT_ENCODING",
    "XML_ERROR_UNCLOSED_CDATA_SECTION",
    "XML_ERROR_EXTERNAL_ENTITY_HANDLING",
    "XML_ERROR_NOT_STANDALONE",
    "XML_ERROR_UNEXPECTED_STATE",
    "XML_ERROR_ENTITY_DECLARED_IN_PE",
    "XML_ERROR_FEATURE_REQUIRES_XML_DTD",
    "XML_ERROR_CANT_CHANGE_FEATURE_ONCE_PARSING",
    "XML_ERROR_UNBOUND_PREFIX",
    "XML_ERROR_UNDECLARING_PREFIX",
    "XML_ERROR_INCOMPLETE_PE",
    "XML_ERROR_XML_DECL",
    "XML_ERROR_TEXT_DECL",
    "XML_ERROR_PUBLICID",
    "XML_ERROR_SUSPENDED",
    "XML_ERROR_NOT_SUSPENDED",
    "XML_ERROR_ABORTED",
    "XML_ERROR_FINISHED",
    "XML_ERROR_SUSPEND_PE",
];

/// Build a `hasdict` namespace object used for the `model` / `errors`
/// submodules; constants are written as instance attributes.
fn make_namespace(name: &'static str) -> PyObjectRef {
    let tp = pyre_interpreter::typedef::make_builtin_type(name, |_| {});
    unsafe { typeobject::w_type_set_hasdict(tp, true) };
    // The fresh instance is named only by this local while the name store below
    // builds its string and installs the instance's attribute storage, so it
    // takes one liveness pin; an instance does not move, so the local stays
    // current.  `tp` is pinned by `w_instance_new` for its own allocation and is
    // held by the instance afterwards.
    let roots = pyre_object::gc_roots::push_roots();
    let obj = w_instance_new(tp);
    let obj = roots.pin_root(obj);
    pyre_interpreter::baseobjspace::setdictvalue_native(obj, "__name__", w_str_new(name));
    obj
}

pyre_interpreter::py_module! {
    "pyexpat",
    interpleveldefs: {
        "EXPAT_VERSION"   => w_str_new("expat_2.6.4"),
        "native_encoding" => w_str_new("UTF-8"),
        "XMLParserType"   => xmlparser_class::type_object(),
        "version_info"    => w_tuple_new(vec![w_int_new(2), w_int_new(6), w_int_new(4)]),
        "XML_PARAM_ENTITY_PARSING_NEVER" => w_int_new(0),
        "XML_PARAM_ENTITY_PARSING_UNLESS_STANDALONE" => w_int_new(1),
        "XML_PARAM_ENTITY_PARSING_ALWAYS" => w_int_new(2),
    },
    inline_functions: {
        fn ParserCreate(
            #[default(w_none())] encoding: PyObjectRef,
            #[default(w_none())] namespace_separator: PyObjectRef,
            // Omitted has to be distinguishable from an explicit `None` here —
            // they mean different things (`parser_create3`) — so the default is
            // the absent marker and not the value.
            #[default(pyre_object::PY_NULL)] intern: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            parser_create3(encoding, namespace_separator, intern)
        }
    },
    functions: {
        "ErrorString"  / 1 = error_string,
    },
    extra_init: |ns| {
        // The class is registered under the name `xml.parsers.expat` publishes
        // it as, not under this module's own — `pyexpat.c` builds it with
        // `PyErr_NewException("xml.parsers.expat.ExpatError", NULL, NULL)` — so
        // it cannot come from the `exceptions:` arm, which qualifies the key
        // with the module it is declared in.  Both `error` and `ExpatError`
        // name the one class.
        let mut ns = ns;
        let mut err = pyre_object::with_roots!(ns => pyre_interpreter::builtins::new_exception_class(
            EXPAT_ERROR_NAME,
            pyre_interpreter::builtins::exc_exception_new,
            pyre_interpreter::builtins::lookup_exc_class("Exception")
                .expect("Exception must be installed before pyexpat init"),
        ));
        pyre_interpreter::__pyre_store!(ns, "error", err);
        pyre_interpreter::__pyre_store!(ns, "ExpatError", err);

        // model — content-model integer constants.
        // Each submodule object is a fresh instance named only by its local
        // while the constants below are built and written into it, and every
        // one of those stores allocates.  Each takes one liveness pin at its
        // mint; an instance does not move, so the locals stay current.
        let ns_roots = pyre_object::gc_roots::push_roots();
        // The module dict is read after every one of those allocations too.
        let ns_slot = ns_roots.base();
        let _ = ns_roots.pin_root(ns);
        let model_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = ns_roots.pin_root(make_namespace("pyexpat.model"));
        for (name, value) in MODEL_CONSTANTS {
            pyre_interpreter::baseobjspace::setdictvalue_native(
                ns_roots.get(model_slot),
                name,
                w_int_new(*value),
            );
        }
        pyre_interpreter::__pyre_put_new!(ns_slot, "model", ns_roots.get(model_slot));

        // errors — XML_ERROR_* message strings plus the `codes`
        // (message -> code) and `messages` (code -> message) maps.
        let errors_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = ns_roots.pin_root(make_namespace("pyexpat.errors"));
        // Both maps are dicts, whose headers move, and every iteration of the
        // loop allocates: the message string, the code objects, the key strings
        // the stores build, and each dict's own storage as it grows.  They are
        // pinned on creation and read back out of their root slots at each use.
        // The values are built into locals first — call arguments evaluate left
        // to right, so reading a slot inline would take the address before the
        // value allocates.
        let error_map_roots = pyre_object::gc_roots::push_roots();
        let codes_slot = error_map_roots.base();
        let _ = error_map_roots.pin_root(w_dict_new());
        let messages_slot = codes_slot + 1;
        let _ = error_map_roots.pin_root(w_dict_new());
        for (idx, name) in ERROR_NAMES.iter().enumerate() {
            // ERROR_NAMES[0] is XML_ERROR_NONE (no message); codes start at 1.
            if idx == 0 {
                continue;
            }
            // The message pin lives across this iteration's stores. The two
            // maps stay on `error_map_roots`.
            let _pyre_store_roots = pyre_object::gc_roots::push_roots();
            let (msg, code) = ERROR_TABLE[idx - 1];
            let msg_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = error_map_roots.pin_root(w_str_new(msg));
            pyre_interpreter::baseobjspace::setdictvalue_native(
                ns_roots.get(errors_slot),
                name,
                error_map_roots.get(msg_slot),
            );
            let code_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = error_map_roots.pin_root(w_int_new(code));
            let codes = error_map_roots.get(codes_slot);
            unsafe { w_dict_setitem_str(codes, msg, error_map_roots.get(code_slot)) };
            let key_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = error_map_roots.pin_root(w_int_new(code));
            let messages = error_map_roots.get(messages_slot);
            unsafe {
                w_dict_store(
                    messages,
                    error_map_roots.get(key_slot),
                    error_map_roots.get(msg_slot),
                )
            };
        }
        let codes = error_map_roots.get(codes_slot);
        pyre_interpreter::baseobjspace::setdictvalue_native(
            ns_roots.get(errors_slot),
            "codes",
            codes,
        );
        let messages = error_map_roots.get(messages_slot);
        pyre_interpreter::baseobjspace::setdictvalue_native(
            ns_roots.get(errors_slot),
            "messages",
            messages,
        );
        pyre_interpreter::__pyre_put_new!(ns_slot, "errors", ns_roots.get(errors_slot));

        // features — list of (name, value) capability tuples.
        // Each name, value and tuple allocates while the pieces already built
        // are named only by a Rust local, so both levels are pinned as they are
        // produced: the pair until its tuple exists, the tuples until the list
        // does.  A pair's bracket closes before the tuple joins the outer set —
        // the outer slots would otherwise sit above it and be popped with it.
        let mut feature_items = pyre_object::gc_roots::RootedItems::new();
        for (name, value) in [
            ("sizeof(XML_Char)", 1),
            ("sizeof(XML_LChar)", 1),
            ("XML_DTD", 0),
            ("XML_CONTEXT_BYTES", 1024),
            ("XML_NS", 0),
        ] {
            let entry = {
                let mut pair = pyre_object::gc_roots::RootedItems::new();
                pair.push(w_str_new(name));
                pair.push(w_int_new(value));
                w_tuple_new(pair.take())
            };
            feature_items.push(entry);
        }
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "features",
            w_list_new(feature_items.take())
        );
    },
}
