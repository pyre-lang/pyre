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

/// `INIT_DATA_BUF_SIZE` (`xmlparse.c`): `XmlConvert` writes at most this
/// many UTF-8 bytes into `m_dataBuf` when the encoding `MUST_CONVERT`.
const INIT_DATA_BUF_SIZE: usize = 1024;

fn is_char_or_predefined_entity(ent: &str) -> bool {
    matches!(ent, "amp" | "lt" | "gt" | "quot" | "apos") || ent.starts_with('#')
}

/// `checkCharRefNumber`.
fn check_char_ref_number(code: u32) -> Option<char> {
    if code > 0x10FFFF {
        return None;
    }
    if (0xD800..=0xDFFF).contains(&code) {
        return None;
    }
    if code == 0xFFFE || code == 0xFFFF {
        return None;
    }
    if code < 0x20 && !matches!(code, 0x09 | 0x0A | 0x0D) {
        return None;
    }
    char::from_u32(code)
}

/// `XmlCharRefNumber`.
fn char_from_char_ref(ent: &str) -> Option<char> {
    let (digits, radix) = if let Some(hex) = ent.strip_prefix("#x") {
        (hex, 16)
    } else if let Some(dec) = ent.strip_prefix('#') {
        (dec, 10)
    } else {
        return None;
    };
    let code = u32::from_str_radix(digits, radix).ok()?;
    check_char_ref_number(code)
}

#[derive(Clone, Copy)]
struct XmlPos {
    index: usize,
    line: usize,
    col: usize,
}

/// Advance `start` by `text` with `XmlUpdatePosition` rules: CR is a line
/// break, and LF immediately after CR is not.
fn pos_after(start: XmlPos, text: &str) -> XmlPos {
    let mut pos = start;
    let mut last_cr = false;
    for ch in text.chars() {
        match ch {
            '\r' => {
                pos.index += ch.len_utf8();
                pos.line += 1;
                pos.col = 0;
                last_cr = true;
            }
            '\n' => {
                pos.index += ch.len_utf8();
                if !last_cr {
                    pos.line += 1;
                    pos.col = 0;
                }
                last_cr = false;
            }
            _ => {
                pos.index += ch.len_utf8();
                pos.col += 1;
                last_cr = false;
            }
        }
    }
    pos
}

fn decl_token_len(rest: &str) -> usize {
    let Some(ch) = rest.chars().next() else {
        return 0;
    };
    if rest.starts_with("<!") {
        let mut j = 2;
        for c in rest[2..].chars() {
            if MiniXmlParser::is_name_char(c) {
                j += c.len_utf8();
            } else {
                break;
            }
        }
        return j;
    }
    if matches!(ch, ' ' | '\t' | '\r' | '\n') {
        let mut j = 0;
        for c in rest.chars() {
            if matches!(c, ' ' | '\t' | '\r' | '\n') {
                j += c.len_utf8();
            } else {
                break;
            }
        }
        return j;
    }
    if ch == '"' || ch == '\'' {
        let mut j = ch.len_utf8();
        if let Some(rel) = rest[j..].find(ch) {
            j += rel + ch.len_utf8();
        } else {
            j = rest.len();
        }
        return j;
    }
    if ch == '#' {
        let mut j = 1;
        for c in rest[1..].chars() {
            if MiniXmlParser::is_name_char(c) {
                j += c.len_utf8();
            } else {
                break;
            }
        }
        return j;
    }
    if MiniXmlParser::is_name_char(ch) {
        let mut j = ch.len_utf8();
        for c in rest[j..].chars() {
            if MiniXmlParser::is_name_char(c) {
                j += c.len_utf8();
            } else {
                break;
            }
        }
        if rest[j..].starts_with(['?', '*', '+']) {
            j += 1;
        }
        return j;
    }
    if ch == ')' {
        let mut j = 1;
        if rest[j..].starts_with(['?', '*', '+']) {
            j += 1;
        }
        return j;
    }
    ch.len_utf8()
}

/// `ENTITY` `textPtr` / `open`.
struct InternalEntity {
    text: String,
    open: bool,
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
    internal_entities: HashMap<String, InternalEntity>,
    external_entities: HashMap<String, (String, Option<String>)>,
    root_closed: bool,
    char_buffer: String,
    deferred: bool,
    convert_chunks: bool,
    /// Pending tail from the last decode: `0` none, `1` `XML_TOK_PARTIAL`
    /// (odd UTF-16 byte), `2` `XML_TOK_PARTIAL_CHAR` (incomplete sequence).
    partial_tail_kind: i64,
    /// `externalEntityContentProcessor`: a sub-parser whose context is not
    /// `None`.
    is_external_entity: bool,
    /// `externalParEntInitProcessor`: the subparser was created with a
    /// `None` context.
    is_external_dtd: bool,
    /// `m_accounting.countBytesIndirect`
    bytes_indirect: u64,
    /// `dtd->hasParamEntityRefs`
    has_param_entity_refs: bool,
    /// `dtd->standalone`, set from `standalone="yes"` in the XML declaration.
    standalone: bool,
    /// `(element, attribute, is_cdata, default)` for each ATTLIST attribute
    /// (`defineAttribute`). The first declaration of an element/attribute
    /// pair wins.
    attlist_decls: Vec<(String, String, bool, Option<String>)>,
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

    fn new(
        parser_slot: usize,
        input: &'a str,
        isfinal: bool,
        suppress_until: usize,
        convert_chunks: bool,
    ) -> Self {
        let has_param_entity_refs = pyre_interpreter::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(parser_slot),
            "_pyre_use_foreign_dtd",
        )
        .map(is_true_obj)
        .unwrap_or(false);
        let partial_tail_kind = get_parser_int(
            pyre_object::gc_roots::shadow_stack_get(parser_slot),
            "_pyre_partial_tail_kind",
            0,
        );
        let is_subparser = pyre_interpreter::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(parser_slot),
            "_pyre_is_subparser",
        )
        .map(is_true_obj)
        .unwrap_or(false);
        let context_none = match pyre_interpreter::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(parser_slot),
            "_pyre_external_context",
        ) {
            Ok(obj) => obj.is_null() || unsafe { is_none(obj) },
            Err(_) => true,
        };
        let is_external_entity = is_subparser && !context_none;
        let is_external_dtd = is_subparser && context_none;
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
            deferred: false,
            convert_chunks,
            partial_tail_kind,
            is_external_entity,
            is_external_dtd,
            bytes_indirect: 0,
            has_param_entity_refs,
            standalone: false,
            attlist_decls: Vec::new(),
        }
    }

    fn parse(mut self) -> Result<usize, pyre_interpreter::PyError> {
        let partial_char_final = self.isfinal && self.partial_tail_kind == 2;
        if partial_char_final {
            let cut = match self.input.rfind('<') {
                Some(i) if !self.input[i..].contains('>') => i,
                _ => self.input.len(),
            };
            self.input = &self.input[..cut];
            self.isfinal = false;
        }
        if self.is_external_dtd {
            return self.parse_external_subset(partial_char_final);
        }
        if !self.is_external_entity {
            self.skip_ws_report(true)?;
        }
        if self.starts_with("<?xml") {
            self.parse_xml_decl()?;
        }
        loop {
            if self.eof() || self.deferred {
                break;
            }
            if !self.is_external_entity && self.root_closed {
                self.skip_ws_report(true)?;
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
            if !self.is_external_entity && self.stack.is_empty() && self.peek_is_ws() {
                self.skip_ws_report(true)?;
                continue;
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
            } else if !self.is_external_entity && self.stack.is_empty() && !self.root_closed {
                self.fail_prolog_text()?;
            } else {
                self.parse_chardata()?;
            }
        }
        self.finish(partial_char_final)
    }

    /// `doProlog` returns `XML_ERROR_PARTIAL_CHAR` / `XML_ERROR_UNCLOSED_TOKEN`
    /// at the end of a final buffer.  An external subset has no root element,
    /// so the "no element found" / "asynchronous entity" checks do not fire.
    fn finish(mut self, partial_char_final: bool) -> Result<usize, pyre_interpreter::PyError> {
        if self.isfinal && self.partial_tail_kind == 1 {
            return self.fail("unclosed token");
        }
        if partial_char_final {
            self.pos = pos_after(
                XmlPos {
                    index: 0,
                    line: 1,
                    col: 0,
                },
                self.input,
            );
            self.flush_character_buffer()?;
            return self.fail("partial character");
        }
        if self.isfinal && self.pos.index == 0 && !self.input.is_empty() {
            return self.fail("unclosed token");
        }
        if self.isfinal && !self.stack.is_empty() && !self.is_external_dtd {
            if self.is_external_entity {
                return self.fail("asynchronous entity");
            }
            return self.fail("no element found");
        }
        if self.isfinal
            && self.stack.is_empty()
            && !self.root_closed
            && !self.is_external_entity
            && !self.is_external_dtd
        {
            return self.fail("no element found");
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

    /// `XmlUpdatePosition` (`PREFIX(updatePosition)`): `BT_CR` increments
    /// the line; a `BT_LF` immediately after a CR is consumed with it.
    fn bump_char(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        match ch {
            '\r' => {
                self.pos.index += ch.len_utf8();
                self.pos.line += 1;
                self.pos.col = 0;
            }
            '\n' => {
                let after_cr =
                    self.pos.index > 0 && self.input.as_bytes()[self.pos.index - 1] == b'\r';
                self.pos.index += ch.len_utf8();
                if !after_cr {
                    self.pos.line += 1;
                    self.pos.col = 0;
                }
            }
            _ => {
                self.pos.index += ch.len_utf8();
                self.pos.col += 1;
            }
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
        while self.peek_is_ws() {
            self.bump_char();
        }
    }

    fn peek_is_ws(&self) -> bool {
        matches!(self.peek_char(), Some(' ' | '\t' | '\r' | '\n'))
    }

    fn is_name_char(ch: char) -> bool {
        ch.is_alphanumeric() || matches!(ch, '_' | '-' | ':' | '.')
    }

    fn read_name(&mut self) -> Result<String, pyre_interpreter::PyError> {
        let mut name = String::new();
        while let Some(ch) = self.peek_char() {
            if Self::is_name_char(ch) {
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

    fn read_attlist_enum(&mut self) -> Result<Vec<String>, pyre_interpreter::PyError> {
        self.expect("(")?;
        let mut names = Vec::new();
        loop {
            self.skip_ws();
            names.push(self.read_name()?);
            self.skip_ws();
            if self.consume(")") {
                return Ok(names);
            }
            self.expect("|")?;
        }
    }

    fn fail_prolog_text(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let start = self.pos;
        let Some(ch) = self.peek_char() else {
            return Ok(());
        };
        if ch.is_alphabetic() || ch == '_' || ch == ':' {
            while let Some(next) = self.peek_char() {
                if Self::is_name_char(next) {
                    self.bump_char();
                } else {
                    break;
                }
            }
            if self.peek_char() == Some('<') {
                return self.fail("not well-formed (invalid token)");
            }
        }
        self.pos = start;
        self.fail("syntax error")
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
                return self.expand_entities(raw, false, self.pos);
            }
            self.bump_char();
        }
        self.fail("unterminated quoted string")
    }

    /// `appendAttributeValue`.
    fn append_attribute_value(
        &mut self,
        raw: &str,
        is_cdata: bool,
        out: &mut String,
        tag_pos: XmlPos,
        undef_pos: XmlPos,
        value_start: Option<XmlPos>,
    ) -> Result<(), pyre_interpreter::PyError> {
        let mut frames: Vec<(String, usize, XmlPos, Option<XmlPos>, Option<String>)> =
            vec![(raw.to_string(), 0, undef_pos, value_start, None)];
        let result = (|| {
            while !frames.is_empty() {
                let top = frames.len() - 1;
                let i = frames[top].1;
                if i >= frames[top].0.len() {
                    let name = frames.pop().unwrap().4;
                    if let Some(ref name) = name {
                        if let Some(ent) = self.internal_entities.get_mut(name) {
                            ent.open = false;
                        }
                    }
                    continue;
                }
                let ch = match frames[top].0[i..].chars().next() {
                    Some(ch) => ch,
                    None => break,
                };
                if matches!(ch, ' ' | '\t' | '\n' | '\r') {
                    let mut i = i;
                    i += ch.len_utf8();
                    if ch == '\r' && i < frames[top].0.len() && frames[top].0.as_bytes()[i] == b'\n'
                    {
                        i += 1;
                    }
                    if is_cdata || !(out.is_empty() || out.ends_with(' ')) {
                        out.push(' ');
                    }
                    frames[top].1 = i;
                    continue;
                }
                if ch == '&' {
                    let ref_pos_of =
                        |frames: &[(String, usize, XmlPos, Option<XmlPos>, Option<String>)]| {
                            match frames[top].3 {
                                Some(start) => pos_after(start, &frames[top].0[..i]),
                                None => frames[top].2,
                            }
                        };
                    let Some(end) = frames[top].0[i + 1..].find(';') else {
                        // `XML_ERROR_UNDEFINED_ENTITY` leaves `m_eventPtr` at the
                        // start tag (or the `<!ATTLIST` that owns a default).
                        return Err(self.make_error_at("undefined entity", tag_pos));
                    };
                    let ent = frames[top].0[i + 1..i + 1 + end].to_string();
                    let next_i = i + 1 + end + 1;
                    match ent.as_str() {
                        "amp" => out.push('&'),
                        "lt" => out.push('<'),
                        "gt" => out.push('>'),
                        "quot" => out.push('"'),
                        "apos" => out.push('\''),
                        _ if ent.starts_with('#') => {
                            let ch = char_from_char_ref(&ent).ok_or_else(|| {
                                self.make_error_at(
                                    "reference to invalid character number",
                                    ref_pos_of(&frames),
                                )
                            })?;
                            if is_cdata || ch != ' ' || !(out.is_empty() || out.ends_with(' ')) {
                                out.push(ch);
                            }
                        }
                        _ if self.internal_entities.contains_key(&ent) => {
                            if self.internal_entities[&ent].open {
                                return Err(self.make_error_at(
                                    "recursive entity reference",
                                    ref_pos_of(&frames),
                                ));
                            }
                            let stored = self.internal_entities[&ent].text.clone();
                            self.account_indirect(stored.len())
                                .map_err(|m| self.make_error_at(&m, ref_pos_of(&frames)))?;
                            if let Some(entity) = self.internal_entities.get_mut(&ent) {
                                entity.open = true;
                            }
                            frames[top].1 = next_i;
                            frames.push((stored, 0, tag_pos, None, Some(ent)));
                            continue;
                        }
                        _ if self.external_entities.contains_key(&ent) => {
                            return Err(self.make_error_at(
                                "reference to external entity in attribute",
                                ref_pos_of(&frames),
                            ));
                        }
                        _ => {
                            // `appendAttributeValue` `checkEntityDecl`.
                            if !self.has_param_entity_refs || self.standalone {
                                return Err(self.make_error_at("undefined entity", tag_pos));
                            }
                        }
                    }
                    frames[top].1 = next_i;
                    continue;
                }
                out.push(ch);
                frames[top].1 = i + ch.len_utf8();
            }
            Ok(())
        })();
        if result.is_err() {
            for (_, _, _, _, name) in &frames {
                if let Some(name) = name {
                    if let Some(ent) = self.internal_entities.get_mut(name) {
                        ent.open = false;
                    }
                }
            }
        }
        result
    }

    /// `storeAttributeValue`.
    fn store_attribute_value(
        &mut self,
        raw: &str,
        is_cdata: bool,
        value_start: XmlPos,
        tag_pos: XmlPos,
    ) -> Result<String, pyre_interpreter::PyError> {
        let mut out = String::new();
        self.append_attribute_value(
            raw,
            is_cdata,
            &mut out,
            tag_pos,
            value_start,
            Some(value_start),
        )?;
        if !is_cdata && out.ends_with(' ') {
            out.pop();
        }
        Ok(out)
    }

    fn read_attribute_value(
        &mut self,
        is_cdata: bool,
        tag_pos: XmlPos,
    ) -> Result<String, pyre_interpreter::PyError> {
        let quote = match self.bump_char() {
            Some(q @ ('"' | '\'')) => q,
            _ => return self.fail("expected quoted string"),
        };
        let value_start = self.pos;
        let start = self.pos.index;
        while let Some(ch) = self.peek_char() {
            if ch == quote {
                let raw = self.input[start..self.pos.index].to_string();
                self.bump_char();
                return self.store_attribute_value(&raw, is_cdata, value_start, tag_pos);
            }
            self.bump_char();
        }
        self.fail("unterminated quoted string")
    }

    /// `storeEntityValue`: expand character references (`XML_TOK_CHAR_REF`)
    /// and normalise newlines (`XML_TOK_DATA_NEWLINE` / `XML_TOK_TRAILING_CR`);
    /// copy general entity references (`XML_TOK_ENTITY_REF`) through unchanged.
    fn read_entity_value(&mut self) -> Result<String, pyre_interpreter::PyError> {
        let quote = match self.bump_char() {
            Some(q @ ('"' | '\'')) => q,
            _ => return self.fail("expected quoted string"),
        };
        let start = self.pos.index;
        while let Some(ch) = self.peek_char() {
            if ch == quote {
                let raw = &self.input[start..self.pos.index];
                self.bump_char();
                let mut out = String::new();
                let mut rest = raw;
                while let Some(pos) = rest.find('&') {
                    out.push_str(&rest[..pos].replace("\r\n", "\n").replace('\r', "\n"));
                    let after = &rest[pos + 1..];
                    let Some(end) = after.find(';') else {
                        return self.fail("undefined entity");
                    };
                    let ent = &after[..end];
                    if ent.starts_with('#') {
                        let ch = char_from_char_ref(ent).ok_or_else(|| {
                            self.make_error("reference to invalid character number")
                        })?;
                        out.push(ch);
                    } else {
                        out.push('&');
                        out.push_str(ent);
                        out.push(';');
                    }
                    rest = &after[end + 1..];
                }
                out.push_str(&rest.replace("\r\n", "\n").replace('\r', "\n"));
                return Ok(out);
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
        let mut encoding_slot = roots.pin_roots(&[w_none()]);
        let mut standalone_slot = roots.pin_roots(&[w_int_new(-1)]);
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
                "encoding" => {
                    encoding_slot = roots.pin_roots(&[w_str_new_managed(&value)]);
                }
                "standalone" => {
                    self.standalone = value == "yes";
                    standalone_slot =
                        roots.pin_roots(&[w_int_new(if self.standalone { 1 } else { 0 })]);
                }
                _ => {}
            }
        }
        self.set_event_position(event_pos);
        let encoding = roots.get(encoding_slot);
        if let Some(enc) = unsafe {
            if is_none(encoding) {
                None
            } else {
                Some(pyre_interpreter::baseobjspace::str_utf8_w(encoding)?.to_string())
            }
        } {
            let has_forced_encoding = match pyre_interpreter::baseobjspace::getattr_str(
                self.parser(),
                "_pyre_forced_encoding",
            ) {
                Ok(obj) if unsafe { is_str(obj) } => true,
                _ => false,
            };
            if !has_forced_encoding {
                validate_decl_encoding(&enc).map_err(|m| self.make_error(m))?;
            }
        }
        if self.handler_is_set("XmlDeclHandler") {
            self.handler_args("XmlDeclHandler")
                .arg(w_str_new_managed(&version))
                .arg(roots.get(encoding_slot))
                .arg(roots.get(standalone_slot))
                .call()?;
        } else {
            let text = self.input[event_pos.index..self.pos.index].to_string();
            self.report_default(&text)?;
        }
        let standalone = roots.get(standalone_slot);
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
                self.pos = event_pos;
                self.fail("unclosed token")
            } else {
                Ok(())
            };
        };
        let text = self.input[start..start + rel].to_string();
        let end = start + rel + 3;
        while self.pos.index < end {
            self.bump_char();
        }
        self.set_event_position(event_pos);
        if self.handler_is_set("CommentHandler") {
            self.handler_args("CommentHandler")
                .arg(w_str_new_managed(&normalize_lines(&text)))
                .call()
        } else {
            let raw = self.input[event_pos.index..self.pos.index].to_string();
            self.report_default(&raw)
        }
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
        let data = self.input[start..start + rel].to_string();
        let end = start + rel + 2;
        while self.pos.index < end {
            self.bump_char();
        }
        self.set_event_position(event_pos);
        if self.handler_is_set("ProcessingInstructionHandler") {
            self.handler_args("ProcessingInstructionHandler")
                .arg(w_str_new_managed(&target))
                .arg(w_str_new_managed(&normalize_lines(&data)))
                .call()
        } else {
            let raw = self.input[event_pos.index..self.pos.index].to_string();
            self.report_default(&raw)
        }
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
        let close_pos = pos_after(self.pos, &text);
        self.set_event_position(event_pos);
        self.flush_character_buffer()?;
        if self.handler_is_set("StartCdataSectionHandler") {
            self.call_handler_raw("StartCdataSectionHandler", &[])?;
        } else {
            self.report_default("<![CDATA[")?;
        }
        let end = start + rel + 3;
        while self.pos.index < end {
            self.bump_char();
        }
        self.report_text(&text)?;
        self.flush_character_buffer()?;
        if self.handler_is_set("EndCdataSectionHandler") {
            self.call_handler_raw("EndCdataSectionHandler", &[])
        } else {
            self.set_event_position(close_pos);
            self.report_default("]]>")
        }
    }

    fn parse_doctype(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let event_pos = self.pos;
        let report = !self.handler_is_set("StartDoctypeDeclHandler");
        let tok = self.pos;
        self.expect("<!DOCTYPE")?;
        self.report_consumed(tok, report)?;
        self.skip_ws_report(report)?;
        let name_pos = self.pos;
        let name = self.read_name()?;
        self.report_consumed(name_pos, report)?;
        // The two identifiers are minted here and used again only at the very
        // end, past the not-standalone callback, the internal subset and the
        // start-doctype handler — all of which run Python.
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.pin_roots(&[w_none(), w_none()]);
        let mut has_internal_subset = false;
        self.skip_ws_report(report)?;
        if self.starts_with("PUBLIC") {
            let kw = self.pos;
            self.expect("PUBLIC")?;
            self.report_consumed(kw, report)?;
            self.skip_ws_report(report)?;
            let q = self.pos;
            roots.set(base + 1, w_str_new_managed(&self.read_quoted()?));
            self.report_consumed(q, report)?;
            self.skip_ws_report(report)?;
            let q = self.pos;
            roots.set(base, w_str_new_managed(&self.read_quoted()?));
            self.report_consumed(q, report)?;
            self.has_param_entity_refs = true;
        } else if self.starts_with("SYSTEM") {
            let kw = self.pos;
            self.expect("SYSTEM")?;
            self.report_consumed(kw, report)?;
            self.skip_ws_report(report)?;
            let q = self.pos;
            roots.set(base, w_str_new_managed(&self.read_quoted()?));
            self.report_consumed(q, report)?;
            self.has_param_entity_refs = true;
        }
        self.skip_ws_report(report)?;
        let not_standalone_pending = pyre_interpreter::baseobjspace::getattr_str(
            self.parser(),
            "_pyre_not_standalone_pending",
        )
        .map(is_true_obj)
        .unwrap_or(false);
        let has_sysid = unsafe { !is_none(roots.get(base)) };
        let param_parsing = get_parser_int(self.parser(), "_pyre_param_entity_parsing", 0);
        // `doProlog` `XML_ROLE_DOCTYPE_SYSTEM_ID`: not-standalone when the
        // doctype has a system id, `!dtd->standalone`, and
        // `m_paramEntityParsing` is `XML_PARAM_ENTITY_PARSING_NEVER`.
        if not_standalone_pending || (has_sysid && !self.standalone && param_parsing == 0) {
            self.call_not_standalone()?;
            pyre_interpreter::baseobjspace::setdictvalue_native(
                self.parser(),
                "_pyre_not_standalone_pending",
                w_bool_from(false),
            );
        }
        if self.starts_with("[") {
            has_internal_subset = true;
            let br = self.pos;
            self.consume("[");
            self.report_consumed(br, report)?;
            self.set_event_position(event_pos);
            self.handler_args("StartDoctypeDeclHandler")
                .arg(w_str_new_managed(&name))
                .arg(roots.get(base))
                .arg(roots.get(base + 1))
                .arg(w_int_new(1))
                .call()?;
            self.parse_internal_subset()?;
            let br = self.pos;
            self.expect("]")?;
            self.report_consumed(br, report)?;
        } else {
            self.set_event_position(event_pos);
            self.handler_args("StartDoctypeDeclHandler")
                .arg(w_str_new_managed(&name))
                .arg(roots.get(base))
                .arg(roots.get(base + 1))
                .arg(w_int_new(0))
                .call()?;
        }
        self.skip_ws_report(report)?;
        let gt = self.pos;
        self.expect(">")?;
        // `doProlog` `XML_ROLE_DOCTYPE_CLOSE`: `handleDefault` is false when
        // `EndDoctypeDeclHandler` is set.
        self.report_consumed(gt, !self.handler_is_set("EndDoctypeDeclHandler"))?;
        self.maybe_external_dtd(roots.get(base + 1), roots.get(base))?;
        let _ = has_internal_subset;
        self.call_handler("EndDoctypeDeclHandler", &[])
    }

    fn parse_subset_decl(&mut self) -> Result<bool, pyre_interpreter::PyError> {
        if self.starts_with("<!ENTITY") {
            self.parse_entity_decl()?;
            Ok(true)
        } else if self.starts_with("%") {
            self.skip_parameter_entity_ref()?;
            Ok(true)
        } else if self.starts_with("<!ELEMENT") {
            self.parse_element_decl()?;
            Ok(true)
        } else if self.starts_with("<!ATTLIST") {
            self.parse_attlist_decl()?;
            Ok(true)
        } else if self.starts_with("<!NOTATION") {
            self.parse_notation_decl()?;
            Ok(true)
        } else if self.starts_with("<!--") {
            self.parse_comment()?;
            Ok(true)
        } else if self.starts_with("<?") {
            self.parse_pi()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn parse_internal_subset(&mut self) -> Result<(), pyre_interpreter::PyError> {
        loop {
            self.skip_ws_report(true)?;
            if self.starts_with("]") || self.eof() {
                return Ok(());
            }
            if !self.parse_subset_decl()? {
                self.skip_declaration()?;
            }
        }
    }

    /// `externalParEntInitProcessor`: parse an external subset (DTD).
    fn parse_external_subset(
        mut self,
        partial_char_final: bool,
    ) -> Result<usize, pyre_interpreter::PyError> {
        if self.starts_with("<?xml") {
            self.parse_xml_decl()?;
        }
        loop {
            self.skip_ws_report(true)?;
            if self.eof() {
                break;
            }
            if !self.parse_subset_decl()? {
                return self.fail("syntax error");
            }
        }
        self.finish(partial_char_final)
    }

    /// `doProlog` roles of `<!ENTITY`: each token is reported as it is
    /// consumed, and the handler fires at the token that completes the
    /// declaration (`XML_ROLE_ENTITY_VALUE`, `XML_ROLE_ENTITY_NOTATION_NAME`,
    /// `XML_ROLE_ENTITY_COMPLETE`).
    fn parse_entity_decl(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let tok = self.pos;
        self.expect("<!ENTITY")?;
        self.report_consumed(tok, !self.handler_is_set("EntityDeclHandler"))?;
        self.skip_ws_report(!self.handler_is_set("EntityDeclHandler"))?;
        let is_param = if self.starts_with("%") {
            let pct = self.pos;
            self.consume("%");
            self.report_consumed(pct, !self.handler_is_set("EntityDeclHandler"))?;
            self.skip_ws_report(!self.handler_is_set("EntityDeclHandler"))?;
            1
        } else {
            0
        };
        let name_pos = self.pos;
        let name = self.read_name()?;
        // `XML_ROLE_GENERAL_ENTITY_NAME`: a predefined name or a redeclaration
        // leaves `m_declEntity` NULL, so nothing is stored and no declaration
        // handler fires; the tokens go to the default handler.
        let redeclared = is_param == 0
            && (matches!(name.as_str(), "amp" | "lt" | "gt" | "quot" | "apos")
                || self.internal_entities.contains_key(&name)
                || self.external_entities.contains_key(&name));
        let decl_handler = !redeclared && self.handler_is_set("EntityDeclHandler");
        self.report_consumed(name_pos, !decl_handler)?;
        self.skip_ws_report(!decl_handler)?;
        // Every identifier this declaration mints is reachable only from its
        // local until the handler call, and the reads that follow — another
        // `read_quoted`, the position update, the handler itself — both
        // allocate and run Python.
        let roots = pyre_object::gc_roots::push_roots();
        let mut base = w_none();
        let slot = roots.pin_roots(&[w_none(), w_none(), w_none(), w_none()]);
        let mut is_external = false;
        if matches!(self.peek_char(), Some('"' | '\'')) {
            let q = self.pos;
            let v = self.read_entity_value()?;
            if is_param == 0 && !redeclared {
                self.internal_entities.insert(
                    name.clone(),
                    InternalEntity {
                        text: v.clone(),
                        open: false,
                    },
                );
            }
            roots.set(slot, w_str_new_managed(&v));
            // `XML_ROLE_ENTITY_VALUE`: the handler fires at the literal.
            if decl_handler {
                self.set_event_position(q);
                self.handler_args("EntityDeclHandler")
                    .arg(w_str_new_managed(&name))
                    .arg(w_int_new(is_param))
                    .arg(roots.get(slot))
                    .arg(base)
                    .arg(roots.get(slot + 1))
                    .arg(roots.get(slot + 2))
                    .arg(roots.get(slot + 3))
                    .call()?;
            } else {
                self.report_consumed(q, true)?;
            }
        } else if self.starts_with("PUBLIC") {
            let kw = self.pos;
            self.expect("PUBLIC")?;
            self.report_consumed(kw, !decl_handler)?;
            self.skip_ws_report(!decl_handler)?;
            let q = self.pos;
            roots.set(slot + 2, w_str_new_managed(&self.read_quoted()?));
            self.report_consumed(q, !decl_handler)?;
            self.skip_ws_report(!decl_handler)?;
            let q = self.pos;
            let system = self.read_quoted()?;
            self.report_consumed(q, !decl_handler)?;
            if is_param == 0 && !redeclared {
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
            }
            roots.set(slot + 1, w_str_new_managed(&system));
            is_external = true;
        } else if self.starts_with("SYSTEM") {
            let kw = self.pos;
            self.expect("SYSTEM")?;
            self.report_consumed(kw, !decl_handler)?;
            self.skip_ws_report(!decl_handler)?;
            let q = self.pos;
            let system = self.read_quoted()?;
            self.report_consumed(q, !decl_handler)?;
            if is_param == 0 && !redeclared {
                self.external_entities
                    .insert(name.clone(), (system.clone(), None));
            }
            roots.set(slot + 1, w_str_new_managed(&system));
            is_external = true;
        }
        self.skip_ws_report(!decl_handler)?;
        if self.starts_with("NDATA") {
            let kw = self.pos;
            self.expect("NDATA")?;
            self.report_consumed(kw, !decl_handler)?;
            self.skip_ws_report(!decl_handler)?;
            let n = self.pos;
            roots.set(slot + 3, w_str_new_managed(&self.read_name()?));
            // `XML_ROLE_ENTITY_NOTATION_NAME`: UnparsedEntityDeclHandler, else
            // EntityDeclHandler (`is_param` is 0), else Default.
            if !redeclared && self.handler_is_set("UnparsedEntityDeclHandler") {
                self.set_event_position(n);
                self.handler_args("UnparsedEntityDeclHandler")
                    .arg(w_str_new_managed(&name))
                    .arg(base)
                    .arg(roots.get(slot + 1))
                    .arg(roots.get(slot + 2))
                    .arg(roots.get(slot + 3))
                    .call()?;
            } else if decl_handler {
                self.set_event_position(n);
                self.handler_args("EntityDeclHandler")
                    .arg(w_str_new_managed(&name))
                    .arg(w_int_new(0))
                    .arg(roots.get(slot))
                    .arg(base)
                    .arg(roots.get(slot + 1))
                    .arg(roots.get(slot + 2))
                    .arg(roots.get(slot + 3))
                    .call()?;
            } else {
                self.report_consumed(n, true)?;
            }
        }
        self.skip_ws_report(!decl_handler)?;
        let gt = self.pos;
        self.expect(">")?;
        let has_ndata = !unsafe { is_none(roots.get(slot + 3)) };
        let _ = &mut base;
        // `XML_ROLE_ENTITY_COMPLETE`: an external entity with no NDATA calls
        // EntityDeclHandler at `>`; otherwise `>` is an ordinary token.
        if is_external && !has_ndata && decl_handler {
            self.set_event_position(gt);
            self.handler_args("EntityDeclHandler")
                .arg(w_str_new_managed(&name))
                .arg(w_int_new(is_param))
                .arg(roots.get(slot))
                .arg(base)
                .arg(roots.get(slot + 1))
                .arg(roots.get(slot + 2))
                .arg(roots.get(slot + 3))
                .call()?;
        } else {
            self.report_consumed(gt, !decl_handler)?;
        }
        Ok(())
    }

    fn skip_parameter_entity_ref(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let start = self.pos;
        self.expect("%")?;
        let _name = self.read_name()?;
        self.expect(";")?;
        self.has_param_entity_refs = true;
        self.set_event_position(start);
        let text = self.input[start.index..self.pos.index].to_string();
        self.report_default(&text)?;
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
                        return Err(pyre_interpreter::PyError::recursion_error(
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
        if self.handler_is_set("ElementDeclHandler") {
            let name = w_str_new_managed(&name);
            self.handler_args("ElementDeclHandler")
                .arg(name)
                .arg(roots.get(model_slot))
                .call()
        } else {
            self.report_decl_tokens(event_pos, self.pos.index)
        }
    }

    fn parse_attlist_decl(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let event_pos = self.pos;
        let mut seg_start: XmlPos = event_pos;
        self.expect("<!ATTLIST")?;
        self.skip_ws();
        let elem = self.read_name()?;
        loop {
            self.skip_ws();
            if self.consume(">") {
                self.report_decl_tokens_unless(seg_start, self.pos.index, "AttlistDeclHandler")?;
                return Ok(());
            }
            let attr = self.read_name()?;
            self.skip_ws();
            let kind = if self.starts_with("(") {
                let names = self.read_attlist_enum()?;
                format!("({})", names.join("|"))
            } else {
                let type_name = self.read_name()?;
                if type_name == "NOTATION" {
                    self.skip_ws();
                    let names = self.read_attlist_enum()?;
                    format!("NOTATION({})", names.join("|"))
                } else {
                    type_name
                }
            };
            self.skip_ws();
            let is_cdata = kind == "CDATA";
            let (default, required) = if self.consume("#REQUIRED") {
                (None, 1)
            } else if self.consume("#IMPLIED") {
                (None, 0)
            } else if self.consume("#FIXED") {
                self.skip_ws();
                (Some(self.read_attribute_value(is_cdata, event_pos)?), 1)
            } else {
                (Some(self.read_attribute_value(is_cdata, event_pos)?), 0)
            };
            if !self
                .attlist_decls
                .iter()
                .any(|(e, a, _, _)| e == &elem && a == &attr)
            {
                self.attlist_decls
                    .push((elem.clone(), attr.clone(), is_cdata, default.clone()));
            }
            self.report_decl_tokens_unless(seg_start, self.pos.index, "AttlistDeclHandler")?;
            if self.handler_is_set("AttlistDeclHandler") {
                self.set_event_position(event_pos);
                self.handler_args("AttlistDeclHandler")
                    .arg(w_str_new_managed(&elem))
                    .arg(w_str_new_managed(&attr))
                    .arg(w_str_new_managed(&kind))
                    .arg(match &default {
                        Some(v) => w_str_new_managed(v),
                        None => w_none(),
                    })
                    .arg(w_int_new(required))
                    .call()?;
            }
            seg_start = self.pos;
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
        if self.handler_is_set("NotationDeclHandler") {
            self.handler_args("NotationDeclHandler")
                .arg(w_str_new_managed(&name))
                .arg(w_none())
                .arg(roots.get(base))
                .arg(roots.get(base + 1))
                .call()
        } else {
            self.report_decl_tokens(event_pos, self.pos.index)
        }
    }

    fn skip_declaration(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let start = self.pos;
        if self.starts_with("<!") {
            self.skip_until_gt()?;
            self.set_event_position(start);
            let text = self.input[start.index..self.pos.index].to_string();
            self.report_default(&text)
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
                let n_specified = self.apply_attlist_defaults(&raw_name, &mut attrs);
                self.set_event_position(event_pos);
                self.apply_namespace_decls(&attrs, &mut ns_declared)?;
                let name = self.expand_name(&raw_name, false, event_pos)?;
                let (expanded_attrs, n_expanded_specified) =
                    self.expand_attributes(&attrs, n_specified, event_pos)?;
                let reported = if self.specified_attributes() {
                    &expanded_attrs[..n_expanded_specified]
                } else {
                    &expanded_attrs[..]
                };
                let start_set = self.handler_is_set("StartElementHandler");
                self.set_event_position(event_pos);
                if start_set {
                    // The attribute container is a `dict` or a `list`, both of which
                    // move, and the position update and the interned element name
                    // that follow allocate.  Pin it on creation and read it back out
                    // of the root slot at the call; the name is built into a local
                    // first, since the argument list evaluates left to right.
                    let roots = pyre_object::gc_roots::push_roots();
                    let attrs_slot = roots.base();
                    let _ = roots.pin_root(self.convert_attributes(reported));
                    let w_name = self.intern_string(&name);
                    self.call_handler("StartElementHandler", &[w_name, roots.get(attrs_slot)])?;
                }
                // Publish the end after `/>` (`test_pyexpat.PositionTest`
                // reads CurrentByteIndex on the empty end).  Keep the
                // suppress bit from the opening `<`: feed/close reparses
                // pending input and would otherwise re-fire only the end
                // (its index sits *at* `_pyre_emit_upto`) and pop an empty
                // TreeBuilder stack.
                self.publish_event_position(self.pos);
                // `doContent` `XML_TOK_EMPTY_ELEMENT_NO_ATTS`: read the end
                // handler after the start callback.
                let end_set = self.handler_is_set("EndElementHandler");
                if end_set {
                    self.call_handler("EndElementHandler", &[self.intern_string(&name)])?;
                }
                // `doContent` `noElmHandlers`: report the raw tag when no
                // start handler was set before the call and no end handler
                // is set after it.
                if !start_set && !end_set {
                    let text = self.input[event_pos.index..self.pos.index].to_string();
                    self.set_event_position(event_pos);
                    self.report_default(&text)?;
                }
                self.end_namespace_scope(ns_declared)?;
                if self.stack.is_empty() && !self.is_external_entity {
                    self.root_closed = true;
                }
                return Ok(());
            }
            if self.consume(">") {
                let n_specified = self.apply_attlist_defaults(&raw_name, &mut attrs);
                self.set_event_position(event_pos);
                self.apply_namespace_decls(&attrs, &mut ns_declared)?;
                let name = self.expand_name(&raw_name, false, event_pos)?;
                let (expanded_attrs, n_expanded_specified) =
                    self.expand_attributes(&attrs, n_specified, event_pos)?;
                let reported = if self.specified_attributes() {
                    &expanded_attrs[..n_expanded_specified]
                } else {
                    &expanded_attrs[..]
                };
                self.stack.push(name.clone());
                self.ns_stack.push(ns_declared);
                self.set_event_position(event_pos);
                if self.handler_is_set("StartElementHandler") {
                    let roots = pyre_object::gc_roots::push_roots();
                    let attrs_slot = roots.base();
                    let _ = roots.pin_root(self.convert_attributes(reported));
                    let w_name = self.intern_string(&name);
                    self.call_handler("StartElementHandler", &[w_name, roots.get(attrs_slot)])?;
                } else {
                    let text = self.input[event_pos.index..self.pos.index].to_string();
                    self.report_default(&text)?;
                }
                return Ok(());
            }
            let attr_pos = self.pos;
            let attr_name = self.read_name()?;
            if attrs.iter().any(|(existing, _)| existing == &attr_name) {
                self.pos = attr_pos;
                return self.fail("duplicate attribute");
            }
            self.skip_ws();
            self.expect("=")?;
            self.skip_ws();
            // `storeAtts` looks `attId` up in `elementType->defaultAtts`.
            let is_cdata = self
                .attlist_decls
                .iter()
                .find(|(elem, attr, _, _)| elem == &raw_name && attr == &attr_name)
                .map(|(_, _, is_cdata, _)| *is_cdata)
                .unwrap_or(true);
            let attr_value = self.read_attribute_value(is_cdata, event_pos)?;
            attrs.push((attr_name, attr_value));
        }
    }

    fn parse_end_element(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let event_pos = self.pos;
        self.expect("</")?;
        let name_pos = self.pos;
        let raw_name = self.read_name()?;
        self.skip_ws();
        self.expect(">")?;
        let name = self.expand_name(&raw_name, false, self.pos)?;
        match self.stack.pop() {
            Some(open) if open == name => {
                self.set_event_position(event_pos);
                if self.handler_is_set("EndElementHandler") {
                    self.call_handler("EndElementHandler", &[self.intern_string(&name)])?;
                } else {
                    let text = self.input[event_pos.index..self.pos.index].to_string();
                    self.report_default(&text)?;
                }
                if let Some(declared) = self.ns_stack.pop() {
                    self.end_namespace_scope(declared)?;
                }
                if self.stack.is_empty() && !self.is_external_entity {
                    self.root_closed = true;
                }
                Ok(())
            }
            _ => {
                self.pos = name_pos;
                self.fail("mismatched tag")
            }
        }
    }

    /// `XmlContentTok` / `doContent`: character data is one token at a time.
    fn parse_chardata(&mut self) -> Result<(), pyre_interpreter::PyError> {
        loop {
            if self.eof() || self.deferred {
                return Ok(());
            }
            let Some(ch) = self.peek_char() else {
                return Ok(());
            };
            if ch == '<' {
                return Ok(());
            }
            if ch == '\0' {
                return self.fail("unclosed token");
            }
            if ch == '\r' || ch == '\n' {
                let tok_start = self.pos;
                if ch == '\r' && !self.isfinal && self.pos.index + ch.len_utf8() == self.input.len()
                {
                    // `XML_TOK_TRAILING_CR`
                    self.deferred = true;
                    return Ok(());
                }
                self.bump_char();
                if ch == '\r' && self.peek_char() == Some('\n') {
                    self.bump_char();
                }
                self.set_event_position(tok_start);
                if self.handler_is_set("CharacterDataHandler") {
                    self.report_text("\n")?;
                } else {
                    let raw = self.input[tok_start.index..self.pos.index].to_string();
                    self.report_default_maybe_chunked(&raw)?;
                }
                continue;
            }
            if ch == '&' {
                let tok_start = self.pos;
                self.bump_char();
                if self.peek_char() == Some('#') {
                    self.bump_char();
                    if self.peek_char() == Some('x') {
                        self.bump_char();
                        if !matches!(self.peek_char(), Some('0'..='9' | 'a'..='f' | 'A'..='F')) {
                            if !self.eof() {
                                return self.fail("not well-formed (invalid token)");
                            }
                        } else {
                            self.bump_char();
                            while matches!(
                                self.peek_char(),
                                Some('0'..='9' | 'a'..='f' | 'A'..='F')
                            ) {
                                self.bump_char();
                            }
                        }
                    } else if !matches!(self.peek_char(), Some('0'..='9')) {
                        if !self.eof() {
                            return self.fail("not well-formed (invalid token)");
                        }
                    } else {
                        self.bump_char();
                        while matches!(self.peek_char(), Some('0'..='9')) {
                            self.bump_char();
                        }
                    }
                } else if match self.peek_char() {
                    Some(ch) => Self::is_name_char(ch) && !matches!(ch, '0'..='9' | '-' | '.'),
                    None => false,
                } {
                    self.bump_char();
                    while let Some(next) = self.peek_char() {
                        if Self::is_name_char(next) {
                            self.bump_char();
                        } else {
                            break;
                        }
                    }
                } else if !self.eof() {
                    return self.fail("not well-formed (invalid token)");
                }
                if self.peek_char() == Some(';') {
                    self.bump_char();
                } else if self.eof() {
                    self.pos = tok_start;
                    if !self.isfinal {
                        self.deferred = true;
                        return Ok(());
                    }
                    return self.fail("unclosed token");
                } else {
                    return self.fail("not well-formed (invalid token)");
                }
                let ref_text = self.input[tok_start.index..self.pos.index].to_owned();
                self.set_event_position(tok_start);
                let ent = if ref_text.len() >= 2 {
                    &ref_text[1..ref_text.len() - 1]
                } else {
                    ""
                };
                if is_char_or_predefined_entity(ent) && !self.handler_is_set("CharacterDataHandler")
                {
                    if ent.starts_with('#') && char_from_char_ref(ent).is_none() {
                        return Err(
                            self.make_error_at("reference to invalid character number", tok_start)
                        );
                    }
                    self.report_default_maybe_chunked(&ref_text)?;
                    continue;
                }
                let expanded = self.expand_entities(&ref_text, true, tok_start)?;
                // `doContent` `XML_TOK_CHAR_REF` does not newline-normalise.
                if is_char_or_predefined_entity(ent) {
                    self.emit_reported_run(&expanded)?;
                } else {
                    self.report_entity_text(&expanded)?;
                }
                continue;
            }
            let tok_start = self.pos;
            let mut split_pos = None;
            while let Some(next) = self.peek_char() {
                if matches!(next, '<' | '&' | '\r' | '\n' | '\0') {
                    break;
                }
                if self.pos.index == self.suppress_until && self.pos.index > tok_start.index {
                    split_pos = Some(self.pos);
                }
                self.bump_char();
            }
            self.emit_data_chars(tok_start, split_pos, self.pos.index)?;
        }
    }

    fn emit_data_chars(
        &mut self,
        start: XmlPos,
        split: Option<XmlPos>,
        end: usize,
    ) -> Result<(), pyre_interpreter::PyError> {
        if end <= self.suppress_until {
            self.set_event_position(start);
            return Ok(());
        }
        let from = split.unwrap_or(start);
        self.set_event_position(from);
        let text = self.input[from.index..end].to_string();
        if !self.is_external_entity && self.stack.is_empty() && text.trim().is_empty() {
            return Ok(());
        }
        if !self.handler_is_set("CharacterDataHandler") {
            return self.report_default_maybe_chunked(&text);
        }
        self.emit_maybe_chunked(&text)
    }

    /// Internal entity replacement text is in the internal encoding, so it is
    /// never `MUST_CONVERT`-chunked (`processEntity` uses `m_internalEncoding`).
    fn report_entity_text(&mut self, text: &str) -> Result<(), pyre_interpreter::PyError> {
        let convert = std::mem::replace(&mut self.convert_chunks, false);
        let result = self.report_text(text);
        self.convert_chunks = convert;
        result
    }

    /// Split expansion / CDATA text the same way `doContent` splits
    /// `XML_TOK_DATA_CHARS` and `XML_TOK_DATA_NEWLINE`.
    fn report_text(&mut self, text: &str) -> Result<(), pyre_interpreter::PyError> {
        let bytes = text.as_bytes();
        let mut run_start = 0;
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'\r' => {
                    if run_start < i {
                        self.emit_reported_run(&text[run_start..i])?;
                    }
                    let nl_start = i;
                    i += 1;
                    if i < bytes.len() && bytes[i] == b'\n' {
                        i += 1;
                    }
                    self.emit_reported_newline(&text[nl_start..i])?;
                    run_start = i;
                }
                b'\n' => {
                    if run_start < i {
                        self.emit_reported_run(&text[run_start..i])?;
                    }
                    let nl_start = i;
                    i += 1;
                    self.emit_reported_newline(&text[nl_start..i])?;
                    run_start = i;
                }
                _ => i += 1,
            }
        }
        if run_start < text.len() {
            self.emit_reported_run(&text[run_start..])?;
        }
        Ok(())
    }

    fn emit_reported_run(&mut self, text: &str) -> Result<(), pyre_interpreter::PyError> {
        if !self.is_external_entity && self.stack.is_empty() && text.trim().is_empty() {
            return Ok(());
        }
        if !self.handler_is_set("CharacterDataHandler") {
            return self.report_default_maybe_chunked(text);
        }
        self.emit_maybe_chunked(text)
    }

    /// `doCdataSection` `XML_TOK_DATA_NEWLINE`: `raw` is the newline as it
    /// appears in the source (`"\r\n"`, `"\r"`, or `"\n"`). Without a
    /// character-data handler it is reported unchanged; with one, `"\n"`.
    fn emit_reported_newline(&mut self, raw: &str) -> Result<(), pyre_interpreter::PyError> {
        if !self.is_external_entity && self.stack.is_empty() {
            return Ok(());
        }
        if !self.handler_is_set("CharacterDataHandler") {
            return self.report_default(raw);
        }
        self.emit_character_data("\n")
    }

    /// `MUST_CONVERT`: `XmlConvert` fills `m_dataBuf` (`INIT_DATA_BUF_SIZE`).
    fn emit_maybe_chunked(&mut self, text: &str) -> Result<(), pyre_interpreter::PyError> {
        if text.is_empty() {
            return Ok(());
        }
        if !self.convert_chunks {
            return self.emit_character_data(text);
        }
        let mut start = 0;
        while start < text.len() {
            let mut end = (start + INIT_DATA_BUF_SIZE).min(text.len());
            while end > start && !text.is_char_boundary(end) {
                end -= 1;
            }
            self.emit_character_data(&text[start..end])?;
            start = end;
        }
        Ok(())
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
        if handler_int_rv(ret, "NotStandalone")? == 0 {
            return self.fail("document is not standalone");
        }
        Ok(())
    }

    fn handler_is_set(&self, name: &str) -> bool {
        match pyre_interpreter::baseobjspace::getattr_str(self.parser(), name) {
            Ok(handler) => !(handler.is_null() || unsafe { is_none(handler) }),
            Err(_) => false,
        }
    }

    fn default_handler_name(&self) -> Option<&'static str> {
        let kind = match pyre_interpreter::baseobjspace::getattr_str(
            self.parser(),
            "_pyre_default_kind",
        ) {
            Ok(obj) if !obj.is_null() && unsafe { !is_none(obj) && is_str(obj) } => {
                match pyre_interpreter::baseobjspace::str_utf8_w(obj) {
                    Ok(s) => s.to_string(),
                    Err(_) => return None,
                }
            }
            _ => return None,
        };
        if !self.handler_is_set(&kind) {
            return None;
        }
        match kind.as_str() {
            "DefaultHandler" => Some("DefaultHandler"),
            "DefaultHandlerExpand" => Some("DefaultHandlerExpand"),
            _ => None,
        }
    }

    fn report_default(&mut self, text: &str) -> Result<(), pyre_interpreter::PyError> {
        match self.default_handler_name() {
            Some(name) => self.call_handler(name, &[w_str_new_managed(text)]),
            None => Ok(()),
        }
    }

    /// `reportDefault` converts through `m_dataBuf` (`INIT_DATA_BUF_SIZE`)
    /// when `MUST_CONVERT`.
    fn report_default_maybe_chunked(
        &mut self,
        text: &str,
    ) -> Result<(), pyre_interpreter::PyError> {
        if text.is_empty() {
            return Ok(());
        }
        if !self.convert_chunks {
            return self.report_default(text);
        }
        let mut start = 0;
        while start < text.len() {
            let mut end = (start + INIT_DATA_BUF_SIZE).min(text.len());
            while end > start && !text.is_char_boundary(end) {
                end -= 1;
            }
            self.report_default(&text[start..end])?;
            start = end;
        }
        Ok(())
    }

    /// `XML_SetDefaultHandler` clears `m_defaultExpandInternalEntities`;
    /// `XML_SetDefaultHandlerExpand` sets it.
    fn expands_internal_entities(&self) -> bool {
        pyre_interpreter::baseobjspace::getattr_str(self.parser(), "_pyre_default_expand")
            .map(is_true_obj)
            .unwrap_or(true)
    }

    /// `doProlog` `handleDefault`: ELEMENT / ATTLIST / NOTATION tokens.
    fn report_decl_tokens(
        &mut self,
        start: XmlPos,
        end: usize,
    ) -> Result<(), pyre_interpreter::PyError> {
        self.report_decl_tokens_body(start, end, None)
    }

    /// Like `report_decl_tokens`, but re-checks `handler` before each token
    /// and reports only while it is unset (`doProlog` ATTLIST roles).
    fn report_decl_tokens_unless(
        &mut self,
        start: XmlPos,
        end: usize,
        handler: &str,
    ) -> Result<(), pyre_interpreter::PyError> {
        self.report_decl_tokens_body(start, end, Some(handler))
    }

    fn report_decl_tokens_body(
        &mut self,
        start: XmlPos,
        end: usize,
        handler: Option<&str>,
    ) -> Result<(), pyre_interpreter::PyError> {
        let text = self.input[start.index..end].to_string();
        let mut i = 0;
        let mut pos = start;
        while i < text.len() {
            let rest = &text[i..];
            let n = decl_token_len(rest);
            let token = rest[..n].to_string();
            let report = match handler {
                Some(name) => !self.handler_is_set(name),
                None => true,
            };
            if report {
                self.set_event_position(pos);
                self.report_default(&token)?;
            }
            pos = pos_after(pos, &rest[..n]);
            i += n;
        }
        Ok(())
    }

    fn skip_ws_report(&mut self, report: bool) -> Result<(), pyre_interpreter::PyError> {
        let start = self.pos;
        self.skip_ws();
        self.report_consumed(start, report)
    }

    fn report_consumed(
        &mut self,
        start: XmlPos,
        report: bool,
    ) -> Result<(), pyre_interpreter::PyError> {
        if report && self.pos.index > start.index {
            self.set_event_position(start);
            let text = self.input[start.index..self.pos.index].to_string();
            self.report_default(&text)?;
        }
        Ok(())
    }

    fn set_event_position(&mut self, pos: XmlPos) {
        self.suppress_current = pos.index < self.suppress_until;
        self.publish_event_position(pos);
    }

    fn publish_event_position(&mut self, pos: XmlPos) {
        // `XML_GetErrorLineNumber` / `XML_GetErrorColumnNumber` /
        // `XML_GetErrorByteIndex` are `#define`d to the `XML_GetCurrent*`
        // functions (`expat.h`).
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
        pyre_interpreter::baseobjspace::setdictvalue_native(
            self.parser(),
            "ErrorLineNumber",
            w_int_new(pos.line as i64),
        );
        pyre_interpreter::baseobjspace::setdictvalue_native(
            self.parser(),
            "ErrorColumnNumber",
            w_int_new(pos.col as i64),
        );
        pyre_interpreter::baseobjspace::setdictvalue_native(
            self.parser(),
            "ErrorByteIndex",
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

    fn specified_attributes(&self) -> bool {
        pyre_interpreter::baseobjspace::getattr_str(self.parser(), "specified_attributes")
            .map(is_true_obj)
            .unwrap_or(false)
    }

    fn apply_attlist_defaults(&self, raw_name: &str, attrs: &mut Vec<(String, String)>) -> usize {
        let n_specified = attrs.len();
        for (elem, attr, _, default) in &self.attlist_decls {
            if let Some(value) = default {
                if elem == raw_name && !attrs.iter().any(|(existing, _)| existing == attr) {
                    attrs.push((attr.clone(), value.clone()));
                }
            }
        }
        n_specified
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

    fn expand_name(
        &self,
        name: &str,
        is_attr: bool,
        at: XmlPos,
    ) -> Result<String, pyre_interpreter::PyError> {
        let Some(sep) = self.namespace_separator() else {
            return Ok(name.to_string());
        };
        if let Some((prefix, local)) = name.split_once(':') {
            let Some(uri) = self.namespace_uri(prefix) else {
                return Err(self.make_error_at("unbound prefix", at));
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

    /// `storeAtts` / `XML_GetSpecifiedAttributeCount`: expand every attribute
    /// before filtering to the specified prefix (`m_nSpecifiedAtts = attIndex`).
    fn expand_attributes(
        &self,
        attrs: &[(String, String)],
        n_specified: usize,
        at: XmlPos,
    ) -> Result<(Vec<(String, String)>, usize), pyre_interpreter::PyError> {
        let mut out = Vec::new();
        let mut n_expanded_specified = 0;
        for (i, (name, value)) in attrs.iter().enumerate() {
            if self.namespace_separator().is_some()
                && (name == "xmlns" || name.starts_with("xmlns:"))
            {
                continue;
            }
            let expanded = self.expand_name(name, true, at)?;
            if out.iter().any(|(existing, _)| existing == &expanded) {
                return Err(self.make_error_at("duplicate attribute", at));
            }
            out.push((expanded, value.clone()));
            if i < n_specified {
                n_expanded_specified += 1;
            }
        }
        Ok((out, n_expanded_specified))
    }

    fn expand_entities(
        &mut self,
        raw: &str,
        content: bool,
        undef_pos: XmlPos,
    ) -> Result<String, pyre_interpreter::PyError> {
        let err_pos = if content { undef_pos } else { self.pos };
        let mut out = String::new();
        // frames: (text, byte offset of the unread rest, entity name or None for `raw`)
        let mut frames: Vec<(String, usize, Option<String>)> = vec![(raw.to_string(), 0, None)];
        let result = (|| {
            while !frames.is_empty() {
                let top = frames.len() - 1;
                let offset = frames[top].1;
                let Some(rel) = frames[top].0[offset..].find('&') else {
                    out.push_str(&frames[top].0[offset..]);
                    let name = frames.pop().unwrap().2;
                    if let Some(ref name) = name {
                        if let Some(ent) = self.internal_entities.get_mut(name) {
                            ent.open = false;
                        }
                    }
                    if content && name.is_some() && !out.is_empty() {
                        self.report_entity_text(&out)?;
                        out.clear();
                    }
                    continue;
                };
                let amp_at = offset + rel;
                out.push_str(&frames[top].0[offset..amp_at]);
                let after_start = amp_at + 1;
                let Some(end) = frames[top].0[after_start..].find(';') else {
                    return Err(self.make_error("undefined entity"));
                };
                let ent = frames[top].0[after_start..after_start + end].to_string();
                let next_offset = after_start + end + 1;
                match ent.as_str() {
                    "amp" => out.push('&'),
                    "lt" => out.push('<'),
                    "gt" => out.push('>'),
                    "quot" => out.push('"'),
                    "apos" => out.push('\''),
                    _ if ent.starts_with('#') => {
                        let ch = char_from_char_ref(&ent).ok_or_else(|| {
                            self.make_error_at("reference to invalid character number", undef_pos)
                        })?;
                        out.push(ch);
                    }
                    _ if self.internal_entities.contains_key(&ent) => {
                        // `doContent` `! parser->m_defaultExpandInternalEntities`.
                        if content && !self.expands_internal_entities() {
                            if !out.is_empty() {
                                self.report_entity_text(&out)?;
                                out.clear();
                            }
                            if self.handler_is_set("SkippedEntityHandler") {
                                self.handler_args("SkippedEntityHandler")
                                    .arg(w_str_new_managed(&ent))
                                    .arg(w_int_new(0))
                                    .call()?;
                            } else {
                                self.report_default(&format!("&{ent};"))?;
                            }
                        } else {
                            if self.internal_entities[&ent].open {
                                return Err(
                                    self.make_error_at("recursive entity reference", err_pos)
                                );
                            }
                            let text = self.internal_entities[&ent].text.clone();
                            self.account_indirect(text.len())
                                .map_err(|m| self.make_error_at(&m, err_pos))?;
                            if let Some(entity) = self.internal_entities.get_mut(&ent) {
                                entity.open = true;
                            }
                            frames[top].1 = next_offset;
                            if content && !out.is_empty() {
                                self.report_entity_text(&out)?;
                                out.clear();
                            }
                            frames.push((text, 0, Some(ent)));
                            continue;
                        }
                    }
                    _ if content && self.external_entities.contains_key(&ent) => {
                        if !out.is_empty() {
                            self.report_entity_text(&out)?;
                            out.clear();
                        }
                        if self.handler_is_set("ExternalEntityRefHandler") {
                            let (sysid, pubid) = self.external_entities[&ent].clone();
                            self.handle_external_entity(&ent, &sysid, pubid.as_deref(), undef_pos)?;
                        } else {
                            self.report_default(&format!("&{ent};"))?;
                        }
                    }
                    _ if content => {
                        // `doContent` `XML_TOK_ENTITY_REF`: undefined general entity.
                        if !self.has_param_entity_refs || self.standalone {
                            return Err(self.make_error_at("undefined entity", undef_pos));
                        } else {
                            if !out.is_empty() {
                                self.report_entity_text(&out)?;
                                out.clear();
                            }
                            if self.handler_is_set("SkippedEntityHandler") {
                                self.handler_args("SkippedEntityHandler")
                                    .arg(w_str_new_managed(&ent))
                                    .arg(w_int_new(0))
                                    .call()?;
                            } else {
                                self.report_default(&format!("&{ent};"))?;
                            }
                        }
                    }
                    _ => return Err(self.make_error("undefined entity")),
                }
                frames[top].1 = next_offset;
            }
            Ok(())
        })();
        if result.is_err() {
            for (_, _, name) in &frames {
                if let Some(name) = name {
                    if let Some(ent) = self.internal_entities.get_mut(name) {
                        ent.open = false;
                    }
                }
            }
        }
        result?;
        Ok(out)
    }

    /// `accountingDiffTolerated` + `accountingGetCurrentAmplification`.
    /// `lenOfShortestInclude` is `sizeof("<!ENTITY a SYSTEM 'b'>") - 1` = 22.
    fn account_indirect(&mut self, n: usize) -> Result<(), String> {
        self.bytes_indirect += n as u64;
        let direct = self.pos.index as u64;
        let output = direct + self.bytes_indirect;
        let factor = if direct > 0 {
            output as f32 / direct as f32
        } else {
            (22 + self.bytes_indirect) as f32 / 22.0f32
        };
        // `m_accounting.activationThresholdBytes`
        let threshold =
            get_parser_int(self.parser(), "_pyre_billion_threshold", 8388608).max(0) as u64;
        // `m_accounting.maximumAmplificationFactor`
        let max_factor = get_parser_float(self.parser(), "_pyre_billion_max_factor", 100.0) as f32;
        if output >= threshold && factor > max_factor {
            Err("limit on input amplification factor (from DTD and entities) breached".to_string())
        } else {
            Ok(())
        }
    }

    fn handle_external_entity(
        &self,
        context: &str,
        sysid: &str,
        pubid: Option<&str>,
        pos: XmlPos,
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
        if handler_int_rv(ret, "ExternalEntityRef")? == 0 {
            return Err(self.make_error_at("error in processing external entity reference", pos));
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
        if handler_int_rv(ret, "ExternalEntityRef")? == 0 {
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
        self.make_error_at(msg, self.pos)
    }

    fn make_error_at(&self, msg: &str, pos: XmlPos) -> pyre_interpreter::PyError {
        publish_error(self.parser(), msg, pos)
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
    }
}

fn object_to_xml_string(
    mut parser: PyObjectRef,
    obj: PyObjectRef,
    isfinal: bool,
) -> Result<String, pyre_interpreter::PyError> {
    unsafe {
        if is_str(obj) {
            let text = pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::str_utf8_w(obj).map(|s| s.to_string()))?;
            decode_xml_bytes(parser, text.as_bytes(), isfinal)
        } else if pyre_object::bytesobject::is_bytes_like(obj) {
            decode_xml_bytes(
                parser,
                pyre_object::bytesobject::bytes_like_data(obj),
                isfinal,
            )
        } else {
            Err(pyre_interpreter::PyError::type_error(
                "Parse() argument must be str or bytes",
            ))
        }
    }
}

/// `asciitab.h`: the ASCII bytes whose type is `BT_OTHER` or `BT_NONXML`,
/// the only ones `XmlInitUnknownEncoding` lets an encoding remap.
fn ascii_byte_may_remap(b: usize) -> bool {
    matches!(
        b,
        0x00..=0x08 | 0x0B | 0x0C | 0x0E..=0x1F | 0x24 | 0x40 | 0x5C | 0x5E | 0x60 | 0x7B | 0x7D..=0x7F
    )
}

/// `PyUnknownEncodingHandler` plus `XmlInitUnknownEncoding`.
fn unknown_encoding_map(
    mut parser: PyObjectRef,
    name: &str,
) -> Result<[i32; 256], pyre_interpreter::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let parser_slot = roots.pin_roots(&[parser]);
    let decoded_slot = if let Ok(obj) = pyre_interpreter::baseobjspace::getattr_str(
        roots.get(parser_slot),
        "_pyre_unknown_map_text",
    ) && unsafe { is_str(obj) }
    {
        roots.pin_roots(&[obj])
    } else {
        let payload: [u8; 256] = std::array::from_fn(|i| i as u8);
        let bytes_slot = roots.pin_roots(&[pyre_object::bytesobject::w_bytes_from_bytes(&payload)]);
        let name_slot = roots.pin_roots(&[w_str_new_managed(name)]);
        let replace_slot = roots.pin_roots(&[w_str_new_managed("replace")]);
        let decoded = pyre_interpreter::baseobjspace::call_method_result(
            roots.get(bytes_slot),
            "decode",
            &[roots.get(name_slot), roots.get(replace_slot)],
        )?;
        roots.pin_roots(&[decoded])
    };
    let text = match pyre_interpreter::baseobjspace::str_utf8_w(roots.get(decoded_slot)) {
        Ok(s) => s.to_string(),
        Err(_) => {
            return Err(pyexpat_error(
                "unknown encoding: line 1, column 0".to_string(),
                error_code_for_message("unknown encoding"),
                1,
                0,
            ));
        }
    };
    let chars: Vec<char> = text.chars().collect();
    if chars.len() != 256 {
        return Err(pyre_interpreter::PyError::value_error(
            "multi-byte encodings are not supported",
        ));
    }
    let mut map = [0i32; 256];
    for (i, ch) in chars.iter().copied().enumerate() {
        map[i] = if ch == '\u{FFFD}' { -1 } else { ch as i32 };
    }
    for i in 0..256 {
        let c = map[i];
        if i < 0x80 && !ascii_byte_may_remap(i) && c != i as i32 {
            return Err(pyexpat_error(
                "unknown encoding: line 1, column 0".to_string(),
                error_code_for_message("unknown encoding"),
                1,
                0,
            ));
        }
        if (0..0x80).contains(&c) && !ascii_byte_may_remap(c as usize) && c != i as i32 {
            return Err(pyexpat_error(
                "unknown encoding: line 1, column 0".to_string(),
                error_code_for_message("unknown encoding"),
                1,
                0,
            ));
        }
        // `XmlInitUnknownEncoding`: mapped values must sit in the BMP.
        if c > 0xFFFF {
            return Err(pyexpat_error(
                "unknown encoding: line 1, column 0".to_string(),
                error_code_for_message("unknown encoding"),
                1,
                0,
            ));
        }
    }
    parser = roots.get(parser_slot);
    let mut decoded = roots.get(decoded_slot);
    pyre_object::with_roots!(parser, decoded => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_unknown_map_text",
        decoded,
    ));
    Ok(map)
}

fn parser_pending_raw(mut parser: PyObjectRef) -> Vec<u8> {
    let mut obj = match pyre_object::with_roots!(parser => {
        pyre_interpreter::baseobjspace::getattr_str(parser, "_pyre_pending_raw")
    }) {
        Ok(obj) => obj,
        Err(_) => return Vec::new(),
    };
    if !unsafe { pyre_object::bytesobject::is_bytes_like(obj) } {
        return Vec::new();
    }
    pyre_object::with_roots!(obj => unsafe { pyre_object::bytesobject::bytes_like_data(obj).to_vec() })
}

fn set_parser_pending_raw(mut parser: PyObjectRef, data: &[u8]) {
    let mut raw = pyre_object::bytesobject::w_bytes_from_bytes(data);
    pyre_object::with_roots!(parser, raw => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_pending_raw",
        raw,
    ));
}

/// Publish the error position on the parser (`XML_GetErrorLineNumber` after
/// a failed `XML_Parse`).  Each `w_int_new` allocates, so the parser is
/// pinned and read back out of its root slot.
fn publish_error(parser: PyObjectRef, msg: &str, pos: XmlPos) -> pyre_interpreter::PyError {
    let code = error_code_for_message(msg);
    let roots = pyre_object::gc_roots::push_roots();
    let slot = roots.base();
    let _ = roots.pin_root(parser);
    pyre_interpreter::baseobjspace::setdictvalue_native(
        roots.get(slot),
        "ErrorLineNumber",
        w_int_new(pos.line as i64),
    );
    pyre_interpreter::baseobjspace::setdictvalue_native(
        roots.get(slot),
        "ErrorColumnNumber",
        w_int_new(pos.col as i64),
    );
    pyre_interpreter::baseobjspace::setdictvalue_native(
        roots.get(slot),
        "ErrorByteIndex",
        w_int_new(pos.index as i64),
    );
    pyre_interpreter::baseobjspace::setdictvalue_native(
        roots.get(slot),
        "CurrentLineNumber",
        w_int_new(pos.line as i64),
    );
    pyre_interpreter::baseobjspace::setdictvalue_native(
        roots.get(slot),
        "CurrentColumnNumber",
        w_int_new(pos.col as i64),
    );
    pyre_interpreter::baseobjspace::setdictvalue_native(
        roots.get(slot),
        "CurrentByteIndex",
        w_int_new(pos.index as i64),
    );
    pyre_interpreter::baseobjspace::setdictvalue_native(
        roots.get(slot),
        "ErrorCode",
        w_int_new(code),
    );
    pyexpat_error(
        format!("{msg}: line {}, column {}", pos.line, pos.col),
        code,
        pos.line as i64,
        pos.col as i64,
    )
}

/// `XML_ERROR_FINISHED`: a finished parser fails at its current position.
fn parsing_finished_error(parser: PyObjectRef) -> pyre_interpreter::PyError {
    let roots = pyre_object::gc_roots::push_roots();
    let slot = roots.base();
    let _ = roots.pin_root(parser);
    let pos = XmlPos {
        index: get_parser_int(roots.get(slot), "CurrentByteIndex", 0) as usize,
        line: get_parser_int(roots.get(slot), "CurrentLineNumber", 1) as usize,
        col: get_parser_int(roots.get(slot), "CurrentColumnNumber", 0) as usize,
    };
    publish_error(roots.get(slot), "parsing finished", pos)
}

/// Position of a decoding error: the text decoded so far, from the
/// document start (`parser_pending` holds the whole document text so far).
fn decode_error(parser: PyObjectRef, prev: &str, prefix: &str) -> pyre_interpreter::PyError {
    let origin = XmlPos {
        index: 0,
        line: 1,
        col: 0,
    };
    let pos = pos_after(pos_after(origin, prev), prefix);
    publish_error(parser, "not well-formed (invalid token)", pos)
}

fn decode_xml_bytes(
    mut parser: PyObjectRef,
    data: &[u8],
    isfinal: bool,
) -> Result<String, pyre_interpreter::PyError> {
    let mut buf = pyre_object::with_roots!(parser => parser_pending_raw(parser));
    buf.extend_from_slice(data);
    let encoding_detected = pyre_object::with_roots!(parser => {
        pyre_interpreter::baseobjspace::getattr_str(parser, "_pyre_encoding_detected")
            .map(is_true_obj)
            .unwrap_or(false)
    });
    // `initScan` `XML_TOK_PARTIAL`: wait for a BOM, a UTF-16 null, or a
    // complete XML declaration before choosing an encoding.
    if !encoding_detected && !isfinal {
        let keep_pending = if buf.len() < 2 {
            true
        } else if looks_like_utf16(&buf) {
            false
        } else if buf.len() < 3 && [0xefu8, 0xbb, 0xbf].starts_with(buf.as_slice()) {
            true
        } else {
            let rest = buf.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&buf);
            (rest.len() < b"<?xml".len() && b"<?xml".starts_with(rest))
                || (rest.starts_with(b"<?xml") && !rest.windows(2).any(|w| w == b"?>"))
        };
        if keep_pending {
            pyre_object::with_roots!(parser => set_parser_pending_raw(parser, &buf));
            pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
                parser,
                "_pyre_partial_tail_kind",
                w_int_new(if buf.is_empty() { 0 } else { 1 }),
            ));
            return Ok(String::new());
        }
    }
    pyre_object::with_roots!(parser => set_parser_pending_raw(parser, &[]));
    let enc = pyre_object::with_roots!(parser => declared_or_forced_encoding(parser, &buf))?;
    let normalized = normalize_encoding(&enc);
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_forced_encoding",
        w_str_new_managed(&normalized),
    ));
    let prev = pyre_object::with_roots!(parser => parser_pending(parser));
    let (text, pending, kind) = match normalized.as_str() {
        "utf-8" | "us-ascii" => {
            let data = if !encoding_detected {
                buf.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&buf[..])
            } else {
                &buf[..]
            };
            match std::str::from_utf8(data) {
                Ok(s) => (s.to_string(), Vec::new(), 0),
                Err(error) if error.error_len().is_none() => {
                    let valid = error.valid_up_to();
                    let pending = data[valid..].to_vec();
                    let text = match std::str::from_utf8(&data[..valid]) {
                        Ok(s) => s.to_string(),
                        Err(_) => String::new(),
                    };
                    (text, pending, 2)
                }
                Err(error) => {
                    let prefix = std::str::from_utf8(&data[..error.valid_up_to()]).unwrap_or("");
                    return Err(
                        pyre_object::with_roots!(parser => decode_error(parser, &prev, prefix)),
                    );
                }
            }
        }
        "iso-8859-1" => (buf.iter().map(|b| *b as char).collect(), Vec::new(), 0),
        "utf-16" | "utf-16le" | "utf-16be" => {
            match decode_utf16_bytes(&buf, &normalized, !encoding_detected) {
                Ok(decoded) => decoded,
                Err(prefix) => {
                    return Err(pyre_object::with_roots!(
                        parser => decode_error(parser, &prev, &prefix)
                    ));
                }
            }
        }
        _ => {
            let map = pyre_object::with_roots!(parser => unknown_encoding_map(parser, &enc))?;
            let mut text = String::with_capacity(buf.len());
            for &byte in &buf {
                let c = map[byte as usize];
                let Some(ch) = u32::try_from(c).ok().and_then(char::from_u32) else {
                    return Err(
                        pyre_object::with_roots!(parser => decode_error(parser, &prev, &text)),
                    );
                };
                text.push(ch);
            }
            (text, Vec::new(), 0)
        }
    };
    if buf.len() >= 2 {
        pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
            parser,
            "_pyre_encoding_detected",
            w_bool_from(true),
        ));
    }
    pyre_object::with_roots!(parser => set_parser_pending_raw(parser, &pending));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_partial_tail_kind",
        w_int_new(kind),
    ));
    Ok(text)
}

fn declared_or_forced_encoding(
    mut parser: PyObjectRef,
    data: &[u8],
) -> Result<String, pyre_interpreter::PyError> {
    let detected = pyre_object::with_roots!(parser => {
        pyre_interpreter::baseobjspace::getattr_str(parser, "_pyre_encoding_detected")
            .map(is_true_obj)
            .unwrap_or(false)
    });
    if detected {
        return pyre_object::with_roots!(parser => {
            if let Ok(obj) =
                pyre_interpreter::baseobjspace::getattr_str(parser, "_pyre_forced_encoding")
                    && unsafe { is_str(obj) }
            {
                pyre_interpreter::baseobjspace::str_utf8_w(obj).map(|s| s.to_string())
            } else {
                Ok("utf-8".to_string())
            }
        });
    }
    // `initScan`: a BOM first, else a leading zero byte is UTF-16BE and a
    // zero in the second byte is UTF-16LE. Runs once at the start of the entity.
    if data.starts_with(&[0xff, 0xfe]) {
        return Ok("utf-16le".to_string());
    }
    if data.starts_with(&[0xfe, 0xff]) {
        return Ok("utf-16be".to_string());
    }
    if data.len() >= 2 && data[0] == 0 {
        return Ok("utf-16be".to_string());
    }
    if data.len() >= 2 && data[1] == 0 {
        return Ok("utf-16le".to_string());
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
        || (data.len() >= 2 && (data[0] == 0 || data[1] == 0))
}

fn decode_utf16_bytes(
    data: &[u8],
    enc: &str,
    strip_bom: bool,
) -> Result<(String, Vec<u8>, i64), String> {
    let (little, start) = if strip_bom && data.starts_with(&[0xff, 0xfe]) {
        (true, 2)
    } else if strip_bom && data.starts_with(&[0xfe, 0xff]) {
        (false, 2)
    } else {
        (enc != "utf-16be", 0)
    };
    let rest_len = data.len() - start;
    let leftover = rest_len % 2 == 1;
    let even_end = start + (rest_len & !1);
    let words: Vec<u16> = data[start..even_end]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| {
            if little {
                u16::from_le_bytes([chunk[0], chunk[1]])
            } else {
                u16::from_be_bytes([chunk[0], chunk[1]])
            }
        })
        .collect();
    let high_surrogate_tail = words
        .last()
        .copied()
        .is_some_and(|w| (0xD800..=0xDBFF).contains(&w));
    let complete = if high_surrogate_tail {
        &words[..words.len() - 1]
    } else {
        &words[..]
    };
    let mut text = String::new();
    for ch in char::decode_utf16(complete.iter().copied()) {
        match ch {
            Ok(c) => text.push(c),
            Err(_) => return Err(text),
        }
    }
    let pending_from = if high_surrogate_tail {
        even_end - 2
    } else {
        even_end
    };
    let pending = data[pending_from..].to_vec();
    let kind = if high_surrogate_tail {
        2
    } else if leftover {
        1
    } else {
        0
    };
    Ok((text, pending, kind))
}

/// `normalizeLines`: `"\r\n"` and a lone `"\r"` become `"\n"`.
fn normalize_lines(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\r' {
            out.push('\n');
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
        } else {
            out.push(ch);
        }
    }
    out
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

fn get_parser_float(parser: PyObjectRef, name: &str, default: f64) -> f64 {
    match pyre_interpreter::baseobjspace::getattr_str(parser, name) {
        Ok(obj) if unsafe { is_float(obj) } => unsafe { w_float_get_value(obj) },
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
        return Err(parsing_finished_error(roots.get(base + 1)));
    }
    // `pyexpat_xmlparser_Parse_impl`: a `str` argument calls
    // `XML_SetEncoding(..., "utf-8")` before the first Parse.
    if unsafe { is_str(roots.get(base)) }
        && !pyre_interpreter::baseobjspace::getattr_str(
            roots.get(base + 1),
            "_pyre_parsing_started",
        )
        .map(is_true_obj)
        .unwrap_or(false)
    {
        pyre_interpreter::baseobjspace::setdictvalue_native(
            roots.get(base + 1),
            "_pyre_forced_encoding",
            w_str_new_managed("utf-8"),
        );
    }
    pyre_interpreter::baseobjspace::setdictvalue_native(
        roots.get(base + 1),
        "_pyre_parsing_started",
        w_bool_from(true),
    );
    let final_flag = is_true_obj(roots.get(base + 2));
    let mut input = parser_pending(roots.get(base + 1));
    input.push_str(&object_to_xml_string(
        roots.get(base + 1),
        roots.get(base),
        final_flag,
    )?);
    let convert_chunks = match pyre_interpreter::baseobjspace::getattr_str(
        roots.get(base + 1),
        "_pyre_forced_encoding",
    ) {
        Ok(obj) if unsafe { is_str(obj) } => {
            let enc = pyre_interpreter::baseobjspace::str_utf8_w(obj)?;
            enc != "utf-8" && enc != "us-ascii"
        }
        _ => false,
    };
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
    let parsed = MiniXmlParser::new(
        base + 1,
        &parse_input,
        final_flag,
        suppress_until,
        convert_chunks,
    )
    .parse()?;
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

/// `INT_HANDLER` / `RC_HANDLER` `PyLong_AsLong` plus `invalid_expat_handler_rv`.
fn handler_int_rv(mut ret: PyObjectRef, name: &str) -> Result<i64, pyre_interpreter::PyError> {
    match pyre_object::with_roots!(ret => pyre_interpreter::baseobjspace::int_w(ret)) {
        // Narrow to `int` (`rffi.cast(rffi.INT_real, ...)`).
        Ok(value) => Ok(value as i32 as i64),
        Err(mut err) => {
            pyre_interpreter::baseobjspace::add_internal_exception_note(
                &mut err,
                &format!("invalid '{name}' event handler return value"),
            )?;
            Err(err)
        }
    }
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
    if handler_int_rv(ret, "ExternalEntityRef")? == 0 {
        return Err(pyexpat_error(
            "error in processing external entity reference: line 1, column 0".to_string(),
            error_code_for_message("error in processing external entity reference"),
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
        "_pyre_reparse_deferral",
        "_pyre_default_kind",
        "_pyre_default_expand",
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
                // pyexpat_xmlparser_ParseFile_impl looks `read` up once
                // (`PyObject_GetOptionalAttr(file, "read")`) and the loop
                // calls that bound method. Each chunk's `read` and
                // `parse_impl` run Python and can collect, so the bound
                // method is kept in a slot and reloaded.
                let roots = pyre_object::gc_roots::push_roots();
                let self_slot = roots.base();
                let _ = roots.pin_root(self_obj);
                let file_slot = self_slot + 1;
                let _ = roots.pin_root(file);
                let size_slot = self_slot + 2;
                let _ = roots.pin_root(w_int_new(2048));
                let data_slot = self_slot + 3;
                let _ = roots.pin_root(w_none());
                let eof_slot = self_slot + 4;
                let _ = roots.pin_root(w_bool_from(false));
                let read_slot = match pyre_interpreter::baseobjspace::getattr_str(
                    roots.get(file_slot),
                    "read",
                ) {
                    Ok(read) => {
                        let _ = roots.pin_root(read);
                        self_slot + 5
                    }
                    Err(err) if err.kind == pyre_interpreter::PyErrorKind::AttributeError => {
                        return Err(pyre_interpreter::PyError::type_error(
                            "argument must have 'read' attribute",
                        ));
                    }
                    Err(err) => return Err(err),
                };
                if pyre_interpreter::baseobjspace::getattr_str(roots.get(self_slot), "_pyre_finished")
                    .map(is_true_obj)
                    .unwrap_or(false)
                {
                    return Err(parsing_finished_error(roots.get(self_slot)));
                }
                let mut result;
                loop {
                    let data = pyre_interpreter::call::call_function_impl_result(
                        roots.get(read_slot),
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
            fn SetBase(mut self_obj: PyObjectRef, mut base: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
                if unsafe { !is_str(base) } {
                    return Err(pyre_interpreter::PyError::type_error("SetBase() argument must be str"));
                }
                let has_nul = pyre_object::with_roots!(self_obj, base => pyre_interpreter::baseobjspace::str_utf8_w(base).map(|s| s.contains('\0')))?;
                if has_nul {
                    return Err(pyre_interpreter::PyError::value_error("embedded null character"));
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
            fn SetParamEntityParsing(
                mut self_obj: PyObjectRef,
                mut flag: PyObjectRef,
            ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
                // interp_pyexpat.py `SetParamEntityParsing` `@unwrap_spec(flag=int)`.
                let flag = pyre_object::with_roots!(
                    self_obj, flag => pyre_interpreter::baseobjspace::gateway_int_w(flag)
                )?;
                pyre_interpreter::baseobjspace::setdictvalue_native(
                    self_obj,
                    "_pyre_param_entity_parsing",
                    w_int_new(flag),
                );
                Ok(w_int_new(1))
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
                pyre_object::with_roots!(self_obj => pyre_interpreter::baseobjspace::setdictvalue_native(
                    self_obj,
                    "_pyre_billion_max_factor",
                    w_float_new(value),
                ));
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
                if name_s == "DefaultHandler" || name_s == "DefaultHandlerExpand" {
                    let expand = name_s == "DefaultHandlerExpand";
                    let clear = unsafe { is_none(value) };
                    let mut kind = if clear {
                        w_none()
                    } else {
                        w_str_new_managed(name_s)
                    };
                    pyre_object::with_roots!(self_obj, value, kind => {
                        pyre_interpreter::baseobjspace::setdictvalue_native(self_obj, name_s, value);
                        pyre_interpreter::baseobjspace::setdictvalue_native(self_obj, "_pyre_default_kind", kind);
                        pyre_interpreter::baseobjspace::setdictvalue_native(
                            self_obj,
                            "_pyre_default_expand",
                            w_bool_from(expand),
                        );
                    });
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
        "_pyre_default_kind",
        w_none(),
    ));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_default_expand",
        w_bool_from(true),
    ));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_encoding_detected",
        w_bool_from(false),
    ));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_unknown_map_text",
        w_none(),
    ));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_pending_raw",
        pyre_object::bytesobject::w_bytes_from_bytes(&[]),
    ));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_partial_tail_kind",
        w_int_new(0),
    ));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_parsing_started",
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
        // EXPAT_BILLION_LAUGHS_ATTACK_PROTECTION_ACTIVATION_THRESHOLD_DEFAULT
        w_int_new(8388608),
    ));
    pyre_object::with_roots!(parser => pyre_interpreter::baseobjspace::setdictvalue_native(
        parser,
        "_pyre_billion_max_factor",
        // EXPAT_BILLION_LAUGHS_ATTACK_PROTECTION_MAXIMUM_AMPLIFICATION_DEFAULT
        w_float_new(100.0),
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
    // interp_pyexpat.py `ErrorString` `@unwrap_spec(code=int)`.
    let mut w_code = args
        .first()
        .copied()
        .ok_or_else(|| pyre_interpreter::PyError::type_error("ErrorString() missing argument"))?;
    let code = pyre_object::with_roots!(
        w_code => pyre_interpreter::baseobjspace::gateway_int_w(w_code)
    )?;
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
    (
        "reserved prefix (xml) must not be undeclared or bound to another namespace name",
        38,
    ),
    (
        "reserved prefix (xmlns) must not be declared or undeclared",
        39,
    ),
    (
        "prefix must not be bound to one of the reserved namespace names",
        40,
    ),
    ("invalid argument", 41),
    (
        "a successful prior call to function XML_GetBuffer is required",
        42,
    ),
    (
        "limit on input amplification factor (from DTD and entities) breached",
        43,
    ),
    ("parser not started", 44),
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
    "XML_ERROR_RESERVED_PREFIX_XML",
    "XML_ERROR_RESERVED_PREFIX_XMLNS",
    "XML_ERROR_RESERVED_NAMESPACE_URI",
    "XML_ERROR_INVALID_ARGUMENT",
    "XML_ERROR_NO_BUFFER",
    "XML_ERROR_AMPLIFICATION_LIMIT_BREACH",
    "XML_ERROR_NOT_STARTED",
];

/// Build a `hasdict` namespace object used for the `model` / `errors`
/// submodules; constants are written as instance attributes.
fn make_namespace(name: &'static str) -> PyObjectRef {
    let tp = pyre_interpreter::typedef::make_builtin_type(name, |_| {});
    unsafe { pyre_interpreter::objspace::std::mapdict::set_type_hasdict(tp, true) };
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
