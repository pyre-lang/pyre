//! Data-pointer and length offsets of a fat pointer (`Box<[T]>`, `Box<str>`,
//! `Box<dyn Trait>`).
//!
//! The two words are the data address and the metadata. Charon does not
//! record that split, and a `Vec<T>`'s `(capacity, pointer, length)` order
//! is a different value. The word index is measured on the host the way
//! [`crate::vec_layout`] measures `Vec`; the byte offset is that index
//! times [`crate::layout::target_word_size`].

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FatPtrLayout {
    pub data_offset: usize,
    pub len_offset: usize,
}

impl FatPtrLayout {
    /// Index of the metadata (`true`) or data (`false`) word in the
    /// two-word fat value `two_word_fat_struct` / `dyn_fat_field_annotation`
    /// spell as a tuple. `rtuple.py` `rtype_getitem` uses this index.
    pub fn word_index(self, metadata: bool) -> i64 {
        let len_first = self.len_offset < self.data_offset;
        match (metadata, len_first) {
            (true, true) | (false, false) => 0,
            (true, false) | (false, true) => 1,
        }
    }
}

pub fn probe() -> FatPtrLayout {
    static CELL: OnceLock<FatPtrLayout> = OnceLock::new();
    *CELL.get_or_init(|| layout_for_word(crate::layout::target_word_size()))
}

/// Fat-pointer component offsets for a target whose pointer is `word` bytes.
pub fn layout_for_word(word: usize) -> FatPtrLayout {
    let (data_index, len_index) = component_indices();
    FatPtrLayout {
        data_offset: data_index * word,
        len_offset: len_index * word,
    }
}

/// `&dyn Trait` / `&mut dyn Trait` / `Box<dyn Trait>` occupy two words:
/// the data pointer and the vtable (metadata) pointer.
///
/// `get_type_flag` uses this arm. Raw `*mut dyn` / `Option<*mut dyn>` stay
/// off it: those spellings are two-word `lltype.Struct` fields in the
/// bookkeeper, not FLAG_POINTER descrs of size two words.
///
/// A lifetime on the reference (`&'static dyn DictStrategy`) is the same
/// two-word value: [`pointer_pointee`] strips it the way `spelling_is_fat_ptr`
/// already does.
pub fn spelling_is_dyn_fat_ptr(s: &str) -> bool {
    let s = s.trim();
    if s.starts_with("*const ") || s.starts_with("*mut ") {
        return false;
    }
    if let Some(inner) = wrapper_inner(s, &["Box"]) {
        return is_dyn_pointee(inner);
    }
    if pointer_pointee(s).is_some_and(is_dyn_pointee) {
        return true;
    }
    // Layout rows may spell a `&dyn Trait` field as the pointee after
    // `charon_dyn_trait_to_ast_string` (`dyn DictStrategy`): the field is
    // still the two-word fat pointer.
    is_dyn_pointee(s)
}

/// A pointer to an unsized pointee (`dyn Trait`, `[T]`, `str`) is two
/// words: data then metadata (`lltype` models that as a struct).
/// `Option` of such a pointer keeps the same two-word value.
pub fn spelling_is_fat_ptr(s: &str) -> bool {
    unsized_pointee(s).is_some()
}

fn unsized_pointee(s: &str) -> Option<&str> {
    let s = s.trim();
    let inner = wrapper_inner(s, &["Option"]).unwrap_or(s);
    let pointee = pointer_pointee(inner)?;
    is_unsized_pointee(pointee).then_some(pointee)
}

fn wrapper_inner<'a>(s: &'a str, leaves: &[&str]) -> Option<&'a str> {
    let (head, rest) = s.split_once('<')?;
    if !rest.ends_with('>') {
        return None;
    }
    let leaf = head.trim().rsplit("::").next().unwrap_or(head.trim());
    if !leaves.contains(&leaf) {
        return None;
    }
    let inner = &rest[..rest.len() - 1];
    if top_level_comma(inner) {
        return None;
    }
    Some(inner.trim())
}

fn pointer_pointee(s: &str) -> Option<&str> {
    let s = s.trim();
    if let Some(rest) = s
        .strip_prefix("*const ")
        .or_else(|| s.strip_prefix("*mut "))
    {
        return Some(rest.trim());
    }
    if let Some(rest) = s.strip_prefix("&mut ") {
        return Some(rest.trim());
    }
    if let Some(rest) = s.strip_prefix('&') {
        let rest = rest.trim();
        if let Some(after_life) = rest.strip_prefix('\'') {
            return Some(
                after_life
                    .split_once(char::is_whitespace)
                    .map(|(_, rhs)| rhs.trim())
                    .unwrap_or(after_life),
            );
        }
        return Some(rest);
    }
    wrapper_inner(s, &["Box", "NonNull"])
}

fn is_dyn_pointee(inner: &str) -> bool {
    let inner = inner.trim();
    inner == "dyn" || inner.starts_with("dyn ") || inner.starts_with("(dyn ")
}

fn is_unsized_pointee(inner: &str) -> bool {
    let inner = inner.trim();
    if inner == "str" || inner.ends_with("::str") || is_dyn_pointee(inner) {
        return true;
    }
    inner
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .is_some_and(|body| !body.contains(';'))
}

fn top_level_comma(s: &str) -> bool {
    let mut depth = 0i32;
    for ch in s.chars() {
        match ch {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => return true,
            _ => {}
        }
    }
    false
}

fn component_indices() -> (usize, usize) {
    static CELL: OnceLock<(usize, usize)> = OnceLock::new();
    *CELL.get_or_init(measure_indices)
}

fn measure_indices() -> (usize, usize) {
    const WORD: usize = std::mem::size_of::<usize>();
    const {
        assert!(std::mem::size_of::<Box<[u8]>>() == 2 * WORD);
        assert!(std::mem::align_of::<Box<[u8]>>() == WORD);
    }
    let sample: Box<[u8]> = vec![1u8, 2, 3, 4, 5].into_boxed_slice();
    assert_ne!(sample.len(), sample.as_ptr() as usize);
    let words =
        unsafe { std::slice::from_raw_parts((&sample as *const Box<[u8]>).cast::<usize>(), 2) };
    let data = sample.as_ptr() as usize;
    let len = sample.len();
    let mut data_index = None;
    let mut len_index = None;
    for (index, word) in words.iter().copied().enumerate() {
        if word == data {
            data_index = Some(index);
        } else if word == len {
            len_index = Some(index);
        }
    }
    let data_index = data_index.expect("fat-pointer data word");
    let len_index = len_index.expect("fat-pointer length word");
    assert_ne!(data_index, len_index);
    (data_index, len_index)
}

#[cfg(test)]
mod tests {
    use super::layout_for_word;

    #[test]
    fn dyn_trait_spellings_are_fat() {
        assert!(super::spelling_is_dyn_fat_ptr("&dyn"));
        assert!(super::spelling_is_dyn_fat_ptr("&dyn Storage"));
        assert!(super::spelling_is_dyn_fat_ptr("&mut dyn Storage"));
        assert!(super::spelling_is_dyn_fat_ptr("Box<dyn Storage>"));
        assert!(super::spelling_is_dyn_fat_ptr("&'static dyn DictStrategy"));
        assert!(super::spelling_is_dyn_fat_ptr("&'a dyn Storage"));
        assert!(super::spelling_is_dyn_fat_ptr("dyn DictStrategy"));
        assert!(super::spelling_is_dyn_fat_ptr("dyn"));
        assert!(super::spelling_is_dyn_fat_ptr("&dyn"));
        assert!(super::spelling_is_dyn_fat_ptr("Box<dyn>"));
        assert!(super::spelling_is_dyn_fat_ptr("&'static dyn"));
        assert!(!super::spelling_is_dyn_fat_ptr("*mut dyn Storage"));
        assert!(!super::spelling_is_dyn_fat_ptr(
            "Option<*mut dyn AsyncActionOps>"
        ));
        assert!(!super::spelling_is_dyn_fat_ptr("&Holder"));
        assert!(!super::spelling_is_dyn_fat_ptr("Box<Holder>"));
    }

    #[test]
    fn unsized_raw_pointers_are_fat() {
        assert!(super::spelling_is_fat_ptr("*const [u8]"));
        assert!(super::spelling_is_fat_ptr("&[String]"));
        assert!(super::spelling_is_fat_ptr("Box<[u8]>"));
        assert!(super::spelling_is_fat_ptr("&dyn"));
        assert!(super::spelling_is_fat_ptr("*mut dyn Storage"));
        assert!(super::spelling_is_fat_ptr(
            "Option<*mut dyn AsyncActionOps>"
        ));
        assert!(super::spelling_is_fat_ptr("Option<&dyn Storage>"));
        assert!(super::spelling_is_fat_ptr("&dyn"));
        assert!(!super::spelling_is_fat_ptr("*mut u8"));
        assert!(!super::spelling_is_fat_ptr("Option<*mut u8>"));
        assert!(!super::spelling_is_fat_ptr("&Holder"));
    }

    #[test]
    fn component_offsets_follow_the_target_word() {
        let wide = layout_for_word(8);
        assert_eq!(
            (wide.data_offset, wide.len_offset),
            (0, 8),
            "a 64-bit target places the data pointer then the length"
        );
        let wasm = layout_for_word(4);
        assert_eq!(
            (wasm.data_offset, wasm.len_offset),
            (0, 4),
            "a wasm32 target places the data pointer then the length"
        );
        assert_eq!(wide.word_index(false), 0, "data word is item 0");
        assert_eq!(wide.word_index(true), 1, "metadata word is item 1");
        assert_eq!(wasm.word_index(false), 0);
        assert_eq!(wasm.word_index(true), 1);
    }
}
