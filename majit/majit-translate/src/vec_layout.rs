//! Buffer-pointer and length offsets of `alloc::vec::Vec<T>`.
//!
//! Charon's type decl for `alloc::vec::Vec` is `Opaque` with no field
//! layout, so the component order is taken from the three-word value this
//! crate is compiled against (pointer, length, capacity — `Global` is a
//! ZST). Offsets are that word index times [`crate::layout::target_word_size`],
//! not the host `usize` the measurement ran on.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VecLayout {
    pub ptr_offset: usize,
    pub len_offset: usize,
    pub cap_offset: usize,
}

pub fn probe() -> VecLayout {
    static CELL: OnceLock<VecLayout> = OnceLock::new();
    *CELL.get_or_init(|| layout_for_word(crate::layout::target_word_size()))
}

/// `Vec` component offsets for a target whose pointer is `word` bytes.
///
/// The word index of each component is fixed by the host measurement
/// (`measure_indices`); only the byte stride changes with the target.
pub fn layout_for_word(word: usize) -> VecLayout {
    let (ptr_index, len_index, cap_index) = component_indices();
    VecLayout {
        ptr_offset: ptr_index * word,
        len_offset: len_index * word,
        cap_offset: cap_index * word,
    }
}

fn component_indices() -> (usize, usize, usize) {
    static CELL: OnceLock<(usize, usize, usize)> = OnceLock::new();
    *CELL.get_or_init(measure_indices)
}

fn measure_indices() -> (usize, usize, usize) {
    const WORD: usize = std::mem::size_of::<usize>();
    const {
        assert!(std::mem::size_of::<Vec<u8>>() == 3 * WORD);
        assert!(std::mem::size_of::<Vec<i64>>() == 3 * WORD);
        assert!(std::mem::align_of::<Vec<u8>>() == WORD);
        assert!(std::mem::align_of::<Vec<i64>>() == WORD);
    }
    let mut sample = Vec::<u8>::with_capacity(4);
    sample.push(1);
    assert_ne!(
        sample.len(),
        sample.capacity(),
        "len and cap must occupy different words"
    );
    let words =
        unsafe { std::slice::from_raw_parts((&sample as *const Vec<u8>).cast::<usize>(), 3) };
    let ptr = sample.as_ptr() as usize;
    let len = sample.len();
    let cap = sample.capacity();
    let mut ptr_index = None;
    let mut len_index = None;
    let mut cap_index = None;
    for (index, word) in words.iter().copied().enumerate() {
        if word == ptr {
            ptr_index = Some(index);
        } else if word == len {
            len_index = Some(index);
        } else if word == cap {
            cap_index = Some(index);
        }
    }
    let ptr_index = ptr_index.expect("Vec buffer-pointer word");
    let len_index = len_index.expect("Vec length word");
    let cap_index = cap_index.expect("Vec capacity word");
    assert!(
        ptr_index != len_index && ptr_index != cap_index && len_index != cap_index,
        "Vec components must occupy three distinct words"
    );
    (ptr_index, len_index, cap_index)
}

/// Pointer and length offsets of a `&[T]` / `*const [T]` value.
///
/// A slice reference is the two words `(ptr, len)`. Their order is taken
/// from the value this crate is compiled against, the same way
/// [`probe`] measures `Vec`, and scaled by the target word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceLayout {
    pub ptr_offset: usize,
    pub len_offset: usize,
}

pub fn slice_probe() -> SliceLayout {
    static CELL: OnceLock<SliceLayout> = OnceLock::new();
    *CELL.get_or_init(|| slice_layout_for_word(crate::layout::target_word_size()))
}

/// Slice-reference word offsets for a target whose pointer is `word` bytes.
pub fn slice_layout_for_word(word: usize) -> SliceLayout {
    let (ptr_index, len_index) = slice_component_indices();
    SliceLayout {
        ptr_offset: ptr_index * word,
        len_offset: len_index * word,
    }
}

fn slice_component_indices() -> (usize, usize) {
    static CELL: OnceLock<(usize, usize)> = OnceLock::new();
    *CELL.get_or_init(measure_slice_indices)
}

fn measure_slice_indices() -> (usize, usize) {
    const WORD: usize = std::mem::size_of::<usize>();
    const {
        assert!(std::mem::size_of::<&[u8]>() == 2 * WORD);
        assert!(std::mem::size_of::<&[i64]>() == 2 * WORD);
    }
    let items = [7u8, 8, 9];
    let sample: &[u8] = &items;
    let words: [usize; 2] = unsafe { std::mem::transmute(sample) };
    let ptr = sample.as_ptr() as usize;
    let len = sample.len();
    let ptr_index = words
        .iter()
        .position(|&word| word == ptr)
        .expect("slice pointer word");
    let len_index = words
        .iter()
        .position(|&word| word == len)
        .expect("slice length word");
    assert_ne!(ptr_index, len_index, "slice words must be distinct");
    (ptr_index, len_index)
}

/// Constructor path of `alloc::vec::Vec`: unique intern `Vec<T>`,
/// crate-stripped `vec::Vec` / `vec::Vec<T>`, or `alloc::vec::Vec<T>`.
/// Type-argument `::` is not a constructor segment, so
/// `Vec<module::marshal::Rooted>` is still this ADT. Another crate's
/// `foo::Vec<T>` is not.
pub fn spelling_is_alloc_vec(type_str: &str) -> bool {
    let type_str = type_str.trim();
    let head = type_str.split('<').next().unwrap_or(type_str);
    let mut segs = head.rsplit("::");
    let last = segs.next().unwrap_or("");
    if last != "Vec" {
        return false;
    }
    match segs.next() {
        None => type_str.contains('<'),
        Some("vec") => true,
        _ => false,
    }
}

/// A field-layout spelling that is an inline `Vec<T>`, not `Box<Vec<T>>`
/// and not `&Vec<T>`.
pub fn field_layout_is_inline_vec(type_str: &str) -> bool {
    let type_str = type_str.trim();
    if type_str.starts_with('&')
        || type_str.starts_with('*')
        || type_str.starts_with("Box<")
        || type_str.starts_with("Arc<")
        || type_str.starts_with("Rc<")
    {
        return false;
    }
    spelling_is_alloc_vec(type_str)
}

#[cfg(test)]
mod tests {
    use super::layout_for_word;

    #[test]
    fn component_offsets_match_the_majit_ir_header_words() {
        use majit_ir::rvec::{VEC_CAP_WORD, VEC_LEN_WORD, VEC_PTR_WORD, vec_word_offset};
        for word in [8, 4] {
            let layout = layout_for_word(word);
            assert_eq!(layout.ptr_offset, vec_word_offset(VEC_PTR_WORD, word));
            assert_eq!(layout.len_offset, vec_word_offset(VEC_LEN_WORD, word));
            assert_eq!(layout.cap_offset, vec_word_offset(VEC_CAP_WORD, word));
        }
        let word = crate::layout::target_word_size();
        let probed = super::probe();
        assert_eq!(probed.ptr_offset, vec_word_offset(VEC_PTR_WORD, word));
        assert_eq!(probed.len_offset, vec_word_offset(VEC_LEN_WORD, word));
        assert_eq!(probed.cap_offset, vec_word_offset(VEC_CAP_WORD, word));
    }

    #[test]
    fn slice_words_are_two_distinct_target_words() {
        for word in [8, 4] {
            let layout = super::slice_layout_for_word(word);
            let mut offsets = [layout.ptr_offset, layout.len_offset];
            offsets.sort();
            assert_eq!(offsets, [0, word]);
        }
    }

    #[test]
    fn component_offsets_follow_the_target_word() {
        let wide = layout_for_word(8);
        assert_eq!(
            (wide.ptr_offset, wide.len_offset),
            (8, 16),
            "a 64-bit target places the buffer and length at 8 and 16"
        );
        let wasm = layout_for_word(4);
        assert_eq!(
            (wasm.ptr_offset, wasm.len_offset),
            (4, 8),
            "a wasm32 target places the buffer and length at 4 and 8"
        );
    }

    #[test]
    fn inline_vec_matches_the_alloc_vec_declaration_path() {
        use super::{field_layout_is_inline_vec, spelling_is_alloc_vec};
        for spelling in [
            "Vec<u8>",
            "Vec<module::marshal::Rooted>",
            "vec::Vec<module::marshal::Rooted>",
            "alloc::vec::Vec<usize>",
        ] {
            assert!(
                field_layout_is_inline_vec(spelling),
                "{spelling} is inline alloc::vec::Vec"
            );
            assert!(
                spelling_is_alloc_vec(spelling),
                "{spelling} is the alloc::vec::Vec constructor"
            );
        }
        for spelling in [
            "Vec",
            "VecDeque<u8>",
            "Box<Vec<u8>>",
            "&Vec<u8>",
            "&mut Vec<u8>",
            "*const Vec<u8>",
            "foo::Vec<u8>",
            "Option<Vec<u8>>",
        ] {
            assert!(
                !field_layout_is_inline_vec(spelling),
                "{spelling} is not an inline alloc::vec::Vec value"
            );
        }
        assert!(!spelling_is_alloc_vec("&mut Vec<i64>"));
        assert!(spelling_is_alloc_vec("Vec<i64>"));
    }
}
