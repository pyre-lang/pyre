//! Host Rust layout of a raw struct or enum: field offsets and the tag
//! rustc actually stores, including a niche.

use majit_charon_reader::ullbc::{TagEncoding, TagLayout};

/// Byte layout of one concrete type, as Charon recorded it for the
/// extraction target. Independent of the explicit sum shell the front
/// uses when a payload would overwrite a tag.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HostLayout {
    pub size: u64,
    pub align: u64,
    /// `variant_field_offsets[variant][field]` is the byte offset of that
    /// field. A struct has one variant.
    pub variant_field_offsets: Vec<Vec<u64>>,
    pub tag: Option<TagLayout>,
}

pub fn host_layout_from_type(layout: &majit_charon_reader::ullbc::TypeLayout) -> HostLayout {
    HostLayout {
        size: layout.size.unwrap_or(0),
        align: layout.align.unwrap_or(1),
        variant_field_offsets: layout
            .variant_layouts
            .iter()
            .map(|variant| variant.field_offsets.clone())
            .collect(),
        tag: layout.tag(),
    }
}

fn mask(bits: u32, raw: u128) -> u128 {
    if bits >= 128 {
        raw
    } else {
        raw & ((1u128 << bits) - 1)
    }
}

/// Variant index of `raw_tag_value` under rustc's niche rule.
///
/// `relative = raw.wrapping_sub(niche_start)` in the tag width.
/// `relative <= end - start` selects `start + relative`; every other
/// value is the untagged variant. A direct tag matches `tags[index]`.
pub fn host_variant_index(tag: &TagLayout, raw_tag_value: u128) -> usize {
    let raw = mask(tag.bits, raw_tag_value);
    match &tag.encoding {
        TagEncoding::Direct { tags } => tags
            .iter()
            .position(|known| mask(tag.bits, *known) == raw)
            .unwrap_or_else(|| {
                panic!("host_variant_index: direct tag {raw:#x} is not in the tag list")
            }),
        TagEncoding::Niche {
            untagged_variant,
            niche_variants,
            niche_start,
        } => {
            let relative = mask(tag.bits, raw.wrapping_sub(*niche_start));
            let span = (niche_variants.end() - niche_variants.start()) as u128;
            if relative <= span {
                niche_variants.start() + relative as usize
            } else {
                *untagged_variant
            }
        }
    }
}

/// Tag word a niche or direct variant stores. `None` for the untagged
/// niche variant, which has no single tag value.
pub fn host_tag_value_for_variant(tag: &TagLayout, variant: usize) -> Option<u128> {
    match &tag.encoding {
        TagEncoding::Direct { tags } => tags.get(variant).map(|word| mask(tag.bits, *word)),
        TagEncoding::Niche {
            untagged_variant,
            niche_variants,
            niche_start,
        } => {
            if variant == *untagged_variant || !niche_variants.contains(&variant) {
                None
            } else {
                let step = (variant - niche_variants.start()) as u128;
                Some(mask(tag.bits, niche_start.wrapping_add(step)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, discriminant, size_of};
    use std::ptr::NonNull;

    /// Read the tag integer at `offset`. The layouts under test place that
    /// integer on an aligned boundary, so `ptr::read` is defined.
    fn read_tag<T>(value: &T, offset: usize, bits: u32) -> u128 {
        let ptr = unsafe { (value as *const T as *const u8).add(offset) };
        unsafe {
            match bits {
                8 => u128::from(std::ptr::read(ptr)),
                32 => u128::from(u32::from_le(std::ptr::read(ptr.cast::<u32>()))),
                64 => u128::from(u64::from_le(std::ptr::read(ptr.cast::<u64>()))),
                _ => panic!("untested tag width {bits}"),
            }
        }
    }

    #[test]
    fn niche_and_direct_tags_match_real_values() {
        #[derive(Clone, Copy)]
        #[allow(dead_code)] // payloads only shape the layout under test
        enum BoolNiche {
            T(bool),
            U,
            V,
        }
        assert_eq!(size_of::<BoolNiche>(), 1);
        assert_eq!(align_of::<BoolNiche>(), 1);
        assert_eq!(
            discriminant(&BoolNiche::T(false)),
            discriminant(&BoolNiche::T(true))
        );
        assert_ne!(discriminant(&BoolNiche::U), discriminant(&BoolNiche::V));
        let bool_tag = TagLayout {
            offset: 0,
            signed: false,
            bits: 8,
            encoding: TagEncoding::Niche {
                untagged_variant: 0,
                niche_variants: 1..=2,
                niche_start: 2,
            },
        };
        let samples = [
            (BoolNiche::T(false), 0usize, 0u128),
            (BoolNiche::T(true), 0, 1),
            (BoolNiche::U, 1, 2),
            (BoolNiche::V, 2, 3),
        ];
        for (value, variant, raw) in samples {
            assert_eq!(read_tag(&value, 0, 8), raw);
            assert_eq!(host_variant_index(&bool_tag, raw), variant);
        }
        assert!(host_tag_value_for_variant(&bool_tag, 0).is_none());
        assert_eq!(host_tag_value_for_variant(&bool_tag, 1), Some(2));
        assert_eq!(host_tag_value_for_variant(&bool_tag, 2), Some(3));

        #[derive(Clone, Copy)]
        #[allow(dead_code)] // payloads only shape the layout under test
        enum CharNiche {
            A(char),
            B,
            C,
        }
        assert_eq!(size_of::<CharNiche>(), 4);
        assert_eq!(align_of::<CharNiche>(), 4);
        assert_eq!(
            discriminant(&CharNiche::A('\0')),
            discriminant(&CharNiche::A('a'))
        );
        assert_ne!(discriminant(&CharNiche::B), discriminant(&CharNiche::C));
        let char_tag = TagLayout {
            offset: 0,
            signed: false,
            bits: 32,
            encoding: TagEncoding::Niche {
                untagged_variant: 0,
                niche_variants: 1..=2,
                niche_start: 0x11_0000,
            },
        };
        assert_eq!(read_tag(&CharNiche::A('\0'), 0, 32), 0);
        assert_eq!(
            read_tag(&CharNiche::A('a'), 0, 32),
            u128::from(u32::from('a'))
        );
        assert_eq!(read_tag(&CharNiche::B, 0, 32), 0x11_0000);
        assert_eq!(read_tag(&CharNiche::C, 0, 32), 0x11_0001);
        assert_eq!(host_variant_index(&char_tag, 0), 0);
        assert_eq!(host_variant_index(&char_tag, u128::from(u32::from('a'))), 0);
        assert_eq!(host_variant_index(&char_tag, 0x11_0000), 1);
        assert_eq!(host_variant_index(&char_tag, 0x11_0001), 2);
        assert_eq!(host_tag_value_for_variant(&char_tag, 1), Some(0x11_0000));
        assert_eq!(host_tag_value_for_variant(&char_tag, 2), Some(0x11_0001));
        assert!(host_tag_value_for_variant(&char_tag, 0).is_none());

        assert_eq!(size_of::<Option<NonNull<u8>>>(), 8);
        assert_eq!(align_of::<Option<NonNull<u8>>>(), 8);
        let none_tag = TagLayout {
            offset: 0,
            signed: false,
            bits: 64,
            encoding: TagEncoding::Niche {
                untagged_variant: 1,
                niche_variants: 0..=0,
                niche_start: 0,
            },
        };
        let none = Option::<NonNull<u8>>::None;
        let some = NonNull::new(0x10 as *mut u8).map(Some).unwrap();
        assert_ne!(discriminant(&none), discriminant(&some));
        assert_eq!(read_tag(&none, 0, 64), 0);
        assert_eq!(read_tag(&some, 0, 64), 0x10);
        assert_eq!(host_variant_index(&none_tag, 0), 0);
        assert_eq!(host_variant_index(&none_tag, 0x10), 1);
        assert_eq!(host_tag_value_for_variant(&none_tag, 0), Some(0));
        assert!(host_tag_value_for_variant(&none_tag, 1).is_none());

        // A reference has a single invalid value, so two extra variants do
        // not fit in the niche. The tag is a direct `u8` and the pointer
        // sits at offset 8.
        #[derive(Clone, Copy)]
        #[allow(dead_code)] // payloads only shape the layout under test
        enum RefDirect {
            A(&'static u8),
            B,
            C,
        }
        assert_eq!(size_of::<RefDirect>(), 16);
        assert_eq!(align_of::<RefDirect>(), 8);
        let static_byte: &'static u8 = &7;
        let a = RefDirect::A(static_byte);
        assert_ne!(discriminant(&a), discriminant(&RefDirect::B));
        assert_ne!(discriminant(&RefDirect::B), discriminant(&RefDirect::C));
        assert_eq!(read_tag(&a, 0, 8), 0);
        assert_eq!(read_tag(&RefDirect::B, 0, 8), 1);
        assert_eq!(read_tag(&RefDirect::C, 0, 8), 2);
        let ptr =
            unsafe { std::ptr::read(((&a) as *const RefDirect as *const u8).add(8).cast::<u64>()) };
        assert_ne!(ptr, 0);
        let direct_ref = TagLayout {
            offset: 0,
            signed: false,
            bits: 8,
            encoding: TagEncoding::Direct {
                tags: vec![0, 1, 2],
            },
        };
        assert_eq!(host_variant_index(&direct_ref, 0), 0);
        assert_eq!(host_variant_index(&direct_ref, 1), 1);
        assert_eq!(host_variant_index(&direct_ref, 2), 2);
        assert_eq!(host_tag_value_for_variant(&direct_ref, 2), Some(2));
        let unknown = std::panic::catch_unwind(|| host_variant_index(&direct_ref, 9));
        assert!(unknown.is_err(), "an unknown direct tag must panic");

        #[derive(Clone, Copy)]
        #[allow(dead_code)] // payloads only shape the layout under test
        enum DirectPay {
            A(u64),
            B(u64),
            C(u64),
        }
        assert_eq!(size_of::<DirectPay>(), 16);
        assert_eq!(align_of::<DirectPay>(), 8);
        let a_pay = DirectPay::A(0x1111);
        assert_eq!(discriminant(&a_pay), discriminant(&DirectPay::A(0)));
        assert_ne!(
            discriminant(&DirectPay::A(0)),
            discriminant(&DirectPay::B(0))
        );
        assert_eq!(read_tag(&a_pay, 0, 8), 0);
        assert_eq!(read_tag(&DirectPay::B(0), 0, 8), 1);
        assert_eq!(read_tag(&DirectPay::C(0), 0, 8), 2);
        let payload = unsafe {
            std::ptr::read(
                ((&a_pay) as *const DirectPay as *const u8)
                    .add(8)
                    .cast::<u64>(),
            )
        };
        assert_eq!(payload, 0x1111);
        let direct_pay = TagLayout {
            offset: 0,
            signed: false,
            bits: 8,
            encoding: TagEncoding::Direct {
                tags: vec![0, 1, 2],
            },
        };
        assert_eq!(host_variant_index(&direct_pay, read_tag(&a_pay, 0, 8)), 0);
        assert_eq!(
            host_variant_index(&direct_pay, read_tag(&DirectPay::B(7), 0, 8)),
            1
        );
        assert_eq!(host_tag_value_for_variant(&direct_pay, 1), Some(1));

        // `Big` is the untagged bool. Niche variants on both sides share
        // one span, and the dead tag `niche_start + 7` is not stored.
        #[derive(Clone, Copy)]
        #[allow(dead_code)] // payloads only shape the layout under test
        enum Hole {
            A,
            B,
            C,
            D,
            E,
            F,
            G,
            Big(bool),
            H,
        }
        assert_eq!(size_of::<Hole>(), 1);
        let hole_tag = TagLayout {
            offset: 0,
            signed: false,
            bits: 8,
            encoding: TagEncoding::Niche {
                untagged_variant: 7,
                niche_variants: 0..=8,
                niche_start: 2,
            },
        };
        let hole_samples = [
            (Hole::A, 0usize, 2u128),
            (Hole::B, 1, 3),
            (Hole::C, 2, 4),
            (Hole::D, 3, 5),
            (Hole::E, 4, 6),
            (Hole::F, 5, 7),
            (Hole::G, 6, 8),
            (Hole::Big(false), 7, 0),
            (Hole::Big(true), 7, 1),
            (Hole::H, 8, 0x0a),
        ];
        for (value, variant, raw) in hole_samples {
            assert_eq!(read_tag(&value, 0, 8), raw);
            assert_eq!(host_variant_index(&hole_tag, raw), variant);
            if variant != 7 {
                assert_eq!(host_tag_value_for_variant(&hole_tag, variant), Some(raw));
                assert_eq!(host_variant_index(&hole_tag, raw), variant);
            }
        }
        assert!(host_tag_value_for_variant(&hole_tag, 7).is_none());
        // Dead value `niche_start + untagged` is inside the span, so the
        // sum is the untagged index. It is not a tag any variant stores.
        assert_eq!(host_variant_index(&hole_tag, 2 + 7), 7);
        assert_ne!(read_tag(&Hole::H, 0, 8), 2 + 7);
    }

    #[test]
    fn parsed_branch_round_trips_through_the_niche_rule() {
        let layout: majit_charon_reader::ullbc::TypeLayout = serde_json::from_str(
            r#"{"discriminator":{"Branch":{"offset":{"guarantee":null,"chosen":0},"int_ty":{"Signed":"Isize"},"children":[[{"start":{"Signed":["Isize","0"]},"end":{"Signed":["Isize","0"]}},{"Known":0}]],"fallback":{"Known":1}}},"variant_layouts":[{"field_offsets":[{"chosen":8}]},{"field_offsets":[{"chosen":0}]}]}"#,
        )
        .unwrap();
        let host = host_layout_from_type(&layout);
        assert_eq!(host.variant_field_offsets, vec![vec![8], vec![0]]);
        let tag = host.tag.expect("niche tag");
        assert_eq!(host_variant_index(&tag, 0), 0);
        assert_eq!(host_variant_index(&tag, 1), 1);
        assert_eq!(host_tag_value_for_variant(&tag, 0), Some(0));
        assert!(host_tag_value_for_variant(&tag, 1).is_none());

        let holed = TagLayout {
            offset: 0,
            signed: false,
            bits: 64,
            encoding: TagEncoding::Niche {
                untagged_variant: 7,
                niche_variants: 0..=8,
                niche_start: 9223372036854775808,
            },
        };
        let start = 9223372036854775808u128;
        assert_eq!(host_variant_index(&holed, start), 0);
        assert_eq!(host_variant_index(&holed, start + 6), 6);
        assert_eq!(host_variant_index(&holed, start + 7), 7);
        assert_eq!(host_variant_index(&holed, start + 8), 8);
        assert_eq!(host_variant_index(&holed, 0), 7);
        assert_eq!(host_tag_value_for_variant(&holed, 8), Some(start + 8));
        assert!(host_tag_value_for_variant(&holed, 7).is_none());
    }
}
