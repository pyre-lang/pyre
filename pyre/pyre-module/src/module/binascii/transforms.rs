//! Byte transforms for the 3.15 `binascii` surface that are not already in
//! `rustpython_common::binascii` (`binascii_a2b_base32_impl`,
//! `binascii_b2a_base32_impl`, `binascii_a2b_ascii85_impl`,
//! `binascii_b2a_ascii85_impl`, `binascii_a2b_base85_impl`,
//! `binascii_b2a_base85_impl`, `binascii_a2b_hex_impl`).

pub const BASE64_ALPHABET: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
pub const URLSAFE_BASE64_ALPHABET: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
pub const CRYPT_ALPHABET: &[u8] =
    b"./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
pub const UU_ALPHABET: &[u8] =
    b" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_";
pub const BINHEX_ALPHABET: &[u8] =
    b"!\"#$%&'()*+,-012345689@ABCDEFGHIJKLMNPQRSTUVXYZ[`abcdefhijklmpqr";
pub const BASE85_ALPHABET: &[u8] =
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~";
pub const ASCII85_ALPHABET: &[u8] =
    b"!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstu";
pub const Z85_ALPHABET: &[u8] =
    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#";
pub const BASE32_ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
pub const BASE32HEX_ALPHABET: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUV";

const BASE32_PAD: u8 = b'=';
const POW85: [u32; 5] = [1, 85, 7225, 614125, 52_200_625];
const ASCII85_Z: u32 = 0x0000_0000;
const ASCII85_Y: u32 = 0x2020_2020;

const fn reverse_table(alphabet: &[u8], pad: i16) -> [u8; 256] {
    let mut out = [0xffu8; 256];
    let mut i = 0;
    while i < alphabet.len() {
        out[alphabet[i] as usize] = i as u8;
        i += 1;
    }
    if pad >= 0 {
        out[pad as usize] = alphabet.len() as u8;
    }
    out
}

const TABLE_A2B_ASCII85: [u8; 256] = reverse_table(ASCII85_ALPHABET, -1);

/// Insert `\\n` every `width` characters from the left (`wraplines`).
pub fn wrap_lines(data: &[u8], width: usize) -> Vec<u8> {
    if width == 0 || data.len() <= width {
        return data.to_vec();
    }
    let newlines = (data.len() - 1) / width;
    let mut out = Vec::with_capacity(data.len() + newlines);
    for (i, chunk) in data.chunks(width).enumerate() {
        if i > 0 {
            out.push(b'\n');
        }
        out.extend_from_slice(chunk);
    }
    out
}

pub fn ignore_set(chars: &[u8]) -> [bool; 256] {
    let mut set = [false; 256];
    for &c in chars {
        set[c as usize] = true;
    }
    set
}

pub fn reverse_table_runtime(alphabet: &[u8], pad: Option<u8>) -> [u8; 256] {
    let pad = pad.map(|p| p as i16).unwrap_or(-1);
    reverse_table(alphabet, pad)
}

fn too_much(kind: &str) -> String {
    format!("Too much data for {kind}")
}

/// `binascii_a2b_hex_impl`
pub fn a2b_hex(data: &[u8], ignorechars: &[u8]) -> Result<Vec<u8>, String> {
    let ignored = ignore_set(ignorechars);
    let mut out = Vec::with_capacity(data.len() / 2);
    let mut pair_pos = false;
    let mut left = 0u8;
    for &ch in data {
        let digit = match ch {
            b'0'..=b'9' => ch - b'0',
            b'a'..=b'f' => ch - b'a' + 10,
            b'A'..=b'F' => ch - b'A' + 10,
            _ => {
                if ignored[ch as usize] {
                    continue;
                }
                return Err("Non-hexadecimal digit found".to_owned());
            }
        };
        if !pair_pos {
            pair_pos = true;
            left = digit;
        } else {
            pair_pos = false;
            out.push((left << 4) | digit);
        }
    }
    if pair_pos {
        return Err("Odd number of hexadecimal digits".to_owned());
    }
    Ok(out)
}

/// `binascii_b2a_base32_impl`
pub fn b2a_base32(
    data: &[u8],
    padded: bool,
    wrapcol: usize,
    alphabet: &[u8],
) -> Result<Vec<u8>, String> {
    let table = alphabet;
    let mut ascii_len = data.len().div_ceil(5).saturating_mul(8);
    let mut pads = (5 - (data.len() % 5)) % 5 * 8 / 5;
    if !padded {
        ascii_len = ascii_len.saturating_sub(pads);
        pads = 0;
    }
    let mut wrapcol = wrapcol;
    if wrapcol != 0 && ascii_len != 0 {
        wrapcol = if wrapcol < 8 { 8 } else { wrapcol / 8 * 8 };
        ascii_len += (ascii_len - 1) / wrapcol;
    }
    if ascii_len > isize::MAX as usize {
        return Err(too_much("base32"));
    }
    let mut out = Vec::with_capacity(ascii_len);
    let (full, rem) = data.split_at(data.len() - data.len() % 5);
    for chunk in full.chunks_exact(5) {
        let val = (u64::from(chunk[0]) << 32)
            | (u64::from(chunk[1]) << 24)
            | (u64::from(chunk[2]) << 16)
            | (u64::from(chunk[3]) << 8)
            | u64::from(chunk[4]);
        for i in 0..8 {
            out.push(table[((val >> (35 - 5 * i)) & 0x1f) as usize]);
        }
    }
    match rem.len() {
        0 => {}
        1 => {
            let val = u32::from(rem[0]);
            out.push(table[((val >> 3) & 0x1f) as usize]);
            out.push(table[((val << 2) & 0x1f) as usize]);
        }
        2 => {
            let val = (u32::from(rem[0]) << 8) | u32::from(rem[1]);
            out.push(table[((val >> 11) & 0x1f) as usize]);
            out.push(table[((val >> 6) & 0x1f) as usize]);
            out.push(table[((val >> 1) & 0x1f) as usize]);
            out.push(table[((val << 4) & 0x1f) as usize]);
        }
        3 => {
            let val = (u32::from(rem[0]) << 16) | (u32::from(rem[1]) << 8) | u32::from(rem[2]);
            out.push(table[((val >> 19) & 0x1f) as usize]);
            out.push(table[((val >> 14) & 0x1f) as usize]);
            out.push(table[((val >> 9) & 0x1f) as usize]);
            out.push(table[((val >> 4) & 0x1f) as usize]);
            out.push(table[((val << 1) & 0x1f) as usize]);
        }
        4 => {
            let val = (u32::from(rem[0]) << 24)
                | (u32::from(rem[1]) << 16)
                | (u32::from(rem[2]) << 8)
                | u32::from(rem[3]);
            out.push(table[((val >> 27) & 0x1f) as usize]);
            out.push(table[((val >> 22) & 0x1f) as usize]);
            out.push(table[((val >> 17) & 0x1f) as usize]);
            out.push(table[((val >> 12) & 0x1f) as usize]);
            out.push(table[((val >> 7) & 0x1f) as usize]);
            out.push(table[((val >> 2) & 0x1f) as usize]);
            out.push(table[((val << 3) & 0x1f) as usize]);
        }
        _ => {}
    }
    for _ in 0..pads {
        out.push(BASE32_PAD);
    }
    if wrapcol != 0 && !out.is_empty() {
        out = wrap_lines(&out, wrapcol);
    }
    Ok(out)
}

/// `binascii_a2b_base32_impl`
pub fn a2b_base32(
    data: &[u8],
    padded: bool,
    table: &[u8; 256],
    ignorechars: &[u8],
    canonical: bool,
) -> Result<Vec<u8>, String> {
    let ignored = ignore_set(ignorechars);
    let mut out = Vec::with_capacity(data.len().div_ceil(8) * 5);
    let mut leftchar = 0u8;
    let mut octa_pos = 0i32;
    let mut pads = 0i32;
    for &this_ch in data {
        if padded && this_ch == BASE32_PAD {
            pads += 1;
            if (octa_pos == 2 || octa_pos == 4 || octa_pos == 5 || octa_pos == 7)
                && octa_pos + pads <= 8
            {
                continue;
            }
            if ignored[BASE32_PAD as usize] {
                continue;
            }
            if octa_pos == 1 || octa_pos == 3 || octa_pos == 6 {
                break;
            }
            return Err(if octa_pos == 0 && out.is_empty() {
                "Leading padding not allowed".to_owned()
            } else {
                "Excess padding not allowed".to_owned()
            });
        }
        let v = table[this_ch as usize];
        if v >= 32 {
            if !ignored[this_ch as usize] {
                return Err(if this_ch == BASE32_PAD {
                    "Padding not allowed".to_owned()
                } else {
                    "Only base32 data is allowed".to_owned()
                });
            }
            continue;
        }
        if pads != 0 && !ignored[BASE32_PAD as usize] {
            return Err(if octa_pos + pads == 8 {
                "Excess data after padding".to_owned()
            } else {
                "Discontinuous padding not allowed".to_owned()
            });
        }
        match octa_pos {
            0 => {
                octa_pos = 1;
                leftchar = v;
            }
            1 => {
                octa_pos = 2;
                out.push((leftchar << 3) | (v >> 2));
                leftchar = v & 0x03;
            }
            2 => {
                octa_pos = 3;
                leftchar = (leftchar << 5) | v;
            }
            3 => {
                octa_pos = 4;
                out.push((leftchar << 1) | (v >> 4));
                leftchar = v & 0x0f;
            }
            4 => {
                octa_pos = 5;
                out.push((leftchar << 4) | (v >> 1));
                leftchar = v & 0x01;
            }
            5 => {
                octa_pos = 6;
                leftchar = (leftchar << 5) | v;
            }
            6 => {
                octa_pos = 7;
                out.push((leftchar << 2) | (v >> 3));
                leftchar = v & 0x07;
            }
            7 => {
                octa_pos = 0;
                out.push((leftchar << 5) | v);
                leftchar = 0;
            }
            _ => {}
        }
    }
    if octa_pos == 1 || octa_pos == 3 || octa_pos == 6 {
        let n = out.len() / 5 * 8 + octa_pos as usize;
        return Err(format!(
            "Invalid base32-encoded string: number of data characters ({n}) cannot be 1, 3, or 6 more than a multiple of 8"
        ));
    }
    if padded && octa_pos != 0 && octa_pos + pads < 8 {
        return Err("Incorrect padding".to_owned());
    }
    if canonical && leftchar != 0 {
        return Err("Non-zero padding bits".to_owned());
    }
    Ok(out)
}

/// `binascii_b2a_base85_impl`
pub fn b2a_base85(
    data: &[u8],
    pad: bool,
    wrapcol: usize,
    alphabet: &[u8],
) -> Result<Vec<u8>, String> {
    let table = alphabet;
    let mut out_len = data.len().div_ceil(4).saturating_mul(5);
    if !pad && data.len() % 4 != 0 {
        out_len = out_len.saturating_sub(4 - (data.len() % 4));
    }
    let mut wrapcol = wrapcol;
    if wrapcol != 0 && out_len != 0 {
        wrapcol = if wrapcol < 5 { 5 } else { wrapcol / 5 * 5 };
        out_len += (out_len - 1) / wrapcol;
    }
    if out_len > isize::MAX as usize {
        return Err(too_much("Base85"));
    }
    let mut out = Vec::with_capacity(out_len);
    let (full, rem) = data.split_at(data.len() - data.len() % 4);
    for chunk in full.chunks_exact(4) {
        let mut leftchar = (u32::from(chunk[0]) << 24)
            | (u32::from(chunk[1]) << 16)
            | (u32::from(chunk[2]) << 8)
            | u32::from(chunk[3]);
        let mut buf = [0u8; 5];
        buf[4] = table[(leftchar % 85) as usize];
        leftchar /= 85;
        buf[3] = table[(leftchar % 85) as usize];
        leftchar /= 85;
        buf[2] = table[(leftchar % 85) as usize];
        leftchar /= 85;
        buf[1] = table[(leftchar % 85) as usize];
        leftchar /= 85;
        buf[0] = table[leftchar as usize];
        out.extend_from_slice(&buf);
    }
    if !rem.is_empty() {
        let mut leftchar = 0u32;
        for i in 0..4 {
            leftchar <<= 8;
            if i < rem.len() {
                leftchar |= u32::from(rem[i]);
            }
        }
        let group_len = if pad { 5 } else { rem.len() + 1 };
        let mut buf = [0u8; 5];
        for i in (0..5).rev() {
            if i < group_len {
                buf[i] = table[(leftchar % 85) as usize];
            }
            leftchar /= 85;
        }
        out.extend_from_slice(&buf[..group_len]);
    }
    if wrapcol != 0 && !out.is_empty() {
        out = wrap_lines(&out, wrapcol);
    }
    Ok(out)
}

/// `binascii_a2b_base85_impl`
pub fn a2b_base85(
    data: &[u8],
    table: &[u8; 256],
    ignorechars: &[u8],
    canonical: bool,
) -> Result<Vec<u8>, String> {
    decode_base85_inner(data, table, ignorechars, canonical, false, false)
}

/// `binascii_b2a_ascii85_impl`
pub fn b2a_ascii85(
    data: &[u8],
    foldspaces: bool,
    wrapcol: usize,
    pad: bool,
    adobe: bool,
) -> Result<Vec<u8>, String> {
    let table = ASCII85_ALPHABET;
    let mut wrapcol = wrapcol;
    if adobe && wrapcol == 1 {
        wrapcol = 2;
    }
    let mut out_len = data.len().div_ceil(4).saturating_mul(5);
    if adobe {
        out_len += 4;
    }
    if !pad && data.len() % 4 != 0 {
        out_len = out_len.saturating_sub(4 - (data.len() % 4));
    }
    if wrapcol != 0 && out_len != 0 && out_len <= isize::MAX as usize {
        out_len += (out_len - 1) / wrapcol;
    }
    if out_len > isize::MAX as usize {
        return Err(too_much("Ascii85"));
    }
    let mut out = Vec::with_capacity(out_len);
    if adobe {
        out.push(b'<');
        out.push(b'~');
    }
    let (full, rem) = data.split_at(data.len() - data.len() % 4);
    for chunk in full.chunks_exact(4) {
        let mut leftchar = (u32::from(chunk[0]) << 24)
            | (u32::from(chunk[1]) << 16)
            | (u32::from(chunk[2]) << 8)
            | u32::from(chunk[3]);
        if leftchar == ASCII85_Z {
            out.push(b'z');
        } else if foldspaces && leftchar == ASCII85_Y {
            out.push(b'y');
        } else {
            let mut buf = [0u8; 5];
            buf[4] = table[(leftchar % 85) as usize];
            leftchar /= 85;
            buf[3] = table[(leftchar % 85) as usize];
            leftchar /= 85;
            buf[2] = table[(leftchar % 85) as usize];
            leftchar /= 85;
            buf[1] = table[(leftchar % 85) as usize];
            leftchar /= 85;
            buf[0] = table[leftchar as usize];
            out.extend_from_slice(&buf);
        }
    }
    if !rem.is_empty() {
        let mut leftchar = 0u32;
        for i in 0..4 {
            leftchar <<= 8;
            if i < rem.len() {
                leftchar |= u32::from(rem[i]);
            }
        }
        if pad && leftchar == ASCII85_Z {
            out.push(b'z');
        } else {
            let group_len = if pad { 5 } else { rem.len() + 1 };
            let mut buf = [0u8; 5];
            for i in (0..5).rev() {
                if i < group_len {
                    buf[i] = table[(leftchar % 85) as usize];
                }
                leftchar /= 85;
            }
            out.extend_from_slice(&buf[..group_len]);
        }
    }
    if adobe {
        out.push(b'~');
        out.push(b'>');
    }
    if wrapcol != 0 && !out.is_empty() {
        out = wrap_lines(&out, wrapcol);
        if adobe && out.len() >= 3 && out[out.len() - 2] == b'\n' {
            let n = out.len();
            out[n - 3] = b'\n';
            out[n - 2] = b'~';
        }
    }
    Ok(out)
}

/// `binascii_a2b_ascii85_impl`
pub fn a2b_ascii85(
    data: &[u8],
    foldspaces: bool,
    adobe: bool,
    ignorechars: &[u8],
    canonical: bool,
) -> Result<Vec<u8>, String> {
    let mut data = data;
    if adobe {
        if data.len() < 2 || data[data.len() - 2] != b'~' || data[data.len() - 1] != b'>' {
            return Err("Ascii85 encoded byte sequences must end with b'~>'".to_owned());
        }
        data = &data[..data.len() - 2];
        if data.len() >= 2 && data[0] == b'<' && data[1] == b'~' {
            data = &data[2..];
        }
    }
    decode_base85_inner(
        data,
        &TABLE_A2B_ASCII85,
        ignorechars,
        canonical,
        true,
        foldspaces,
    )
}

fn decode_base85_inner(
    data: &[u8],
    table: &[u8; 256],
    ignorechars: &[u8],
    canonical: bool,
    ascii85: bool,
    foldspaces: bool,
) -> Result<Vec<u8>, String> {
    let ignored = ignore_set(ignorechars);
    let orig_len = data.len() as isize;
    let mut ascii_len = orig_len;
    let mut pos = 0usize;
    let mut out = Vec::with_capacity(data.len().div_ceil(5) * 4 + 16);
    let mut leftchar: u32 = 0;
    let mut group_pos: i32 = 0;
    let mut from_z = false;
    while ascii_len > 0 || group_pos != 0 {
        let (this_ch, this_digit) = if ascii_len > 0 {
            let ch = data[pos];
            (ch, table[ch as usize])
        } else {
            (0u8, 84u8)
        };
        if this_digit < 85 {
            if group_pos == 4 {
                let next = leftchar as u64 * 85 + u64::from(this_digit);
                if next > u32::MAX as u64 {
                    return Err(if ascii85 {
                        "Ascii85 overflow".to_owned()
                    } else {
                        format!(
                            "Base85 overflow in hunk starting at byte {}",
                            (orig_len - ascii_len) / 5 * 5
                        )
                    });
                }
                leftchar = next as u32;
            } else {
                leftchar = leftchar * 85 + u32::from(this_digit);
            }
            group_pos += 1;
        } else if ascii85 && ((this_ch == b'y' && foldspaces) || this_ch == b'z') {
            if group_pos != 0 {
                return Err(format!("'{}' inside Ascii85 5-tuple", this_ch as char));
            }
            leftchar = if this_ch == b'y' {
                ASCII85_Y
            } else {
                ASCII85_Z
            };
            from_z = this_ch == b'z';
            group_pos = 5;
        } else if !ignored[this_ch as usize] {
            return Err(if ascii85 {
                format!("Non-Ascii85 digit found: {}", this_ch as char)
            } else {
                format!("bad Base85 character at position {}", orig_len - ascii_len)
            });
        }
        if group_pos == 5 {
            let chunk_len = if ascii_len < 1 {
                3 + ascii_len as i32
            } else {
                4
            };
            if chunk_len == 0 {
                return Err(if ascii85 {
                    "Incomplete Ascii85 group".to_owned()
                } else {
                    "Incomplete Base85 group".to_owned()
                });
            }
            for i in 0..chunk_len {
                out.push(((leftchar >> (24 - 8 * i)) & 0xff) as u8);
            }
            if canonical {
                if ascii85 && chunk_len == 4 && leftchar == 0 && !from_z {
                    return Err("Non-canonical encoding, use 'z' for all-zero groups".to_owned());
                }
                if chunk_len < 4 {
                    let n_pad = 4 - chunk_len;
                    let canonical_top = (leftchar >> (n_pad * 8)) << (n_pad * 8);
                    if canonical_top / POW85[n_pad as usize] != leftchar / POW85[n_pad as usize] {
                        return Err("Non-zero padding bits".to_owned());
                    }
                }
            }
            from_z = false;
            group_pos = 0;
            leftchar = 0;
        }
        if ascii_len > 0 {
            pos += 1;
        }
        ascii_len -= 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alphabets_match_3_15() {
        assert_eq!(
            BASE64_ALPHABET,
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        );
        assert_eq!(
            URLSAFE_BASE64_ALPHABET,
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        );
        assert_eq!(BASE32_ALPHABET, b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567");
        assert_eq!(BASE32HEX_ALPHABET, b"0123456789ABCDEFGHIJKLMNOPQRSTUV");
        assert_eq!(
            BASE85_ALPHABET,
            b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~"
        );
        assert_eq!(
            ASCII85_ALPHABET,
            b"!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstu"
        );
        assert_eq!(
            Z85_ALPHABET,
            b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#"
        );
        assert_eq!(
            reverse_table_runtime(BASE32_ALPHABET, Some(BASE32_PAD))[b'A' as usize],
            0
        );
        assert_eq!(
            reverse_table_runtime(BASE85_ALPHABET, None)[b'0' as usize],
            0
        );
        assert_eq!(TABLE_A2B_ASCII85[b'!' as usize], 0);
    }

    #[test]
    fn b2a_base32_vectors() {
        assert_eq!(
            b2a_base32(b"hello", true, 0, BASE32_ALPHABET).unwrap(),
            b"NBSWY3DP"
        );
        assert_eq!(
            b2a_base32(b"hello", true, 0, BASE32HEX_ALPHABET).unwrap(),
            b"D1IMOR3F"
        );
        assert_eq!(
            b2a_base32(b"h", true, 0, BASE32_ALPHABET).unwrap(),
            b"NA======"
        );
        assert_eq!(b2a_base32(b"h", false, 0, BASE32_ALPHABET).unwrap(), b"NA");
        assert_eq!(
            b2a_base32(b"0123456789abcdef", true, 8, BASE32_ALPHABET).unwrap(),
            b"GAYTEMZU\nGU3DOOBZ\nMFRGGZDF\nMY======"
        );
    }

    #[test]
    fn a2b_base32_vectors() {
        let table = reverse_table_runtime(BASE32_ALPHABET, Some(BASE32_PAD));
        assert_eq!(
            a2b_base32(b"NBSWY3DP", true, &table, b"", false).unwrap(),
            b"hello"
        );
        assert_eq!(
            a2b_base32(b"NA======", true, &table, b"", false).unwrap(),
            b"h"
        );
        assert_eq!(a2b_base32(b"NA", false, &table, b"", false).unwrap(), b"h");
        assert_eq!(
            a2b_base32(b"NBSWY3DP\n", true, &table, b"\n", false).unwrap(),
            b"hello"
        );
        assert_eq!(
            a2b_base32(b"=NBSWY3DP", true, &table, b"", false).unwrap_err(),
            "Leading padding not allowed"
        );
    }

    #[test]
    fn b2a_base85_vectors() {
        assert_eq!(
            b2a_base85(b"hello", false, 0, BASE85_ALPHABET).unwrap(),
            b"Xk~0{Zv"
        );
        assert_eq!(
            b2a_base85(b"hello", false, 0, Z85_ALPHABET).unwrap(),
            b"xK#0@zV"
        );
        assert_eq!(
            b2a_base85(b"h", true, 0, BASE85_ALPHABET).unwrap(),
            b"XaE2J"
        );
        assert_eq!(b2a_base85(b"h", false, 0, BASE85_ALPHABET).unwrap(), b"Xa");
    }

    #[test]
    fn a2b_base85_vectors() {
        let table = reverse_table_runtime(BASE85_ALPHABET, None);
        assert_eq!(
            a2b_base85(b"Xk~0{Zv", &table, b"", false).unwrap(),
            b"hello"
        );
        assert_eq!(
            a2b_base85(b"XaE2J", &table, b"", false).unwrap(),
            b"h\0\0\0"
        );
        assert_eq!(a2b_base85(b"Xa", &table, b"", false).unwrap(), b"h");
    }

    #[test]
    fn b2a_ascii85_vectors() {
        assert_eq!(
            b2a_ascii85(b"hello", false, 0, false, false).unwrap(),
            b"BOu!rDZ"
        );
        assert_eq!(
            b2a_ascii85(b"hello", false, 0, false, true).unwrap(),
            b"<~BOu!rDZ~>"
        );
        assert_eq!(b2a_ascii85(b"    ", true, 0, false, false).unwrap(), b"y");
        assert_eq!(b2a_ascii85(b"h", false, 0, true, false).unwrap(), b"BE/#4");
        assert_eq!(b2a_ascii85(b"h", false, 0, false, false).unwrap(), b"BE");
        assert_eq!(
            b2a_ascii85(b"hello", false, 1, false, false).unwrap(),
            b"B\nO\nu\n!\nr\nD\nZ"
        );
        assert_eq!(
            b2a_ascii85(b"hello", false, 1, false, true).unwrap(),
            b"<~\nBO\nu!\nrD\nZ\n~>"
        );
    }

    #[test]
    fn a2b_ascii85_vectors() {
        assert_eq!(
            a2b_ascii85(b"BOu!rDZ", false, false, b"", false).unwrap(),
            b"hello"
        );
        assert_eq!(
            a2b_ascii85(b"<~BOu!rDZ~>", false, true, b"", false).unwrap(),
            b"hello"
        );
        assert_eq!(a2b_ascii85(b"y", true, false, b"", false).unwrap(), b"    ");
        assert_eq!(
            a2b_ascii85(b"B", false, false, b"", false).unwrap_err(),
            "Incomplete Ascii85 group"
        );
        assert_eq!(
            a2b_ascii85(b"hello", false, true, b"", false).unwrap_err(),
            "Ascii85 encoded byte sequences must end with b'~>'"
        );
    }

    #[test]
    fn a2b_hex_vectors() {
        assert_eq!(a2b_hex(b"abcd", b"").unwrap(), b"\xab\xcd");
        assert_eq!(a2b_hex(b"ab cd", b" ").unwrap(), b"\xab\xcd");
        assert_eq!(
            a2b_hex(b"abc", b"").unwrap_err(),
            "Odd number of hexadecimal digits"
        );
        assert_eq!(
            a2b_hex(b"abxd", b"").unwrap_err(),
            "Non-hexadecimal digit found"
        );
    }

    #[test]
    fn wrap_lines_left_chunks() {
        assert_eq!(wrap_lines(b"abcdefgh", 3), b"abc\ndef\ngh");
        assert_eq!(wrap_lines(b"ab", 8), b"ab");
        assert_eq!(wrap_lines(b"abcd", 0), b"abcd");
    }
}
