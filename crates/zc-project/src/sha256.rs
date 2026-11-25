//! Pure-Rust FIPS 180-4 SHA-256 digest implementation (zero `unsafe`, O(1) stack workspace).

/// Maximum prefix length (256 MiB) hashed by CM3 when deriving 32-hex asset keys.
pub const CM3_ASSET_HASH_PREFIX_BYTES: usize = 256 * 1024 * 1024;

const HEX_LOWER: &[u8; 16] = b"0123456789abcdef";

const INIT_H: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

macro_rules! sha256_round {
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident, $k:expr, $w:expr) => {{
        let s1 = $e.rotate_right(6) ^ $e.rotate_right(11) ^ $e.rotate_right(25);
        let ch = ($e & $f) ^ ((!$e) & $g);
        let temp1 = $h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add($k)
            .wrapping_add($w);
        let s0 = $a.rotate_right(2) ^ $a.rotate_right(13) ^ $a.rotate_right(22);
        let maj = ($a & $b) ^ ($a & $c) ^ ($b & $c);
        let temp2 = s0.wrapping_add(maj);
        $d = $d.wrapping_add(temp1);
        $h = temp1.wrapping_add(temp2);
    }};
}

macro_rules! schedule_word {
    ($w:ident, $i:expr) => {{
        let w15 = $w[($i + 1) & 15];
        let w2 = $w[($i + 14) & 15];
        let s0 = w15.rotate_right(7) ^ w15.rotate_right(18) ^ (w15 >> 3);
        let s1 = w2.rotate_right(17) ^ w2.rotate_right(19) ^ (w2 >> 10);
        let next = $w[$i]
            .wrapping_add(s0)
            .wrapping_add($w[($i + 9) & 15])
            .wrapping_add(s1);
        $w[$i] = next;
        next
    }};
}

macro_rules! sha256_rounds_initial_8 {
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident, $w:ident, $base:expr) => {{
        sha256_round!($a, $b, $c, $d, $e, $f, $g, $h, K[$base], $w[$base]);
        sha256_round!($h, $a, $b, $c, $d, $e, $f, $g, K[$base + 1], $w[$base + 1]);
        sha256_round!($g, $h, $a, $b, $c, $d, $e, $f, K[$base + 2], $w[$base + 2]);
        sha256_round!($f, $g, $h, $a, $b, $c, $d, $e, K[$base + 3], $w[$base + 3]);
        sha256_round!($e, $f, $g, $h, $a, $b, $c, $d, K[$base + 4], $w[$base + 4]);
        sha256_round!($d, $e, $f, $g, $h, $a, $b, $c, K[$base + 5], $w[$base + 5]);
        sha256_round!($c, $d, $e, $f, $g, $h, $a, $b, K[$base + 6], $w[$base + 6]);
        sha256_round!($b, $c, $d, $e, $f, $g, $h, $a, K[$base + 7], $w[$base + 7]);
    }};
}

macro_rules! sha256_rounds_scheduled_8 {
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident, $w:ident, $k_base:expr, $w_base:expr) => {{
        let w0 = schedule_word!($w, $w_base);
        sha256_round!($a, $b, $c, $d, $e, $f, $g, $h, K[$k_base], w0);
        let w1 = schedule_word!($w, $w_base + 1);
        sha256_round!($h, $a, $b, $c, $d, $e, $f, $g, K[$k_base + 1], w1);
        let w2 = schedule_word!($w, $w_base + 2);
        sha256_round!($g, $h, $a, $b, $c, $d, $e, $f, K[$k_base + 2], w2);
        let w3 = schedule_word!($w, $w_base + 3);
        sha256_round!($f, $g, $h, $a, $b, $c, $d, $e, K[$k_base + 3], w3);
        let w4 = schedule_word!($w, $w_base + 4);
        sha256_round!($e, $f, $g, $h, $a, $b, $c, $d, K[$k_base + 4], w4);
        let w5 = schedule_word!($w, $w_base + 5);
        sha256_round!($d, $e, $f, $g, $h, $a, $b, $c, K[$k_base + 5], w5);
        let w6 = schedule_word!($w, $w_base + 6);
        sha256_round!($c, $d, $e, $f, $g, $h, $a, $b, K[$k_base + 6], w6);
        let w7 = schedule_word!($w, $w_base + 7);
        sha256_round!($b, $c, $d, $e, $f, $g, $h, $a, K[$k_base + 7], w7);
    }};
}

#[inline(always)]
#[allow(unused_assignments)]
fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 16];
    let (words, _) = block.as_chunks::<4>();
    for (dst, word) in w.iter_mut().zip(words.iter()) {
        *dst = u32::from_be_bytes(*word);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;

    sha256_rounds_initial_8!(a, b, c, d, e, f, g, h, w, 0);
    sha256_rounds_initial_8!(a, b, c, d, e, f, g, h, w, 8);
    sha256_rounds_scheduled_8!(a, b, c, d, e, f, g, h, w, 16, 0);
    sha256_rounds_scheduled_8!(a, b, c, d, e, f, g, h, w, 24, 8);
    sha256_rounds_scheduled_8!(a, b, c, d, e, f, g, h, w, 32, 0);
    sha256_rounds_scheduled_8!(a, b, c, d, e, f, g, h, w, 40, 8);
    sha256_rounds_scheduled_8!(a, b, c, d, e, f, g, h, w, 48, 0);
    sha256_rounds_scheduled_8!(a, b, c, d, e, f, g, h, w, 56, 8);

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

/// Computes the 32-byte SHA-256 digest of `data`.
#[must_use]
pub fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let mut state = INIT_H;
    let (chunks, remainder) = data.as_chunks::<64>();
    for block in chunks {
        compress(&mut state, block);
    }

    let rem_len = remainder.len();
    let bit_len = (data.len() as u64).wrapping_mul(8).to_be_bytes();
    if rem_len < 56 {
        let mut block = [0u8; 64];
        block[..rem_len].copy_from_slice(remainder);
        block[rem_len] = 0x80;
        block[56..64].copy_from_slice(&bit_len);
        compress(&mut state, &block);
    } else {
        let mut block1 = [0u8; 64];
        block1[..rem_len].copy_from_slice(remainder);
        block1[rem_len] = 0x80;
        compress(&mut state, &block1);
        let mut block2 = [0u8; 64];
        block2[56..64].copy_from_slice(&bit_len);
        compress(&mut state, &block2);
    }

    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Formats a 32-byte SHA-256 digest as a 64-character lowercase hexadecimal string.
#[must_use]
pub fn sha256_hex_from_bytes(digest: &[u8; 32]) -> String {
    let mut out = Vec::with_capacity(64);
    for &byte in digest {
        out.push(HEX_LOWER[(byte >> 4) as usize]);
        out.push(HEX_LOWER[(byte & 0x0f) as usize]);
    }
    String::from_utf8(out).expect("HEX_LOWER contains only ASCII")
}

/// Computes the 64-character lowercase hexadecimal SHA-256 digest of `data`.
#[must_use]
pub fn sha256_hex(data: &[u8]) -> String {
    let digest = sha256_bytes(data);
    sha256_hex_from_bytes(&digest)
}

/// Derives a CM3-compatible 32-character lowercase hex asset key from a precomputed
/// 32-byte SHA-256 digest.
///
/// CM3 (`PZ.asset.hash`) interprets the SHA-256 digest buffer as a little-endian
/// `Uint32Array` and formats the first 4 `u32` words as 8 lowercase hex characters
/// each (32 hex chars total).
#[must_use]
pub fn cm3_asset_key_from_bytes(digest: &[u8; 32]) -> String {
    let mut out = Vec::with_capacity(32);
    for chunk in digest[..16].as_chunks::<4>().0 {
        for &byte in chunk.iter().rev() {
            out.push(HEX_LOWER[(byte >> 4) as usize]);
            out.push(HEX_LOWER[(byte & 0x0f) as usize]);
        }
    }
    String::from_utf8(out).expect("HEX_LOWER contains only ASCII")
}

/// Derives a CM3-compatible 32-character lowercase hex asset key from the first
/// 256 MiB of `data`.
#[must_use]
pub fn cm3_asset_key(data: &[u8]) -> String {
    let prefix_len = data.len().min(CM3_ASSET_HASH_PREFIX_BYTES);
    let digest = sha256_bytes(&data[..prefix_len]);
    cm3_asset_key_from_bytes(&digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nist_vectors_match() {
        // FIPS 180-4 / NIST CAVP short & multi-block test vectors.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // 448-bit (56 bytes — exact 2-block padding boundary)
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // 896-bit (112 bytes — 2 full blocks + padding)
        assert_eq!(
            sha256_hex(
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
            ),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
        );
        // 1,000,000 repetitions of 'a' (NIST long vector)
        let one_million_a = vec![b'a'; 1_000_000];
        assert_eq!(
            sha256_hex(&one_million_a),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
        // 55 bytes (single-block boundary) vs 64 bytes (exact block)
        assert_eq!(
            sha256_hex(&[b'a'; 55]),
            "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"
        );
        assert_eq!(
            sha256_hex(&[b'a'; 64]),
            "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"
        );
    }

    #[test]
    fn cm3_asset_key_uses_little_endian_u32_words() {
        // SHA-256("abc") first 16 bytes: ba7816bf 8f01cfea 414140de 5dae2223
        // Little-endian Uint32Array words: bf1678ba eacf018f de404141 2322ae5d
        let key = cm3_asset_key(b"abc");
        assert_eq!(key.len(), 32);
        assert_eq!(key, "bf1678baeacf018fde4041412322ae5d");
        let digest = sha256_bytes(b"abc");
        assert_eq!(cm3_asset_key_from_bytes(&digest), key);
        assert_eq!(sha256_hex_from_bytes(&digest), sha256_hex(b"abc"));

        // SHA-256("") first 16 bytes: e3b0c442 98fc1c14 9afbf4c8 996fb924
        // Little-endian Uint32Array words: 42c4b0e3 141cfc98 c8f4fb9a 24b96f99
        assert_eq!(cm3_asset_key(b""), "42c4b0e3141cfc98c8f4fb9a24b96f99");
    }
}
