//! Odd-dimension 1px right/bottom edge-replicate padding for 4:2:0 video codecs (H.264, etc.)
//! without scaling the source image.

use crate::ExportError;

/// Rounds odd `width` or `height` up by 1 pixel to the nearest even dimension
/// (clamping `u32::MAX` to `u32::MAX - 1` so addition never overflows).
#[must_use]
pub const fn even_padded_dimensions(width: u32, height: u32) -> (u32, u32) {
    let w = if width == u32::MAX {
        u32::MAX - 1
    } else {
        width + (width & 1)
    };
    let h = if height == u32::MAX {
        u32::MAX - 1
    } else {
        height + (height & 1)
    };
    (w, h)
}

/// Copies a tightly packed RGBA8 frame (`src_size`) into `dst` (`dst_size`),
/// padding any extra right column and/or bottom row by replicating the rightmost column
/// and bottom row edge pixels for opaque output (eliminating visible 1px black borders
/// on odd-dimension `yuv420p` exports), or with `[0, 0, 0, 0]` when `transparent_pad` is `true`,
/// without scaling or filtering the original pixels.
///
/// Reuses `dst`'s existing heap allocation across frames.
pub fn pad_rgba_frame_into(
    src: &[u8],
    src_size: (u32, u32),
    dst_size: (u32, u32),
    transparent_pad: bool,
    dst: &mut Vec<u8>,
) -> Result<(), ExportError> {
    let (src_w, src_h) = src_size;
    let (dst_w, dst_h) = dst_size;

    if dst_w < src_w || dst_h < src_h {
        return Err(ExportError::InvalidSettings(format!(
            "padded dimensions ({dst_w}x{dst_h}) cannot be smaller than source ({src_w}x{src_h})"
        )));
    }

    let expected_src_len = (src_w as usize)
        .checked_mul(src_h as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or_else(|| ExportError::InvalidSettings("source frame byte size overflow".into()))?;
    if src.len() != expected_src_len {
        return Err(ExportError::InvalidFrameSize {
            frame_index: 0,
            expected: expected_src_len,
            actual: src.len(),
        });
    }

    let dst_len = (dst_w as usize)
        .checked_mul(dst_h as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or_else(|| ExportError::InvalidSettings("padded frame byte size overflow".into()))?;

    dst.resize(dst_len, 0);

    let fallback_pixel: [u8; 4] = if transparent_pad {
        [0, 0, 0, 0]
    } else {
        [0, 0, 0, 255]
    };

    let src_row_bytes = (src_w as usize) * 4;
    let dst_row_bytes = (dst_w as usize) * 4;
    let replicate_edge = !transparent_pad && src_w > 0 && src_h > 0;

    for y in 0..(src_h as usize) {
        let src_start = y * src_row_bytes;
        let dst_start = y * dst_row_bytes;
        dst[dst_start..dst_start + src_row_bytes]
            .copy_from_slice(&src[src_start..src_start + src_row_bytes]);

        let right_pixel = if replicate_edge {
            let last_src = src_start + (src_w as usize - 1) * 4;
            [
                src[last_src],
                src[last_src + 1],
                src[last_src + 2],
                src[last_src + 3],
            ]
        } else {
            fallback_pixel
        };

        for x in (src_w as usize)..(dst_w as usize) {
            let p = dst_start + x * 4;
            dst[p..p + 4].copy_from_slice(&right_pixel);
        }
    }

    if replicate_edge {
        let last_row_start = (src_h as usize - 1) * dst_row_bytes;
        for y in (src_h as usize)..(dst_h as usize) {
            let dst_start = y * dst_row_bytes;
            dst.copy_within(last_row_start..last_row_start + dst_row_bytes, dst_start);
        }
    } else {
        for y in (src_h as usize)..(dst_h as usize) {
            let dst_start = y * dst_row_bytes;
            for x in 0..(dst_w as usize) {
                let p = dst_start + x * 4;
                dst[p..p + 4].copy_from_slice(&fallback_pixel);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_padded_dimensions_only_bumps_odd_axes() {
        assert_eq!(even_padded_dimensions(160, 90), (160, 90));
        assert_eq!(even_padded_dimensions(161, 90), (162, 90));
        assert_eq!(even_padded_dimensions(160, 91), (160, 92));
        assert_eq!(even_padded_dimensions(161, 91), (162, 92));
        assert_eq!(even_padded_dimensions(321, 241), (322, 242));
        assert_eq!(
            even_padded_dimensions(u32::MAX, u32::MAX),
            (u32::MAX - 1, u32::MAX - 1)
        );
    }

    #[test]
    fn pad_rgba_frame_edge_replicates_right_and_bottom_without_scaling() {
        // 3x3 distinct source frame where every pixel has unique (R=x*50, G=y*60, B=128, A=255).
        let mut src = Vec::with_capacity(3 * 3 * 4);
        for y in 0..3u8 {
            for x in 0..3u8 {
                src.extend_from_slice(&[x * 50 + 20, y * 60 + 10, 128, 255]);
            }
        }
        let mut dst = Vec::new();

        pad_rgba_frame_into(&src, (3, 3), (4, 4), false, &mut dst).expect("padding succeeds");
        assert_eq!(dst.len(), 4 * 4 * 4);

        for y in 0..4usize {
            for x in 0..4usize {
                let offset = (y * 4 + x) * 4;
                let px = &dst[offset..offset + 4];
                let sx = x.min(2) as u8;
                let sy = y.min(2) as u8;
                let expected = [sx * 50 + 20, sy * 60 + 10, 128, 255];
                assert_eq!(
                    px, &expected,
                    "pixel ({x},{y}) expected edge-replicated {expected:?}, got {px:?}"
                );
            }
        }
    }

    #[test]
    fn pad_rgba_frame_transparent_uses_zero_alpha_padding_and_321x241_edge_replicates() {
        // Transparent padding still uses [0, 0, 0, 0].
        let src_small: Vec<u8> = [200_u8, 100, 50, 255].repeat(9);
        let mut dst = Vec::new();
        pad_rgba_frame_into(&src_small, (3, 3), (4, 4), true, &mut dst).expect("transparent pad");
        assert_eq!(&dst[(3 * 4 + 3) * 4..(3 * 4 + 3) * 4 + 4], &[0, 0, 0, 0]);

        // Odd 321x241 -> 322x242 opaque frame replicates right column & bottom row (no 1px black border).
        let (sw, sh) = (321usize, 241usize);
        let (dw, dh) = even_padded_dimensions(sw as u32, sh as u32);
        assert_eq!((dw, dh), (322, 242));
        let mut odd_src = vec![0u8; sw * sh * 4];
        for y in 0..sh {
            for x in 0..sw {
                let o = (y * sw + x) * 4;
                odd_src[o] = ((x + 10) % 250) as u8;
                odd_src[o + 1] = ((y + 20) % 250) as u8;
                odd_src[o + 2] = 200;
                odd_src[o + 3] = 255;
            }
        }
        pad_rgba_frame_into(&odd_src, (sw as u32, sh as u32), (dw, dh), false, &mut dst)
            .expect("321x241 pad");
        // Rightmost padded column equals column 320.
        for y in 0..sh {
            let orig = (y * (dw as usize) + (sw - 1)) * 4;
            let pad = (y * (dw as usize) + sw) * 4;
            assert_eq!(&dst[pad..pad + 4], &dst[orig..orig + 4]);
        }
        // Bottom padded row (including corner 321,241) equals row 240.
        let row_bytes = (dw as usize) * 4;
        assert_eq!(
            &dst[sh * row_bytes..(sh + 1) * row_bytes],
            &dst[(sh - 1) * row_bytes..sh * row_bytes]
        );
    }
}
