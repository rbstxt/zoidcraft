//! Odd-dimension 1px right/bottom black padding for 4:2:0 video codecs (H.264, etc.)
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
/// padding any extra right column and/or bottom row with black pixels (`[0, 0, 0, 255]`
/// for opaque output, or `[0, 0, 0, 0]` when `transparent_pad` is `true`) without
/// scaling or filtering the original pixels.
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

    let pad_pixel: [u8; 4] = if transparent_pad {
        [0, 0, 0, 0]
    } else {
        [0, 0, 0, 255]
    };

    let src_row_bytes = (src_w as usize) * 4;
    let dst_row_bytes = (dst_w as usize) * 4;

    for y in 0..(src_h as usize) {
        let src_start = y * src_row_bytes;
        let dst_start = y * dst_row_bytes;
        dst[dst_start..dst_start + src_row_bytes]
            .copy_from_slice(&src[src_start..src_start + src_row_bytes]);

        for x in (src_w as usize)..(dst_w as usize) {
            let p = dst_start + x * 4;
            dst[p..p + 4].copy_from_slice(&pad_pixel);
        }
    }

    for y in (src_h as usize)..(dst_h as usize) {
        let dst_start = y * dst_row_bytes;
        for x in 0..(dst_w as usize) {
            let p = dst_start + x * 4;
            dst[p..p + 4].copy_from_slice(&pad_pixel);
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
        assert_eq!(
            even_padded_dimensions(u32::MAX, u32::MAX),
            (u32::MAX - 1, u32::MAX - 1)
        );
    }

    #[test]
    fn pad_rgba_frame_adds_1px_black_right_and_bottom_without_scaling() {
        // 3x3 solid red source frame
        let src: Vec<u8> = [255_u8, 10, 20, 255].repeat(9);
        let mut dst = Vec::new();

        pad_rgba_frame_into(&src, (3, 3), (4, 4), false, &mut dst).expect("padding succeeds");
        assert_eq!(dst.len(), 4 * 4 * 4);

        for y in 0..4 {
            for x in 0..4 {
                let offset = (y * 4 + x) * 4;
                let px = &dst[offset..offset + 4];
                if x < 3 && y < 3 {
                    assert_eq!(px, &[255, 10, 20, 255], "original pixel ({x},{y}) altered");
                } else {
                    assert_eq!(
                        px,
                        &[0, 0, 0, 255],
                        "padded pixel ({x},{y}) not opaque black"
                    );
                }
            }
        }
    }
}
