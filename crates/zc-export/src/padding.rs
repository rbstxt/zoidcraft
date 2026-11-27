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
    } else if transparent_pad {
        for y in (src_h as usize)..(dst_h as usize) {
            let dst_start = y * dst_row_bytes;
            dst[dst_start..dst_start + dst_row_bytes].fill(0);
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

/// Computes the intermediate render resolution for an export aspect-ratio fit mode:
/// - `0` (Stretch): render directly at `(dst_w, dst_h)`.
/// - `1` (Crop / Cover): preserve `seq_res` aspect ratio so the rendered image covers `(dst_w, dst_h)`.
/// - `2` (Black fill / Contain): preserve `seq_res` aspect ratio so the rendered image fits inside `(dst_w, dst_h)`.
#[must_use]
pub fn compute_aspect_fit_render_size(
    seq_res: [u32; 2],
    dst_w: u32,
    dst_h: u32,
    aspect_mode: usize,
) -> (u32, u32) {
    let dst_w = dst_w.max(1);
    let dst_h = dst_h.max(1);
    if aspect_mode == 0 {
        return (dst_w, dst_h);
    }
    let src_w = f64::from(seq_res[0].max(1));
    let src_h = f64::from(seq_res[1].max(1));
    let src_aspect = src_w / src_h;
    let dst_aspect = f64::from(dst_w) / f64::from(dst_h);
    if (src_aspect - dst_aspect).abs() < 1e-4 {
        return (dst_w, dst_h);
    }
    let max_dim = dst_w.max(dst_h).saturating_mul(4).max(16);
    if aspect_mode == 1 {
        // Crop (cover): expand the shorter axis so both axes are >= dst.
        if src_aspect > dst_aspect {
            let rw = (f64::from(dst_h) * src_aspect).round() as u32;
            (rw.clamp(dst_w, max_dim), dst_h)
        } else {
            let rh = (f64::from(dst_w) / src_aspect).round() as u32;
            (dst_w, rh.clamp(dst_h, max_dim))
        }
    } else {
        // Black fill (contain): shrink the longer axis so both axes are <= dst.
        if src_aspect > dst_aspect {
            let rh = (f64::from(dst_w) / src_aspect).round() as u32;
            (dst_w, rh.clamp(1, dst_h))
        } else {
            let rw = (f64::from(dst_h) * src_aspect).round() as u32;
            (rw.clamp(1, dst_w), dst_h)
        }
    }
}

/// Center-crops or center-pads an RGBA8 frame from `src_size` into `dst_size`.
pub fn fit_rgba_frame_into(
    src: &[u8],
    src_size: (u32, u32),
    dst_size: (u32, u32),
    transparent_fill: bool,
    dst: &mut Vec<u8>,
) {
    let (src_w, src_h) = src_size;
    let (dst_w, dst_h) = dst_size;
    let expected_src = (src_w as usize)
        .saturating_mul(src_h as usize)
        .saturating_mul(4);
    let dst_len = (dst_w as usize)
        .saturating_mul(dst_h as usize)
        .saturating_mul(4);
    if src_size == dst_size && src.len() == dst_len {
        dst.clear();
        dst.extend_from_slice(src);
        return;
    }
    dst.resize(dst_len, 0);
    if transparent_fill {
        dst.fill(0);
    } else {
        for px in dst.as_chunks_mut::<4>().0 {
            px[0] = 0;
            px[1] = 0;
            px[2] = 0;
            px[3] = 255;
        }
    }
    if src.len() < expected_src || src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return;
    }
    let copy_w = src_w.min(dst_w) as usize;
    let copy_h = src_h.min(dst_h) as usize;
    let src_x0 = (src_w.saturating_sub(dst_w) / 2) as usize;
    let src_y0 = (src_h.saturating_sub(dst_h) / 2) as usize;
    let dst_x0 = (dst_w.saturating_sub(src_w) / 2) as usize;
    let dst_y0 = (dst_h.saturating_sub(src_h) / 2) as usize;
    let row_bytes = copy_w * 4;
    let src_stride = (src_w as usize) * 4;
    let dst_stride = (dst_w as usize) * 4;
    for y in 0..copy_h {
        let s_off = (src_y0 + y) * src_stride + src_x0 * 4;
        let d_off = (dst_y0 + y) * dst_stride + dst_x0 * 4;
        dst[d_off..d_off + row_bytes].copy_from_slice(&src[s_off..s_off + row_bytes]);
    }
}

/// Downsamples an RGBA8 image to at most `max_width` pixels wide for live export preview thumbnails.
#[must_use]
pub fn downsample_rgba_thumbnail(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    max_width: u32,
) -> (u32, u32, Vec<u8>) {
    if src_w == 0 || src_h == 0 || max_width == 0 {
        return (0, 0, Vec::new());
    }
    let expected = (src_w as usize)
        .saturating_mul(src_h as usize)
        .saturating_mul(4);
    if src.len() < expected {
        return (0, 0, Vec::new());
    }
    if src_w <= max_width {
        return (src_w, src_h, src[..expected].to_vec());
    }
    let tw = max_width.max(1);
    let th = ((u64::from(src_h) * u64::from(tw)) / u64::from(src_w)).max(1) as u32;
    let mut out = vec![0u8; (tw as usize) * (th as usize) * 4];
    for y in 0..th {
        let sy =
            ((u64::from(y) * u64::from(src_h)) / u64::from(th)).min(u64::from(src_h - 1)) as usize;
        for x in 0..tw {
            let sx = ((u64::from(x) * u64::from(src_w)) / u64::from(tw)).min(u64::from(src_w - 1))
                as usize;
            let s_idx = (sy * (src_w as usize) + sx) * 4;
            let d_idx = ((y as usize) * (tw as usize) + (x as usize)) * 4;
            out[d_idx..d_idx + 4].copy_from_slice(&src[s_idx..s_idx + 4]);
        }
    }
    (tw, th, out)
}

/// Encodes straight-alpha RGBA8 pixels to PNG (`0`), JPEG (`1`), or WebP (`2`).
#[cfg(not(target_arch = "wasm32"))]
pub fn encode_single_frame_image(
    rgba: &[u8],
    width: u32,
    height: u32,
    format_index: usize,
) -> Result<Vec<u8>, ExportError> {
    use image::ImageEncoder;

    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or_else(|| ExportError::InvalidSettings("image dimensions overflow".into()))?;
    if rgba.len() != expected {
        return Err(ExportError::InvalidFrameSize {
            frame_index: 0,
            expected,
            actual: rgba.len(),
        });
    }
    let mut buf = Vec::new();
    match format_index {
        1 => {
            // JPEG: composite straight RGBA onto black RGB8.
            let mut rgb = Vec::with_capacity((width as usize) * (height as usize) * 3);
            for px in rgba.as_chunks::<4>().0 {
                let a = u16::from(px[3]);
                rgb.push(((u16::from(px[0]) * a + 127) / 255) as u8);
                rgb.push(((u16::from(px[1]) * a + 127) / 255) as u8);
                rgb.push(((u16::from(px[2]) * a + 127) / 255) as u8);
            }
            let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 92);
            encoder
                .write_image(&rgb, width, height, image::ExtendedColorType::Rgb8)
                .map_err(|e| ExportError::ImageEncode(e.to_string()))?;
        }
        2 => {
            let encoder = image::codecs::webp::WebPEncoder::new_lossless(&mut buf);
            encoder
                .write_image(rgba, width, height, image::ExtendedColorType::Rgba8)
                .map_err(|e| ExportError::ImageEncode(e.to_string()))?;
        }
        _ => {
            let encoder = image::codecs::png::PngEncoder::new(&mut buf);
            encoder
                .write_image(rgba, width, height, image::ExtendedColorType::Rgba8)
                .map_err(|e| ExportError::ImageEncode(e.to_string()))?;
        }
    }
    Ok(buf)
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

    #[test]
    fn aspect_fit_thumbnail_and_single_frame_encoders_work() {
        // 16:9 sequence exported to 1:1 square (400x400):
        assert_eq!(
            compute_aspect_fit_render_size([1920, 1080], 400, 400, 0),
            (400, 400)
        );
        // Crop (cover): width expands to ~711, height stays 400.
        let (cw, ch) = compute_aspect_fit_render_size([1920, 1080], 400, 400, 1);
        assert!(cw > 700 && ch == 400);
        // Black fill (contain): width stays 400, height shrinks to ~225.
        let (fw, fh) = compute_aspect_fit_render_size([1920, 1080], 400, 400, 2);
        assert!(fw == 400 && fh < 230 && fh > 220);

        // fit_rgba_frame_into center-pads 2x2 red square into 4x4 opaque black.
        let red_2x2 = [255_u8, 0, 0, 255].repeat(4);
        let mut fitted = Vec::new();
        fit_rgba_frame_into(&red_2x2, (2, 2), (4, 4), false, &mut fitted);
        assert_eq!(&fitted[0..4], &[0, 0, 0, 255]);
        let center_px = (4 + 1) * 4;
        assert_eq!(&fitted[center_px..center_px + 4], &[255, 0, 0, 255]);

        // Downsample thumbnail preserves aspect ratio.
        let (tw, th, thumb) = downsample_rgba_thumbnail(&fitted, 4, 4, 2);
        assert_eq!((tw, th, thumb.len()), (2, 2, 16));

        // PNG, JPEG, and WebP single-frame encoders all emit valid headers.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let png = encode_single_frame_image(&fitted, 4, 4, 0).expect("encode png");
            assert_eq!(&png[..4], &[0x89, b'P', b'N', b'G']);
            let jpg = encode_single_frame_image(&fitted, 4, 4, 1).expect("encode jpg");
            assert_eq!(&jpg[..2], &[0xFF, 0xD8]);
            let webp = encode_single_frame_image(&fitted, 4, 4, 2).expect("encode webp");
            assert_eq!(&webp[..4], b"RIFF");
        }
    }
}
