//! Activation quantizers — the exact functions llama.cpp runs on x86 when a
//! quantized weight matrix is multiplied by an f32 activation vector
//! (`ggml_compute_forward_mul_mat` → `from_float` on `vec_dot_type`).
//!
//! Numeric contracts transcribed from b12502:
//! - Q8_0 activations: `arch/x86/quants.c:302 quantize_row_q8_0`
//!   (AVX2 path: `_mm256_round_ps(_MM_ROUND_NEAREST)` = round-half-to-even).
//!   Scalar fallback mirrors `ggml-quants.c quantize_row_q8_0_ref` (roundf).
//! - Q8_K activations: `ggml-quants.c:2784 quantize_row_q8_K_ref`
//!   (the x86 `quantize_row_q8_K` is exactly this ref function).
//!
//! Block layouts (little-endian):
//!   block_q8_0 { d: f16, qs: [i8; 32] }                     = 34 bytes
//!   block_q8_K { d: f32, qs: [i8; 256], bsums: [i16; 16] } = 292 bytes

/// f32 → f16 bits, round-to-nearest-even — the semantics of
/// `GGML_CPU_FP32_TO_FP16` (F16C `_mm_cvtss_sh(x, 0)` on x86).
pub fn f32_to_f16_bits(x: f32) -> u16 {
    #[cfg(target_arch = "x86_64")]
    {
        static F16C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let have = *F16C.get_or_init(|| std::arch::is_x86_feature_detected!("f16c"));
        if have {
            #[target_feature(enable = "f16c")]
            unsafe fn cvt(x: f32) -> u16 {
                use std::arch::x86_64::*;
                // _mm_cvtps_ph rounds to nearest even with imm=0
                let v = _mm_cvtps_ph(_mm_set_ss(x), 0);
                _mm_extract_epi16(v, 0) as u16
            }
            // SAFETY: feature detected above
            return unsafe { cvt(x) };
        }
    }
    f32_to_f16_bits_sw(x)
}

/// Software f32→f16 with round-to-nearest-even (IEEE 754 half conversion).
pub fn f32_to_f16_bits_sw(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let rest = bits & 0x7fff_ffff;
    let e = (rest >> 23) as i32;
    let m = rest & 0x007f_ffff;

    if e == 0xff {
        // inf / nan
        let m16 = if m != 0 { 0x0200 } else { 0 };
        return sign | 0x7c00 | m16;
    }
    // unbiased f16 exponent the value would have as a normal
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        // |x| >= 2^16: beyond the f16 range; the round-to-nearest-even
        // boundary (65520) lies below 2^16, so this is always infinity.
        return sign | 0x7c00;
    }
    if e16 >= 1 {
        // normal f16
        // round the low 13 bits of the f32 mantissa to nearest even
        let half = 0x1000; // half of the discarded field
        let mut m16 = m >> 13;
        let round_bit = m & half;
        let sticky = m & (half - 1);
        if round_bit != 0 && (sticky != 0 || (m16 & 1) != 0) {
            m16 += 1;
        }
        if m16 == 0x400 {
            // mantissa overflow into the exponent
            return sign | (((e16 + 1) as u16) << 10);
        }
        return sign | (((e16 as u16) << 10) | (m16 as u16));
    }
    if e16 < -10 {
        // |x| < 2^-25 → rounds to zero (exactly 2^-25 is a tie → even → 0)
        // values ≥ 2^-25 handled below
        if rest > 0x3300_0000 {
            return sign | 1; // smallest subnormal (nearest subnormal below is 0)
        }
        return sign;
    }
    // subnormal f16 (|x| in [2^-24, 2^-14))
    // m16 = value / 2^-24 = (2^23 + m) * 2^(e-126)
    let shift = (126 - e) as u32; // 14 ..= 24 for e in 102..=112
    let mprime = m | 0x0080_0000; // 2^23 + m
    let half = 1u32 << (shift - 1);
    let mut m16 = mprime >> shift;
    let round_bit = mprime & half;
    let sticky = mprime & (half - 1);
    if round_bit != 0 && (sticky != 0 || (m16 & 1) != 0) {
        m16 += 1;
    }
    if m16 == 0x400 {
        // rounded up into the smallest normal
        return sign | (1 << 10);
    }
    return sign | m16 as u16;
}

/// nearest_int — verbatim port of `ggml-quants.c:630`
/// (the magic-number trick; bit-exact with the C version).
#[inline]
pub fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0f32;
    let i = val.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// Q8_0 quantization of `x` (`x.len() % 32 == 0`) into `x.len()/32` blocks
/// written into `out`. With AVX2+FMA available this matches the x86
/// `quantize_row_q8_0` (round-half-to-even); the scalar fallback matches the
/// `_ref` quantizer (roundf). Differences are at most 1 LSB on ties.
pub fn quantize_row_q8_0(x: &[f32], out: &mut [u8]) {
    let k = x.len();
    assert!(k % 32 == 0 && out.len() >= k / 32 * 34, "q8_0 quantize sizes");
    let nb = k / 32;
    let round_even = use_round_even();
    for i in 0..nb {
        let xb = &x[i * 32..i * 32 + 32];
        let mut amax = 0f32;
        for &v in xb {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        let d = amax / 127.0;
        let d_bits = f32_to_f16_bits(d);
        out[i * 34..i * 34 + 2].copy_from_slice(&d_bits.to_le_bytes());
        let id = if amax != 0.0 { 127.0f32 / amax } else { 0.0 };
        let qs = &mut out[i * 34 + 2..i * 34 + 34];
        for (j, &v) in xb.iter().enumerate() {
            let r = v * id;
            let q: i8 = if round_even {
                r.round_ties_even() as i8
            } else {
                r.round() as i8
            };
            qs[j] = q as u8;
        }
    }
}

/// Q8_K quantization of `x` (`x.len() % 256 == 0`) into `x.len()/256` blocks
/// written into `out` — verbatim port of `quantize_row_q8_K_ref` (which is
/// exactly what the x86 build runs, see `arch/x86/quants.c:505`).
pub fn quantize_row_q8_k(x: &[f32], out: &mut [u8]) {
    let k = x.len();
    assert!(k % 256 == 0 && out.len() >= k / 256 * 292, "q8_K quantize sizes");
    let nb = k / 256;
    for i in 0..nb {
        let xb = &x[i * 256..(i + 1) * 256];
        let mut max = 0f32;
        let mut amax = 0f32;
        for &v in xb {
            let a = v.abs();
            if a > amax {
                amax = a;
                max = v;
            }
        }
        let base = i * 292;
        if amax == 0.0 {
            out[base..base + 4].copy_from_slice(&0f32.to_le_bytes());
            out[base + 4..base + 260].fill(0);
            out[base + 260..base + 292].fill(0);
            continue;
        }
        let iscale = -127.0f32 / max;
        let qs_base = base + 4;
        for (j, &v) in xb.iter().enumerate() {
            let vq = nearest_int(iscale * v);
            out[qs_base + j] = vq.min(127) as u8; // MIN(127, v) as in the C
        }
        // bsums: sums of 16 quantized values, i16
        let bsums_base = base + 260;
        for j in 0..16 {
            let mut sum = 0i32;
            for q in &out[qs_base + j * 16..qs_base + j * 16 + 16] {
                sum += *q as i8 as i32;
            }
            out[bsums_base + 2 * j..bsums_base + 2 * j + 2]
                .copy_from_slice(&(sum as i16).to_le_bytes());
        }
        let d = 1.0f32 / iscale;
        out[base..base + 4].copy_from_slice(&d.to_le_bytes());
    }
}

fn use_round_even() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static R: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *R.get_or_init(|| {
            std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
