//! AVX-512 VNNI vec_dot kernel for Q4_K×Q8_K — the dominant tensor type of
//! the K-quant models (LFM3-1.7B Q4_K_M is ~90% Q4_K).
//!
//! Structure: one 512-bit pass per pair of 64-value groups
//! (`_mm512_dpbusd_epi32` computes the u8×i8 products of 64 bytes in a
//! single instruction); the per-32-value-group scales are applied with
//! `vpermd`-built lane vectors + `mullo_epi32`, and the `-32`-free K-quant
//! min correction follows the same `bsums` math as the AVX2 kernel.
//!
//! The integer dot products are exact; only the f32 lane reduction order
//! differs from the AVX2 version (16 lanes vs 8 + separate acc_m).

#[cfg(target_arch = "x86_64")]
pub use inner::*;

#[cfg(target_arch = "x86_64")]
mod inner {
    use llama_quant::f16_to_f32;
    use std::arch::x86_64::*;

    const QK_K: usize = 256;

    /// Does this CPU have everything the 512-bit VNNI kernel needs?
    pub fn supported() -> bool {
        static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *S.get_or_init(|| {
            is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vnni")
                && is_x86_feature_detected!("fma")
        })
    }

    #[inline]
    unsafe fn hsum_float_16(x: __m512) -> f32 {
        let lo = _mm512_castps512_ps256(x);
        let hi = _mm512_extractf32x8_ps(x, 1);
        let sum256 = _mm256_add_ps(lo, hi);
        let mut res = _mm_add_ps(_mm256_extractf128_ps(sum256, 1), _mm256_castps256_ps128(sum256));
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        res = _mm_add_ss(res, _mm_movehdup_ps(res));
        _mm_cvtss_f32(res)
    }

    /// utmp scale/min extraction (quants.c:2151-2156) → (scales[8], mins[8])
    #[inline]
    fn scales_mins_12(s12: &[u8]) -> ([u8; 8], [u8; 8]) {
        let u0 = u32::from_le_bytes([s12[0], s12[1], s12[2], s12[3]]);
        let u1 = u32::from_le_bytes([s12[4], s12[5], s12[6], s12[7]]);
        let u2 = u32::from_le_bytes([s12[8], s12[9], s12[10], s12[11]]);
        const KMASK1: u32 = 0x3f3f_3f3f;
        const KMASK2: u32 = 0x0f0f_0f0f;
        const KMASK3: u32 = 0x0303_0303;
        let u3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
        let uaux = u1 & KMASK1;
        let u1 = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
        let u2 = uaux;
        let u0 = u0 & KMASK1;
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for i in 0..4 {
            sc[i] = (u0 >> (8 * i)) as u8;
            sc[4 + i] = (u1 >> (8 * i)) as u8;
            mn[i] = (u2 >> (8 * i)) as u8;
            mn[4 + i] = (u3 >> (8 * i)) as u8;
        }
        (sc, mn)
    }

    /// Q4_K weights × Q8_K activations, 512-bit VNNI.
    ///
    /// Per 256-value block:
    /// - lanes 0-31 of the 512-bit q4 operands carry group G's low nibbles
    ///   (values 64G..64G+32, scale s2G), lanes 32-63 carry group G+1's
    ///   low nibbles (values 64G+64..64G+96, scale s2G+2); the high-nibble
    ///   register analogously carries scale s2G+1 / s2G+3 halves.
    /// - the matching q8 bytes are two 32-byte pieces, combined with
    ///   `inserti64x4`.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,fma")]
    pub unsafe fn dot_q4_k_q8_k(w: &[u8], y: &[u8], n: usize) -> f32 {
        let nb = n / QK_K;
        let m4 = _mm512_set1_epi8(0xF);
        let zero = _mm512_setzero_si512();
        let mut acc = _mm512_setzero_ps();
        let mut acc_m = 0f32; // scalar min-correction accumulator (q5_K-style)

        for i in 0..nb {
            let wx = &w[i * 144..(i + 1) * 144];
            let wy = &y[i * 292..(i + 1) * 292];
            if i + 2 < nb {
                _mm_prefetch(w.as_ptr().add(144 * (i + 1)) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(144 * (i + 1) + 64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(144 * (i + 2)) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(144 * (i + 2) + 64) as *const i8, _MM_HINT_T0);
            }
            let yd = f32::from_le_bytes([wy[0], wy[1], wy[2], wy[3]]);
            let d = yd * f16_to_f32(u16::from_le_bytes([wx[0], wx[1]]));
            let dmin = -yd * f16_to_f32(u16::from_le_bytes([wx[2], wx[3]]));
            let (sc, mn) = scales_mins_12(&wx[4..16]);

            // ---- min correction: Σ_j bsums[j]*mins[j/2] (quants.c:792) ----
            let bsums = &wy[260..292];
            let mut sumi_min = 0i32;
            for j in 0..16 {
                let b = i16::from_le_bytes([bsums[2 * j], bsums[2 * j + 1]]) as i32;
                sumi_min += b * mn[j / 2] as i32;
            }
            acc_m += dmin * sumi_min as f32;

            // ---- main dot: 2 passes of 128 values ----
            // scales512 lanes = [s0..s7, s0..s7] as i32
            let s8 = _mm_loadl_epi64(sc.as_ptr() as *const __m128i);
            let s256 = _mm256_cvtepu8_epi32(s8); // 8 i32 lanes
            let s512 = _mm512_inserti64x4(_mm512_castsi256_si512(s256), s256, 1);

            let mut sumi = zero;
            // pass G ∈ {0, 2}: groups G and G+1
            for g in [0usize, 2usize] {
                // w-side: 64 bytes = qs of groups G, G+1
                let q4bits = _mm512_loadu_si512(wx.as_ptr().add(16 + 32 * g) as *const __m512i);
                let q4l = _mm512_and_si512(q4bits, m4);
                let q4h = _mm512_and_si512(_mm512_srli_epi16(q4bits, 4), m4);

                // y-side: lo pieces y[64G..+32] ++ y[64G+64..+96]
                let yl0 = _mm256_loadu_si256(wy.as_ptr().add(4 + 64 * g) as *const __m256i);
                let yl1 = _mm256_loadu_si256(wy.as_ptr().add(4 + 64 * g + 64) as *const __m256i);
                let yl = _mm512_inserti64x4(_mm512_castsi256_si512(yl0), yl1, 1);
                // hi pieces y[64G+32..+64] ++ y[64G+96..+128]
                let yh0 = _mm256_loadu_si256(wy.as_ptr().add(4 + 64 * g + 32) as *const __m256i);
                let yh1 = _mm256_loadu_si256(wy.as_ptr().add(4 + 64 * g + 96) as *const __m256i);
                let yh = _mm512_inserti64x4(_mm512_castsi256_si512(yh0), yh1, 1);

                // VNNI: u8 × i8 dot products into i32 lanes
                let dpl = _mm512_dpbusd_epi32(zero, q4l, yl); // lanes 0-7: s2G, 8-15: s2G+2
                let dph = _mm512_dpbusd_epi32(zero, q4h, yh); // lanes 0-7: s2G+1, 8-15: s2G+3

                // scale lane vectors: [sA×8 | sB×8] for the two 256-bit halves
                // (g=0: [s0|s2]/[s1|s3], g=2: [s4|s6]/[s5|s7])
                let idx_l: __m512i = if g == 2 {
                    _mm512_setr_epi32(4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6)
                } else {
                    _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2)
                };
                let idx_h: __m512i = if g == 2 {
                    _mm512_setr_epi32(5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7)
                } else {
                    _mm512_setr_epi32(1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3)
                };
                let sc_l = _mm512_permutexvar_epi32(idx_l, s512);
                let sc_h = _mm512_permutexvar_epi32(idx_h, s512);

                sumi = _mm512_add_epi32(
                    sumi,
                    _mm512_add_epi32(
                        _mm512_mullo_epi32(dpl, sc_l),
                        _mm512_mullo_epi32(dph, sc_h),
                    ),
                );
            }
            let vd = _mm512_set1_ps(d);
            acc = _mm512_fmadd_ps(vd, _mm512_cvtepi32_ps(sumi), acc);
        }
        hsum_float_16(acc) + acc_m
    }
}
