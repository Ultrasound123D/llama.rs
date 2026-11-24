//! AVX2+FMA vec_dot kernels — transcriptions of the b12502 kernels in
//! `ggml/src/ggml-cpu/arch/x86/quants.c`:
//! - q4_0×q8_0 (line 783), q5_0×q8_0 (1224), q8_0×q8_0 (1390)
//! - q4_K×q8_K (2120), q5_K×q8_K (2298), q6_K×q8_K (2508)
//! plus the float dots (`vec.cpp:11 / 380`).
//!
//! When AVX512-VNNI + AVX512-VL are available, `mul_sum_us8_pairs_float`
//! uses the same `_mm256_dpbusd_epi32` fast path the C code compiles to
//! (`__AVX512VNNI__ && __AVX512VL__` branch).
//!
//! Numerical note: these kernels keep the integer dot products of the C
//! versions exact; the f32 accumulation order follows the same lane
//! structure (8-lane f32 accumulator + horizontal sum at the end).

#![allow(clippy::too_many_arguments, unused_assignments)]

#[cfg(target_arch = "x86_64")]
pub use inner::*;

#[cfg(target_arch = "x86_64")]
mod inner {
    use llama_quant::f16_to_f32;
    use std::arch::x86_64::*;

    const QK_K: usize = 256;

    // ---------- helpers (quants.c:41-300) ----------

    #[inline]
    pub unsafe fn hsum_float_8(x: __m256) -> f32 {
        let mut res = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        res = _mm_add_ss(res, _mm_movehdup_ps(res));
        _mm_cvtss_f32(res)
    }

    /// Unpack 32 4-bit fields into 32 bytes in [0..15] (quants.c:90)
    #[inline]
    unsafe fn bytes_from_nibbles_32(qs: &[u8]) -> __m256i {
        debug_assert!(qs.len() >= 16);
        let tmp = _mm_loadu_si128(qs.as_ptr() as *const __m128i);
        let bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
        let low_mask = _mm256_set1_epi8(0xF);
        _mm256_and_si256(low_mask, bytes)
    }

    /// Spread 32 bits into 32 bytes {0x00, 0xFF} (quants.c:76)
    #[inline]
    unsafe fn bytes_from_bits_32(x: &[u8; 4]) -> __m256i {
        let x32 = u32::from_le_bytes(*x);
        let shuf_mask = _mm256_set_epi64x(
            0x0303_0303_0303_0303,
            0x0202_0202_0202_0202,
            0x0101_0101_0101_0101,
            0x0000_0000_0000_0000,
        );
        let mut bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32 as i32), shuf_mask);
        let bit_mask = _mm256_set1_epi64x(0x7fbf_dfef_f7fb_fdfe_u64 as i64);
        bytes = _mm256_or_si256(bytes, bit_mask);
        _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1))
    }

    /// multiply int8 pairs, add pairwise, return as f32 (quants.c:122)
    #[inline]
    unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
        let ax = _mm256_sign_epi8(x, x);
        let sy = _mm256_sign_epi8(y, x);
        mul_sum_us8_pairs_float(ax, sy)
    }

    #[inline]
    unsafe fn mul_sum_us8_pairs_float(ax: __m256i, sy: __m256i) -> __m256 {
        static VNNI: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let vnni = *VNNI.get_or_init(|| {
            is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl")
        });
        if vnni {
            #[target_feature(enable = "avx512vnni,avx512vl")]
            unsafe fn dpbusd(ax: __m256i, sy: __m256i) -> __m256 {
                let zero = _mm256_setzero_si256();
                let summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
                _mm256_cvtepi32_ps(summed_pairs)
            }
            return unsafe { dpbusd(ax, sy) };
        }
        let dot = _mm256_maddubs_epi16(ax, sy);
        let ones = _mm256_set1_epi16(1);
        let summed_pairs = _mm256_madd_epi16(ones, dot);
        _mm256_cvtepi32_ps(summed_pairs)
    }

    // ---------- simple quants ----------

    /// `ggml_vec_dot_q4_0_q8_0` AVX2 path + prefetch
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q4_0_q8_0(w: &[u8], y: &[u8], n: usize) -> f32 {
        let nb = n / 32;
        let mut acc = _mm256_setzero_ps();
        for ib in 0..nb {
            let wx = &w[ib * 18..(ib + 1) * 18];
            let wy = &y[ib * 34..(ib + 1) * 34];
            if ib + 3 < nb {
                _mm_prefetch(w.as_ptr().add(18 * (ib + 2)) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(18 * (ib + 3)) as *const i8, _MM_HINT_T0);
            }
            let d = f16_to_f32(u16::from_le_bytes([wx[0], wx[1]]))
                * f16_to_f32(u16::from_le_bytes([wy[0], wy[1]]));
            let d = _mm256_set1_ps(d);
            let mut qx = bytes_from_nibbles_32(&wx[2..18]);
            let off = _mm256_set1_epi8(8);
            qx = _mm256_sub_epi8(qx, off);
            let qy = _mm256_loadu_si256(wy[2..].as_ptr() as *const __m256i);
            let q = mul_sum_i8_pairs_float(qx, qy);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        hsum_float_8(acc)
    }

    /// `ggml_vec_dot_q5_0_q8_0` AVX2 path + prefetch
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q5_0_q8_0(w: &[u8], y: &[u8], n: usize) -> f32 {
        let nb = n / 32;
        let mut acc = _mm256_setzero_ps();
        for ib in 0..nb {
            let wx = &w[ib * 22..(ib + 1) * 22];
            let wy = &y[ib * 34..(ib + 1) * 34];
            if ib + 3 < nb {
                _mm_prefetch(w.as_ptr().add(22 * (ib + 2)) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(22 * (ib + 3)) as *const i8, _MM_HINT_T0);
            }
            let d = f16_to_f32(u16::from_le_bytes([wx[0], wx[1]]))
                * f16_to_f32(u16::from_le_bytes([wy[0], wy[1]]));
            let d = _mm256_set1_ps(d);
            let mut qx = bytes_from_nibbles_32(&wx[6..22]);
            let qh: [u8; 4] = [wx[2], wx[3], wx[4], wx[5]];
            let bxhi = bytes_from_bits_32(&qh);
            let bxhi = _mm256_andnot_si256(bxhi, _mm256_set1_epi8(0xF0u8 as i8));
            qx = _mm256_or_si256(qx, bxhi);
            let qy = _mm256_loadu_si256(wy[2..].as_ptr() as *const __m256i);
            let q = mul_sum_i8_pairs_float(qx, qy);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        hsum_float_8(acc)
    }

    /// `ggml_vec_dot_q8_0_q8_0` AVX2 path + prefetch
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q8_0_q8_0(w: &[u8], y: &[u8], n: usize) -> f32 {
        let nb = n / 32;
        let mut acc = _mm256_setzero_ps();
        for ib in 0..nb {
            let wx = &w[ib * 34..(ib + 1) * 34];
            let wy = &y[ib * 34..(ib + 1) * 34];
            if ib + 3 < nb {
                _mm_prefetch(w.as_ptr().add(34 * (ib + 2)) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(34 * (ib + 3)) as *const i8, _MM_HINT_T0);
            }
            let d = f16_to_f32(u16::from_le_bytes([wx[0], wx[1]]))
                * f16_to_f32(u16::from_le_bytes([wy[0], wy[1]]));
            let d = _mm256_set1_ps(d);
            let qx = _mm256_loadu_si256(wx[2..].as_ptr() as *const __m256i);
            let qy = _mm256_loadu_si256(wy[2..].as_ptr() as *const __m256i);
            let q = mul_sum_i8_pairs_float(qx, qy);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        hsum_float_8(acc)
    }

    /// Is the 256-bit VNNI dpbusd (AVX512VNNI + AVX512VL) available?
    pub fn vnni256() -> bool {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| {
            is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl")
        })
    }

    /// q8_0×q8_0 with the VNNI dot-product inlined (the `__AVX512VNNI__ &&
    /// __AVX512VL__` branch of `mul_sum_us8_pairs_float`). The sign trick
    /// (|x|, y·sign(x)) keeps dpbusd's u8×i8 operands correct.
    #[target_feature(enable = "avx2,fma,avx512vnni,avx512vl")]
    pub unsafe fn dot_q8_0_q8_0_vnni(w: &[u8], y: &[u8], n: usize) -> f32 {
        let nb = n / 32;
        let zero = _mm256_setzero_si256();
        let mut acc = _mm256_setzero_ps();
        for ib in 0..nb {
            let wx = &w[ib * 34..(ib + 1) * 34];
            let wy = &y[ib * 34..(ib + 1) * 34];
            if ib + 3 < nb {
                _mm_prefetch(w.as_ptr().add(34 * (ib + 2)) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(34 * (ib + 3)) as *const i8, _MM_HINT_T0);
            }
            let d = f16_to_f32(u16::from_le_bytes([wx[0], wx[1]]))
                * f16_to_f32(u16::from_le_bytes([wy[0], wy[1]]));
            let d = _mm256_set1_ps(d);
            let qx = _mm256_loadu_si256(wx[2..].as_ptr() as *const __m256i);
            let qy = _mm256_loadu_si256(wy[2..].as_ptr() as *const __m256i);
            // sign trick then VNNI dot product
            let ax = _mm256_sign_epi8(qx, qx);
            let sy = _mm256_sign_epi8(qy, qx);
            let summed = _mm256_dpbusd_epi32(zero, ax, sy);
            let q = _mm256_cvtepi32_ps(summed);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        hsum_float_8(acc)
    }

    /// q4_0×q8_0 with the VNNI dot-product inlined (q4 nibbles are already
    /// unsigned, so no sign trick is needed for the u8×i8 dpbusd).
    #[target_feature(enable = "avx2,fma,avx512vnni,avx512vl")]
    pub unsafe fn dot_q4_0_q8_0_vnni(w: &[u8], y: &[u8], n: usize) -> f32 {
        let nb = n / 32;
        let zero = _mm256_setzero_si256();
        let mut acc = _mm256_setzero_ps();
        for ib in 0..nb {
            let wx = &w[ib * 18..(ib + 1) * 18];
            let wy = &y[ib * 34..(ib + 1) * 34];
            if ib + 3 < nb {
                _mm_prefetch(w.as_ptr().add(18 * (ib + 2)) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(18 * (ib + 3)) as *const i8, _MM_HINT_T0);
            }
            let d = f16_to_f32(u16::from_le_bytes([wx[0], wx[1]]))
                * f16_to_f32(u16::from_le_bytes([wy[0], wy[1]]));
            let d = _mm256_set1_ps(d);
            // nibbles-8: qx = nibble - 8 as i8 — the sum trick: (q+8) unsigned
            // with the -8×Σq8 correction via the y-block sum.
            // Simpler and exact: use the sign trick on the signed q4 values.
            let qx_u = bytes_from_nibbles_32(&wx[2..18]); // 0..15
            let off = _mm256_set1_epi8(8);
            let qx = _mm256_sub_epi8(qx_u, off); // -8..7 (signed)
            let qy = _mm256_loadu_si256(wy[2..].as_ptr() as *const __m256i);
            let ax = _mm256_sign_epi8(qx, qx);
            let sy = _mm256_sign_epi8(qy, qx);
            let summed = _mm256_dpbusd_epi32(zero, ax, sy);
            let q = _mm256_cvtepi32_ps(summed);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        hsum_float_8(acc)
    }

    /// q5_0×q8_0 with the VNNI dot-product inlined (signed values → sign trick).
    #[target_feature(enable = "avx2,fma,avx512vnni,avx512vl")]
    pub unsafe fn dot_q5_0_q8_0_vnni(w: &[u8], y: &[u8], n: usize) -> f32 {
        let nb = n / 32;
        let zero = _mm256_setzero_si256();
        let mut acc = _mm256_setzero_ps();
        for ib in 0..nb {
            let wx = &w[ib * 22..(ib + 1) * 22];
            let wy = &y[ib * 34..(ib + 1) * 34];
            if ib + 3 < nb {
                _mm_prefetch(w.as_ptr().add(22 * (ib + 2)) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(22 * (ib + 3)) as *const i8, _MM_HINT_T0);
            }
            let d = f16_to_f32(u16::from_le_bytes([wx[0], wx[1]]))
                * f16_to_f32(u16::from_le_bytes([wy[0], wy[1]]));
            let d = _mm256_set1_ps(d);
            let mut qx = bytes_from_nibbles_32(&wx[6..22]);
            let qh: [u8; 4] = [wx[2], wx[3], wx[4], wx[5]];
            let bxhi = bytes_from_bits_32(&qh);
            let bxhi = _mm256_andnot_si256(bxhi, _mm256_set1_epi8(0xF0u8 as i8));
            qx = _mm256_or_si256(qx, bxhi); // signed -16..15
            let qy = _mm256_loadu_si256(wy[2..].as_ptr() as *const __m256i);
            let ax = _mm256_sign_epi8(qx, qx);
            let sy = _mm256_sign_epi8(qy, qx);
            let summed = _mm256_dpbusd_epi32(zero, ax, sy);
            let q = _mm256_cvtepi32_ps(summed);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        hsum_float_8(acc)
    }

    // ---------- K-quants ----------

    // get_scale_shuffle_k4 table (quants.c:528): 8 chunks of 32 bytes;
    // chunk i repeats the byte pair (2i, 2i+1).
    static K_SHUFFLE_K4: [u8; 256] = {
        let mut t = [0u8; 256];
        let mut i = 0;
        while i < 8 {
            let mut j = 0;
            while j < 32 {
                t[i * 32 + j] = if j % 2 == 0 { (2 * i) as u8 } else { (2 * i + 1) as u8 };
                j += 1;
            }
            i += 1;
        }
        t
    };

    // get_scale_shuffle table (quants.c:540): 8 rows of 16 bytes;
    // row i = [2i ×8, (2i+1) ×8].
    static SCALE_SHUFFLE: [u8; 128] = {
        let mut t = [0u8; 128];
        let mut i = 0;
        while i < 8 {
            let mut j = 0;
            while j < 16 {
                t[i * 16 + j] = if j < 8 { (2 * i) as u8 } else { (2 * i + 1) as u8 };
                j += 1;
            }
            i += 1;
        }
        t
    };

    #[inline]
    unsafe fn k4_scale_chunk(i: usize) -> __m256i {
        _mm256_loadu_si256(K_SHUFFLE_K4[32 * i..].as_ptr() as *const __m256i)
    }

    #[inline]
    unsafe fn scale_row(i: usize) -> __m128i {
        _mm_loadu_si128(SCALE_SHUFFLE[16 * i..].as_ptr() as *const __m128i)
    }

    /// utmp scale/min extraction shared by q4_K/q5_K (quants.c:2151-2156)
    #[inline]
    unsafe fn mins_and_scales(s12: &[u8]) -> __m256i {
        const KMASK1: u32 = 0x3f3f_3f3f;
        const KMASK2: u32 = 0x0f0f_0f0f;
        const KMASK3: u32 = 0x0303_0303;
        let u0 = u32::from_le_bytes([s12[0], s12[1], s12[2], s12[3]]);
        let u1 = u32::from_le_bytes([s12[4], s12[5], s12[6], s12[7]]);
        let u2 = u32::from_le_bytes([s12[8], s12[9], s12[10], s12[11]]);
        let u3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
        let uaux = u1 & KMASK1;
        let u1 = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
        let u2 = uaux;
        let u0 = u0 & KMASK1;
        _mm256_cvtepu8_epi16(_mm_set_epi32(u3 as i32, u2 as i32, u1 as i32, u0 as i32))
    }

    /// `ggml_vec_dot_q4_K_q8_K` AVX2 path (quants.c:2146-2202)
    /// plus software prefetch of the next weight blocks (DRAM latency hiding).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q4_k_q8_k(w: &[u8], y: &[u8], n: usize) -> f32 {
        let nb = n / QK_K;
        let m4 = _mm256_set1_epi8(0xF);
        let mut acc = _mm256_setzero_ps();
        let mut acc_m = _mm_setzero_ps();
        for i in 0..nb {
            let wx = &w[i * 144..(i + 1) * 144];
            let wy = &y[i * 292..(i + 1) * 292];
            // prefetch the next two weight blocks (w is the DRAM stream)
            if i + 2 < nb {
                _mm_prefetch(w.as_ptr().add(144 * (i + 1)) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(144 * (i + 1) + 64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(144 * (i + 2)) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(144 * (i + 2) + 64) as *const i8, _MM_HINT_T0);
            }
            let yd = f32::from_le_bytes([wy[0], wy[1], wy[2], wy[3]]);
            let d = yd * f16_to_f32(u16::from_le_bytes([wx[0], wx[1]]));
            let dmin = -yd * f16_to_f32(u16::from_le_bytes([wx[2], wx[3]]));

            let mins_and_scales = mins_and_scales(&wx[4..16]);

            let q8sums = _mm256_loadu_si256(wy[260..].as_ptr() as *const __m256i);
            let q8s = _mm_hadd_epi16(
                _mm256_castsi256_si128(q8sums),
                _mm256_extractf128_si256(q8sums, 1),
            );
            let prod = _mm_madd_epi16(_mm256_extractf128_si256(mins_and_scales, 1), q8s);
            acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            let scales = _mm256_set_m128i(sc128, sc128);

            let mut sumi = _mm256_setzero_si256();
            let mut qs_off = 16usize; // wx qs cursor
            let mut q8_off = 4usize;  // wy qs cursor
            for j in 0..4 {
                let scale_l = _mm256_shuffle_epi8(scales, k4_scale_chunk(2 * j));
                let scale_h = _mm256_shuffle_epi8(scales, k4_scale_chunk(2 * j + 1));

                let q4bits = _mm256_loadu_si256(wx[qs_off..].as_ptr() as *const __m256i);
                qs_off += 32;
                let q4l = _mm256_and_si256(q4bits, m4);
                let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                let q8l = _mm256_loadu_si256(wy[q8_off..].as_ptr() as *const __m256i);
                q8_off += 32;
                let mut p16l = _mm256_maddubs_epi16(q4l, q8l);
                p16l = _mm256_madd_epi16(scale_l, p16l);

                let q8h = _mm256_loadu_si256(wy[q8_off..].as_ptr() as *const __m256i);
                q8_off += 32;
                let mut p16h = _mm256_maddubs_epi16(q4h, q8h);
                p16h = _mm256_madd_epi16(scale_h, p16h);
                let sumj = _mm256_add_epi32(p16l, p16h);

                sumi = _mm256_add_epi32(sumi, sumj);
            }
            let vd = _mm256_set1_ps(d);
            acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
        }
        // horizontal sum of the 4-lane acc_m (the C q4_K tail)
        let mut res = acc_m;
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        res = _mm_add_ss(res, _mm_movehdup_ps(res));
        hsum_float_8(acc) + _mm_cvtss_f32(res)
    }

    /// `ggml_vec_dot_q5_K_q8_K` AVX2 path (quants.c:2327-2394) + prefetch
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q5_k_q8_k(w: &[u8], y: &[u8], n: usize) -> f32 {
        let nb = n / QK_K;
        let m4 = _mm256_set1_epi8(0xF);
        let mzero = _mm_setzero_si128();
        let mone = _mm256_set1_epi8(1);
        let mut acc = _mm256_setzero_ps();
        let mut summs = 0f32;
        for i in 0..nb {
            let wx = &w[i * 176..(i + 1) * 176];
            let wy = &y[i * 292..(i + 1) * 292];
            if i + 2 < nb {
                _mm_prefetch(w.as_ptr().add(176 * (i + 1)) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(176 * (i + 1) + 64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(176 * (i + 2)) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(176 * (i + 2) + 64) as *const i8, _MM_HINT_T0);
            }
            let yd = f32::from_le_bytes([wy[0], wy[1], wy[2], wy[3]]);
            let d = yd * f16_to_f32(u16::from_le_bytes([wx[0], wx[1]]));
            let dmin = -yd * f16_to_f32(u16::from_le_bytes([wx[2], wx[3]]));

            let mins_and_scales = mins_and_scales(&wx[4..16]);

            let q8sums = _mm256_loadu_si256(wy[260..].as_ptr() as *const __m256i);
            let q8s = _mm_hadd_epi16(
                _mm256_castsi256_si128(q8sums),
                _mm256_extractf128_si256(q8sums, 1),
            );
            let prod = _mm_madd_epi16(_mm256_extractf128_si256(mins_and_scales, 1), q8s);
            let hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            summs += dmin * _mm_extract_epi32(hsum, 0) as f32;

            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            let scales = _mm256_set_m128i(sc128, sc128);

            let hbits = _mm256_loadu_si256(wx[16..48].as_ptr() as *const __m256i);
            let mut hmask = mone;
            let mut sumi = _mm256_setzero_si256();
            let mut qs_off = 48usize;
            let mut q8_off = 4usize;
            // The C uses a runtime `bit` shift; intrinsics need const counts,
            // so the 4 groups are unrolled with literal bits (same codegen).
            macro_rules! q5k_group {
                ($j:literal, $BL:literal, $BH:literal) => {{
                    let scale_0 = _mm256_shuffle_epi8(scales, k4_scale_chunk(2 * $j));
                    let scale_1 = _mm256_shuffle_epi8(scales, k4_scale_chunk(2 * $j + 1));

                    let q5bits = _mm256_loadu_si256(wx[qs_off..].as_ptr() as *const __m256i);
                    qs_off += 32;

                    let q5l_0 = _mm256_and_si256(q5bits, m4);
                    let q5h_0 = _mm256_slli_epi16(
                        _mm256_srli_epi16(_mm256_and_si256(hbits, hmask), $BL),
                        4,
                    );
                    let q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
                    hmask = _mm256_slli_epi16(hmask, 1);

                    let q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
                    let q5h_1 = _mm256_slli_epi16(
                        _mm256_srli_epi16(_mm256_and_si256(hbits, hmask), $BH),
                        4,
                    );
                    let q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
                    hmask = _mm256_slli_epi16(hmask, 1);

                    let q8_0 = _mm256_loadu_si256(wy[q8_off..].as_ptr() as *const __m256i);
                    q8_off += 32;
                    let q8_1 = _mm256_loadu_si256(wy[q8_off..].as_ptr() as *const __m256i);
                    q8_off += 32;

                    let mut p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
                    let mut p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);
                    p16_0 = _mm256_madd_epi16(scale_0, p16_0);
                    p16_1 = _mm256_madd_epi16(scale_1, p16_1);
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                }};
            }
            q5k_group!(0, 0, 1);
            q5k_group!(1, 2, 3);
            q5k_group!(2, 4, 5);
            q5k_group!(3, 6, 7);
            let vd = _mm256_set1_ps(d);
            acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum_float_8(acc) + summs
    }

    /// `ggml_vec_dot_q6_K_q8_K` AVX2 path (quants.c:2528-2590) + prefetch
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q6_k_q8_k(w: &[u8], y: &[u8], n: usize) -> f32 {
        let nb = n / QK_K;
        let m3 = _mm256_set1_epi8(3);
        let m15 = _mm256_set1_epi8(15);
        let m12 = _mm256_set1_epi8(12);
        let m48 = _mm256_set1_epi8(48);
        let mc64 = _mm256_set1_epi8(0xC0u8 as i8);
        let mut acc = _mm256_setzero_ps();
        for i in 0..nb {
            let wx = &w[i * 210..(i + 1) * 210];
            let wy = &y[i * 292..(i + 1) * 292];
            if i + 2 < nb {
                _mm_prefetch(w.as_ptr().add(210 * (i + 1)) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(210 * (i + 1) + 64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(210 * (i + 2)) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(210 * (i + 2) + 64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w.as_ptr().add(210 * (i + 2) + 128) as *const i8, _MM_HINT_T0);
            }
            let yd = f32::from_le_bytes([wy[0], wy[1], wy[2], wy[3]]);
            let d = yd * f16_to_f32(u16::from_le_bytes([wx[208], wx[209]]));

            let q8sums = _mm256_loadu_si256(wy[260..].as_ptr() as *const __m256i);
            let scales = _mm_loadu_si128(wx[192..208].as_ptr() as *const __m128i);
            let scales_16 = _mm256_cvtepi8_epi16(scales);
            // 32 * madd(bsums, scales): the q6 -32 offset correction
            let q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

            let mut sumi = _mm256_setzero_si256();
            let mut is = 0usize;
            let mut ql_off = 0usize;   // wx ql cursor
            let mut qh_off = 128usize; // wx qh cursor
            let mut q8_off = 4usize;   // wy qs cursor
            for _j in 0..2 {
                let q4bits1 = _mm256_loadu_si256(wx[ql_off..].as_ptr() as *const __m256i);
                ql_off += 32;
                let q4bits2 = _mm256_loadu_si256(wx[ql_off..].as_ptr() as *const __m256i);
                ql_off += 32;
                let q4bits_h = _mm256_loadu_si256(wx[qh_off..].as_ptr() as *const __m256i);
                qh_off += 32;

                let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m3), 4);
                let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m12), 2);
                let q4h_2 = _mm256_and_si256(q4bits_h, m48);
                let q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bits_h, mc64), 2);

                let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
                let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
                let q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
                let q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

                let q8_0 = _mm256_loadu_si256(wy[q8_off..].as_ptr() as *const __m256i);
                q8_off += 32;
                let q8_1 = _mm256_loadu_si256(wy[q8_off..].as_ptr() as *const __m256i);
                q8_off += 32;
                let q8_2 = _mm256_loadu_si256(wy[q8_off..].as_ptr() as *const __m256i);
                q8_off += 32;
                let q8_3 = _mm256_loadu_si256(wy[q8_off..].as_ptr() as *const __m256i);
                q8_off += 32;

                let mut p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
                let mut p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
                let mut p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
                let mut p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

                let scale_0 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_row(is + 0)));
                let scale_1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_row(is + 1)));
                let scale_2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_row(is + 2)));
                let scale_3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_row(is + 3)));
                is += 4;

                p16_0 = _mm256_madd_epi16(scale_0, p16_0);
                p16_1 = _mm256_madd_epi16(scale_1, p16_1);
                p16_2 = _mm256_madd_epi16(scale_2, p16_2);
                p16_3 = _mm256_madd_epi16(scale_3, p16_3);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
            }
            sumi = _mm256_sub_epi32(sumi, q8sclsub);
            let vd = _mm256_set1_ps(d);
            acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum_float_8(acc)
    }

    // ---------- float dots ----------

    /// f32 dot with 8-lane FMA accumulator (ggml_vec_dot_f32 structure)
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
        let n = x.len();
        let np = n & !31;
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i < np {
            // 4 FMA steps per iteration (GGML_F32_ARR unroll)
            let ax0 = _mm256_loadu_ps(x.as_ptr().add(i) as *const f32);
            let ay0 = _mm256_loadu_ps(y.as_ptr().add(i) as *const f32);
            acc = _mm256_fmadd_ps(ax0, ay0, acc);
            let ax1 = _mm256_loadu_ps(x.as_ptr().add(i + 8) as *const f32);
            let ay1 = _mm256_loadu_ps(y.as_ptr().add(i + 8) as *const f32);
            acc = _mm256_fmadd_ps(ax1, ay1, acc);
            let ax2 = _mm256_loadu_ps(x.as_ptr().add(i + 16) as *const f32);
            let ay2 = _mm256_loadu_ps(y.as_ptr().add(i + 16) as *const f32);
            acc = _mm256_fmadd_ps(ax2, ay2, acc);
            let ax3 = _mm256_loadu_ps(x.as_ptr().add(i + 24) as *const f32);
            let ay3 = _mm256_loadu_ps(y.as_ptr().add(i + 24) as *const f32);
            acc = _mm256_fmadd_ps(ax3, ay3, acc);
            i += 32;
        }
        let mut sumf = hsum_float_8(acc);
        while i < n {
            sumf += x[i] * y[i];
            i += 1;
        }
        sumf
    }

    /// f16 weights × f32 activations: convert 16 halves at once with F16C
    /// (`_mm256_cvtph_ps`) then FMA — the structure of `ggml_vec_dot_f16_f32`.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f16_f32(w_f16: &[u8], y: &[f32]) -> f32 {
        let n = y.len();
        let np = n & !15;
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i < np {
            if i + 64 < np {
                _mm_prefetch(w_f16.as_ptr().add(2 * i + 64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(w_f16.as_ptr().add(2 * i + 128) as *const i8, _MM_HINT_T0);
            }
            let h = _mm_loadu_si128(w_f16.as_ptr().add(2 * i) as *const __m128i);
            let ax = _mm256_cvtph_ps(h); // 8 halves → f32
            let ay = _mm256_loadu_ps(y.as_ptr().add(i) as *const f32);
            acc = _mm256_fmadd_ps(ax, ay, acc);
            let h2 = _mm_loadu_si128(w_f16.as_ptr().add(2 * i + 16) as *const __m128i);
            let ax2 = _mm256_cvtph_ps(h2);
            let ay2 = _mm256_loadu_ps(y.as_ptr().add(i + 8) as *const f32);
            acc = _mm256_fmadd_ps(ax2, ay2, acc);
            i += 16;
        }
        let mut sumf = hsum_float_8(acc);
        while i < n {
            sumf += f16_to_f32(u16::from_le_bytes([w_f16[2 * i], w_f16[2 * i + 1]])) * y[i];
            i += 1;
        }
        sumf
    }
}
