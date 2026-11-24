//! Scalar vec_dot implementations — the *generic* numeric contracts from
//! `ggml/src/ggml-cpu/quants.c` (b12502). They serve as:
//! 1. the fallback path on hardware without AVX2, and
//! 2. the reference oracle for the AVX2 kernel tests (the integer parts are
//!    exact; only the final f32 accumulation order differs).
//!
//! The AVX2 kernels implement the same contracts with wider lanes.

use llama_quant::f16_to_f32;

const QK_K: usize = 256;

// ---------- simple quants × q8_0 ----------

/// `ggml_vec_dot_q4_0_q8_0_generic` (quants.c:230)
pub fn dot_q4_0_q8_0(w: &[u8], y: &[u8], n: usize) -> f32 {
    let nb = n / 32;
    let mut sumf = 0f32;
    for ib in 0..nb {
        let wx = &w[ib * 18..ib * 18 + 18];
        let wy = &y[ib * 34..ib * 34 + 34];
        let d = f16_to_f32(u16::from_le_bytes([wx[0], wx[1]])) * f16_to_f32(u16::from_le_bytes([wy[0], wy[1]]));
        let qs = &wx[2..18];
        let qy = &wy[2..34];
        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;
        for j in 0..16 {
            let v0 = (qs[j] & 0x0F) as i32 - 8;
            let v1 = (qs[j] >> 4) as i32 - 8;
            sumi0 += v0 * (qy[j] as i8) as i32;
            sumi1 += v1 * (qy[j + 16] as i8) as i32;
        }
        sumf += (sumi0 + sumi1) as f32 * d;
    }
    sumf
}

/// `ggml_vec_dot_q5_0_q8_0_generic` (quants.c:413)
pub fn dot_q5_0_q8_0(w: &[u8], y: &[u8], n: usize) -> f32 {
    let nb = n / 32;
    let mut sumf = 0f32;
    for ib in 0..nb {
        let wx = &w[ib * 22..ib * 22 + 22];
        let wy = &y[ib * 34..ib * 34 + 34];
        let d = f16_to_f32(u16::from_le_bytes([wx[0], wx[1]])) * f16_to_f32(u16::from_le_bytes([wy[0], wy[1]]));
        let qh = u32::from_le_bytes([wx[2], wx[3], wx[4], wx[5]]);
        let qs = &wx[6..22];
        let qy = &wy[2..34];
        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;
        for j in 0..16 {
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = ((qs[j] & 0x0F) as i32 | xh_0 as i32) - 16;
            let x1 = ((qs[j] >> 4) as i32 | xh_1 as i32) - 16;
            sumi0 += x0 * (qy[j] as i8) as i32;
            sumi1 += x1 * (qy[j + 16] as i8) as i32;
        }
        sumf += (sumi0 + sumi1) as f32 * d;
    }
    sumf
}

/// `ggml_vec_dot_q8_0_q8_0_generic` (quants.c:499)
pub fn dot_q8_0_q8_0(w: &[u8], y: &[u8], n: usize) -> f32 {
    let nb = n / 32;
    let mut sumf = 0f32;
    for ib in 0..nb {
        let wx = &w[ib * 34..ib * 34 + 34];
        let wy = &y[ib * 34..ib * 34 + 34];
        let d = f16_to_f32(u16::from_le_bytes([wx[0], wx[1]])) * f16_to_f32(u16::from_le_bytes([wy[0], wy[1]]));
        let mut sumi = 0i32;
        for j in 0..32 {
            sumi += (wx[2 + j] as i8) as i32 * (wy[2 + j] as i8) as i32;
        }
        sumf += sumi as f32 * d;
    }
    sumf
}

// ---------- K-quants × q8_K ----------
// Scale/min extraction shared by q4_K/q5_K (quants.c utmp dance).

#[inline]
fn scales_mins_from12(s12: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut utmp = [0u32; 4];
    utmp[0] = u32::from_le_bytes([s12[0], s12[1], s12[2], s12[3]]);
    utmp[1] = u32::from_le_bytes([s12[4], s12[5], s12[6], s12[7]]);
    utmp[2] = u32::from_le_bytes([s12[8], s12[9], s12[10], s12[11]]);
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;
    // scales = bytes of utmp[0..2], mins = bytes of utmp[2..4]
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for i in 0..4 {
        scales[i] = (utmp[0] >> (8 * i as u32)) as u8;
        scales[4 + i] = (utmp[1] >> (8 * i as u32)) as u8;
        mins[i] = (utmp[2] >> (8 * i as u32)) as u8;
        mins[4 + i] = (utmp[3] >> (8 * i as u32)) as u8;
    }
    (scales, mins)
}

/// `ggml_vec_dot_q4_K_q8_K_generic` (quants.c:744) — same math, structured
/// per group of 32 with exact int32 accumulation (as the generic does through
/// aux32 lanes; lane partitioning only affects the f32 reduction).
pub fn dot_q4_k_q8_k(w: &[u8], y: &[u8], n: usize) -> f32 {
    let nb = n / QK_K;
    let mut sumf = 0f32;
    for i in 0..nb {
        let wx = &w[i * 144..(i + 1) * 144];
        let wy = &y[i * 292..(i + 1) * 292];
        let d = f16_to_f32(u16::from_le_bytes([wx[0], wx[1]]))
            * f32::from_le_bytes([wy[0], wy[1], wy[2], wy[3]]);
        let dmin = f16_to_f32(u16::from_le_bytes([wx[2], wx[3]]))
            * f32::from_le_bytes([wy[0], wy[1], wy[2], wy[3]]);
        let (scales, mins) = scales_mins_from12(&wx[4..16]);
        let qs = &wx[16..144];
        let qy = &wy[4..260];
        let bsums = &wy[260..292];
        // min correction: sumi += Σ_j bsums[j]*mins[j/2]
        let mut sumi_min = 0i32;
        for j in 0..16 {
            let b = i16::from_le_bytes([bsums[2 * j], bsums[2 * j + 1]]) as i32;
            sumi_min += b * mins[j / 2] as i32;
        }
        // main: 8 groups of 32 values; q4 qs layout: group g uses bytes
        // [32*(g/2) .. +32): lo nibbles = first 32, hi nibbles = next 32 of g*64
        let mut acc = [0i32; 8]; // like aux32 lanes
        for j in 0..4usize {
            // two groups of 32 per j: values 64j..64j+64
            let sc_lo = scales[2 * j] as i32;
            let sc_hi = scales[2 * j + 1] as i32;
            let q4 = &qs[32 * j..32 * j + 32];
            let q8_lo = &qy[64 * j..64 * j + 32];
            let q8_hi = &qy[64 * j + 32..64 * j + 64];
            for l in 0..32 {
                let a_lo = (q4[l] & 0xF) as i32;
                let a_hi = (q4[l] >> 4) as i32;
                let lane = l % 8; // spread across 8 lanes like the generic
                acc[lane] += sc_lo * a_lo * (q8_lo[l] as i8) as i32;
                acc[lane] += sc_hi * a_hi * (q8_hi[l] as i8) as i32;
            }
        }
        let mut sumf_blk = 0f32;
        for l in 0..8 {
            sumf_blk += d * acc[l] as f32;
        }
        sumf += sumf_blk - dmin * sumi_min as f32;
    }
    sumf
}

/// `ggml_vec_dot_q5_K_q8_K_generic` (quants.c:819)
pub fn dot_q5_k_q8_k(w: &[u8], y: &[u8], n: usize) -> f32 {
    let nb = n / QK_K;
    let mut sumf = 0f32;
    for i in 0..nb {
        let wx = &w[i * 176..(i + 1) * 176];
        let wy = &y[i * 292..(i + 1) * 292];
        let d = f16_to_f32(u16::from_le_bytes([wx[0], wx[1]]))
            * f32::from_le_bytes([wy[0], wy[1], wy[2], wy[3]]);
        let dmin = f16_to_f32(u16::from_le_bytes([wx[2], wx[3]]))
            * f32::from_le_bytes([wy[0], wy[1], wy[2], wy[3]]);
        let (scales, mins) = scales_mins_from12(&wx[4..16]);
        let qh = &wx[16..48]; // 32 bytes: bit 2j+0 = group j lo value l, bit 2j+1 = group j hi value l
        let qs = &wx[48..176];
        let qy = &wy[4..260];
        let bsums = &wy[260..292];
        let mut sumi_min = 0i32;
        for j in 0..16 {
            let b = i16::from_le_bytes([bsums[2 * j], bsums[2 * j + 1]]) as i32;
            sumi_min += b * mins[j / 2] as i32;
        }
        let mut acc = [0i32; 8];
        let mut m = 1u8; // hi-bit mask advances by 2 bits per 64-value group
        for j in 0..4usize {
            let sc_lo = scales[2 * j] as i32;
            let sc_hi = scales[2 * j + 1] as i32;
            let q5 = &qs[32 * j..32 * j + 32];
            let q8_lo = &qy[64 * j..64 * j + 32];
            let q8_hi = &qy[64 * j + 32..64 * j + 64];
            for l in 0..32 {
                let a_lo = (q5[l] & 0xF) as i32 + if qh[l] & m != 0 { 16 } else { 0 };
                let a_hi = (q5[l] >> 4) as i32 + if qh[l] & (m << 1) != 0 { 16 } else { 0 };
                let lane = l % 8;
                acc[lane] += sc_lo * a_lo * (q8_lo[l] as i8) as i32;
                acc[lane] += sc_hi * a_hi * (q8_hi[l] as i8) as i32;
            }
            m <<= 2;
        }
        let mut sumf_blk = 0f32;
        for l in 0..8 {
            sumf_blk += d * acc[l] as f32;
        }
        sumf += sumf_blk - dmin * sumi_min as f32;
    }
    sumf
}

/// `ggml_vec_dot_q6_K_q8_K_generic` (quants.c:899)
pub fn dot_q6_k_q8_k(w: &[u8], y: &[u8], n: usize) -> f32 {
    let nb = n / QK_K;
    let mut sums = [0f32; 8];
    for i in 0..nb {
        let wx = &w[i * 210..(i + 1) * 210];
        let wy = &y[i * 292..(i + 1) * 292];
        let ql = &wx[0..128];
        let qh = &wx[128..192];
        let sc = &wx[192..208]; // int8 scales, signed
        let d = f16_to_f32(u16::from_le_bytes([wx[208], wx[209]]))
            * f32::from_le_bytes([wy[0], wy[1], wy[2], wy[3]]);
        let qy = &wy[4..260];
        let mut aux8 = [0i8; 256];
        // two 128-value chunks (quants.c:925-935): within each chunk,
        // values 0-31 ← ql[0..32] lo + qh bits 0-1; 32-63 ← ql[32..64] lo +
        // qh bits 2-3; 64-95 ← ql[0..32] hi + qh bits 4-5; 96-127 ←
        // ql[32..64] hi + qh bits 6-7 (qh advances 32 bytes per chunk)
        let mut off = 0usize;
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        for _chunk in 0..2 {
            for l in 0..32 {
                aux8[off + l] = ((((ql[ql_off + l] & 0xF) as i32)
                    | ((((qh[qh_off + l] >> 0) & 3) as i32) << 4)) - 32) as i8;
                aux8[off + 32 + l] = ((((ql[ql_off + 32 + l] & 0xF) as i32)
                    | ((((qh[qh_off + l] >> 2) & 3) as i32) << 4)) - 32) as i8;
                aux8[off + 64 + l] = ((((ql[ql_off + l] >> 4) as i32)
                    | ((((qh[qh_off + l] >> 4) & 3) as i32) << 4)) - 32) as i8;
                aux8[off + 96 + l] = ((((ql[ql_off + 32 + l] >> 4) as i32)
                    | ((((qh[qh_off + l] >> 6) & 3) as i32) << 4)) - 32) as i8;
            }
            off += 128;
            ql_off += 64;
            qh_off += 32;
        }
        let mut aux32 = [0i32; 8];
        for j in 0..16usize {
            let scale = sc[j] as i8 as i32;
            let base = j * 16;
            for l in 0..16 {
                aux32[l % 8] += scale * (aux8[base + l] as i32) * (qy[base + l] as i8) as i32;
            }
        }
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
    }
    let mut sumf = 0f32;
    for l in 0..8 {
        sumf += sums[l];
    }
    sumf
}

// ---------- float dots ----------

pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    let mut sumf = 0f32;
    let mut i = 0;
    while i + 8 <= n {
        let mut lane = [0f32; 8];
        for (l, v) in lane.iter_mut().enumerate() {
            // 8-way unrolled like ggml_vec_dot_f32's SIMD lanes
            *v = x[i + l] * y[i + l] + *v;
        }
        for v in lane {
            sumf += v;
        }
        i += 8;
    }
    while i < n {
        sumf += x[i] * y[i];
        i += 1;
    }
    sumf
}

pub fn dot_f16_f32(w_f16: &[u8], y: &[f32]) -> f32 {
    let n = y.len();
    debug_assert_eq!(w_f16.len(), 2 * n);
    let mut buf = [0f32; 32];
    let mut acc = [0f32; 8];
    let mut i = 0;
    while i + 32 <= n {
        for j in 0..32 {
            buf[j] = f16_to_f32(u16::from_le_bytes([w_f16[2 * (i + j)], w_f16[2 * (i + j) + 1]]));
        }
        for l in 0..8 {
            for j in 0..4 {
                acc[l] += buf[l * 4 + j] * y[i + l * 4 + j];
            }
        }
        i += 32;
    }
    while i < n {
        let w = f16_to_f32(u16::from_le_bytes([w_f16[2 * i], w_f16[2 * i + 1]]));
        acc[i % 8] += w * y[i];
        i += 1;
    }
    let mut s = 0f32;
    for v in acc {
        s += v;
    }
    s
}
