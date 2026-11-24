//! AMX INT8 backend (Phase 6.3) — the decode path of llama.cpp's AMX
//! backend, ported from `ggml/src/ggml-cpu/amx/mmq.cpp` (b12502).
//!
//! What llama.cpp actually does on this class of CPU for DECODE (M=1):
//! the "AMX backend" repacks quantized weights at LOAD time into a VNNI
//! int8 layout (`convert_B_packed_format` / `pack_B`) and the M==1 path
//! runs `tinygemm_kernel_vnni` — AVX512-VNNI 512-bit `_mm512_dpbusd_epi32`
//! over 16 columns per instruction. The AMX *tile* instructions
//! (ldtilecfg / tdpbusd, 16x64 tiles) only serve M>1 (batched prefill),
//! which this engine does not issue (prefill is token-by-token). The
//! tile/XSTATE capability is still detected and reported so the
//! capability picture is honest and ready for a future batched path.
//!
//! Packed layout per (16-column group n, 32-value block k) — TILE=576 B:
//!   [ 512 B quants: VNNI int8, 8 rows x 64 B — row j holds values 4j..4j+3
//!     of columns 0..7 (32 B) then columns 8..15 (32 B) ]
//!   [ 32  B d0: 16 f16 block scales ]
//!   [ 64  B comp: 16 x i32 = 128·Σb per column (s8s8 compensation) ]
//!
//! Numeric contract (mmq.cpp `s8s8_compensation`): dpbusd is u8(a)×s8(b);
//! a' = a + 0x80 makes the activation unsigned; Σ(a+128)·b − 128·Σb = Σ a·b
//! exactly. Block scales are applied in f32 afterwards — the same structure
//! as the AVX2 kernels (integer dot exact, f32 rounding at the scale step).
//!
//! Supported: Q8_0 (as in llama.cpp) and Q5_0 (llama.cpp does not repack
//! Q5_0; codes dequantize to int8 in [−16,15] and the same construction
//! applies verbatim).

use crate::kstats;
use crate::pool;
use crate::quant;

use llama_gguf::GgmlType;

pub const TILE_N: usize = 16; // columns per packed group
pub const TILE_K: usize = 32; // values per block (Q8_0/Q5_0 block size)
pub const TILE_BYTES: usize = TILE_N * TILE_K + TILE_N * 2 + TILE_N * 4; // 576

// ---------- capability detection ----------

/// Raw cpuid via the stable `__cpuid_count` intrinsic.
#[inline]
unsafe fn cpuid(leaf: u32, subleaf: u32) -> (u32, u32, u32, u32) {
    let r = std::arch::x86_64::__cpuid_count(leaf, subleaf);
    (r.eax, r.ebx, r.ecx, r.edx)
}

/// XCR0 via xgetbv (user-mode allowed for ecx=0)
#[inline]
unsafe fn xcr0() -> u64 {
    let (lo, hi): (u32, u32);
    std::arch::asm!(
        "xgetbv",
        lateout("eax") lo,
        lateout("edx") hi,
        in("ecx") 0,
        options(nostack, preserves_flags)
    );
    (hi as u64) << 32 | lo as u64
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AmxCaps {
    /// cpuid leaf-7 ebx bit 24: AMX_INT8 (TDPBUSD tiles)
    pub amx_int8: bool,
    /// cpuid leaf-7 ebx bit 25: AMX_BF16
    pub amx_bf16: bool,
    /// AVX512-VNNI (leaf-7 ecx bit 11) + AVX512-VL (ebx bit 31)
    pub avx512_vnni: bool,
    /// XCR0 bits 17/18 (XTILECFG/XTILEDATA) — OS permission for AMX tiles
    pub xstate_tile: bool,
}

pub fn detect() -> AmxCaps {
    static C: std::sync::OnceLock<AmxCaps> = std::sync::OnceLock::new();
    *C.get_or_init(|| unsafe {
        // leaf 1 ecx bit 27 = OSXSAVE (required for xgetbv)
        let (_, _b1, c1, _d1) = cpuid(1, 0);
        let osxsave = (c1 & (1 << 27)) != 0;
        let (_, b7, c7, _d7) = cpuid(7, 0);
        let amx_int8 = (b7 & (1 << 24)) != 0;
        let amx_bf16 = (b7 & (1 << 25)) != 0;
        let avx512_vnni = (c7 & (1 << 11)) != 0 && (b7 & (1 << 31)) != 0;
        let xstate_tile = if osxsave {
            let x = xcr0();
            (x & (0b11 << 17)) == (0b11 << 17)
        } else {
            false
        };
        AmxCaps { amx_int8, amx_bf16, avx512_vnni, xstate_tile }
    })
}

/// The repacked-VNNI decode kernels need AVX512-VNNI (+VL). AMX tiles are
/// not used by the M==1 path (they serve M>1 batching); detection and XSTATE
/// validation are exported for capability reporting and future tiles.
pub fn amx_path_available() -> bool {
    static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *A.get_or_init(|| detect().avx512_vnni)
}

/// Full AMX tile readiness (would be required by a batched M>1 path).
pub fn amx_tiles_available() -> bool {
    let c = detect();
    c.amx_int8 && c.avx512_vnni && c.xstate_tile
}

// ---------- repack (load time; scalar interleave = clarity) ----------

/// VNNI interleave of one 16-column × 32-value block.
/// Row j (64 B): [bytes 4j..4j+3 of columns 0..7] ++ [bytes 4j..4j+3 of
/// columns 8..15] — exactly the layout `pack_qs` produces via the 8x8
/// 32-bit transpose in mmq.cpp.
fn pack_group_16x32(dst: &mut [u8], col_codes: &[[u8; TILE_K]; TILE_N]) {
    for j in 0..8usize {
        for n in 0..8usize {
            dst[j * 64 + 4 * n..j * 64 + 4 * n + 4]
                .copy_from_slice(&col_codes[n][4 * j..4 * j + 4]);
        }
        for n in 8..16usize {
            dst[j * 64 + 32 + 4 * (n - 8)..j * 64 + 32 + 4 * (n - 8) + 4]
                .copy_from_slice(&col_codes[n][4 * j..4 * j + 4]);
        }
    }
}

/// Repack a Q8_0 tensor [n_in, n_out] (gguf layout) into packed VNNI.
/// Requires n_in % 32 == 0 and n_out % 16 == 0.
pub fn repack_q8_0(data: &[u8], n_in: usize, n_out: usize) -> Vec<u8> {
    assert!(n_in % TILE_K == 0, "n_in {n_in} not multiple of 32");
    assert!(n_out % TILE_N == 0, "n_out {n_out} not multiple of 16");
    let kb = n_in / TILE_K;
    let ng = n_out / TILE_N;
    let mut packed = vec![0u8; ng * kb * TILE_BYTES];
    let row_bytes = 34usize;
    let mut col_codes = [[0u8; TILE_K]; TILE_N];
    for g in 0..ng {
        for k in 0..kb {
            let dst = &mut packed[(g * kb + k) * TILE_BYTES..][..TILE_BYTES];
            for n in 0..TILE_N {
                let col = g * TILE_N + n;
                let b = &data[col * row_bytes * kb + k * row_bytes..][..row_bytes];
                col_codes[n].copy_from_slice(&b[2..34]);
            }
            pack_group_16x32(dst, &col_codes);
            for n in 0..TILE_N {
                let col = g * TILE_N + n;
                let b = &data[col * row_bytes * kb + k * row_bytes..][..row_bytes];
                dst[512 + n * 2..514 + n * 2].copy_from_slice(&b[0..2]);
            }
            for n in 0..TILE_N {
                let s: i32 = col_codes[n].iter().map(|&b| b as i8 as i32).sum();
                let c = (s as i64 * 128) as i32;
                dst[544 + n * 4..548 + n * 4].copy_from_slice(&c.to_le_bytes());
            }
        }
    }
    packed
}

/// Repack a Q5_0 tensor into the same layout: codes dequantized to
/// s8 in [−16, 15] (raw code − 16), block scale d kept as f16.
pub fn repack_q5_0(data: &[u8], n_in: usize, n_out: usize) -> Vec<u8> {
    assert!(n_in % TILE_K == 0, "n_in {n_in} not multiple of 32");
    assert!(n_out % TILE_N == 0, "n_out {n_out} not multiple of 16");
    let kb = n_in / TILE_K;
    let ng = n_out / TILE_N;
    let mut packed = vec![0u8; ng * kb * TILE_BYTES];
    let row_bytes = 22usize;
    let mut col_codes = [[0u8; TILE_K]; TILE_N];
    for g in 0..ng {
        for k in 0..kb {
            let dst = &mut packed[(g * kb + k) * TILE_BYTES..][..TILE_BYTES];
            for n in 0..TILE_N {
                let col = g * TILE_N + n;
                let b = &data[col * row_bytes * kb + k * row_bytes..][..row_bytes];
                // block_q5_0: { d:2, qh:4, qs:16 } — qh BEFORE qs
                let qh32 = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                let ql = &b[6..22];
                for i in 0..16usize {
                    let lo = (ql[i] & 0xF) as i32;
                    let hi = (((qh32 >> i) & 1) << 4) as i32;
                    col_codes[n][i] = ((lo | hi) - 16) as i8 as u8;
                    let hi2 = (ql[i] >> 4) as i32;
                    let bit2 = (((qh32 >> (i + 16)) & 1) << 4) as i32;
                    col_codes[n][i + 16] = ((hi2 | bit2) - 16) as i8 as u8;
                }
            }
            pack_group_16x32(dst, &col_codes);
            for n in 0..TILE_N {
                let col = g * TILE_N + n;
                let b = &data[col * row_bytes * kb + k * row_bytes..][..row_bytes];
                dst[512 + n * 2..514 + n * 2].copy_from_slice(&b[0..2]);
            }
            for n in 0..TILE_N {
                let s: i32 = col_codes[n].iter().map(|&b| b as i8 as i32).sum();
                let c = (s as i64 * 128) as i32;
                dst[544 + n * 4..548 + n * 4].copy_from_slice(&c.to_le_bytes());
            }
        }
    }
    packed
}

// ---------- decode kernel (M == 1) ----------

/// f16 → f32 (F16C; the exact GGML_CPU_FP16_TO_FP32)
#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn h_f16(b: [u8; 2]) -> f32 {
    use std::arch::x86_64::*;
    let v = _mm_cvtsi32_si128(u16::from_le_bytes(b) as i32);
    _mm_cvtss_f32(_mm_cvtph_ps(v))
}

/// One (16-column group) × full activation: returns 16 f32 results.
/// The inner compute of `tinygemm_kernel_vnni<q8_0, q8_0>::apply`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn group_dot_q8_0(q: &[u8], kb: usize, packed: &[u8], group: usize) -> [f32; TILE_N] {
    use std::arch::x86_64::*;
    let off = _mm512_set1_epi8(-128i8); // 0x80 as u8 (s8s8 offset)
    let mut acc = _mm512_setzero_ps();
    for k in 0..kb {
        let ab = &q[k * 34..(k + 1) * 34];
        let a_ptr = ab[2..34].as_ptr() as *const i32;
        let mut va = [_mm512_setzero_si512(); 8];
        for w in 0..8 {
            va[w] = _mm512_set1_epi32(*a_ptr.add(w));
            va[w] = _mm512_add_epi8(va[w], off);
        }
        let vd1 = _mm512_set1_ps(h_f16([ab[0], ab[1]]));
        let tile = &packed[(group * kb + k) * TILE_BYTES..][..TILE_BYTES];
        let mut vsum = _mm512_setzero_si512();
        for w in 0..8 {
            let vb = _mm512_loadu_si512(tile.as_ptr().add(w * 64) as *const __m512i);
            vsum = _mm512_dpbusd_epi32(vsum, va[w], vb);
        }
        let vcomp = _mm512_loadu_si512(tile.as_ptr().add(544) as *const __m512i);
        vsum = _mm512_sub_epi32(vsum, vcomp);
        let vd0 = _mm512_cvtph_ps(_mm256_loadu_si256(tile.as_ptr().add(512) as *const __m256i));
        let vd = _mm512_mul_ps(vd0, vd1);
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), vd, acc);
    }
    let mut r = [0f32; TILE_N];
    _mm512_storeu_ps(r.as_mut_ptr() as *mut f32, acc);
    r
}

/// Compute the group band [g_lo, g_hi) of the packed matvec into `out`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn amx_band(packed: &[u8], q: &[u8], kb: usize, g_lo: usize, g_hi: usize, out: *mut f32, is_caller: bool) {
    let t0 = if kstats::enabled() { kstats::tsc() } else { 0 };
    for g in g_lo..g_hi {
        let r = group_dot_q8_0(q, kb, packed, g);
        let dst = out.add(g * TILE_N);
        std::ptr::copy_nonoverlapping(r.as_ptr(), dst, TILE_N);
    }
    if kstats::enabled() {
        let dt = kstats::tsc().wrapping_sub(t0);
        kstats::DOT_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        if is_caller {
            kstats::CALLER_BAND_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Band-job factory for the pool (caller band + worker band).
#[allow(clippy::too_many_arguments)]
fn amx_band_job<'a>(
    packed: &'a [u8],
    q: &'a [u8],
    kb: usize,
    out: *mut f32,
    g_lo: usize,
    g_hi: usize,
    is_caller: bool,
) -> impl FnMut() + 'a {
    move || unsafe { amx_band(packed, q, kb, g_lo, g_hi, out, is_caller) }
}

/// Packed matvec entry: `packed` from repack_*, x = f32 activation (n_in),
/// out (n_out). Dispatched on the worker pool with a balanced band split.
pub fn matvec_amx(
    packed: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    n_threads: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), n_out);
    let prof = kstats::enabled();
    let tw0 = if prof { kstats::tsc() } else { 0 };
    if prof {
        kstats::MATVECS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    // activation quantized ONCE to Q8_0 — the same numeric contract as the
    // AVX2 path (quantize_row_q8_0, round-to-nearest-even)
    assert!(n_in % 32 == 0);
    let kb = n_in / 32;
    let mut q = vec![0u8; kb * 34];
    quant::quantize_row_q8_0(x, &mut q);

    let ng = n_out / TILE_N;
    let n_threads = n_threads.max(1);
    if n_threads <= 1 || ng < 2 {
        let ptr = out.as_mut_ptr();
        unsafe { amx_band(packed, &q, kb, 0, ng, ptr, true) };
    } else {
        let g_mid = (ng / 2).clamp(1, ng - 1);
        let ptr = out.as_mut_ptr();
        let caller = amx_band_job(packed, &q, kb, ptr, 0, g_mid, true);
        let worker = amx_band_job(packed, &q, kb, ptr, g_mid, ng, false);
        pool::par_chunks2(caller, worker);
    }
    if prof {
        kstats::MATVEC_WALL_TICKS.fetch_add(kstats::tsc().wrapping_sub(tw0), std::sync::atomic::Ordering::Relaxed);
    }
}

/// Band over two packed tensors (fused gate/up pair): the concatenated group
/// space [0, ng1+ng2); groups [0, ng1) → out1, the rest → out2.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn amx_band2(
    p1: &[u8], ng1: usize,
    p2: &[u8],
    q: &[u8], kb: usize,
    g_lo: usize, g_hi: usize,
    out1: *mut f32, out2: *mut f32,
    is_caller: bool,
) {
    let t0 = if kstats::enabled() { kstats::tsc() } else { 0 };
    for g in g_lo..g_hi {
        if g < ng1 {
            let r = group_dot_q8_0(q, kb, p1, g);
            std::ptr::copy_nonoverlapping(r.as_ptr(), out1.add(g * TILE_N), TILE_N);
        } else {
            let g2 = g - ng1;
            let r = group_dot_q8_0(q, kb, p2, g2);
            std::ptr::copy_nonoverlapping(r.as_ptr(), out2.add(g2 * TILE_N), TILE_N);
        }
    }
    if kstats::enabled() {
        let dt = kstats::tsc().wrapping_sub(t0);
        kstats::DOT_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        if is_caller {
            kstats::CALLER_BAND_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Fused packed pair matvec (gate/up): one Q8_0 activation, one dispatch.
#[allow(clippy::too_many_arguments)]
pub fn matvec2_amx(
    p1: &[u8], n_out1: usize,
    p2: &[u8], n_out2: usize,
    n_in: usize,
    x: &[f32],
    n_threads: usize,
    out1: &mut [f32],
    out2: &mut [f32],
) {
    debug_assert_eq!(out1.len(), n_out1);
    debug_assert_eq!(out2.len(), n_out2);
    let prof = kstats::enabled();
    let tw0 = if prof { kstats::tsc() } else { 0 };
    if prof {
        kstats::MATVECS.fetch_add(2, std::sync::atomic::Ordering::Relaxed);
    }
    assert!(n_in % 32 == 0);
    let kb = n_in / 32;
    let mut q = vec![0u8; kb * 34];
    quant::quantize_row_q8_0(x, &mut q);

    let ng1 = n_out1 / TILE_N;
    let ng2 = n_out2 / TILE_N;
    let ng = ng1 + ng2;
    let n_threads = n_threads.max(1);
    if n_threads <= 1 || ng < 2 {
        let o1 = out1.as_mut_ptr();
        let o2 = out2.as_mut_ptr();
        unsafe { amx_band2(p1, ng1, p2, &q, kb, 0, ng, o1, o2, true) };
    } else {
        let g_mid = (ng / 2).clamp(1, ng - 1);
        let o1 = out1.as_mut_ptr();
        let o2 = out2.as_mut_ptr();
        let q1 = &q;
        let q2 = &q;
        let caller = move || unsafe { amx_band2(p1, ng1, p2, q1, kb, 0, g_mid, o1, o2, true) };
        let worker = move || unsafe { amx_band2(p1, ng1, p2, q2, kb, g_mid, ng, o1, o2, false) };
        pool::par_chunks2(caller, worker);
    }
    if prof {
        kstats::MATVEC_WALL_TICKS.fetch_add(kstats::tsc().wrapping_sub(tw0), std::sync::atomic::Ordering::Relaxed);
    }
}

// unused param placeholder (ty kept for API symmetry at call sites)
#[allow(unused)]
fn _ty_hack(t: GgmlType) {}

/// A repacked weight tensor (the engine's analog of llama.cpp's AMX buffer).
#[derive(Clone)]
pub struct PackedTensor {
    pub data: Vec<u8>,
    pub n_in: usize,
    pub n_out: usize,
}

/// Repack a tensor by type; returns None when the type/shape has no packed
/// path (caller falls back to the AVX2 kernels).
pub fn repack_tensor(
    ty: GgmlType,
    data: &[u8],
    n_in: usize,
    n_out: usize,
) -> Option<PackedTensor> {
    if !amx_path_available() {
        return None;
    }
    if n_in % TILE_K != 0 || n_out % TILE_N != 0 {
        return None;
    }
    let data = match ty {
        GgmlType::Q8_0 => repack_q8_0(data, n_in, n_out),
        GgmlType::Q5_0 => repack_q5_0(data, n_in, n_out),
        _ => return None,
    };
    Some(PackedTensor { data, n_in, n_out })
}
