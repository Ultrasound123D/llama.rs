//! Phase 6.5 batched-prefill SIMD helpers — the f32-domain ops of the
//! attention core (score dots, softmax, value accumulation) for the M>1
//! prefill path.
//!
//! Numerics note: the decode path keeps its f64-stable scalar softmax and
//! sequential dot order (unchanged); the batched path uses f32 SIMD with
//! the standard `2^k` exp polynomial — the same accuracy class as
//! ggml's own vectorized `exp_ps`/`soft_max` f32 code. Gates: every
//! intrinsic path is runtime-feature-detected with a scalar fallback.

#[cfg(target_arch = "x86_64")]
pub use inner::*;
#[cfg(not(target_arch = "x86_64"))]
pub use scalar_only::*;

#[cfg(target_arch = "x86_64")]
mod inner {
    use crate::avx2;
    use std::arch::x86_64::*;

    fn have_avx2_fma() -> bool {
        static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *A.get_or_init(|| {
            std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        })
    }

    /// f32 dot (batched attention scores) — AVX2+FMA 8-lane accumulator,
    /// scalar fallback. The same kernel structure as `ggml_vec_dot_f32`.
    pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
        debug_assert_eq!(x.len(), y.len());
        if have_avx2_fma() {
            #[target_feature(enable = "avx2,fma")]
            unsafe fn k(x: &[f32], y: &[f32]) -> f32 {
                avx2::dot_f32(x, y)
            }
            return unsafe { k(x, y) };
        }
        let mut s = 0f64;
        for i in 0..x.len() {
            s += (x[i] as f64) * (y[i] as f64);
        }
        s as f32
    }

    /// `out[i] += w * v[i]` (softmax-weighted value accumulation) — AVX2 FMA.
    /// Phase 6.7 fix: the intrinsics MUST live inside a
    /// `#[target_feature]` fn — without it LLVM lowers every `_mm256_*`
    /// to an out-of-line shim call (~10x below vector speed; disassembly-
    /// verified root cause of the old batched-path attention cost).
    pub fn axpy_f32(w: f32, v: &[f32], out: &mut [f32]) {
        debug_assert_eq!(v.len(), out.len());
        if have_avx2_fma() {
            #[target_feature(enable = "avx2,fma")]
            unsafe fn k(w: f32, v: &[f32], out: &mut [f32]) {
                let n = v.len();
                let np = n & !15;
                let vw = _mm256_set1_ps(w);
                let mut i = 0;
                while i < np {
                    let vv = _mm256_loadu_ps(v.as_ptr().add(i) as *const f32);
                    let oo = _mm256_loadu_ps(out.as_ptr().add(i) as *const f32);
                    let r = _mm256_fmadd_ps(vw, vv, oo);
                    _mm256_storeu_ps(out.as_mut_ptr().add(i) as *mut f32, r);
                    i += 8;
                    let vv = _mm256_loadu_ps(v.as_ptr().add(i) as *const f32);
                    let oo = _mm256_loadu_ps(out.as_ptr().add(i) as *const f32);
                    let r = _mm256_fmadd_ps(vw, vv, oo);
                    _mm256_storeu_ps(out.as_mut_ptr().add(i) as *mut f32, r);
                    i += 8;
                }
                while i < n {
                    out[i] += w * v[i];
                    i += 1;
                }
            }
            return unsafe { k(w, v, out) };
        }
        for i in 0..v.len() {
            out[i] += w * v[i];
        }
    }

    /// Vectorized `exp(x)` for 8 lanes — the classic `2^k` decomposition
    /// (structure of ggml's `exp_ps`): t = x·log2(e), k = round(t),
    /// r = (t−k)·ln2, p(r) = 1+r+r²/2+…+r⁷/5040, result = p·2^k.
    /// Valid for the softmax domain (x ≤ 0 after max subtraction); k is
    /// clamped to ≥ −126 so subnormal-flushed terms stay correct.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp8(x: __m256) -> __m256 {
        const LOG2E: f32 = 1.4426950408889634; // log2(e)
        const LN2: f32 = 0.6931471805599453; // ln(2)
        let t = _mm256_mul_ps(x, _mm256_set1_ps(LOG2E));
        let k = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let r = _mm256_mul_ps(_mm256_sub_ps(t, k), _mm256_set1_ps(LN2));
        // Horner for p(r) with the binomial coefficients 1/C(n,k)
        let p = _mm256_set1_ps(1.0 / 5040.0);
        let p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0 / 720.0));
        let p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0 / 120.0));
        let p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0 / 24.0));
        let p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0 / 6.0));
        let p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0 / 2.0));
        let p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0));
        let p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0));
        // 2^k via exponent bits: (k + 127) << 23 (the ggml exp_ps
        // construction — correct for negative k; k clamped to ≥ −126)
        let ki = _mm256_cvtps_epi32(k);
        let ki = _mm256_max_epi32(ki, _mm256_set1_epi32(-126));
        let bits = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
        let two_k = _mm256_castsi256_ps(bits);
        _mm256_mul_ps(p, two_k)
    }

    /// Vectorized f32 softmax over the score slice (assumes the caller
    /// passed only the unmasked scores): max-subtract, exp, sum, scale.
    /// Numerically stable; denominators and weights are f32 (the same
    /// domain ggml's vectorized softmax computes in).
    /// Phase 6.7 fix: the vectorized body now lives inside a
    /// `#[target_feature]` fn (same class of shim-call bug as `axpy_f32`).
    pub fn softmax_inplace_f32(x: &mut [f32]) {
        if x.is_empty() {
            return;
        }
        if have_avx2_fma() {
            #[target_feature(enable = "avx2,fma")]
            unsafe fn k(x: &mut [f32]) {
                unsafe fn hmax8(x: __m256) -> f32 {
                    // horizontal MAX (not sum): 128-bit halves, then lanes
                    let lo = _mm256_castps256_ps128(x);
                    let hi = _mm256_extractf128_ps(x, 1);
                    let m = _mm_max_ps(lo, hi);
                    let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
                    let m = _mm_max_ps(m, _mm_shuffle_ps(m, m, 1));
                    _mm_cvtss_f32(m)
                }
                let n = x.len();
                let np = n & !7;
                // max
                let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
                let mut i = 0;
                while i < np {
                    let v = _mm256_loadu_ps(x.as_ptr().add(i) as *const f32);
                    m = _mm256_max_ps(m, v);
                    i += 8;
                }
                let mut max = hmax8(m);
                while i < n {
                    if x[i] > max {
                        max = x[i];
                    }
                    i += 1;
                }
                // exp(x - max) + sum
                let mut s = _mm256_setzero_ps();
                let mut i = 0;
                while i < np {
                    let v = _mm256_loadu_ps(x.as_ptr().add(i) as *const f32);
                    let e = exp8(_mm256_sub_ps(v, _mm256_set1_ps(max)));
                    _mm256_storeu_ps(x.as_mut_ptr().add(i) as *mut f32, e);
                    s = _mm256_add_ps(s, e);
                    i += 8;
                }
                let mut sum = avx2::hsum_float_8(s);
                while i < n {
                    x[i] = (x[i] - max).exp();
                    sum += x[i];
                    i += 1;
                }
                let inv = 1.0 / sum;
                let vin = _mm256_set1_ps(inv);
                let mut i = 0;
                while i < np {
                    let v = _mm256_loadu_ps(x.as_ptr().add(i) as *const f32);
                    _mm256_storeu_ps(x.as_mut_ptr().add(i) as *mut f32, _mm256_mul_ps(v, vin));
                    i += 8;
                }
                while i < n {
                    x[i] *= inv;
                    i += 1;
                }
            }
            return unsafe { k(x) };
        }
        // scalar fallback (f32 domain, stable)
        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for v in x.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }

    // ===== Phase 6.7 P0: the flash-style blocked attention core =====
    //
    // Every attention call in the engine reduces to: one query row against
    // one kv head's cache rows. The old core did this per key —
    // `scores.push(dot); softmax; axpy per key` — a per-key Vec push and
    // (on the decode path) scalar dot/axpy. The Phase 6.6 audit measured
    // that loop at 72.7 % of long-ctx decode wall and 54.3 % of Qwen
    // prefill (PERFORMANCE_REPORT.md §9.5/§9.4) and ranked its
    // vectorization the P0 fix (§9.8 item 1). This kernel streams key
    // blocks of BK with an online (running max / running sum) softmax:
    // SIMD dots, block exp/sum fused, V accumulated while the K block is
    // still cache-hot, and no score vector materialized at all.
    //
    // Keys before `lo` (the causal/SWA mask) are never visited — identical
    // math to the old -inf masking (masked terms carried exactly zero
    // weight) with none of the work.
    //
    // Numerics: f32 dots, f32 online softmax — the same accuracy class as
    // the batched path's previous f32 two-pass code and ggml's own
    // vectorized f32 attention. The decode path previously used the
    // f64-stable scalar softmax; the change is the audit's sanctioned
    // "f32 first" (§9.8).

    /// Key block size: 64 keys → K-block + V-block of 64×hd×4 bytes each
    /// (16 KB at hd=64) stay L1-resident together.
    const FLASH_BK: usize = 64;

    /// MXCSR read (stmxcsr — memory-operand instruction; `mem` asm class
    /// not yet stable, so the stack slot is addressed through a register).
    #[cfg(target_arch = "x86_64")]
    unsafe fn _mm_getcsr_asm() -> u32 {
        let mut v: u32 = 0;
        let p = &mut v as *mut u32;
        unsafe { std::arch::asm!("stmxcsr [{0}]", in(reg) p, options(nostack)) };
        v
    }

    /// MXCSR write (ldmxcsr — memory-operand instruction).
    #[cfg(target_arch = "x86_64")]
    unsafe fn _mm_setcsr_asm(v: u32) {
        let p = &v as *const u32;
        unsafe { std::arch::asm!("ldmxcsr [{0}]", in(reg) p, options(nostack)) };
    }

    /// `out` receives softmax(scale·(K·q))ᵀ·V over keys [lo, n_kv) of kv
    /// head `kv_head`, where the caches are row-major [rows][nkv*hd] and
    /// this head's row p lives at [p*nkv*hd + kv_head*hd ..][..hd].
    /// `out` is zeroed here; `q.len() == out.len() == hd`.
    pub fn attn_head_flash(
        q: &[f32],
        k_all: &[f32],
        v_all: &[f32],
        nkv: usize,
        kv_head: usize,
        lo: usize,
        n_kv: usize,
        hd: usize,
        scale: f32,
        out: &mut [f32],
    ) {
        debug_assert_eq!(q.len(), hd);
        debug_assert_eq!(out.len(), hd);
        for o in out.iter_mut() {
            *o = 0.0;
        }
        if lo >= n_kv || hd == 0 {
            return;
        }
        // FTZ/DAZ guard (measured necessity): exp(s − max) underflows to
        // subnormals for the wide score spreads of real attention; on this
        // host those operands trigger FP assists (~4x kernel slowdown,
        // flashphases probe: 155→665 c/key). They contribute exactly 0 at
        // f32 accumulation precision (|out| ≫ 1e-38), so flushing them is
        // numerically identical. MXCSR saved/restored around the kernel
        // (stmxcsr/ldmxcsr via inline asm — the intrinsics are deprecated).
        const FTZ_DAZ: u32 = 0x8000 | 0x0040;
        let mxcsr = unsafe { _mm_getcsr_asm() };
        unsafe { _mm_setcsr_asm(mxcsr | FTZ_DAZ) };
        let r = if have_avx2_fma() {
            #[target_feature(enable = "avx2,fma")]
            unsafe fn flash(
                q: &[f32], k_all: &[f32], v_all: &[f32],
                nkv: usize, kv_head: usize, lo: usize, n_kv: usize,
                hd: usize, scale: f32, out: &mut [f32],
            ) {
                let stride = nkv * hd;
                let koff = kv_head * hd;
                let mut s = [0f32; FLASH_BK];
                let mut e = [0f32; FLASH_BK];
                let mut m_run = f32::NEG_INFINITY;
                let mut l_run = 0f32;
                let mut p = lo;
                while p < n_kv {
                    let bsz = usize::min(FLASH_BK, n_kv - p);
                    // 1) block dots (AVX2+FMA, 4-accumulator unrolled)
                    for j in 0..bsz {
                        let krow = std::slice::from_raw_parts(
                            k_all.as_ptr().add((p + j) * stride + koff), hd);
                        s[j] = avx2::dot_f32(q, krow) * scale;
                    }
                    // 2) block max → running-max update
                    let mut m_b = f32::NEG_INFINITY;
                    for j in 0..bsz {
                        if s[j] > m_b { m_b = s[j]; }
                    }
                    let m_new = if m_run > m_b { m_run } else { m_b };
                    // 3) rescale `out` by exp(m_run − m_new) (0 on the
                    //    first block — out is zero there anyway)
                    let corr = (m_run - m_new).exp();
                    let vc = _mm256_set1_ps(corr);
                    let mut i = 0;
                    let np = hd & !7;
                    while i < np {
                        let o_ = _mm256_loadu_ps(out.as_ptr().add(i) as *const f32);
                        _mm256_storeu_ps(out.as_mut_ptr().add(i) as *mut f32,
                            _mm256_mul_ps(o_, vc));
                        i += 8;
                    }
                    while i < hd {
                        out[i] *= corr;
                        i += 1;
                    }
                    // 4) block exps + their sum (vectorized exp8)
                    let mut l_b = 0f32;
                    let mn = _mm256_set1_ps(m_new);
                    let mut j = 0;
                    while j + 8 <= bsz {
                        let v = _mm256_loadu_ps(s.as_ptr().add(j) as *const f32);
                        let ex = exp8(_mm256_sub_ps(v, mn));
                        _mm256_storeu_ps(e.as_mut_ptr().add(j), ex);
                        l_b += avx2::hsum_float_8(ex);
                        j += 8;
                    }
                    for jj in j..bsz {
                        let ex = (s[jj] - m_new).exp();
                        e[jj] = ex;
                        l_b += ex;
                    }
                    l_run = l_run * corr + l_b;
                    // 5) V accumulation for the block (K still hot).
                    // Register-blocked: 4 keys per pass over `out`, the
                    // accumulator held in registers between FMAs — out is
                    // loaded/stored once per 4 keys, not once per key.
                    // (The audit root-cause: the previous per-key
                    // `axpy_f32` call compiled its intrinsics as out-of-line
                    // shim calls — no target_feature on that fn — running
                    // ~10x below vector speed. This loop is inline feature
                    // code: direct instructions, no calls.)
                    let np = hd & !7;
                    let mut j = 0;
                    while j + 4 <= bsz {
                        let e0 = _mm256_set1_ps(e[j]);
                        let e1 = _mm256_set1_ps(e[j + 1]);
                        let e2 = _mm256_set1_ps(e[j + 2]);
                        let e3 = _mm256_set1_ps(e[j + 3]);
                        let v0 = v_all.as_ptr().add((p + j) * stride + koff);
                        let v1 = v_all.as_ptr().add((p + j + 1) * stride + koff);
                        let v2 = v_all.as_ptr().add((p + j + 2) * stride + koff);
                        let v3 = v_all.as_ptr().add((p + j + 3) * stride + koff);
                        let mut i = 0;
                        while i < np {
                            let acc = _mm256_loadu_ps(out.as_ptr().add(i) as *const f32);
                            let a = _mm256_fmadd_ps(e0, _mm256_loadu_ps(v0.add(i)), acc);
                            let a = _mm256_fmadd_ps(e1, _mm256_loadu_ps(v1.add(i)), a);
                            let a = _mm256_fmadd_ps(e2, _mm256_loadu_ps(v2.add(i)), a);
                            let a = _mm256_fmadd_ps(e3, _mm256_loadu_ps(v3.add(i)), a);
                            _mm256_storeu_ps(out.as_mut_ptr().add(i) as *mut f32, a);
                            i += 8;
                        }
                        while i < hd {
                            out[i] += e[j] * *v0.add(i) + e[j + 1] * *v1.add(i)
                                + e[j + 2] * *v2.add(i) + e[j + 3] * *v3.add(i);
                            i += 1;
                        }
                        j += 4;
                    }
                    // remainder keys (1–3) of the block
                    for jj in j..bsz {
                        let ej = _mm256_set1_ps(e[jj]);
                        let vj = v_all.as_ptr().add((p + jj) * stride + koff);
                        let mut i = 0;
                        while i < np {
                            let acc = _mm256_loadu_ps(out.as_ptr().add(i) as *const f32);
                            let a = _mm256_fmadd_ps(ej, _mm256_loadu_ps(vj.add(i)), acc);
                            _mm256_storeu_ps(out.as_mut_ptr().add(i) as *mut f32, a);
                            i += 8;
                        }
                        while i < hd {
                            out[i] += e[jj] * *vj.add(i);
                            i += 1;
                        }
                    }
                    m_run = m_new;
                    p += bsz;
                }
                if l_run > 0.0 {
                    let inv = 1.0 / l_run;
                    let vi = _mm256_set1_ps(inv);
                    let mut i = 0;
                    let np = hd & !7;
                    while i < np {
                        let o_ = _mm256_loadu_ps(out.as_ptr().add(i) as *const f32);
                        _mm256_storeu_ps(out.as_mut_ptr().add(i) as *mut f32,
                            _mm256_mul_ps(o_, vi));
                        i += 8;
                    }
                    while i < hd {
                        out[i] *= inv;
                        i += 1;
                    }
                }
            }
            Some(unsafe { flash(q, k_all, v_all, nkv, kv_head, lo, n_kv, hd, scale, out) })
        } else {
            attn_head_flash_scalar(q, k_all, v_all, nkv, kv_head, lo, n_kv, hd, scale, out);
            None
        };
        unsafe { _mm_setcsr_asm(mxcsr) };
        let _ = r;
    }

    /// The runtime-checked-features fallback of `attn_head_flash` — also
    /// the non-x86 implementation. Same online-softmax structure, scalar
    /// ops (f64 dot accumulation, the old decode-path accuracy class).
    pub fn attn_head_flash_scalar(
        q: &[f32],
        k_all: &[f32],
        v_all: &[f32],
        nkv: usize,
        kv_head: usize,
        lo: usize,
        n_kv: usize,
        hd: usize,
        scale: f32,
        out: &mut [f32],
    ) {
        for o in out.iter_mut() { *o = 0.0; }
        if lo >= n_kv || hd == 0 { return; }
        let stride = nkv * hd;
        let koff = kv_head * hd;
        let mut m_run = f32::NEG_INFINITY;
        let mut l_run = 0f64;
        let mut p = lo;
        while p < n_kv {
            let bsz = usize::min(FLASH_BK, n_kv - p);
            let mut s = [0f32; FLASH_BK];
            for j in 0..bsz {
                let krow = &k_all[(p + j) * stride + koff..][..hd];
                let mut dot = 0f64;
                for i in 0..hd {
                    dot += (q[i] as f64) * (krow[i] as f64);
                }
                s[j] = (dot as f32) * scale;
            }
            let mut m_b = f32::NEG_INFINITY;
            for j in 0..bsz {
                if s[j] > m_b { m_b = s[j]; }
            }
            let m_new = if m_run > m_b { m_run } else { m_b };
            let corr = (m_run - m_new).exp();
            for i in 0..hd { out[i] *= corr; }
            let mut l_b = 0f64;
            for j in 0..bsz {
                let ex = ((s[j] - m_new) as f64).exp();
                s[j] = ex as f32;
                l_b += ex;
            }
            l_run = l_run * (corr as f64) + l_b;
            for j in 0..bsz {
                let vrow = &v_all[(p + j) * stride + koff..][..hd];
                let w = s[j];
                for i in 0..hd { out[i] += w * vrow[i]; }
            }
            m_run = m_new;
            p += bsz;
        }
        if l_run > 0.0 {
            let inv = 1.0 / l_run;
            for o in out.iter_mut() { *o *= inv as f32; }
        }
    }
}

#[cfg(test)]
mod attn_flash_tests {
    use super::{attn_head_flash, attn_head_flash_scalar};

    /// The OLD two-pass core (f64-stable scalar dot + softmax, masked keys
    /// dropped) transcribed as the golden reference for the flash kernel.
    fn attn_ref(
        q: &[f32], k_all: &[f32], v_all: &[f32],
        nkv: usize, kv_head: usize, lo: usize, n_kv: usize,
        hd: usize, scale: f32,
    ) -> Vec<f32> {
        let stride = nkv * hd;
        let koff = kv_head * hd;
        let mut scores: Vec<f32> = Vec::with_capacity(n_kv - lo);
        for p in lo..n_kv {
            let krow = &k_all[p * stride + koff..][..hd];
            let mut dot = 0f64;
            for i in 0..hd {
                dot += (q[i] as f64) * (krow[i] as f64);
            }
            scores.push((dot as f32) * scale);
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f64;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += (*s as f64);
        }
        let mut out = vec![0f32; hd];
        for (p, &w) in (lo..n_kv).zip(scores.iter()) {
            let vrow = &v_all[p * stride + koff..][..hd];
            for i in 0..hd {
                out[i] += w * vrow[i];
            }
        }
        for o in out.iter_mut() { *o /= sum as f32; }
        out
    }

    fn rng_state(seed: u64) -> u64 {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s ^= s >> 33;
        s
    }
    fn rand_f32(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = (*s >> 40) as i32;
        (x as f32) / 1e6
    }
    fn make_cache(rows: usize, nkv: usize, hd: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut st = seed;
        let n = rows * nkv * hd;
        let k: Vec<f32> = (0..n).map(|_| rand_f32(&mut st)).collect();
        let v: Vec<f32> = (0..n).map(|_| rand_f32(&mut st)).collect();
        let q: Vec<f32> = (0..hd).map(|_| rand_f32(&mut st)).collect();
        (q, k, v)
    }

    fn check_case(
        rows: usize, nkv: usize, hd: usize, kv_head: usize,
        lo: usize, n_kv: usize, seed: u64,
    ) {
        let (q, k, v) = make_cache(rows, nkv, hd, seed);
        let scale = 1.0 / (hd as f32).sqrt();
        let want = attn_ref(&q, &k, &v, nkv, kv_head, lo, n_kv, hd, scale);
        for kernel in [attn_head_flash as fn(&[f32], &[f32], &[f32], usize, usize, usize, usize, usize, f32, &mut [f32]),
                       attn_head_flash_scalar as fn(&[f32], &[f32], &[f32], usize, usize, usize, usize, usize, f32, &mut [f32])] {
            let mut out = vec![0f32; hd];
            kernel(&q, &k, &v, nkv, kv_head, lo, n_kv, hd, scale, &mut out);
            // element-wise relative tolerance against the f64 reference
            for i in 0..hd {
                let d = (out[i] - want[i]).abs();
                assert!(d <= 1e-3 * (1.0 + want[i].abs()),
                    "seed {seed} hd {hd} lo {lo} n_kv {n_kv}: out[{i}]={:.7} want={:.7} d={:.2e}",
                    out[i], want[i], d);
            }
        }
    }

    #[test]
    fn flash_matches_reference_random() {
        // typical GQA shape: 2 kv heads, hd=64, 300 keys, SWA lo=40
        check_case(300, 2, 64, 1, 40, 300, 0xC0FFEE);
        // multi-block (> 64 keys → 5 blocks), full attention
        check_case(500, 4, 128, 2, 0, 500, 0xBADC0DE);
        // tiny: one key
        check_case(1, 2, 64, 0, 0, 1, 0x1111);
        // exactly one block
        check_case(64, 2, 64, 1, 0, 64, 0x2222);
        // block boundary +1
        check_case(65, 2, 64, 0, 0, 65, 0x3333);
        // SWA window smaller than a block (lo near the end)
        check_case(100, 2, 64, 1, 97, 100, 0x4444);
        // hd not a multiple of 8 (tail paths)
        check_case(130, 2, 36, 0, 0, 130, 0x5555);
        // hd 96, kv head 0 of 3
        check_case(200, 3, 96, 0, 3, 200, 0x6666);
    }

    #[test]
    fn flash_online_rescale_path() {
        // later blocks carry far larger scores → running max updates and
        // the correction/rescale path must stay exact
        let hd = 64usize;
        let nkv = 1usize;
        let rows = 200usize;
        let mut k = vec![0f32; rows * nkv * hd];
        let mut v = vec![0f32; rows * nkv * hd];
        let mut q = vec![0f32; hd];
        for i in 0..hd { q[i] = 0.5; }
        for p in 0..rows {
            for i in 0..hd {
                k[p * hd + i] = if p < 64 { 0.01 } else if p < 128 { 0.5 } else { 1.0 };
                v[p * hd + i] = (p as f32) * 0.01;
            }
        }
        let scale = 1.0 / (hd as f32).sqrt();
        let want = attn_ref(&q, &k, &v, nkv, 0, 0, rows, hd, scale);
        let mut out = vec![0f32; hd];
        attn_head_flash(&q, &k, &v, nkv, 0, 0, rows, hd, scale, &mut out);
        for i in 0..hd {
            let d = (out[i] - want[i]).abs();
            assert!(d <= 1e-4 * (1.0 + want[i].abs()),
                "out[{i}]={:.7} want={:.7}", out[i], want[i]);
        }
    }

    #[test]
    fn flash_empty_range_is_zero() {
        // lo == n_kv: nothing attended, out stays zero
        let (q, k, v) = make_cache(10, 2, 64, 7);
        let mut out = vec![1.25f32; 64];
        attn_head_flash(&q, &k, &v, 2, 0, 5, 5, 64, 0.1, &mut out);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn flash_single_key_is_that_value() {
        // softmax over one key = weight 1 → out = V[0]
        let hd = 32usize;
        let k = vec![0.5f32; hd];
        let mut v = vec![0f32; hd];
        for i in 0..hd { v[i] = (i as f32) - 16.0; }
        let q = vec![0.25f32; hd];
        let mut out = vec![0f32; hd];
        attn_head_flash(&q, &k, &v, 1, 0, 0, 1, hd, 1.0, &mut out);
        for i in 0..hd {
            assert!((out[i] - v[i]).abs() <= 1e-6 * (1.0 + v[i].abs()), "i={i} {} vs {}", out[i], v[i]);
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod scalar_only {
    pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
        let mut s = 0f64;
        for i in 0..x.len() {
            s += (x[i] as f64) * (y[i] as f64);
        }
        s as f32
    }
    pub fn axpy_f32(w: f32, v: &[f32], out: &mut [f32]) {
        for i in 0..v.len() {
            out[i] += w * v[i];
        }
    }
    pub fn attn_head_flash_scalar(
        q: &[f32], k_all: &[f32], v_all: &[f32],
        nkv: usize, kv_head: usize, lo: usize, n_kv: usize,
        hd: usize, scale: f32, out: &mut [f32],
    ) {
        for o in out.iter_mut() { *o = 0.0; }
        if lo >= n_kv || hd == 0 { return; }
        let stride = nkv * hd;
        let koff = kv_head * hd;
        let mut m_run = f32::NEG_INFINITY;
        let mut l_run = 0f64;
        let mut p = lo;
        while p < n_kv {
            let bsz = usize::min(64, n_kv - p);
            let mut s = [0f32; 64];
            for j in 0..bsz {
                let krow = &k_all[(p + j) * stride + koff..][..hd];
                let mut dot = 0f64;
                for i in 0..hd {
                    dot += (q[i] as f64) * (krow[i] as f64);
                }
                s[j] = (dot as f32) * scale;
            }
            let mut m_b = f32::NEG_INFINITY;
            for j in 0..bsz {
                if s[j] > m_b { m_b = s[j]; }
            }
            let m_new = if m_run > m_b { m_run } else { m_b };
            let corr = (m_run - m_new).exp();
            for i in 0..hd { out[i] *= corr; }
            let mut l_b = 0f64;
            for j in 0..bsz {
                let ex = ((s[j] - m_new) as f64).exp();
                s[j] = ex as f32;
                l_b += ex;
            }
            l_run = l_run * (corr as f64) + l_b;
            for j in 0..bsz {
                let vrow = &v_all[(p + j) * stride + koff..][..hd];
                let w = s[j];
                for i in 0..hd { out[i] += w * vrow[i]; }
            }
            m_run = m_new;
            p += bsz;
        }
        if l_run > 0.0 {
            let inv = 1.0 / l_run;
            for o in out.iter_mut() { *o *= inv as f32; }
        }
    }
    pub use attn_head_flash_scalar as attn_head_flash;
    pub fn softmax_inplace_f32(x: &mut [f32]) {
        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for v in x.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}
