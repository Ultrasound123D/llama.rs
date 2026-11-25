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
    pub fn axpy_f32(w: f32, v: &[f32], out: &mut [f32]) {
        debug_assert_eq!(v.len(), out.len());
        if have_avx2_fma() {
            let n = v.len();
            let np = n & !15;
            let vw = unsafe { _mm256_set1_ps(w) };
            let mut i = 0;
            unsafe {
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
            }
            while i < n {
                out[i] += w * v[i];
                i += 1;
            }
            return;
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
    pub fn softmax_inplace_f32(x: &mut [f32]) {
        if x.is_empty() {
            return;
        }
        if have_avx2_fma() {
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
            unsafe {
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
            return;
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
