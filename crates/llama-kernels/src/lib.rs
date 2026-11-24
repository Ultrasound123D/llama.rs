//! llama-kernels — Phase 6 CPU compute kernels.
//!
//! This is the Rust counterpart of `ggml/src/ggml-cpu` (b12502) for the
//! decode path (`ggml_compute_forward_mul_mat` with a single activation
//! vector):
//!
//! 1. the activation `x` is quantized ONCE into the weight type's
//!    `vec_dot_type` (Q8_0 for the simple quants, Q8_K for the K-quants,
//!    F32 passthrough for float weights) — exactly what `from_float` does
//!    inside llama.cpp's mul_mat;
//! 2. every output column is then an int-domain `vec_dot` between one row
//!    of the weight matrix and the quantized activation — the same numeric
//!    contract as the ggml kernels;
//! 3. work is spread over a persistent worker pool (llama.cpp's
//!    threadpool equivalent).
//!
//! Numerical fidelity note: llama.cpp runs these kernels with activations
//! quantized to int8 — its *published* logits already carry that rounding.
//! Reproducing the same int-domain math (instead of exact-f32 dequant)
//! moves llama.rs numerically *closer* to llama.cpp than the v0 scalar
//! dequant kernel was.

pub mod pool;
pub mod quant;
pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;

use llama_gguf::GgmlType;
use llama_quant::bf16_to_f32;

/// SIMD level, detected once at first use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Simd {
    Scalar,
    Avx2Fma,
}

pub fn simd_level() -> Simd {
    static L: std::sync::OnceLock<Simd> = std::sync::OnceLock::new();
    *L.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                return Simd::Avx2Fma;
            }
        }
        Simd::Scalar
    })
}

/// The quantized activation vector for one matvec — the analog of llama.cpp's
/// `wdata` row produced by `from_float`.
#[allow(non_camel_case_types)]
pub enum QAct<'x> {
    /// F32 activations (used for F32/F16/BF16 weights — matching the
    /// `vec_dot_type = F32` of the float types).
    F32(&'x [f32]),
    /// Q8_0-packed activations for Q4_0/Q5_0/Q8_0 weights.
    Q8_0(Vec<u8>),
    /// Q8_K-packed activations for Q4_K/Q5_K/Q6_K weights.
    Q8_K(Vec<u8>),
}

impl<'x> QAct<'x> {
    /// Quantize `x` for a weight matrix of type `ty`
    /// (`ggml-cpu/ggml-cpu.c` type_traits_cpu table):
    /// F32/F16/BF16 → F32, Q4_0/Q5_0/Q8_0 → Q8_0, Q4_K/Q5_K/Q6_K → Q8_K.
    pub fn quantize(ty: GgmlType, x: &'x [f32]) -> QAct<'x> {
        match ty {
            GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0 => {
                assert!(x.len() % 32 == 0, "n_in {} not a multiple of 32", x.len());
                let mut q = vec![0u8; x.len() / 32 * 34];
                quant::quantize_row_q8_0(x, &mut q);
                QAct::Q8_0(q)
            }
            GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => {
                assert!(x.len() % 256 == 0, "n_in {} not a multiple of 256", x.len());
                let mut q = vec![0u8; x.len() / 256 * 292];
                quant::quantize_row_q8_k(x, &mut q);
                QAct::Q8_K(q)
            }
            _ => QAct::F32(x),
        }
    }
}

/// One dot product: weight "row" `w` (n bytes of packed blocks) against the
/// quantized activation `y`. `n` is the number of f32 values represented.
pub fn vec_dot(ty: GgmlType, w: &[u8], y: &QAct, n: usize) -> f32 {
    let level = simd_level();
    #[cfg(target_arch = "x86_64")]
    let use_avx = level == Simd::Avx2Fma;
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx = false;

    match (ty, y) {
        (GgmlType::F32, QAct::F32(yx)) => {
            debug_assert_eq!(w.len(), 4 * n);
            if w.as_ptr() as usize % 4 != 0 {
                // unaligned fallback (GGUF keeps data 32B-aligned; just in case)
                let mut wf = vec![0f32; n];
                for i in 0..n {
                    wf[i] = f32::from_le_bytes([w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3]]);
                }
                return scalar::dot_f32(&wf, yx);
            }
            let wf: &[f32] = unsafe { std::slice::from_raw_parts(w.as_ptr() as *const f32, n) };
            #[cfg(target_arch = "x86_64")]
            if use_avx {
                #[target_feature(enable = "avx2,fma")]
                unsafe fn k(x: &[f32], y: &[f32]) -> f32 {
                    avx2::dot_f32(x, y)
                }
                return unsafe { k(wf, yx) };
            }
            scalar::dot_f32(wf, yx)
        }
        (GgmlType::F16, QAct::F32(yx)) => {
            debug_assert_eq!(w.len(), 2 * n);
            #[cfg(target_arch = "x86_64")]
            {
                static F16C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let f16c = *F16C.get_or_init(|| std::arch::is_x86_feature_detected!("f16c"));
                if use_avx && f16c {
                    #[target_feature(enable = "avx2,fma,f16c")]
                    unsafe fn k(w: &[u8], y: &[f32]) -> f32 {
                        avx2::dot_f16_f32(w, y)
                    }
                    return unsafe { k(w, yx) };
                }
            }
            scalar::dot_f16_f32(w, yx)
        }
        (GgmlType::BF16, QAct::F32(yx)) => {
            // scalar conversion + f32 dot
            let mut buf = vec![0f32; n];
            for i in 0..n {
                buf[i] = bf16_to_f32(u16::from_le_bytes([w[2 * i], w[2 * i + 1]]));
            }
            scalar::dot_f32(&buf, yx)
        }
        (GgmlType::Q4_0, QAct::Q8_0(q)) => {
            #[cfg(target_arch = "x86_64")]
            {
                if use_avx && avx2::vnni256() {
                    #[target_feature(enable = "avx2,fma,avx512vnni,avx512vl")]
                    unsafe fn k(w: &[u8], y: &[u8], n: usize) -> f32 {
                        avx2::dot_q4_0_q8_0_vnni(w, y, n)
                    }
                    return unsafe { k(w, q, n) };
                }
                if use_avx {
                    #[target_feature(enable = "avx2,fma")]
                    unsafe fn k(w: &[u8], y: &[u8], n: usize) -> f32 {
                        avx2::dot_q4_0_q8_0(w, y, n)
                    }
                    return unsafe { k(w, q, n) };
                }
            }
            scalar::dot_q4_0_q8_0(w, q, n)
        }
        (GgmlType::Q5_0, QAct::Q8_0(q)) => {
            #[cfg(target_arch = "x86_64")]
            {
                if use_avx && avx2::vnni256() {
                    #[target_feature(enable = "avx2,fma,avx512vnni,avx512vl")]
                    unsafe fn k(w: &[u8], y: &[u8], n: usize) -> f32 {
                        avx2::dot_q5_0_q8_0_vnni(w, y, n)
                    }
                    return unsafe { k(w, q, n) };
                }
                if use_avx {
                    #[target_feature(enable = "avx2,fma")]
                    unsafe fn k(w: &[u8], y: &[u8], n: usize) -> f32 {
                        avx2::dot_q5_0_q8_0(w, y, n)
                    }
                    return unsafe { k(w, q, n) };
                }
            }
            scalar::dot_q5_0_q8_0(w, q, n)
        }
        (GgmlType::Q8_0, QAct::Q8_0(q)) => {
            #[cfg(target_arch = "x86_64")]
            {
                if use_avx && avx2::vnni256() {
                    #[target_feature(enable = "avx2,fma,avx512vnni,avx512vl")]
                    unsafe fn k(w: &[u8], y: &[u8], n: usize) -> f32 {
                        avx2::dot_q8_0_q8_0_vnni(w, y, n)
                    }
                    return unsafe { k(w, q, n) };
                }
                if use_avx {
                    #[target_feature(enable = "avx2,fma")]
                    unsafe fn k(w: &[u8], y: &[u8], n: usize) -> f32 {
                        avx2::dot_q8_0_q8_0(w, y, n)
                    }
                    return unsafe { k(w, q, n) };
                }
            }
            scalar::dot_q8_0_q8_0(w, q, n)
        }
        (GgmlType::Q4_K, QAct::Q8_K(q)) => {
            // NOTE: a 512-bit VNNI variant exists (avx512::dot_q4_k_q8_k) and
            // is kept as an experiment (examples/q4cmp.rs) but measured ~9%
            // slower on this VM (AVX-512 frequency licensing), so the AVX2
            // kernel is the default.
            #[cfg(target_arch = "x86_64")]
            if use_avx {
                #[target_feature(enable = "avx2,fma")]
                unsafe fn k(w: &[u8], y: &[u8], n: usize) -> f32 {
                    avx2::dot_q4_k_q8_k(w, y, n)
                }
                return unsafe { k(w, q, n) };
            }
            scalar::dot_q4_k_q8_k(w, q, n)
        }
        (GgmlType::Q5_K, QAct::Q8_K(q)) => {
            #[cfg(target_arch = "x86_64")]
            if use_avx {
                #[target_feature(enable = "avx2,fma")]
                unsafe fn k(w: &[u8], y: &[u8], n: usize) -> f32 {
                    avx2::dot_q5_k_q8_k(w, y, n)
                }
                return unsafe { k(w, q, n) };
            }
            scalar::dot_q5_k_q8_k(w, q, n)
        }
        (GgmlType::Q6_K, QAct::Q8_K(q)) => {
            #[cfg(target_arch = "x86_64")]
            if use_avx {
                #[target_feature(enable = "avx2,fma")]
                unsafe fn k(w: &[u8], y: &[u8], n: usize) -> f32 {
                    avx2::dot_q6_k_q8_k(w, y, n)
                }
                return unsafe { k(w, q, n) };
            }
            scalar::dot_q6_k_q8_k(w, q, n)
        }
        (other_ty, other_y) => {
            // unsupported pair — fall back to exact dequant f32 math
            match other_y {
                QAct::F32(x) => llama_quant::dot_column(other_ty, w, 0, n, x),
                _ => unreachable!("activation quantized for unsupported type {other_ty:?}"),
            }
        }
    }
}

/// A raw pointer that is Send+Sync so worker closures can hold it.
/// Sound here because the `&mut` borrow of `out` outlives `par_chunks`
/// (which joins all workers) and each worker writes a disjoint band.
#[derive(Clone, Copy)]
struct SyncMutPtr(*mut f32);
unsafe impl Send for SyncMutPtr {}
unsafe impl Sync for SyncMutPtr {}
impl SyncMutPtr {
    #[inline]
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// Parallel matvec — the Phase 6 replacement for `llama_quant::matvec`.
///
/// `data` is the whole tensor `[n_in, n_out]` in ggml order (column o =
/// n_in consecutive values starting at column offset o).
/// `out[o] = Σ_i x[i] · W[i, o]`.
///
/// The activation vector is quantized once (like llama.cpp's `wdata`), then
/// each column becomes an int-domain vec_dot executed on the worker pool.
pub fn matvec(
    ty: GgmlType,
    data: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    n_threads: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), n_out);
    let y = QAct::quantize(ty, x);
    let row_bytes = ty.block_size() as usize * (n_in / ty.block_vals() as usize);

    let n_threads = n_threads.max(1);
    // keep the chunk count modest: each chunk is a contiguous column band
    let n_chunks = n_threads.min(n_out);
    let chunk = (n_out + n_chunks - 1) / n_chunks;
    let ptr = SyncMutPtr(out.as_mut_ptr());
    pool::par_chunks(n_chunks, n_threads, &|ci: usize| {
        let start = ci * chunk;
        let end = ((ci + 1) * chunk).min(n_out);
        let band = unsafe { std::slice::from_raw_parts_mut(ptr.get().add(start), end - start) };
        for (k, o) in (start..end).enumerate() {
            let w = &data[o * row_bytes..(o + 1) * row_bytes];
            band[k] = vec_dot(ty, w, &y, n_in);
        }
    });
}
