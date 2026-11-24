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
pub mod kstats;
/// AVX-512 VNNI INT8 packed backend (Phase 6.3). NOTE: despite its llama.cpp
/// lineage ("the AMX backend's M=1 decode route"), this module executes
/// AVX-512 VNNI instructions only — no AMX tile instructions. See the module
/// docs for the naming correction note.
pub mod vnni;

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
        let prof = kstats::enabled();
        let t0 = if prof { kstats::tsc() } else { 0 };
        let r = match ty {
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
        };
        if prof {
            kstats::QUANT_TICKS.fetch_add(kstats::tsc().wrapping_sub(t0), std::sync::atomic::Ordering::Relaxed);
            let (allocs, bytes) = match &r {
                QAct::Q8_0(q) => (1u64, q.len() as u64),
                QAct::Q8_K(q) => (1u64, q.len() as u64),
                QAct::F32(_) => (0, 0),
            };
            kstats::QUANT_ALLOC_COUNT.fetch_add(allocs, std::sync::atomic::Ordering::Relaxed);
            kstats::QUANT_ALLOC_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        }
        r
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

/// Caller-side band fraction of the column space (Phase 6.2 knob).
/// The caller runs quantize + enqueue before its band while the worker only
/// pays the wake latency, so a symmetric 0.5 split leaves the caller waiting
/// for the worker's tail on every matvec. Tunable via LLAMA_RS_SPLIT
/// (default 0.5).
fn caller_split() -> f64 {
    static S: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        std::env::var("LLAMA_RS_SPLIT")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|v| (0.1..=0.9).contains(v))
            .unwrap_or(0.5)
    })
}

/// Band job factory: a contiguous column band [lo, hi) of one matvec.
/// Returns the closure that computes it (run on caller or worker).
#[inline]
fn band_job<'a>(
    ptr: SyncMutPtr,
    data: &'a [u8],
    y: &'a QAct,
    ty: GgmlType,
    row_bytes: usize,
    n_in: usize,
    prof: bool,
    lo: usize,
    hi: usize,
    is_caller: bool,
) -> impl FnMut() + 'a {
    move || {
        let t0 = if prof { kstats::tsc() } else { 0 };
        let band = unsafe { std::slice::from_raw_parts_mut(ptr.get().add(lo), hi - lo) };
        for (k, o) in (lo..hi).enumerate() {
            let w = &data[o * row_bytes..(o + 1) * row_bytes];
            band[k] = vec_dot(ty, w, y, n_in);
        }
        if prof {
            let dt = kstats::tsc().wrapping_sub(t0);
            kstats::DOT_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
            if is_caller {
                kstats::CALLER_BAND_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

/// Fused two-tensor band factory: columns [lo, hi) over the concatenated
/// column space (tensor1's n_out1 columns first, then tensor2's).
#[inline]
fn fused_cols_job<'a>(
    ptr1: SyncMutPtr,
    ptr2: SyncMutPtr,
    data1: &'a [u8],
    data2: &'a [u8],
    n_out1: usize,
    y: &'a QAct,
    ty: GgmlType,
    row_bytes: usize,
    n_in: usize,
    prof: bool,
    lo: usize,
    hi: usize,
    is_caller: bool,
) -> impl FnMut() + 'a {
    move || {
        let t0 = if prof { kstats::tsc() } else { 0 };
        for c in lo..hi {
            let (ptr, data, o) = if c < n_out1 {
                (ptr1, data1, c)
            } else {
                (ptr2, data2, c - n_out1)
            };
            let w = &data[o * row_bytes..(o + 1) * row_bytes];
            unsafe { *ptr.get().add(o) = vec_dot(ty, w, y, n_in); }
        }
        if prof {
            let dt = kstats::tsc().wrapping_sub(t0);
            kstats::DOT_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
            if is_caller {
                kstats::CALLER_BAND_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
            }
        }
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
    let prof = kstats::enabled();
    let tw0 = if prof { kstats::tsc() } else { 0 };
    if prof {
        kstats::MATVECS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let y = QAct::quantize(ty, x);
    let row_bytes = ty.block_size() as usize * (n_in / ty.block_vals() as usize);

    let n_threads = n_threads.max(1);
    if n_threads <= 1 || n_out < 64 {
        // sequential path (also tiny matvecs: dispatch overhead dominates)
        for o in 0..n_out {
            let w = &data[o * row_bytes..(o + 1) * row_bytes];
            out[o] = vec_dot(ty, w, &y, n_in);
        }
    } else {
        // 2-thread fast path: caller band + worker band, split-tunable
        let split = caller_split();
        let caller_end = ((n_out as f64 * split).round() as usize).clamp(1, n_out - 1);
        let ptr = SyncMutPtr(out.as_mut_ptr());
        let p = caller_end;
        let caller_job = band_job(ptr, data, &y, ty, row_bytes, n_in, prof, 0, p, true);
        let worker_job = band_job(ptr, data, &y, ty, row_bytes, n_in, prof, p, n_out, false);
        pool::par_chunks2(caller_job, worker_job);
    }
    if prof {
        kstats::MATVEC_WALL_TICKS.fetch_add(kstats::tsc().wrapping_sub(tw0), std::sync::atomic::Ordering::Relaxed);
    }
}

/// Fused double matvec (Phase 6.2): two matrices of the SAME type sharing the
/// SAME input vector `x` (the FFN gate/up pair), one activation quantization
/// and ONE pool dispatch. The column space [0, g+u) is split between caller
/// and worker so both threads finish together.
///
/// `out1[o] = Σ x·W1[:,o]` (n_out1 columns), `out2[o] = Σ x·W2[:,o]` (n_out2).
pub fn matvec2(
    ty: GgmlType,
    data1: &[u8],
    n_out1: usize,
    out1: &mut [f32],
    data2: &[u8],
    n_out2: usize,
    out2: &mut [f32],
    n_in: usize,
    x: &[f32],
    n_threads: usize,
) {
    debug_assert_eq!(out1.len(), n_out1);
    debug_assert_eq!(out2.len(), n_out2);
    let prof = kstats::enabled();
    let tw0 = if prof { kstats::tsc() } else { 0 };
    if prof {
        kstats::MATVECS.fetch_add(2, std::sync::atomic::Ordering::Relaxed);
    }
    let y = QAct::quantize(ty, x);
    let row_bytes = ty.block_size() as usize * (n_in / ty.block_vals() as usize);
    let n_threads = n_threads.max(1);
    let total = n_out1 + n_out2;
    if n_threads <= 1 || total < 128 {
        for (o, w) in (0..n_out1).map(|o| (o, &data1[o * row_bytes..(o + 1) * row_bytes])) {
            out1[o] = vec_dot(ty, w, &y, n_in);
        }
        for (o, w) in (0..n_out2).map(|o| (o, &data2[o * row_bytes..(o + 1) * row_bytes])) {
            out2[o] = vec_dot(ty, w, &y, n_in);
        }
    } else {
        let split = caller_split();
        // split point in the fused column space; map back to (which tensor, col)
        let p = ((total as f64 * split).round() as usize).clamp(1, total - 1);
        let ptr1 = SyncMutPtr(out1.as_mut_ptr());
        let ptr2 = SyncMutPtr(out2.as_mut_ptr());
        let caller_job = fused_cols_job(ptr1, ptr2, data1, data2, n_out1, &y, ty, row_bytes, n_in, prof, 0, p, true);
        let worker_job = fused_cols_job(ptr1, ptr2, data1, data2, n_out1, &y, ty, row_bytes, n_in, prof, p, total, false);
        pool::par_chunks2(caller_job, worker_job);
    }
    if prof {
        kstats::MATVEC_WALL_TICKS.fetch_add(kstats::tsc().wrapping_sub(tw0), std::sync::atomic::Ordering::Relaxed);
    }
}
