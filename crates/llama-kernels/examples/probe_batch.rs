//! Phase 6.7 audit probe: do the batch.rs SIMD helpers really vectorize?
//! (Disassembly evidence: a `_mm256_*` intrinsic in a fn WITHOUT
//! `#[target_feature]` compiles to an out-of-line shim CALL — measured
//! 302 vs 37 cycles/key for `axpy_f32` before/after the Phase 6.7 fix.)
use llama_kernels::batch::{dot_f32, axpy_f32};
use llama_kernels::avx2;

fn tsc() -> u64 { unsafe { std::arch::x86_64::_rdtsc() } }

fn main() {
    // ---- 1) replicate have_avx2_fma's detection exactly
    let avx2 = std::arch::is_x86_feature_detected!("avx2");
    let fma = std::arch::is_x86_feature_detected!("fma");
    println!("is_x86_feature_detected: avx2={avx2} fma={fma}");

    // ---- 2) time dot_f32 vs raw avx2::dot_f32 vs scalar over 4050 keys
    let keys = 4050usize;
    let hd = 64usize;
    let stride = 2 * hd; // nkv=2, kv head 0
    let mut kcache = vec![0f32; keys * stride];
    let mut st = 999u64;
    for v in kcache.iter_mut() {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((st >> 40) as i32) as f32 / 1e6;
    }
    let q: Vec<f32> = (0..hd).map(|i| (i as f32) * 0.001 - 0.032).collect();

    // warm
    let _ = dot_f32(&q, &kcache[0..hd]);
    let t0 = tsc();
    for p in 0..keys {
        let k = &kcache[p * stride..p * stride + hd];
        std::hint::black_box(dot_f32(&q, k));
    }
    let t1 = tsc();
    println!("batch::dot_f32      × {keys} keys: {:>10} cycles ({} /key)", t1 - t0, (t1 - t0) / keys as u64);

    let t2 = tsc();
    for p in 0..keys {
        let k = &kcache[p * stride..p * stride + hd];
        std::hint::black_box(unsafe { avx2::dot_f32(&q, k) });
    }
    let t3 = tsc();
    println!("raw avx2::dot_f32  × {keys} keys: {:>10} cycles ({} /key)", t3 - t2, (t3 - t2) / keys as u64);

    let t4 = tsc();
    for p in 0..keys {
        let k = &kcache[p * stride..p * stride + hd];
        let mut d = 0f32;
        for i in 0..hd { d += q[i] * k[i]; }
        std::hint::black_box(d);
    }
    let t5 = tsc();
    println!("manual scalar     × {keys} keys: {:>10} cycles ({} /key)", t5 - t4, (t5 - t4) / keys as u64);

    // ---- 3) axpy timing
    let mut out = vec![0f32; hd];
    let w = 0.5f32;
    let t6 = tsc();
    for p in 0..keys {
        let v = &kcache[p * stride..p * stride + hd];
        axpy_f32(w, v, &mut out);
    }
    let t7 = tsc();
    println!("batch::axpy_f32    × {keys} keys: {:>10} cycles ({} /key)", t7 - t6, (t7 - t6) / keys as u64);
    std::hint::black_box(out[0]);
}
