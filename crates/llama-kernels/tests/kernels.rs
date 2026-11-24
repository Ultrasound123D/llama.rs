//! Phase 6 kernel tests.
//!
//! Test strategy:
//! 1. AVX2 kernels vs the scalar *generic* contracts (transcriptions of
//!    `ggml-cpu/quants.c`): the integer dot products must be exact; only
//!    the final f32 reduction order differs → tiny tolerance.
//! 2. The full `matvec` vs the exact-f32 dequant reference
//!    (`llama_quant::dot_column`): agreement within int8 activation
//!    quantization error.
//! 3. Activation quantizer semantics (hand-computed cases).
//! 4. f16 encoder correctness (software path vs F16C on this machine).
//! 5. Worker pool correctness and reentrancy safety.

use llama_gguf::GgmlType;
use llama_kernels::pool::par_chunks;
use llama_kernels::quant::{f32_to_f16_bits, f32_to_f16_bits_sw, nearest_int, quantize_row_q8_0, quantize_row_q8_k};
use llama_kernels::{matvec, QAct, Simd, simd_level, vec_dot};
use llama_quant::{dequant_slice, f16_to_f32};

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn rand_bytes(state: &mut u64, n: usize) -> Vec<u8> {
    (0..n).map(|_| (xorshift(state) & 0xFF) as u8).collect()
}

fn rand_f32(state: &mut u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| (xorshift(state) % 2000) as f32 / 1000.0 - 1.0)
        .collect()
}

/// Relative comparison with an absolute floor for near-zero results.
fn rel_close(a: f32, b: f32, tol: f32) {
    if a == b {
        return; // exact match (also handles ±inf)
    }
    if a.is_nan() && b.is_nan() {
        return; // both NaN = same garbage in
    }
    let diff = (a - b).abs();
    let denom = b.abs().max(1.0);
    assert!(
        diff / denom < tol,
        "mismatch: a={a:.6} b={b:.6} (rel {:.2e} >= tol {tol})",
        diff / denom
    );
}

/// Patch the block scale bytes of random weight data to safe small values
/// (random 16-bit patterns would often be NaN/Inf f16 scales).
fn sanitize_scales(ty: GgmlType, data: &mut [u8]) {
    let block = ty.block_size() as usize;
    let vpb = ty.block_vals() as usize;
    match ty {
        GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0 => {
            for b in data.chunks_exact_mut(block) {
                b[0] = 0x38; // f16(0.5) = 0x3800
                b[1] = 0x00;
            }
        }
        GgmlType::Q4_K | GgmlType::Q5_K => {
            for b in data.chunks_exact_mut(block) {
                b[0] = 0x38; // d = 0.5
                b[1] = 0x00;
                b[2] = 0x34; // dmin = 0.25
                b[3] = 0x00;
            }
        }
        GgmlType::Q6_K => {
            for b in data.chunks_exact_mut(block) {
                b[208] = 0x38; // d = 0.5
                b[209] = 0x00;
            }
        }
        _ => {
            let _ = vpb;
        }
    }
}

// ---------- 1) AVX2 vs scalar generic contracts ----------

#[cfg(target_arch = "x86_64")]
fn assert_avx2() {
    assert_eq!(simd_level(), Simd::Avx2Fma, "this machine has AVX2+FMA");
}

#[test]
#[cfg(target_arch = "x86_64")]
fn dots_avx2_match_generic_contracts() {
    assert_avx2();
    let mut st = 0x1234_5678_9abc_def0u64;
    let n = 2048usize; // 64 small blocks / 8 K blocks

    for ty in [GgmlType::Q4_0, GgmlType::Q5_0, GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K] {
        let x = rand_f32(&mut st, n);
        let y = QAct::quantize(ty, &x);
        let mut w = rand_bytes(&mut st, 4096 * 4); // plenty
        let block = ty.block_size() as usize;
        let row_bytes = block * (n / ty.block_vals() as usize);
        sanitize_scales(ty, &mut w);
        let w = &w[..row_bytes];

        let (scalar_res, avx2_res) = match (&y, ty) {
            (QAct::Q8_0(q), GgmlType::Q4_0) => (
                llama_kernels::scalar::dot_q4_0_q8_0(w, q, n),
                unsafe { llama_kernels::avx2::dot_q4_0_q8_0(w, q, n) },
            ),
            (QAct::Q8_0(q), GgmlType::Q5_0) => (
                llama_kernels::scalar::dot_q5_0_q8_0(w, q, n),
                unsafe { llama_kernels::avx2::dot_q5_0_q8_0(w, q, n) },
            ),
            (QAct::Q8_0(q), GgmlType::Q8_0) => (
                llama_kernels::scalar::dot_q8_0_q8_0(w, q, n),
                unsafe { llama_kernels::avx2::dot_q8_0_q8_0(w, q, n) },
            ),
            (QAct::Q8_K(q), GgmlType::Q4_K) => (
                llama_kernels::scalar::dot_q4_k_q8_k(w, q, n),
                unsafe { llama_kernels::avx2::dot_q4_k_q8_k(w, q, n) },
            ),
            (QAct::Q8_K(q), GgmlType::Q5_K) => (
                llama_kernels::scalar::dot_q5_k_q8_k(w, q, n),
                unsafe { llama_kernels::avx2::dot_q5_k_q8_k(w, q, n) },
            ),
            (QAct::Q8_K(q), GgmlType::Q6_K) => (
                llama_kernels::scalar::dot_q6_k_q8_k(w, q, n),
                unsafe { llama_kernels::avx2::dot_q6_k_q8_k(w, q, n) },
            ),
            _ => panic!("unexpected pairing"),
        };
        // int parts are exact; only the f32 sum order differs
        rel_close(avx2_res, scalar_res, 1e-5);
        // and the dispatcher agrees with the AVX2 kernel
        rel_close(vec_dot(ty, w, &y, n), scalar_res, 1e-5);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn float_dots_avx2_match_scalar() {
    assert_avx2();
    let mut st = 0xdead_beef_cafe_f00du64;
    let n = 1024usize;
    let a = rand_f32(&mut st, n);
    let b = rand_f32(&mut st, n);
    let s = llama_kernels::scalar::dot_f32(&a, &b);
    let v = unsafe { llama_kernels::avx2::dot_f32(&a, &b) };
    rel_close(v, s, 2e-4);

    // f16 weights × f32 activations
    let wf16: Vec<u8> = (0..n).map(|i| f32_to_f16_bits_sw(a[i]).to_le_bytes()).flatten().collect();
    let s16 = llama_kernels::scalar::dot_f16_f32(&wf16, &b);
    let v16 = unsafe { llama_kernels::avx2::dot_f16_f32(&wf16, &b) };
    rel_close(v16, s16, 2e-4);
}

// ---------- 2) matvec vs exact-f32 dequant reference ----------

#[test]
fn matvec_matches_dequant_reference() {
    let mut st = 0xfeed_face_1234_5678u64;
    let n_in = 512usize;
    let n_out = 24usize;
    // Exact-representable activations: x_i = k_i/127 with integer k_i and
    // |k| ≤ 127. Both q8_0 (id = 127/amax with amax = 1) and q8_K
    // (iscale = -127/max with max = -1) then quantize EXACTLY, so the
    // int-domain matvec must equal the exact-f32 dequant reference up to
    // f16-scale rounding and f32 lane ordering.
    let mut k: Vec<i32> = (0..n_in).map(|_| (xorshift(&mut st) % 255) as i32 - 127).collect();
    for v in k.iter_mut() {
        if *v == 127 {
            *v = 126; // keep the max-abs unique and negative
        }
    }
    k[0] = -127; // amax = 1.0 and max = -1.0 (q8_K needs a negative max)
    let x: Vec<f32> = k.iter().map(|&v| v as f32 / 127.0).collect();

    for ty in [GgmlType::F32, GgmlType::F16, GgmlType::Q4_0, GgmlType::Q5_0, GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K] {
        let row_bytes = ty.block_size() as usize * (n_in / ty.block_vals() as usize);
        // float types: build from small finite values (random f32 bit patterns
        // are usually Inf/NaN)
        let data_owned: Vec<u8> = match ty {
            GgmlType::F32 => rand_f32(&mut st, n_in * n_out)
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            GgmlType::F16 => rand_f32(&mut st, n_in * n_out)
                .iter()
                .map(|v| f32_to_f16_bits_sw(*v).to_le_bytes())
                .flatten()
                .collect(),
            _ => {
                let mut d = rand_bytes(&mut st, row_bytes * n_out);
                sanitize_scales(ty, &mut d);
                d
            }
        };
        let data = &data_owned[..];
        let mut out = vec![0f32; n_out];
        matvec(ty, data, n_in, n_out, &x, 2, &mut out);

        for o in 0..n_out {
            let w = &data[o * row_bytes..(o + 1) * row_bytes];
            let reference = match ty {
                GgmlType::F32 => {
                    let wf: Vec<f32> = (0..n_in)
                        .map(|i| f32::from_le_bytes([w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3]]))
                        .collect();
                    llama_kernels::scalar::dot_f32(&wf, &x)
                }
                GgmlType::F16 => {
                    let wf = dequant_slice(GgmlType::F16, w, n_in);
                    llama_kernels::scalar::dot_f32(&wf, &x)
                }
                _ => {
                    // exact dequant dot (what llama-quant::dot_column does)
                    llama_quant::dot_column(ty, data, o, n_in, &x)
                }
            };
            // activations quantized exactly → only f16-scale rounding +
            // f32 lane-order differences remain
            rel_close(out[o], reference, 1e-3);
        }
    }
}

// ---------- 3) quantizer semantics ----------

#[test]
fn q8_0_quantizer_hand_case() {
    // 32 values in [-1, 1]: amax = 1.0, d = 1/127, id = 127
    let x: Vec<f32> = (0..32).map(|i| ((i as i32 % 7) - 3) as f32 / 3.0).collect();
    let mut out = vec![0u8; 34];
    quantize_row_q8_0(&x, &mut out);
    // d = 1/127 → f16 bits of 0.0078740157... = 0x3C00? compute:
    let d = 1.0f32 / 127.0;
    assert_eq!(u16::from_le_bytes([out[0], out[1]]), f32_to_f16_bits(d));
    // on AVX2 machines the rounding is round-half-to-even:
    // x[i]*127 rounded — e.g. x=1/3*127 = 42.333… → 42; x=1 → 127; x=-1 → -127
// x[i] = ((i%7)-3)/3 → values -1, -2/3, -1/3, 0, 1/3, 2/3, 1 repeating
    // × 127 (id): -127, -84.666→-85, -42.333→-42, 0, 42.333→42, 84.666→85, 127
    let qs = &out[2..34];
    assert_eq!(qs[0] as i8, -127);
    assert_eq!(qs[1] as i8, -85);
    assert_eq!(qs[2] as i8, -42);
    assert_eq!(qs[3] as i8, 0);
    assert_eq!(qs[4] as i8, 42);
    assert_eq!(qs[5] as i8, 85);
    assert_eq!(qs[6] as i8, 127);
    assert_eq!(qs[7] as i8, -127); // i=7: (0-3)/3 = -1
}

#[test]
fn q8_k_quantizer_hand_case() {
    // 256 values; max positive 2.0 → iscale = -63.5, d = 1/-63.5
    let mut x = vec![0.01f32; 256];
    x[10] = 2.0;
    x[11] = -2.0;
    let mut out = vec![0u8; 292];
    quantize_row_q8_k(&x, &mut out);
    let d = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    let iscale = -127.0f32 / 2.0;
    assert!((d - (1.0 / iscale)).abs() < 1e-6, "d={d}");
    let qs = &out[4..260];
    // nearest_int(iscale * 0.01) = nearest_int(-0.635) = -1
    assert_eq!(qs[0] as i8, -1);
    // x=2.0 → nearest_int(-63.5)?? the C magic trick: -63.5 + 12582912 →
    // check via nearest_int directly
    let v = nearest_int(iscale * 2.0);
    assert_eq!(qs[10] as i8, v.min(127) as i8);
    assert_eq!(qs[11] as i8, nearest_int(iscale * -2.0).min(127) as i8);
    // bsums: first 16 values: 15×(-1) + one value → sum
    let b0 = i16::from_le_bytes([out[260], out[261]]);
    let mut expect = 0i32;
    for i in 0..16 {
        expect += qs[i] as i8 as i32;
    }
    assert_eq!(b0 as i32, expect);
}

#[test]
fn q8_k_quantizer_zero_block() {
    let x = vec![0f32; 256];
    let mut out = vec![0u8; 292];
    quantize_row_q8_k(&x, &mut out);
    assert!(out[..292].iter().all(|&b| b == 0));
}

// ---------- 4) f16 encoder ----------

#[test]
fn f16_encoder_matches_f16c() {
    if !std::arch::is_x86_feature_detected!("f16c") {
        return;
    }
    // exhaustive-ish: sweep bit patterns of interesting f32 ranges
    let mut st = 42u64;
    for _ in 0..200_000 {
        let bits = (xorshift(&mut st) & 0xFFFF_FFFF) as u32;
        // keep to finite, reasonable range for scales
        let f = f32::from_bits(bits & 0x3FFF_FFFF); // positive finite part
        if !f.is_finite() {
            continue;
        }
        let sw = f32_to_f16_bits_sw(f);
        let hw = f32_to_f16_bits(f); // F16C path on this machine
        assert_eq!(sw, hw, "f={f} (bits {bits:08x})");
    }
    // specific boundary values
    for &v in &[
        0.0f32, -0.0, 1.0, 65504.0, 65505.0, 65520.0, 65521.0, 65536.0,
        2.0e-5, 5.96e-8, 6.0e-8, 6.1e-8, 1e-6, 0.5, 0.25, 0.33333334,
    ] {
        assert_eq!(f32_to_f16_bits_sw(v), f32_to_f16_bits(v), "v={v}");
    }
    // round trip: f16→f32 of the encoding must be the closest half
    for &v in &[0.0f32, 1.0, 0.5, 0.3, 1234.5, 65504.0, 2.0e-5] {
        let h = f32_to_f16_bits_sw(v);
        let back = f16_to_f32(h);
        assert!((back - v).abs() <= (v / 1024.0).max(6e-8) * 1.5, "v={v} back={back}");
    }
}

#[test]
fn nearest_int_matches_c_semantics() {
    // the magic trick: ties land on the even? no — the trick adds 1.5*2^23;
    // exact halves resolve by fp addition (rn). Known values:
    assert_eq!(nearest_int(0.5), 0); // 0.5 + 12582912 = 12582912.5 → rn-even → 12582912
    assert_eq!(nearest_int(1.5), 2);
    assert_eq!(nearest_int(2.5), 2);
    assert_eq!(nearest_int(-0.5), 0);
    assert_eq!(nearest_int(-2.5), -2);
    assert_eq!(nearest_int(126.4), 126);
    assert_eq!(nearest_int(-126.6), -127);
}

// ---------- 5) worker pool ----------

#[test]
fn pool_runs_all_chunks() {
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let c2 = counter.clone();
    let f = move |i: usize| {
        c2.fetch_add(i + 1, std::sync::atomic::Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_micros(100));
    };
    par_chunks(8, 4, &f);
    // Σ (i+1) for i in 0..8 = 36
    assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 36);
}

#[test]
fn pool_parallelism_actually_overlaps() {
    // two long jobs on separate workers must overlap in time
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let b2 = barrier.clone();
    let start = std::time::Instant::now();
    let f = move |_i: usize| {
        b2.wait();
    };
    par_chunks(2, 2, &f);
    // if serialized, the first barrier wait would deadlock-timeout; a quick
    // completion means both entered wait() concurrently
    assert!(start.elapsed().as_secs() < 10);
}

#[test]
fn matvec_pool_no_deadlock_many_calls() {
    let mut st = 0x1234u64;
    let n_in = 256;
    let n_out = 64;
    let x = rand_f32(&mut st, n_in);
    let mut data = rand_bytes(&mut st, 144 * (n_in / 256) * n_out);
    sanitize_scales(GgmlType::Q4_K, &mut data);
    let mut out = vec![0f32; n_out];
    for _ in 0..200 {
        matvec(GgmlType::Q4_K, &data, n_in, n_out, &x, 2, &mut out);
    }
    // deterministic check: last call equals a single call
    let mut out2 = vec![0f32; n_out];
    matvec(GgmlType::Q4_K, &data, n_in, n_out, &x, 2, &mut out2);
    for (a, b) in out.iter().zip(out2.iter()) {
        assert_eq!(a, b);
    }
}
