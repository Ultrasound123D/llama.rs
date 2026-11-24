//! Per-kernel micro-benchmark (Phase 6.1): isolated vec_dot/matvec rates for
//! every quant type at the real model shapes of LFM3-1.7B / Qwen2.5-0.5B,
//! with 1 and 2 threads. Reports µs/call and effective streaming GB/s
//! (weight bytes / wall time), median over R repetitions.
//!
//! Usage: cargo run --release --example kprofile [-- large]
//!   (default: model-representative shapes; `large`: DRAM-resident 1 GB
//!    shapes that defeat the 504 MB L3)

use llama_gguf::GgmlType;
use llama_kernels::matvec;

fn make_weights(ty: GgmlType, n_in: usize, n_out: usize) -> Vec<u8> {
    let block = ty.block_size() as usize;
    let vpb = ty.block_vals() as usize;
    let row_bytes = block * (n_in / vpb);
    let mut data = vec![0u8; row_bytes * n_out];
    let mut st = 0x1234_5678u64;
    let mut xr = |st: &mut u64| {
        let mut x = *st;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        *st = x;
        x
    };
    for b in data.chunks_exact_mut(block) {
        for byte in b.iter_mut() {
            *byte = (xr(&mut st) & 0xFF) as u8;
        }
        match ty {
            GgmlType::Q4_K | GgmlType::Q5_K => {
                b[0] = 0x38; b[1] = 0x00; b[2] = 0x34; b[3] = 0x00;
            }
            GgmlType::Q6_K => {
                b[208] = 0x38; b[209] = 0x00;
            }
            GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::Q5_0 => {
                b[0] = 0x38; b[1] = 0x00;
            }
            _ => {}
        }
    }
    data
}

fn bench(ty: GgmlType, n_in: usize, n_out: usize, threads: usize, reps: usize) -> (f64, f64) {
    let data = make_weights(ty, n_in, n_out);
    let mut x = vec![0f32; n_in];
    let mut st = 0xdead_beefu64;
    for v in x.iter_mut() {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        *v = (st % 1000) as f32 / 1000.0 - 0.5;
    }
    let mut out = vec![0f32; n_out];
    // warmup
    matvec(ty, &data, n_in, n_out, &x, threads, &mut out);
    let mut times: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        matvec(ty, &data, n_in, n_out, &x, threads, &mut out);
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    let us = med * 1e6;
    let gbs = (data.len() as f64 / 1e9) / med;
    let _ = out.iter().sum::<f32>();
    (us, gbs)
}

fn main() {
    let large = std::env::args().any(|a| a == "large");
    let reps: usize = std::env::var("REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(15);

    // (label, type, n_in, n_out, MB) — the real LFM3-1.7B/Qwen2.5-0.5B shapes
    let shapes: Vec<(&str, GgmlType, usize, usize)> = if large {
        vec![
            ("q4k-1gb", GgmlType::Q4_K, 2048, 524288),
            ("q5k-1.3gb", GgmlType::Q5_K, 2048, 524288),
            ("q6k-1.5gb", GgmlType::Q6_K, 2048, 436906),
            ("q8_0-2gb", GgmlType::Q8_0, 2048, 327680),
        ]
    } else {
        vec![
            // LFM3-1.7B
            ("lfm3 ffn_gate 2048x7680 Q4_K", GgmlType::Q4_K, 2048, 7680),
            ("lfm3 ffn_down 7680x2048 Q6_K", GgmlType::Q6_K, 7680, 2048),
            ("lfm3 in_proj 2048x6144 Q4_K", GgmlType::Q4_K, 2048, 6144),
            ("lfm3 head 2048x32768 Q6_K", GgmlType::Q6_K, 2048, 32768),
            ("lfm3 head 2048x32768 Q4_K", GgmlType::Q4_K, 2048, 32768),
            ("lfm3 attn_q 2048x2048 Q4_K", GgmlType::Q4_K, 2048, 2048),
            // Qwen2.5-0.5B (real tensor types from the GGUF)
            ("qwen ffn_gate 896x4864 Q5_0", GgmlType::Q5_0, 896, 4864),
            ("qwen ffn_down 4864x896 Q6_K", GgmlType::Q6_K, 4864, 896),
            ("qwen head 896x151936 Q8_0", GgmlType::Q8_0, 896, 151936),
            ("qwen attn_q 896x896 Q5_0", GgmlType::Q5_0, 896, 896),
            // other types exercised by the spec
            ("q5k 2048x7680", GgmlType::Q5_K, 2048, 7680),
            ("q8_0 2048x7680", GgmlType::Q8_0, 2048, 7680),
            ("q4_0 2048x7680", GgmlType::Q4_0, 2048, 7680),
            ("q5_0 2048x7680", GgmlType::Q5_0, 2048, 7680),
            ("q5_0 896x4864 (qwen gate)", GgmlType::Q5_0, 896, 4864),
            ("f16 2048x7680", GgmlType::F16, 2048, 7680),
        ]
    };

    println!("kernel profile — median of {reps} reps, cold-free (weights stay in RAM/L3)\n");
    println!("{:<34} {:>9} {:>9} {:>9} {:>9} {:>9}", "shape", "1t µs", "1t GB/s", "2t µs", "2t GB/s", "MB");
    for (label, ty, n_in, n_out) in &shapes {
        let mb = ty.block_size() as f64 * (*n_in as f64 / ty.block_vals() as f64) * *n_out as f64 / 1e6;
        let (us1, gbs1) = bench(*ty, *n_in, *n_out, 1, reps);
        let (us2, gbs2) = bench(*ty, *n_in, *n_out, 2, reps);
        println!("{:<34} {:>9.1} {:>9.2} {:>9.1} {:>9.2} {:>9.1}", label, us1, gbs1, us2, gbs2, mb);
    }
}
