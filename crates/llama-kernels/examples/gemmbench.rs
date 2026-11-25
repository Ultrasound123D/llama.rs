// Phase 6.8 micro-benchmark: the packed GEMM kernel itself (gemm_prepared /
// gemm2_prepared) on model-realistic shapes and batch sizes. Isolates the
// GEMM from quantization (QBatch built once, outside the timing loop) and
// reports weight-streaming GB/s — the metric the Phase 6.6 audit used
// (§9.4: LFM3 GEMM 63–82 % of prefill wall at 0.55–0.65 GB/s).
use llama_gguf::GgmlType;
use llama_kernels::vnni;

fn make_weights(ty: GgmlType, n_in: usize, n_out: usize, seed: u64) -> Vec<u8> {
    let block = ty.block_size() as usize;
    let vpb = ty.block_vals() as usize;
    let row_bytes = block * (n_in / vpb);
    let mut data = vec![0u8; row_bytes * n_out];
    let mut st = seed;
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
            GgmlType::Q4_K => {
                b[0] = 0x38; b[1] = 0x00; b[2] = 0x34; b[3] = 0x00;
            }
            GgmlType::Q6_K => {
                b[208] = 0x38; b[209] = 0x00;
            }
            GgmlType::Q8_0 | GgmlType::Q5_0 => {
                b[0] = 0x38; b[1] = 0x00;
            }
            _ => {}
        }
    }
    data
}

fn make_x(m: usize, n_in: usize, seed: u64) -> Vec<f32> {
    let mut st = seed;
    (0..m * n_in)
        .map(|_| {
            let mut x = st;
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            st = x;
            (x % 1000) as f32 / 1000.0 - 0.5
        })
        .collect()
}

fn bench_gemm(ty: GgmlType, n_in: usize, n_out: usize, m: usize, label: &str) {
    let data = make_weights(ty, n_in, n_out, 0x1234_5678u64);
    let pt = vnni::PackedTensor {
        data: match ty {
            GgmlType::Q8_0 => vnni::repack_q8_0(&data, n_in, n_out),
            GgmlType::Q5_0 => vnni::repack_q5_0(&data, n_in, n_out),
            GgmlType::Q4_K => vnni::repack_q4_k(&data, n_in, n_out),
            GgmlType::Q6_K => vnni::repack_q6_k(&data, n_in, n_out),
            _ => unreachable!(),
        },
        ty,
        n_in,
        n_out,
    };
    let x = make_x(m, n_in, 0x9e37_79b9u64);
    let q = vnni::quantize_batch(ty, &x, n_in, m, 2);
    let mut out = vec![0f32; m * n_out];
    vnni::gemm_prepared(&pt, &q, 2, &mut out);
    let mut iters = 2usize;
    let mut t;
    loop {
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            vnni::gemm_prepared(&pt, &q, 2, &mut out);
        }
        t = t0.elapsed();
        if t.as_secs_f32() > 0.3 || iters > 500 {
            break;
        }
        iters *= 4;
    }
    let secs = t.as_secs_f64() / iters as f64;
    let wbytes = data.len() as f64;
    let gbs = wbytes / 1e9 / secs;
    let us_tok = secs * 1e6 / m as f64;
    println!(
        "{label:>26}: {n_in:>5}x{n_out:<6} M={m:<4} {us_tok:9.2} us/tok  weights {gbs:6.2} GB/s  (sum {:.0})",
        out.iter().sum::<f32>()
    );
}

fn bench_gemm2(ty: GgmlType, n_in: usize, n_out1: usize, n_out2: usize, m: usize, label: &str) {
    let d1 = make_weights(ty, n_in, n_out1, 0x1111_2222u64);
    let d2 = make_weights(ty, n_in, n_out2, 0x3333_4444u64);
    let p1 = vnni::PackedTensor {
        data: match ty {
            GgmlType::Q8_0 => vnni::repack_q8_0(&d1, n_in, n_out1),
            GgmlType::Q5_0 => vnni::repack_q5_0(&d1, n_in, n_out1),
            GgmlType::Q4_K => vnni::repack_q4_k(&d1, n_in, n_out1),
            GgmlType::Q6_K => vnni::repack_q6_k(&d1, n_in, n_out1),
            _ => unreachable!(),
        },
        ty,
        n_in,
        n_out: n_out1,
    };
    let p2 = vnni::PackedTensor {
        data: match ty {
            GgmlType::Q8_0 => vnni::repack_q8_0(&d2, n_in, n_out2),
            GgmlType::Q5_0 => vnni::repack_q5_0(&d2, n_in, n_out2),
            GgmlType::Q4_K => vnni::repack_q4_k(&d2, n_in, n_out2),
            GgmlType::Q6_K => vnni::repack_q6_k(&d2, n_in, n_out2),
            _ => unreachable!(),
        },
        ty,
        n_in,
        n_out: n_out2,
    };
    let x = make_x(m, n_in, 0x5555_6666u64);
    let q = vnni::quantize_batch(ty, &x, n_in, m, 2);
    let mut o1 = vec![0f32; m * n_out1];
    let mut o2 = vec![0f32; m * n_out2];
    vnni::gemm2_prepared(&p1, &p2, &q, 2, &mut o1, &mut o2);
    let mut iters = 2usize;
    let mut t;
    loop {
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            vnni::gemm2_prepared(&p1, &p2, &q, 2, &mut o1, &mut o2);
        }
        t = t0.elapsed();
        if t.as_secs_f32() > 0.3 || iters > 500 {
            break;
        }
        iters *= 4;
    }
    let secs = t.as_secs_f64() / iters as f64;
    let wbytes = (d1.len() + d2.len()) as f64;
    let gbs = wbytes / 1e9 / secs;
    let us_tok = secs * 1e6 / m as f64;
    println!(
        "{label:>26}: {n_in:>5}x{n_out1}+{n_out2:<4} M={m:<4} {us_tok:9.2} us/tok  weights {gbs:6.2} GB/s  (sum {:.0})",
        o1.iter().sum::<f32>() + o2.iter().sum::<f32>()
    );
}

fn main() {
    println!("threads: 2 (persistent pool) — GEMM kernel only (QBatch pre-built)");
    for m in [8usize, 32, 128, 512] {
        println!("--- M={m} ---");
        // LFM3-1.7B prefill shapes (§9.4's GEMM bucket: gate+up 30.1 %, ffn_down 19.9 %, conv, attn)
        bench_gemm2(GgmlType::Q4_K, 2048, 7680, 7680, m, "LFM3 gate+up pair (Q4_K)");
        bench_gemm(GgmlType::Q6_K, 7680, 2048, m, "LFM3 ffn_down (Q6_K)");
        bench_gemm(GgmlType::Q4_K, 2048, 32768, m, "LFM3 output head (Q4_K)");
        bench_gemm(GgmlType::Q4_K, 2048, 2048, m, "LFM3 attn proj (Q4_K)");
        // Qwen2.5-0.5B prefill shapes (hidden 896, ffn 4864)
        bench_gemm2(GgmlType::Q5_0, 896, 4864, 4864, m, "Qwen gate+up pair (Q5_0)");
        bench_gemm(GgmlType::Q6_K, 4864, 896, m, "Qwen ffn_down (Q6_K)");
        bench_gemm(GgmlType::Q5_0, 896, 896, m, "Qwen attn proj (Q5_0)");
        bench_gemm(GgmlType::Q8_0, 896, 151936, m, "Qwen output head (Q8_0)");
    }
}
