// Phase 6 micro-benchmark: raw matvec throughput on a synthetic
// Q4_K tensor the size of the LFM3 output head (2048 x 32768 = 37.7 MB),
// plus an FFN-sized one. Reports GB/s of weight streaming — the metric
// llama-bench implicitly achieves at its reported tg rate.
use llama_gguf::GgmlType;
use llama_kernels::matvec;

fn bench(ty: GgmlType, n_in: usize, n_out: usize, label: &str) {
    // deterministic pseudo-random weights (any bytes are valid blocks);
    // fix the scales to safe values so values stay finite
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
    // F16: build from small finite values
    if ty == GgmlType::F16 {
        for (i, v) in data.chunks_exact_mut(2).enumerate() {
            let f = ((i % 1000) as f32) / 1000.0 - 0.5;
            v.copy_from_slice(&llama_kernels::quant::f32_to_f16_bits_sw(f).to_le_bytes());
        }
    }
    let mut x = vec![0f32; n_in];
    for v in x.iter_mut() {
        *v = (xr(&mut st) % 1000) as f32 / 1000.0 - 0.5;
    }
    let mut out = vec![0f32; n_out];
    // warmup + timing
    matvec(ty, &data, n_in, n_out, &x, 2, &mut out);
    let iters = 30usize;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        matvec(ty, &data, n_in, n_out, &x, 2, &mut out);
    }
    let secs = t0.elapsed().as_secs_f32() / iters as f32;
    let gbs = (data.len() as f64 / 1e9) / secs as f64;
    let ms = secs * 1000.0;
    println!("{label}: {n_in}x{n_out} {ms:.3} ms/matvec = {gbs:.2} GB/s (sum {:.1})", out.iter().sum::<f32>());
}

fn main() {
    println!("threads: 2 (persistent pool)");
    bench(GgmlType::Q4_K, 2048, 32768, "output-head");
    bench(GgmlType::Q4_K, 2048, 7680, "ffn-gate");
    bench(GgmlType::Q6_K, 2048, 32768, "head-q6");
    bench(GgmlType::Q8_0, 2048, 32768, "head-q8_0");
    bench(GgmlType::F16, 2048, 32768, "head-f16");
    // DRAM-sized tensors (force real memory streaming)
    bench(GgmlType::Q4_K, 2048, 65536 * 8, "dram-q4k-1gb");
    bench(GgmlType::Q6_K, 2048, 65536 * 8, "dram-q6k-1.2gb");
}
