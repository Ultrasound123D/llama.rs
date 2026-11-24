// Qwen-like small matvec sizes: per-call throughput incl. pool overhead
use llama_gguf::GgmlType;
use llama_kernels::matvec;

fn bench(ty: GgmlType, n_in: usize, n_out: usize, iters: usize) {
    let block = ty.block_size() as usize;
    let vpb = ty.block_vals() as usize;
    let row = block * (n_in / vpb);
    let mut data = vec![0u8; row * n_out];
    for (i, b) in data.chunks_exact_mut(block).enumerate() {
        for byte in b.iter_mut() { *byte = (i % 251) as u8; }
        b[0] = 0x38; b[1] = 0x00;
    }
    let x = vec![0.01f32; n_in];
    let mut out = vec![0f32; n_out];
    matvec(ty, &data, n_in, n_out, &x, 2, &mut out);
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        matvec(ty, &data, n_in, n_out, &x, 2, &mut out);
    }
    let us = t0.elapsed().as_secs_f32() * 1e6 / iters as f32;
    let gbs = (data.len() as f64 / 1e9) / (us / 1e6) as f64;
    println!("{ty:?} {n_in}x{n_out} ({:.2} MB): {us:.0} us/call = {gbs:.2} GB/s", data.len() as f64 / 1e6);
}

fn main() {
    bench(GgmlType::Q5_0, 896, 896, 3000);
    bench(GgmlType::Q8_0, 896, 128, 3000);
    bench(GgmlType::Q8_0, 896, 151936, 5);
    bench(GgmlType::Q4_K, 2048, 2048, 300);
    bench(GgmlType::Q4_K, 2048, 7680, 100);
}
