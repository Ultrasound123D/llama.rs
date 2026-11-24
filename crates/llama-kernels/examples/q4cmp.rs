// head-to-head: AVX2 vs AVX-512 VNNI q4_K kernel on DRAM-sized data
use llama_gguf::GgmlType;
use llama_kernels::QAct;

fn main() {
    let n_out = 65536usize;
    let n = 2048usize;
    let mut st = 0x1234u64;
    let mut xr = |st: &mut u64| { let mut x = *st; x ^= x<<13; x^=x>>7; x^=x<<17; *st=x; x };
    let mut data = vec![0u8; 144 * (n / 256) * n_out];
    for b in data.chunks_exact_mut(144) {
        for byte in b.iter_mut() { *byte = (xr(&mut st) & 0xFF) as u8; }
        b[0]=0x38; b[1]=0x00; b[2]=0x34; b[3]=0x00;
    }
    let x: Vec<f32> = (0..n).map(|_| (xr(&mut st) % 2000) as f32 / 1000.0 - 1.0).collect();
    let y = QAct::quantize(GgmlType::Q4_K, &x);
    let QAct::Q8_K(q) = &y else { panic!() };
    let row = 144 * (n / 256);

    // sequential column loop (single thread) to test pure streaming
    for kernel in 0..2 {
        let iters = 8;
        let t0 = std::time::Instant::now();
        let mut acc = 0f32;
        for _ in 0..iters {
            for o in 0..n_out {
                let w = &data[o * row..(o + 1) * row];
                acc += if kernel == 0 {
                    unsafe { llama_kernels::avx2::dot_q4_k_q8_k(w, q, n) }
                } else {
                    unsafe { llama_kernels::avx512::dot_q4_k_q8_k(w, q, n) }
                };
            }
        }
        let secs = t0.elapsed().as_secs_f32() / iters as f32;
        let gbs = (data.len() as f64 / 1e9) / secs as f64;
        let ms = secs * 1000.0;
        println!("kernel={} ({}): {ms:.1} ms = {gbs:.2} GB/s (acc {acc:.0})", kernel,
            if kernel == 0 { "avx2" } else { "avx512" });
    }
}
