// isolate: time dot_q8_0_q8_0 per-column vs its pieces
use llama_gguf::GgmlType;
use llama_kernels::QAct;

fn main() {
    let n = 2048usize;
    let mut st = 0x1234u64;
    let mut xr = |st: &mut u64| { let mut x = *st; x ^= x<<13; x^=x>>7; x^=x<<17; *st=x; x };
    let mut w: Vec<u8> = (0..n / 32 * 34).map(|_| (xr(&mut st) & 0xFF) as u8).collect();
    for b in w.chunks_exact_mut(34) {
        b[0] = 0x38; b[1] = 0x00;
    }
    let x: Vec<f32> = (0..n).map(|_| (xr(&mut st) % 2000) as f32 / 1000.0 - 1.0).collect();
    let y = QAct::quantize(GgmlType::Q8_0, &x);
    let QAct::Q8_0(q) = &y else { panic!() };

    // warmup
    let mut acc = 0f32;
    for _ in 0..1000 {
        acc += unsafe { llama_kernels::avx2::dot_q8_0_q8_0(&w, q, n) };
    }
    let iters = 200_000usize;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        acc += unsafe { llama_kernels::avx2::dot_q8_0_q8_0(&w, q, n) };
    }
    let ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("dot_q8_0_q8_0: {ns:.0} ns/column (2048 values, 2176+2176 bytes) acc={acc:.2}");

    // compare: q4_K dot on same size
    let mut wk: Vec<u8> = (0..n / 256 * 144).map(|_| (xr(&mut st) & 0xFF) as u8).collect();
    for b in wk.chunks_exact_mut(144) {
        b[0] = 0x38; b[1] = 0x00; b[2] = 0x34; b[3] = 0x00;
    }
    let yk = QAct::quantize(GgmlType::Q4_K, &x);
    let QAct::Q8_K(qk) = &yk else { panic!() };
    for _ in 0..1000 {
        acc += unsafe { llama_kernels::avx2::dot_q4_k_q8_k(&wk, qk, n) };
    }
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        acc += unsafe { llama_kernels::avx2::dot_q4_k_q8_k(&wk, qk, n) };
    }
    let ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("dot_q4_k_q8_k: {ns:.0} ns/column (2048 values) acc={acc:.2}");
}
