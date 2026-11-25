//! kq_bench — isolated micro-benchmark of the packed Q4_K/Q6_K VNNI matvec
//! at the real LFM3 tensor shapes (median of 15 reps, 1/2 threads).
use llama_gguf::GgmlType;
use llama_kernels::vnni;

fn xr(st: &mut u64) -> u64 {
    let mut x = *st;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *st = x;
    x
}
fn f16(v: f32) -> u16 { llama_kernels::quant::f32_to_f16_bits_sw(v) }

fn make_q4_k(n_in: usize, n_out: usize, st: &mut u64) -> Vec<u8> {
    let kb = n_in / 256;
    let mut data = vec![0u8; n_out * kb * 144];
    for b in data.chunks_exact_mut(144) {
        b[0..2].copy_from_slice(&f16(0.4 + (xr(st) % 64) as f32 / 64.0).to_le_bytes());
        b[2..4].copy_from_slice(&f16(0.05 + (xr(st) % 32) as f32 / 256.0).to_le_bytes());
        for i in 4..144 { b[i] = (xr(st) & 0xFF) as u8; }
    }
    data
}
fn make_q6_k(n_in: usize, n_out: usize, st: &mut u64) -> Vec<u8> {
    let kb = n_in / 256;
    let mut data = vec![0u8; n_out * kb * 210];
    for b in data.chunks_exact_mut(210) {
        b[208..210].copy_from_slice(&f16(0.3 + (xr(st) % 64) as f32 / 128.0).to_le_bytes());
        for i in 0..208 { b[i] = (xr(st) & 0xFF) as u8; }
    }
    data
}

fn bench(ty: GgmlType, data: &[u8], n_in: usize, n_out: usize, label: &str) {
    let packed = match ty {
        GgmlType::Q4_K => vnni::repack_q4_k(data, n_in, n_out),
        _ => vnni::repack_q6_k(data, n_in, n_out),
    };
    let mb = data.len() as f64 / 1e6;
    let mut x = vec![0f32; n_in];
    let mut st = 0x5eed_5eedu64;
    for v in x.iter_mut() { *v = (xr(&mut st) % 2000) as f32 / 2000.0 - 0.5; }
    let mut out = vec![0f32; n_out];
    for &t in &[1usize, 2] {
        // warmup + median of 15
        for _ in 0..3 { vnni::matvec_vnni(&packed, ty, n_in, n_out, &x, t, &mut out); }
        let mut times = Vec::new();
        for _ in 0..15 {
            let t0 = std::time::Instant::now();
            vnni::matvec_vnni(&packed, ty, n_in, n_out, &x, t, &mut out);
            times.push(t0.elapsed().as_secs_f64() * 1e3); // ms
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = times[7];
        println!("{label:<38} {t}t  median {med:7.3} ms   {mb:6.1} MB  {:6.2} GB/s", mb / med);
    }
}

fn main() {
    let mut st = 0xa5a5_5a5au64;
    let x = &mut st;
    bench(GgmlType::Q4_K, &make_q4_k(2048, 7680, x), 2048, 7680, "Q4_K 2048x7680 (gate/up)");
    bench(GgmlType::Q6_K, &make_q6_k(7680, 2048, x), 7680, 2048, "Q6_K 7680x2048 (ffn_down)");
    bench(GgmlType::Q4_K, &make_q4_k(7680, 2048, x), 7680, 2048, "Q4_K 7680x2048 (ffn_down)");
    bench(GgmlType::Q4_K, &make_q4_k(2048, 6144, x), 2048, 6144, "Q4_K 2048x6144 (in_proj)");
    bench(GgmlType::Q4_K, &make_q4_k(2048, 2048, x), 2048, 2048, "Q4_K 2048x2048 (out_proj/q/o)");
    bench(GgmlType::Q4_K, &make_q4_k(2048, 512, x), 2048, 512, "Q4_K 2048x512 (attn_k/v)");
    bench(GgmlType::Q6_K, &make_q6_k(2048, 2048, x), 2048, 2048, "Q6_K 2048x2048 (head-ish)");
    let mut st2 = 0x1234_4321u64;
    let x2 = &mut st2;
    bench(GgmlType::Q6_K, &make_q6_k(2048, 65536, x2), 2048, 65536, "Q6_K 2048x65536 (token head)");
}
