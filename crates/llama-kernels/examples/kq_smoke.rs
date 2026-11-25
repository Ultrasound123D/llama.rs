//! kq_smoke — quick differential smoke test of the packed Q4_K/Q6_K VNNI
//! path vs the AVX2 K-quant path (the engine's established reference).
//! Run: cargo run --release --example kq_smoke

use llama_gguf::GgmlType;
use llama_kernels::matvec;
use llama_kernels::vnni;

fn xr(st: &mut u64) -> u64 {
    let mut x = *st;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *st = x;
    x
}

fn f16(v: f32) -> u16 {
    llama_kernels::quant::f32_to_f16_bits_sw(v)
}

/// Random Q4_K tensor (valid block structure, random content).
fn make_q4_k(n_in: usize, n_out: usize, st: &mut u64) -> Vec<u8> {
    let kb = n_in / 256;
    let mut data = vec![0u8; n_out * kb * 144];
    for b in data.chunks_exact_mut(144) {
        b[0..2].copy_from_slice(&f16(0.4 + (xr(st) % 64) as f32 / 64.0).to_le_bytes());
        b[2..4].copy_from_slice(&f16(0.05 + (xr(st) % 32) as f32 / 256.0).to_le_bytes());
        for i in 4..144 {
            b[i] = (xr(st) & 0xFF) as u8;
        }
    }
    data
}

/// Random Q6_K tensor.
fn make_q6_k(n_in: usize, n_out: usize, st: &mut u64) -> Vec<u8> {
    let kb = n_in / 256;
    let mut data = vec![0u8; n_out * kb * 210];
    for b in data.chunks_exact_mut(210) {
        b[208..210].copy_from_slice(&f16(0.3 + (xr(st) % 64) as f32 / 128.0).to_le_bytes());
        for i in 0..192 {
            b[i] = (xr(st) & 0xFF) as u8;
        }
        for i in 192..208 {
            // i8 scales: cover the signed range (incl. negatives)
            b[i] = (xr(st) % 256) as u8;
        }
    }
    data
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()).max(1e-3))
        .fold(0f32, f32::max)
}

fn check(ty: GgmlType, data: &[u8], n_in: usize, n_out: usize, x: &[f32]) {
    let mut ref_out = vec![0f32; n_out];
    matvec(ty, data, n_in, n_out, x, 1, &mut ref_out);
    let (packed, tystr) = match ty {
        GgmlType::Q4_K => (vnni::repack_q4_k(data, n_in, n_out), "Q4_K"),
        GgmlType::Q6_K => (vnni::repack_q6_k(data, n_in, n_out), "Q6_K"),
        _ => unreachable!(),
    };
    for &(t, label) in &[(1usize, "1t"), (2usize, "2t")] {
        let mut out = vec![0f32; n_out];
        vnni::matvec_vnni(&packed, ty, n_in, n_out, x, t, &mut out);
        let r = max_rel(&ref_out, &out);
        println!("  {tystr} {n_in}x{n_out} {label}: max_rel = {r:.3e}");
        assert!(r < 1e-3, "{tystr} {label} mismatch: max_rel {r}");
    }
    // fused pair: same tensor twice must equal the single
    let mut o1 = vec![0f32; n_out];
    let mut o2 = vec![0f32; n_out];
    let mut s1 = vec![0f32; n_out];
    let mut s2 = vec![0f32; n_out];
    vnni::matvec_vnni(&packed, ty, n_in, n_out, x, 1, &mut s1);
    vnni::matvec_vnni(&packed, ty, n_in, n_out, x, 1, &mut s2);
    vnni::matvec2_vnni(&packed, ty, n_out, &packed, ty, n_out, n_in, x, 1, &mut o1, &mut o2);
    let (r1, r2) = (max_rel(&s1, &o1), max_rel(&s2, &o2));
    println!("  {tystr} fused-pair: rel {r1:.3e}/{r2:.3e}");
    assert!(r1 < 1e-5 && r2 < 1e-5, "fused pair mismatch {r1}/{r2}");
}

fn main() {
    assert!(vnni::vnni_path_available(), "no AVX512-VNNI on this machine");
    let mut st = 0xa5a5_5a5au64;
    for (ni, no) in [(256usize, 16usize), (256, 32), (512, 16), (768, 48), (2048, 512)] {
        let x: Vec<f32> = (0..ni)
            .map(|_| (xr(&mut st) % 2000) as f32 / 2000.0 - 0.5)
            .collect();
        check(GgmlType::Q4_K, &make_q4_k(ni, no, &mut st), ni, no, &x);
        check(GgmlType::Q6_K, &make_q6_k(ni, no, &mut st), ni, no, &x);
    }
    // zero-weights edge: d=0, dmin=0, qs=0 → exactly zero
    let (ni, no) = (256, 16);
    let mut zero4 = vec![0u8; no * ni / 256 * 144];
    let mut out = vec![0f32; no];
    let x: Vec<f32> = (0..ni).map(|i| ((i * 37) % 200) as f32 / 200.0 - 0.5).collect();
    vnni::matvec_vnni(&vnni::repack_q4_k(&zero4, ni, no), GgmlType::Q4_K, ni, no, &x, 1, &mut out);
    assert!(out.iter().all(|&v| v == 0.0), "Q4_K zero tensor must be exactly 0");
    let mut zero6 = vec![0u8; no * ni / 256 * 210];
    vnni::matvec_vnni(&vnni::repack_q6_k(&zero6, ni, no), GgmlType::Q6_K, ni, no, &x, 1, &mut out);
    assert!(out.iter().all(|&v| v == 0.0), "Q6_K zero tensor must be exactly 0");
    println!("Q4_K/Q6_K packed VNNI smoke: ALL GREEN");
}
