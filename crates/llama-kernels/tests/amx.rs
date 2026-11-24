//! AMX backend differential tests (Phase 6.3 gates).
//!
//! The repacked-VNNI path must match the existing AVX2 int-domain path:
//! integer dot products are exact on both sides; only the f32 scale
//! application order differs, so a small relative tolerance applies.
//! Randomized (deterministic LCG) tensors, multiple repetitions.

use llama_gguf::GgmlType;
use llama_kernels::amx;
use llama_kernels::matvec;

fn xr(st: &mut u64) -> u64 {
    let mut x = *st;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *st = x;
    x
}

fn make_q8_0(n_in: usize, n_out: usize) -> Vec<u8> {
    let kb = n_in / 32;
    let mut st = 0xfeed_beefu64;
    let mut data = vec![0u8; n_out * kb * 34];
    for b in data.chunks_exact_mut(34) {
        // d: f16 scale in a safe range (0.5..2)
        let d = 0.5 + (xr(&mut st) % 64) as f32 / 64.0;
        let h = llama_kernels::quant::f32_to_f16_bits_sw(d);
        b[0..2].copy_from_slice(&h.to_le_bytes());
        for q in b[2..34].iter_mut() {
            *q = (xr(&mut st) & 0xFF) as u8;
        }
    }
    // fix alignment: n_out must be a multiple of 16 for the packed layout
    assert!(n_out % 16 == 0);
    data
}

fn make_q5_0(n_in: usize, n_out: usize) -> Vec<u8> {
    let kb = n_in / 32;
    let mut st = 0x1234_abcdu64;
    let mut data = vec![0u8; n_out * kb * 22];
    for b in data.chunks_exact_mut(22) {
        let d = 0.5 + (xr(&mut st) % 64) as f32 / 64.0;
        let h = llama_kernels::quant::f32_to_f16_bits_sw(d);
        b[0..2].copy_from_slice(&h.to_le_bytes());
        // block_q5_0: d(2) qh(4) qs(16)
        for i in 2..22 {
            b[i] = (xr(&mut st) & 0xFF) as u8;
        }
    }
    data
}

fn check_close(a: &[f32], b: &[f32], tol: f32) -> bool {
    let mut max_rel = 0f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        let m = x.abs().max(y.abs()).max(1e-3);
        let r = d / m;
        if r > max_rel {
            max_rel = r;
        }
    }
    eprintln!("  max_rel_diff = {max_rel:.6}");
    max_rel < tol
}

#[test]
fn amx_capability_report() {
    let c = amx::detect();
    eprintln!(
        "AMX caps: int8={} bf16={} avx512_vnni={} xstate_tile={} | path={} tiles={}",
        c.amx_int8, c.amx_bf16, c.avx512_vnni, c.xstate_tile,
        amx::amx_path_available(), amx::amx_tiles_available()
    );
}

#[test]
fn amx_q8_0_matches_avx2() {
    if !amx::amx_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    for rep in 0..5 {
        let (n_in, n_out) = (896usize, 304 + rep * 16); // multiples required
        let data = make_q8_0(n_in, n_out);
        let packed = amx::repack_q8_0(&data, n_in, n_out);
        let mut st = 0xdead_beefu64 + rep as u64;
        let x: Vec<f32> = (0..n_in)
            .map(|_| (xr(&mut st) % 2000) as f32 / 2000.0 - 0.5)
            .collect();
        let mut out_ref = vec![0f32; n_out];
        let mut out_amx = vec![0f32; n_out];
        matvec(GgmlType::Q8_0, &data, n_in, n_out, &x, 1, &mut out_ref);
        amx::matvec_amx(&packed, n_in, n_out, &x, 1, &mut out_amx);
        assert!(
            check_close(&out_ref, &out_amx, 1e-3),
            "Q8_0 rep {rep}: avx2 vs amx mismatch"
        );
    }
}

#[test]
fn amx_q5_0_matches_avx2() {
    if !amx::amx_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    for rep in 0..5 {
        let (n_in, n_out) = (896usize, 304 + rep * 16);
        let data = make_q5_0(n_in, n_out);
        let packed = amx::repack_q5_0(&data, n_in, n_out);
        let mut st = 0xbead_deadu64 + rep as u64;
        let x: Vec<f32> = (0..n_in)
            .map(|_| (xr(&mut st) % 2000) as f32 / 2000.0 - 0.5)
            .collect();
        let mut out_ref = vec![0f32; n_out];
        let mut out_amx = vec![0f32; n_out];
        matvec(GgmlType::Q5_0, &data, n_in, n_out, &x, 1, &mut out_ref);
        amx::matvec_amx(&packed, n_in, n_out, &x, 1, &mut out_amx);
        assert!(
            check_close(&out_ref, &out_amx, 1e-3),
            "Q5_0 rep {rep}: avx2 vs amx mismatch"
        );
    }
}

#[test]
fn amx_fused_pair_matches_two_singles() {
    if !amx::amx_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    let (n_in, n_out1, n_out2) = (896usize, 320usize, 288usize);
    let d1 = make_q5_0(n_in, n_out1);
    let d2 = make_q5_0(n_in, n_out2);
    let p1 = amx::repack_q5_0(&d1, n_in, n_out1);
    let p2 = amx::repack_q5_0(&d2, n_in, n_out2);
    let mut st = 0x1234_5678u64;
    let x: Vec<f32> = (0..n_in)
        .map(|_| (xr(&mut st) % 2000) as f32 / 2000.0 - 0.5)
        .collect();
    let mut o1a = vec![0f32; n_out1];
    let mut o2a = vec![0f32; n_out2];
    amx::matvec_amx(&p1, n_in, n_out1, &x, 1, &mut o1a);
    amx::matvec_amx(&p2, n_in, n_out2, &x, 1, &mut o2a);
    let mut o1b = vec![0f32; n_out1];
    let mut o2b = vec![0f32; n_out2];
    amx::matvec2_amx(&p1, n_out1, &p2, n_out2, n_in, &x, 1, &mut o1b, &mut o2b);
    assert!(check_close(&o1a, &o1b, 1e-5), "fused pair: out1 mismatch");
    assert!(check_close(&o2a, &o2b, 1e-5), "fused pair: out2 mismatch");
}

#[test]
fn amx_parallel_matches_sequential() {
    if !amx::amx_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    let (n_in, n_out) = (2048usize, 4096usize);
    let data = make_q8_0(n_in, n_out);
    let packed = amx::repack_q8_0(&data, n_in, n_out);
    let mut st = 0x0f0f_0f0fu64;
    let x: Vec<f32> = (0..n_in)
        .map(|_| (xr(&mut st) % 2000) as f32 / 2000.0 - 0.5)
        .collect();
    let mut out_seq = vec![0f32; n_out];
    let mut out_par = vec![0f32; n_out];
    amx::matvec_amx(&packed, n_in, n_out, &x, 1, &mut out_seq);
    amx::matvec_amx(&packed, n_in, n_out, &x, 2, &mut out_par);
    assert!(check_close(&out_seq, &out_par, 1e-6), "parallel vs sequential");
}
