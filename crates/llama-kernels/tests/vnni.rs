//! AVX-512 VNNI packed-backend differential tests (Phase 6.3 gates).
//!
//! The repacked-VNNI path must match the existing AVX2 int-domain path:
//! integer dot products are exact on both sides; only the f32 scale
//! application order differs, so a small relative tolerance applies.
//! Coverage: randomized tensors (deterministic LCG, multiple repetitions)
//! AND edge/boundary cases (minimum shapes, zero/extreme weights and
//! scales, qh patterns, activation extremes, repack gating, fused and
//! parallel determinism).

use llama_gguf::GgmlType;
use llama_kernels::vnni;
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

// ---------- capability reporting (detection, NOT execution) ----------

#[test]
fn vnni_capability_report() {
    let c = vnni::detect();
    eprintln!(
        "AMX caps (detection only — no tile instructions executed): int8={} bf16={} avx512_vnni={} xstate_tile={} | vnni_path={} tiles={}",
        c.amx_int8, c.amx_bf16, c.avx512_vnni, c.xstate_tile,
        vnni::vnni_path_available(), vnni::amx_tiles_available()
    );
}

// ---------- randomized differential (Q8_0 / Q5_0 vs AVX2 reference) ----------

#[test]
fn vnni_q8_0_matches_avx2() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    for rep in 0..5 {
        let (n_in, n_out) = (896usize, 304 + rep * 16); // multiples required
        let data = make_q8_0(n_in, n_out);
        let packed = vnni::repack_q8_0(&data, n_in, n_out);
        let mut st = 0xdead_beefu64 + rep as u64;
        let x: Vec<f32> = (0..n_in)
            .map(|_| (xr(&mut st) % 2000) as f32 / 2000.0 - 0.5)
            .collect();
        let mut out_ref = vec![0f32; n_out];
        let mut out_vnni = vec![0f32; n_out];
        matvec(GgmlType::Q8_0, &data, n_in, n_out, &x, 1, &mut out_ref);
        vnni::matvec_vnni(&packed, n_in, n_out, &x, 1, &mut out_vnni);
        assert!(
            check_close(&out_ref, &out_vnni, 1e-3),
            "Q8_0 rep {rep}: avx2 vs vnni mismatch"
        );
    }
}

#[test]
fn vnni_q5_0_matches_avx2() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    for rep in 0..5 {
        let (n_in, n_out) = (896usize, 304 + rep * 16);
        let data = make_q5_0(n_in, n_out);
        let packed = vnni::repack_q5_0(&data, n_in, n_out);
        let mut st = 0xbead_deadu64 + rep as u64;
        let x: Vec<f32> = (0..n_in)
            .map(|_| (xr(&mut st) % 2000) as f32 / 2000.0 - 0.5)
            .collect();
        let mut out_ref = vec![0f32; n_out];
        let mut out_vnni = vec![0f32; n_out];
        matvec(GgmlType::Q5_0, &data, n_in, n_out, &x, 1, &mut out_ref);
        vnni::matvec_vnni(&packed, n_in, n_out, &x, 1, &mut out_vnni);
        assert!(
            check_close(&out_ref, &out_vnni, 1e-3),
            "Q5_0 rep {rep}: avx2 vs vnni mismatch"
        );
    }
}

#[test]
fn vnni_fused_pair_matches_two_singles() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    let (n_in, n_out1, n_out2) = (896usize, 320usize, 288usize);
    let d1 = make_q5_0(n_in, n_out1);
    let d2 = make_q5_0(n_in, n_out2);
    let p1 = vnni::repack_q5_0(&d1, n_in, n_out1);
    let p2 = vnni::repack_q5_0(&d2, n_in, n_out2);
    let mut st = 0x1234_5678u64;
    let x: Vec<f32> = (0..n_in)
        .map(|_| (xr(&mut st) % 2000) as f32 / 2000.0 - 0.5)
        .collect();
    let mut o1a = vec![0f32; n_out1];
    let mut o2a = vec![0f32; n_out2];
    vnni::matvec_vnni(&p1, n_in, n_out1, &x, 1, &mut o1a);
    vnni::matvec_vnni(&p2, n_in, n_out2, &x, 1, &mut o2a);
    let mut o1b = vec![0f32; n_out1];
    let mut o2b = vec![0f32; n_out2];
    vnni::matvec2_vnni(&p1, n_out1, &p2, n_out2, n_in, &x, 1, &mut o1b, &mut o2b);
    assert!(check_close(&o1a, &o1b, 1e-5), "fused pair: out1 mismatch");
    assert!(check_close(&o2a, &o2b, 1e-5), "fused pair: out2 mismatch");
}

#[test]
fn vnni_parallel_matches_sequential() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    let (n_in, n_out) = (2048usize, 4096usize);
    let data = make_q8_0(n_in, n_out);
    let packed = vnni::repack_q8_0(&data, n_in, n_out);
    let mut st = 0x0f0f_0f0fu64;
    let x: Vec<f32> = (0..n_in)
        .map(|_| (xr(&mut st) % 2000) as f32 / 2000.0 - 0.5)
        .collect();
    let mut out_seq = vec![0f32; n_out];
    let mut out_par = vec![0f32; n_out];
    vnni::matvec_vnni(&packed, n_in, n_out, &x, 1, &mut out_seq);
    vnni::matvec_vnni(&packed, n_in, n_out, &x, 2, &mut out_par);
    assert!(check_close(&out_seq, &out_par, 1e-6), "parallel vs sequential");
}

// ---------- edge / boundary cases (review gate) ----------

/// Build a Q8_0 tensor where every block has the same f16 scale bits and
/// every code byte is `code`.
fn q8_0_uniform(d_bits: u16, code: u8, n_in: usize, n_out: usize) -> Vec<u8> {
    let kb = n_in / 32;
    let mut data = vec![0u8; n_out * kb * 34];
    for b in data.chunks_exact_mut(34) {
        b[0..2].copy_from_slice(&d_bits.to_le_bytes());
        b[2..34].fill(code);
    }
    data
}

/// Build a Q5_0 tensor where every block has scale `d_bits`, `qh` bytes
/// and `qs` (low nibble source) bytes set as given.
fn q5_0_uniform(d_bits: u16, qh: u32, qs: u8, n_in: usize, n_out: usize) -> Vec<u8> {
    let kb = n_in / 32;
    let mut data = vec![0u8; n_out * kb * 22];
    for b in data.chunks_exact_mut(22) {
        b[0..2].copy_from_slice(&d_bits.to_le_bytes());
        b[2..6].copy_from_slice(&qh.to_le_bytes());
        b[6..22].fill(qs);
    }
    data
}

/// Minimum-shape Q8_0 edge weights vs the AVX2 reference path.
#[test]
fn vnni_q8_0_edge_min_shape_and_extremes() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    // minimum legal shape: one 32-value block, one 16-column group
    let (n_in, n_out) = (32usize, 16usize);
    let cases: Vec<(&str, u16, u8)> = vec![
        ("zero weights (d=0, codes=0)", 0, 0),
        ("max f16 scale + max code (0x7F)", 0x7BFF, 0x7F), // d=65504
        ("max f16 scale + min code (0x80)", 0x7BFF, 0x80),
        ("min-normal f16 scale (6.1e-5)", 0x0400, 0x55),
        ("d=0 with max codes", 0, 0x7F),
        ("d=1 with max codes", 0x3C00, 0x7F),
    ];
    for (name, d_bits, code) in cases {
        let data = q8_0_uniform(d_bits, code, n_in, n_out);
        let packed = vnni::repack_q8_0(&data, n_in, n_out);
        // activation: deterministic small values
        let x: Vec<f32> = (0..n_in).map(|i| ((i * 37 % 200) as f32) / 200.0 - 0.5).collect();
        let mut out_ref = vec![0f32; n_out];
        let mut out_v = vec![0f32; n_out];
        matvec(GgmlType::Q8_0, &data, n_in, n_out, &x, 1, &mut out_ref);
        vnni::matvec_vnni(&packed, n_in, n_out, &x, 1, &mut out_v);
        assert!(
            check_close(&out_ref, &out_v, 1e-3),
            "Q8_0 edge '{name}': avx2 vs vnni mismatch"
        );
        // zero weights must produce exactly zero on both paths
        if name.starts_with("zero weights") {
            assert!(out_v.iter().all(|&v| v == 0.0), "zero-weight case must be exactly 0");
        }
    }
}

/// Minimum-shape Q5_0 edge weights incl. all qh bit patterns.
#[test]
fn vnni_q5_0_edge_min_shape_and_qh_patterns() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    let (n_in, n_out) = (32usize, 16usize);
    let f16bits = |v: f32| llama_kernels::quant::f32_to_f16_bits_sw(v);
    let d1 = f16bits(1.0);
    // (name, d_bits, qh, qs)
    let cases: Vec<(&str, u16, u32, u8)> = vec![
        ("zero weights", 0, 0x0000_0000, 0x00),
        ("all qh bits set, qs=0 (codes land at 0)", d1, 0xFFFF_FFFF, 0x00),
        ("all qh bits clear, qs=0xFF (codes -1)", d1, 0x0000_0000, 0xFF),
        ("all qh bits set, qs=0xFF (codes +15)", d1, 0xFFFF_FFFF, 0xFF),
        ("max scale, mixed pattern", 0x7BFF, 0xA5A5_A5A5, 0x5A),
    ];
    for (name, d_bits, qh, qs) in cases {
        let data = q5_0_uniform(d_bits, qh, qs, n_in, n_out);
        let packed = vnni::repack_q5_0(&data, n_in, n_out);
        let x: Vec<f32> = (0..n_in).map(|i| ((i * 53 % 200) as f32) / 200.0 - 0.5).collect();
        let mut out_ref = vec![0f32; n_out];
        let mut out_v = vec![0f32; n_out];
        matvec(GgmlType::Q5_0, &data, n_in, n_out, &x, 1, &mut out_ref);
        vnni::matvec_vnni(&packed, n_in, n_out, &x, 1, &mut out_v);
        assert!(
            check_close(&out_ref, &out_v, 1e-3),
            "Q5_0 edge '{name}': avx2 vs vnni mismatch"
        );
    }
}

/// Activation boundary cases against random Q8_0 weights (the quantizer is
/// shared by both paths, so the packed path must stay on the reference).
#[test]
fn vnni_activation_edge_cases() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    let (n_in, n_out) = (896usize, 64usize);
    let data = make_q8_0(n_in, n_out);
    let packed = vnni::repack_q8_0(&data, n_in, n_out);
    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("all-zero activation", vec![0f32; n_in]),
        ("constant activation", vec![1.5f32; n_in]),
        ("large-magnitude activation (1e4)", (0..n_in).map(|i| if i % 2 == 0 { 1e4 } else { -1e4 }).collect()),
        ("tiny activation (1e-7)", vec![1e-7f32; n_in]),
        ("one-hot activation", {
            let mut v = vec![0f32; n_in];
            v[17] = 1.0;
            v
        }),
    ];
    for (name, x) in cases {
        let mut out_ref = vec![0f32; n_out];
        let mut out_v = vec![0f32; n_out];
        matvec(GgmlType::Q8_0, &data, n_in, n_out, &x, 1, &mut out_ref);
        vnni::matvec_vnni(&packed, n_in, n_out, &x, 1, &mut out_v);
        assert!(
            check_close(&out_ref, &out_v, 1e-3),
            "activation edge '{name}': avx2 vs vnni mismatch"
        );
    }
}

/// Repack gating: unsupported types/shapes must return None (fallback path).
#[test]
fn vnni_repack_gating_fallback() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    let (n_in, n_out) = (64usize, 64usize);
    // supported type + legal shape → Some
    let data_q8 = q8_0_uniform(0x3C00, 0x11, n_in, n_out);
    assert!(vnni::repack_tensor(GgmlType::Q8_0, &data_q8, n_in, n_out).is_some());
    // unsupported type (K-quants outside the Phase 6.3 scope) → None
    assert!(vnni::repack_tensor(GgmlType::Q4_K, &data_q8, n_in, n_out).is_none());
    assert!(vnni::repack_tensor(GgmlType::Q6_K, &data_q8, n_in, n_out).is_none());
    // n_out not a multiple of 16 → None (fallback, never a panic)
    assert!(vnni::repack_tensor(GgmlType::Q8_0, &data_q8, n_in, 17).is_none());
    // n_in not a multiple of 32 → None
    assert!(vnni::repack_tensor(GgmlType::Q8_0, &data_q8, 48, n_out).is_none());
    assert!(vnni::repack_tensor(GgmlType::Q8_0, &data_q8, 48, 17).is_none());
}

/// Fused pair under the worker pool (2 threads) must match the sequential
/// (1-thread) computation — the slot dispatch must not corrupt outputs.
#[test]
fn vnni_fused_parallel_matches_sequential() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    let (n_in, n_out1, n_out2) = (2048usize, 128usize, 96usize);
    let d1 = make_q8_0(n_in, n_out1);
    let d2 = make_q8_0(n_in, n_out2);
    let p1 = vnni::repack_q8_0(&d1, n_in, n_out1);
    let p2 = vnni::repack_q8_0(&d2, n_in, n_out2);
    let mut st = 0x9e37_79b9u64;
    let x: Vec<f32> = (0..n_in)
        .map(|_| (xr(&mut st) % 2000) as f32 / 2000.0 - 0.5)
        .collect();
    let (mut a1, mut a2) = (vec![0f32; n_out1], vec![0f32; n_out2]);
    let (mut b1, mut b2) = (vec![0f32; n_out1], vec![0f32; n_out2]);
    vnni::matvec2_vnni(&p1, n_out1, &p2, n_out2, n_in, &x, 1, &mut a1, &mut a2);
    vnni::matvec2_vnni(&p1, n_out1, &p2, n_out2, n_in, &x, 2, &mut b1, &mut b2);
    assert!(check_close(&a1, &b1, 1e-6), "fused parallel out1 mismatch");
    assert!(check_close(&a2, &b2, 1e-6), "fused parallel out2 mismatch");
}
