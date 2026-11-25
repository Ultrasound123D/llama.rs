//! AVX-512 VNNI packed-backend differential tests (Phase 6.3/6.4 gates).
//!
//! The repacked-VNNI path must match the existing AVX2 int-domain path:
//! integer dot products are exact on both sides; only the f32 scale
//! application order differs, so a small relative tolerance applies.
//! Coverage: randomized tensors (deterministic LCG, multiple repetitions)
//! AND edge/boundary cases (minimum shapes, zero/extreme weights and
//! scales, qh patterns, activation extremes, repack gating, fused and
//! parallel determinism) — for the simple quants (Q8_0/Q5_0) and, since
//! Phase 6.4, the K-quants (Q4_K/Q6_K) with their q8_K activations.

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
        vnni::matvec_vnni(&packed, GgmlType::Q8_0, n_in, n_out, &x, 1, &mut out_vnni);
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
        vnni::matvec_vnni(&packed, GgmlType::Q5_0, n_in, n_out, &x, 1, &mut out_vnni);
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
    vnni::matvec_vnni(&p1, GgmlType::Q5_0, n_in, n_out1, &x, 1, &mut o1a);
    vnni::matvec_vnni(&p2, GgmlType::Q5_0, n_in, n_out2, &x, 1, &mut o2a);
    let mut o1b = vec![0f32; n_out1];
    let mut o2b = vec![0f32; n_out2];
    vnni::matvec2_vnni(&p1, GgmlType::Q5_0, n_out1, &p2, GgmlType::Q5_0, n_out2, n_in, &x, 1, &mut o1b, &mut o2b);
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
    vnni::matvec_vnni(&packed, GgmlType::Q8_0, n_in, n_out, &x, 1, &mut out_seq);
    vnni::matvec_vnni(&packed, GgmlType::Q8_0, n_in, n_out, &x, 2, &mut out_par);
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
        vnni::matvec_vnni(&packed, GgmlType::Q8_0, n_in, n_out, &x, 1, &mut out_v);
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
        vnni::matvec_vnni(&packed, GgmlType::Q5_0, n_in, n_out, &x, 1, &mut out_v);
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
        vnni::matvec_vnni(&packed, GgmlType::Q8_0, n_in, n_out, &x, 1, &mut out_v);
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
    // unsupported type → None
    assert!(vnni::repack_tensor(GgmlType::F32, &data_q8, n_in, n_out).is_none());
    assert!(vnni::repack_tensor(GgmlType::Q4_0, &data_q8, n_in, n_out).is_none());
    // n_out not a multiple of 16 → None (fallback, never a panic)
    assert!(vnni::repack_tensor(GgmlType::Q8_0, &data_q8, n_in, 17).is_none());
    // n_in not a multiple of 32 → None
    assert!(vnni::repack_tensor(GgmlType::Q8_0, &data_q8, 48, n_out).is_none());
    assert!(vnni::repack_tensor(GgmlType::Q8_0, &data_q8, 48, 17).is_none());
    // K-quants: legal 256-multiple shape → Some; illegal → None
    let (ki, ko) = (256usize, 16usize);
    let data_q4k = q4_k_uniform(0x3C00, 0x3C00, 0xFF, 0xFF, ki, ko);
    let data_q6k = q6_k_uniform(0x3C00, 0x7F, ki, ko);
    assert!(vnni::repack_tensor(GgmlType::Q4_K, &data_q4k, ki, ko).is_some());
    assert!(vnni::repack_tensor(GgmlType::Q6_K, &data_q6k, ki, ko).is_some());
    // n_in not a multiple of 256 → None
    assert!(vnni::repack_tensor(GgmlType::Q4_K, &data_q4k, 64, ko).is_none());
    assert!(vnni::repack_tensor(GgmlType::Q6_K, &data_q6k, 64, ko).is_none());
    // n_out not a multiple of 16 → None
    assert!(vnni::repack_tensor(GgmlType::Q4_K, &data_q4k, ki, 17).is_none());
    assert!(vnni::repack_tensor(GgmlType::Q6_K, &data_q6k, ki, 17).is_none());
    // the returned PackedTensor carries the type
    let p = vnni::repack_tensor(GgmlType::Q4_K, &data_q4k, ki, ko).unwrap();
    assert_eq!(p.ty, GgmlType::Q4_K);
    assert_eq!(p.n_in, ki);
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
    vnni::matvec2_vnni(&p1, GgmlType::Q8_0, n_out1, &p2, GgmlType::Q8_0, n_out2, n_in, &x, 1, &mut a1, &mut a2);
    vnni::matvec2_vnni(&p1, GgmlType::Q8_0, n_out1, &p2, GgmlType::Q8_0, n_out2, n_in, &x, 2, &mut b1, &mut b2);
    assert!(check_close(&a1, &b1, 1e-6), "fused parallel out1 mismatch");
    assert!(check_close(&a2, &b2, 1e-6), "fused parallel out2 mismatch");
}

// ---------- K-quants (Phase 6.4): Q4_K / Q6_K packed path vs AVX2 ----------

/// Random Q4_K tensor: valid block structure (d/dmin f16 + 12 scale bytes +
/// 128 qs bytes), random content.
fn make_q4_k(n_in: usize, n_out: usize, seed: u64) -> Vec<u8> {
    let kb = n_in / 256;
    let mut st = seed;
    let mut data = vec![0u8; n_out * kb * 144];
    for b in data.chunks_exact_mut(144) {
        // d, dmin: safe positive f16 ranges
        let d = 0.4 + (xr(&mut st) % 64) as f32 / 64.0;
        let dmin = 0.05 + (xr(&mut st) % 32) as f32 / 256.0;
        b[0..2].copy_from_slice(&llama_kernels::quant::f32_to_f16_bits_sw(d).to_le_bytes());
        b[2..4].copy_from_slice(&llama_kernels::quant::f32_to_f16_bits_sw(dmin).to_le_bytes());
        for i in 4..144 {
            b[i] = (xr(&mut st) & 0xFF) as u8;
        }
    }
    data
}

/// Random Q6_K tensor: ql(128) + qh(64) + 16 i8 scales + d f16.
fn make_q6_k(n_in: usize, n_out: usize, seed: u64) -> Vec<u8> {
    let kb = n_in / 256;
    let mut st = seed;
    let mut data = vec![0u8; n_out * kb * 210];
    for b in data.chunks_exact_mut(210) {
        let d = 0.3 + (xr(&mut st) % 64) as f32 / 128.0;
        b[208..210].copy_from_slice(&llama_kernels::quant::f32_to_f16_bits_sw(d).to_le_bytes());
        for i in 0..192 {
            b[i] = (xr(&mut st) & 0xFF) as u8;
        }
        for i in 192..208 {
            b[i] = (xr(&mut st) % 256) as u8; // i8 scales incl. negatives
        }
    }
    data
}

/// Uniform Q4_K tensor: same d/dmin f16 bits, scale/min bytes and qs bytes
/// for every block.
fn q4_k_uniform(d_bits: u16, dmin_bits: u16, scales: u8, qs: u8, n_in: usize, n_out: usize) -> Vec<u8> {
    let kb = n_in / 256;
    let mut data = vec![0u8; n_out * kb * 144];
    for b in data.chunks_exact_mut(144) {
        b[0..2].copy_from_slice(&d_bits.to_le_bytes());
        b[2..4].copy_from_slice(&dmin_bits.to_le_bytes());
        b[4..16].fill(scales);
        b[16..144].fill(qs);
    }
    data
}

/// Uniform Q6_K tensor: same ql/qh bytes, i8 scale and d for every block.
fn q6_k_uniform(d_bits: u16, scale: i8, n_in: usize, n_out: usize) -> Vec<u8> {
    let kb = n_in / 256;
    let mut data = vec![0u8; n_out * kb * 210];
    for b in data.chunks_exact_mut(210) {
        for i in 0..192 {
            b[i] = 0xA5;
        }
        b[128..192].fill(0x5A);
        for i in 192..208 {
            b[i] = scale as u8;
        }
        b[208..210].copy_from_slice(&d_bits.to_le_bytes());
    }
    data
}

fn kq_case(ty: GgmlType, data: &[u8], n_in: usize, n_out: usize, x: &[f32], tag: &str) {
    let mut out_ref = vec![0f32; n_out];
    let mut out_v = vec![0f32; n_out];
    matvec(ty, data, n_in, n_out, x, 1, &mut out_ref);
    let packed = match ty {
        GgmlType::Q4_K => vnni::repack_q4_k(data, n_in, n_out),
        GgmlType::Q6_K => vnni::repack_q6_k(data, n_in, n_out),
        _ => unreachable!(),
    };
    vnni::matvec_vnni(&packed, ty, n_in, n_out, x, 1, &mut out_v);
    assert!(
        check_close(&out_ref, &out_v, 1e-3),
        "{tag}: avx2 vs packed vnni mismatch"
    );
}

#[test]
fn vnni_q4_k_matches_avx2() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    for rep in 0..5 {
        let (n_in, n_out) = (2048usize, 304 + rep * 16); // LFM3-realistic n_in
        let data = make_q4_k(n_in, n_out, 0xfeed_face + rep as u64);
        let x: Vec<f32> = (0..n_in)
            .map(|i| (((i * 37 + rep) % 2000) as f32) / 2000.0 - 0.5)
            .collect();
        kq_case(GgmlType::Q4_K, &data, n_in, n_out, &x, "Q4_K random");
    }
}

#[test]
fn vnni_q6_k_matches_avx2() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    for rep in 0..5 {
        let (n_in, n_out) = (7680usize, 256 + rep * 32); // LFM3 ffn_down n_in
        let data = make_q6_k(n_in, n_out, 0xbead_cafe + rep as u64);
        let x: Vec<f32> = (0..n_in)
            .map(|i| (((i * 53 + rep) % 2000) as f32) / 2000.0 - 0.5)
            .collect();
        kq_case(GgmlType::Q6_K, &data, n_in, n_out, &x, "Q6_K random");
    }
}

/// Minimum-shape + extreme K-quant weights vs the AVX2 reference path.
#[test]
fn vnni_kq_edge_min_shape_and_extremes() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    let (n_in, n_out) = (256usize, 16usize); // one super-block, one group
    let f16b = |v: f32| llama_kernels::quant::f32_to_f16_bits_sw(v);
    let d1 = f16b(1.0);
    // (name, d_bits, dmin_bits, scales12, qs)
    let q4k_cases: Vec<(&str, u16, u16, u8, u8)> = vec![
        ("zero weights (d=0, qs=0)", 0, 0, 0x00, 0x00),
        ("max d/dmin, max 6-bit scales, qs=0xFF", 0x7BFF, 0x7BFF, 0xFF, 0xFF),
        ("max d, d=0 with max scales", 0x7BFF, 0, 0xFF, 0xA5),
        ("min-normal d, alternating qs", 0x0400, 0x0400, 0x3F, 0x5A),
        ("d=1, scales=0 (sub-dots scaled by 0)", d1, d1, 0x00, 0xFF),
    ];
    let x: Vec<f32> = (0..n_in).map(|i| ((i * 37 % 200) as f32) / 200.0 - 0.5).collect();
    for (name, d, dmin, sc, qs) in q4k_cases {
        let data = q4_k_uniform(d, dmin, sc, qs, n_in, n_out);
        kq_case(GgmlType::Q4_K, &data, n_in, n_out, &x, &format!("Q4_K edge '{name}'"));
        if name.starts_with("zero weights") {
            let packed = vnni::repack_q4_k(&data, n_in, n_out);
            let mut out = vec![0f32; n_out];
            vnni::matvec_vnni(&packed, GgmlType::Q4_K, n_in, n_out, &x, 1, &mut out);
            assert!(out.iter().all(|&v| v == 0.0), "Q4_K zero weights must be exactly 0");
        }
    }
    // Q6_K: i8 scale extremes (±127), d extremes, qh patterns
    let q6k_cases: Vec<(&str, u16, i8)> = vec![
        ("max d, scale=+127", 0x7BFF, 127),
        ("max d, scale=-127", 0x7BFF, -127),
        ("min-normal d, scale=1", 0x0400, 1),
        ("d=1, scale=0", f16b(1.0), 0),
    ];
    for (name, d, sc) in q6k_cases {
        let data = q6_k_uniform(d, sc, n_in, n_out);
        kq_case(GgmlType::Q6_K, &data, n_in, n_out, &x, &format!("Q6_K edge '{name}'"));
    }
    // zero Q6_K tensor (d=0) → exactly zero
    let mut zero = vec![0u8; n_out * n_in / 256 * 210];
    let packed = vnni::repack_q6_k(&zero, n_in, n_out);
    let mut out = vec![0f32; n_out];
    vnni::matvec_vnni(&packed, GgmlType::Q6_K, n_in, n_out, &x, 1, &mut out);
    assert!(out.iter().all(|&v| v == 0.0), "Q6_K zero weights must be exactly 0");
}

/// K-quant fused pair (gate/up) must equal the two singles (LFM3's ffn
/// gate+up are both Q4_K — the exact production configuration).
#[test]
fn vnni_kq_fused_pair_matches_two_singles() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    let (n_in, n_out1, n_out2) = (2048usize, 7680usize, 7680usize);
    let d1 = make_q4_k(n_in, n_out1, 0x1234_0001);
    let d2 = make_q4_k(n_in, n_out2, 0x1234_0002);
    let p1 = vnni::repack_q4_k(&d1, n_in, n_out1);
    let p2 = vnni::repack_q4_k(&d2, n_in, n_out2);
    let x: Vec<f32> = (0..n_in)
        .map(|i| (((i * 61) % 2000) as f32) / 2000.0 - 0.5)
        .collect();
    let mut o1a = vec![0f32; n_out1];
    let mut o2a = vec![0f32; n_out2];
    vnni::matvec_vnni(&p1, GgmlType::Q4_K, n_in, n_out1, &x, 1, &mut o1a);
    vnni::matvec_vnni(&p2, GgmlType::Q4_K, n_in, n_out2, &x, 1, &mut o2a);
    let mut o1b = vec![0f32; n_out1];
    let mut o2b = vec![0f32; n_out2];
    vnni::matvec2_vnni(
        &p1, GgmlType::Q4_K, n_out1,
        &p2, GgmlType::Q4_K, n_out2,
        n_in, &x, 1, &mut o1b, &mut o2b,
    );
    assert!(check_close(&o1a, &o1b, 1e-5), "Q4_K fused pair: out1 mismatch");
    assert!(check_close(&o2a, &o2b, 1e-5), "Q4_K fused pair: out2 mismatch");
    // 2-thread fused must match 1-thread fused
    let mut o1c = vec![0f32; n_out1];
    let mut o2c = vec![0f32; n_out2];
    vnni::matvec2_vnni(
        &p1, GgmlType::Q4_K, n_out1,
        &p2, GgmlType::Q4_K, n_out2,
        n_in, &x, 2, &mut o1c, &mut o2c,
    );
    assert!(check_close(&o1b, &o1c, 1e-6), "Q4_K fused parallel out1 mismatch");
    assert!(check_close(&o2b, &o2c, 1e-6), "Q4_K fused parallel out2 mismatch");
}

/// K-quant 2-thread band split must be identical to the sequential run.
#[test]
fn vnni_kq_parallel_matches_sequential() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    let (n_in, n_out) = (2048usize, 4096usize);
    let data = make_q4_k(n_in, n_out, 0x0f0f_0f0f);
    let packed = vnni::repack_q4_k(&data, n_in, n_out);
    let x: Vec<f32> = (0..n_in)
        .map(|_| 0f32)
        .enumerate()
        .map(|(i, _)| (((i * 31) % 2000) as f32) / 2000.0 - 0.5)
        .collect();
    let mut out_seq = vec![0f32; n_out];
    let mut out_par = vec![0f32; n_out];
    vnni::matvec_vnni(&packed, GgmlType::Q4_K, n_in, n_out, &x, 1, &mut out_seq);
    vnni::matvec_vnni(&packed, GgmlType::Q4_K, n_in, n_out, &x, 2, &mut out_par);
    assert!(check_close(&out_seq, &out_par, 1e-6), "Q4_K parallel vs sequential");
}

/// Activation extremes against random K-quant weights (zero / constant /
/// huge / tiny / one-hot) — exercises the q8_K quantizer's zero and ±127
/// branches on both paths.
#[test]
fn vnni_kq_activation_edge_cases() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    let (n_in, n_out) = (512usize, 64usize);
    let data = make_q4_k(n_in, n_out, 0x9e37_79b9);
    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("all-zero activation", vec![0f32; n_in]),
        ("constant activation", vec![1.5f32; n_in]),
        ("large-magnitude activation (1e4)", (0..n_in).map(|i| if i % 2 == 0 { 1e4 } else { -1e4 }).collect()),
        ("tiny activation (1e-7)", vec![1e-7f32; n_in]),
        ("one-hot activation", {
            let mut v = vec![0f32; n_in];
            v[129] = 1.0;
            v
        }),
    ];
    for (name, x) in &cases {
        kq_case(GgmlType::Q4_K, &data, n_in, n_out, x, &format!("Q4_K act edge '{name}'"));
    }
    let data6 = make_q6_k(n_in, n_out, 0x1357_2468);
    for (name, x) in &cases {
        kq_case(GgmlType::Q6_K, &data6, n_in, n_out, x, &format!("Q6_K act edge '{name}'"));
    }
}

// ---------- Phase 6.5: packed GEMM (M>1 batched prefill) ----------

/// Random activation matrix (M × n_in), values in a safe range.
fn make_x(m: usize, n_in: usize, seed: u64) -> Vec<f32> {
    let mut st = seed;
    (0..m * n_in)
        .map(|_| {
            let v = (xr(&mut st) % 4096) as f32 / 512.0 - 4.0;
            v
        })
        .collect()
}

/// The packed GEMM contract: every output row m of `gemm_prepared` is
/// BIT-IDENTICAL to running `matvec_vnni` on activation row m alone (same
/// quantizer, same group-dot accumulation order). Checked exact.
fn gemm_case(
    ty: GgmlType,
    data: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    m: usize,
    threads: usize,
    tag: &str,
) {
    let packed = match ty {
        GgmlType::Q8_0 => vnni::repack_q8_0(data, n_in, n_out),
        GgmlType::Q5_0 => vnni::repack_q5_0(data, n_in, n_out),
        GgmlType::Q4_K => vnni::repack_q4_k(data, n_in, n_out),
        GgmlType::Q6_K => vnni::repack_q6_k(data, n_in, n_out),
        _ => unreachable!(),
    };
    let pt = vnni::PackedTensor {
        data: packed,
        ty,
        n_in,
        n_out,
    };
    let q = vnni::quantize_batch(ty, x, n_in, m, threads);
    let mut out = vec![0f32; m * n_out];
    vnni::gemm_prepared(&pt, &q, threads, &mut out);
    // per-row reference: matvec_vnni on the single row
    let mut ref_row = vec![0f32; n_out];
    let mut mismatches = 0usize;
    let mut max_abs = 0f64;
    for mm in 0..m {
        let row = &x[mm * n_in..(mm + 1) * n_in];
        vnni::matvec_vnni(&pt.data, ty, n_in, n_out, row, 1, &mut ref_row);
        for o in 0..n_out {
            let a = out[mm * n_out + o];
            let b = ref_row[o];
            if a.to_bits() != b.to_bits() {
                mismatches += 1;
                max_abs = max_abs.max((a - b).abs() as f64);
            }
        }
    }
    assert!(
        mismatches == 0,
        "{tag}: GEMM vs matvec mismatch ({mismatches} of {} entries, max_abs {max_abs:.3e})",
        m * n_out
    );
    eprintln!("  {tag}: M={m} threads={threads} — bit-identical to matvec_vnni");
}

#[test]
fn vnni_gemm_matches_matvec_all_types() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    // one small + one model-realistic shape per family; M sweep incl. odd
    // sizes (non-multiples of the tile/quant blocks) and a big batch
    let shapes: Vec<(&str, GgmlType, usize, usize)> = vec![
        ("Q8_0 64x32", GgmlType::Q8_0, 64, 32),
        ("Q8_0 2048x64", GgmlType::Q8_0, 2048, 64),
        ("Q5_0 64x32", GgmlType::Q5_0, 64, 32),
        ("Q5_0 896x4864", GgmlType::Q5_0, 896, 4864),
        ("Q4_K 256x32", GgmlType::Q4_K, 256, 32),
        ("Q4_K 2048x7680", GgmlType::Q4_K, 2048, 7680),
        ("Q6_K 256x32", GgmlType::Q6_K, 256, 32),
        ("Q6_K 7680x2048", GgmlType::Q6_K, 7680, 2048),
    ];
    for (name, ty, n_in, n_out) in &shapes {
        let (data, seed) = match ty {
            GgmlType::Q8_0 => (make_q8_0(*n_in, *n_out), 0x1111_2222u64),
            GgmlType::Q5_0 => (make_q5_0(*n_in, *n_out), 0x3333_4444u64),
            GgmlType::Q4_K => (make_q4_k(*n_in, *n_out, 0x5555_6666u64), 0x5555_6666u64),
            GgmlType::Q6_K => (make_q6_k(*n_in, *n_out, 0x7777_8888u64), 0x7777_8888u64),
            _ => unreachable!(),
        };
        for m in [1usize, 2, 3, 8, 17, 64] {
            let x = make_x(m, *n_in, seed.wrapping_add(m as u64));
            gemm_case(*ty, &data, *n_in, *n_out, &x, m, 1, name);
            if m >= 3 {
                // 2-thread band split must produce the identical matrix
                gemm_case(*ty, &data, *n_in, *n_out, &x, m, 2, name);
            }
        }
    }
}

#[test]
fn vnni_gemm_fused_pair_matches_two_gems() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    let (n_in, n_out1, n_out2, m) = (256usize, 32usize, 48usize, 9usize);
    let d1 = make_q4_k(n_in, n_out1, 0xaaaa_1010u64);
    let d2 = make_q4_k(n_in, n_out2, 0xbbbb_2020u64);
    let x = make_x(m, n_in, 0xcccc_3030u64);
    let p1 = vnni::PackedTensor {
        data: vnni::repack_q4_k(&d1, n_in, n_out1),
        ty: GgmlType::Q4_K,
        n_in,
        n_out: n_out1,
    };
    let p2 = vnni::PackedTensor {
        data: vnni::repack_q4_k(&d2, n_in, n_out2),
        ty: GgmlType::Q4_K,
        n_in,
        n_out: n_out2,
    };
    let q = vnni::quantize_batch(GgmlType::Q4_K, &x, n_in, m, 2);
    let mut o1 = vec![0f32; m * n_out1];
    let mut o2 = vec![0f32; m * n_out2];
    vnni::gemm2_prepared(&p1, &p2, &q, 2, &mut o1, &mut o2);
    let mut r1 = vec![0f32; m * n_out1];
    let mut r2 = vec![0f32; m * n_out2];
    vnni::gemm_prepared(&p1, &q, 1, &mut r1);
    vnni::gemm_prepared(&p2, &q, 1, &mut r2);
    for i in 0..o1.len() {
        assert_eq!(o1[i].to_bits(), r1[i].to_bits(), "fused pair out1 mismatch @ {i}");
    }
    for i in 0..o2.len() {
        assert_eq!(o2[i].to_bits(), r2[i].to_bits(), "fused pair out2 mismatch @ {i}");
    }
    eprintln!("  fused GEMM pair (Q4_K 256 -> 32/48, M=9, 2t): bit-identical to two singles");
}

#[test]
fn vnni_gemm_zero_rows_and_zero_weights() {
    if !vnni::vnni_path_available() {
        eprintln!("SKIP (no AVX512-VNNI)");
        return;
    }
    let (n_in, n_out, m) = (256usize, 32usize, 5usize);
    let data = q4_k_uniform(0, 0, 0, 0, n_in, n_out); // all-zero weights
    let x = make_x(m, n_in, 0x0bad_f00du64);
    gemm_case(GgmlType::Q4_K, &data, n_in, n_out, &x, m, 1, "Q4_K zero weights");
    let xz = vec![0f32; m * n_in];
    let data2 = make_q4_k(n_in, n_out, 0x1234_5678u64);
    gemm_case(GgmlType::Q4_K, &data2, n_in, n_out, &xz, m, 1, "Q4_K zero activations");
    // M=0 must be a no-op (does not touch out)
    let pt = vnni::PackedTensor {
        data: vnni::repack_q4_k(&data2, n_in, n_out),
        ty: GgmlType::Q4_K,
        n_in,
        n_out,
    };
    let q = vnni::quantize_batch(GgmlType::Q4_K, &[], n_in, 0, 1);
    let mut out = vec![-1f32; 0];
    vnni::gemm_prepared(&pt, &q, 1, &mut out);
    assert!(out.is_empty());
    eprintln!("  zero weights / zero activations / M=0: OK");
}

// ---------- Phase 6.5: batched f32 SIMD helpers ----------

#[test]
fn batch_helpers_match_f64_reference() {
    use llama_kernels::batch;
    // dot vs f64 sequential reference
    let mut st = 0x5f3a_b1c2u64;
    let n = 300usize;
    let a: Vec<f32> = (0..n).map(|_| (xr(&mut st) % 2048) as f32 / 256.0 - 4.0).collect();
    let b: Vec<f32> = (0..n).map(|_| (xr(&mut st) % 2048) as f32 / 256.0 - 4.0).collect();
    let mut refd = 0f64;
    for i in 0..n {
        refd += a[i] as f64 * b[i] as f64;
    }
    let got = batch::dot_f32(&a, &b);
    let rel = ((got as f64 - refd) / refd.abs().max(1e-3)).abs();
    assert!(rel < 1e-5, "dot_f32 rel {rel:.3e}");
    // softmax vs f64 stable reference
    let mut scores: Vec<f32> = (0..257).map(|_| (xr(&mut st) % 4096) as f32 / 256.0 - 8.0).collect();
    let mut c = scores.clone();
    batch::softmax_inplace_f32(&mut scores);
    let max = c.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f64;
    for v in c.iter_mut() {
        *v = ((*v - max) as f64).exp() as f32;
        sum += *v as f64;
    }
    for v in c.iter_mut() {
        *v = (*v as f64 / sum) as f32;
    }
    let mut max_rel = 0f64;
    for (x, y) in scores.iter().zip(c.iter()) {
        let d = (*x - *y).abs() as f64;
        let m = x.abs().max(y.abs()).max(1e-6) as f64;
        max_rel = max_rel.max(d / m);
    }
    assert!(max_rel < 1e-5, "softmax rel {max_rel:.3e}");
    eprintln!("  batch helpers: dot rel {rel:.2e}, softmax rel {max_rel:.2e} (vs f64 reference)");
    // axpy
    let v = vec![1.5f32; 128];
    let mut o = vec![0.25f32; 128];
    batch::axpy_f32(2.0, &v, &mut o);
    for x in &o {
        assert_eq!(*x, 3.25);
    }
}
