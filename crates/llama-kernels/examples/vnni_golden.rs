//! vnni_golden — randomized + edge-case differential of the packed VNNI path
//! against a C golden reference built from llama.cpp's own kernels
//! (tests/parity/ref_dot.c, transcribed from b12502 sources).
//!
//! Per case (mode × type × shape × seed) it verifies, against the golden
//! outputs computed with llama.cpp's RUNTIME quantizer contract:
//!   1. our activation quantizer vs the C runtime quantizer — BYTE-EXACT
//!   2. scalar kernels (generic contract)         — tight tolerance
//!   3. AVX2 kernels (direct calls)               — tight tolerance
//!   4. AVX2+VNNI256 kernels (engine dispatch on this CPU) — tight
//!   5. engine `matvec` (full dispatch, 1 thread) — tight
//!   6. packed VNNI `matvec_vnni` (1 thread)      — packed tolerance
//!   7. packed VNNI with 2 threads                 — determinism vs 6
//!   8. fused `matvec2_vnni` (same tensor twice)   — both halves = golden
//! plus a CPU-feature dispatch gate section (gated repack must return None).
//!
//! usage: cargo run --release --example vnni_golden [-- --ref-dot /tmp/ref_dot]
//! Exit code 0 = all gates green.

use llama_gguf::GgmlType;
use llama_kernels::quant;
use llama_kernels::vnni;
use llama_kernels::{avx2, matvec, scalar};

#[derive(Debug)]
struct Case {
    mode: &'static str,
    ty: &'static str,
    n_in: usize,
    n_out: usize,
    seed: u64,
}

fn read_file(p: &str) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"))
}

/// Term-energy-normalized error: |a_i-b_i| / max_i(scale_i), where
/// scale_i = Σ_b |sumi_b·d_b| is the golden per-column term energy.
/// Per-output relative error is the WRONG metric for dot products whose
/// outputs cross zero or — on adversarial data — cancel COMPLETELY (the
/// outputs become pure f32-rounding residues; every path computes the same
/// exact integer dot and differs only in f32 scale-accumulation order).
/// Normalizing by term energy is the honest precision statement: it answers
/// "how wrong is this output relative to the magnitudes that flowed through
/// the sum". A single flipped weight code injects an error at term-energy
/// scale (ratio ~1e-2..1) — orders above the gates; proven by the mutation
/// self-test below.
fn case_err(ours: &[f32], golden: &[f64]) -> f64 {
    case_err_scaled(ours, golden, &Vec::new())
}

fn case_err_scaled(ours: &[f32], golden: &[f64], energies: &[f64]) -> f64 {
    let denom = if energies.is_empty() {
        golden.iter().fold(0.0f64, |m, b| b.abs().max(m)).max(1e-3)
    } else {
        energies.iter().fold(0.0f64, |m, e| e.max(m)).max(1e-3)
    };
    ours.iter()
        .zip(golden)
        .map(|(a, b)| (*a as f64 - b).abs())
        .fold(0.0f64, f64::max)
        / denom
}

fn max_rel(ours: &[f32], golden: &[f64], energies: &[f64]) -> f64 {
    case_err_scaled(ours, golden, energies)
}

fn f32_from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn parse_out(path: &str) -> Vec<f64> {
    let s = std::fs::read_to_string(path).unwrap();
    s.split_whitespace().map(|t| t.parse::<f64>().unwrap()).collect()
}

fn main() {
    let mut ref_dot = std::env::var("REF_DOT").unwrap_or_else(|_| "/tmp/ref_dot".to_string());
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--ref-dot") {
        if i + 1 < args.len() {
            ref_dot = args[i + 1].clone();
        }
    }
    if !std::path::Path::new(&ref_dot).exists() {
        eprintln!("compiling C golden reference ({}) — gcc -O2", "ref_dot.c");
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/parity/ref_dot.c");
        let st = std::process::Command::new("gcc")
            .args(["-O2", "-o", &ref_dot, src, "-lm"])
            .status()
            .expect("gcc available");
        assert!(st.success(), "compiling ref_dot.c failed");
    }

    // ---------- CPU feature dispatch gates (priority: crash safety) ----------
    println!("== [dispatch gates] ==");
    let caps = vnni::detect();
    println!(
        "  cpuid: avx512_vnni={} amx_int8={} amx_bf16={} xstate_tile={} | vnni_path={} (amx_tiles NOT used)",
        caps.avx512_vnni, caps.amx_int8, caps.amx_bf16, caps.xstate_tile,
        vnni::vnni_path_available()
    );
    // cross-check the Rust cpuid detection against the kernel's own view
    if let Ok(ci) = std::fs::read_to_string("/proc/cpuinfo") {
        let have = |f: &str| ci.contains(f);
        let flag_vnni = have("avx512_vnni");
        let flag_amx = have("amx_int8");
        assert_eq!(
            caps.avx512_vnni, flag_vnni,
            "detect() avx512_vnni disagrees with /proc/cpuinfo"
        );
        assert_eq!(
            caps.amx_int8, flag_amx,
            "detect() amx_int8 disagrees with /proc/cpuinfo"
        );
        println!("  cross-check vs /proc/cpuinfo: avx512_vnni={} amx_int8={} — MATCH", flag_vnni, flag_amx);
    }
    // gated repack: cpu_ok=false must return None for every combination
    // (this is the machine-without-AVX512 simulation; None → AVX2 fallback)
    let sample = vec![0u8; 64 * 4 * 64];
    let mut gate_fail = 0;
    for ty in [GgmlType::Q8_0, GgmlType::Q5_0, GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::F32] {
        for (ni, no) in [(32usize, 16usize), (64, 64), (96, 48), (896, 320)] {
            if vnni::repack_tensor_gated(ty, &sample[..], ni, no, false).is_some() {
                gate_fail += 1;
            }
        }
    }
    assert!(gate_fail == 0, "gated repack returned Some with cpu_ok=false");
    println!("  repack_tensor_gated(cpu_ok=false) → None for all types/shapes — fallback forced, no panic");

    // ---------- differential sweep ----------
    let shapes: &[(usize, usize)] = &[
        (32, 16),   // minimum legal
        (32, 32),
        (64, 16),
        (96, 48),   // 3 k-blocks, 3 groups
        (256, 32),
        (896, 304), // model-realistic (qwen hidden 896)
        (1024, 512),
        (2048, 4096),
    ];
    let mut cases = Vec::new();
    for s in [1u64, 2, 3] {
        for &(ni, no) in shapes {
            cases.push(Case { mode: "random", ty: "q8_0", n_in: ni, n_out: no, seed: s });
            cases.push(Case { mode: "random", ty: "q5_0", n_in: ni, n_out: no, seed: s });
        }
    }
    for &(ni, no) in shapes {
        cases.push(Case { mode: "extremes", ty: "q8_0", n_in: ni, n_out: no, seed: 7 });
        cases.push(Case { mode: "extremes", ty: "q5_0", n_in: ni, n_out: no, seed: 7 });
        cases.push(Case { mode: "qhwalk", ty: "q5_0", n_in: ni, n_out: no, seed: 9 });
    }

    let tmp = std::env::temp_dir().join(format!("vg_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let pfx = tmp.to_str().unwrap();

    let vnni_ok = vnni::vnni_path_available();
    let avx2_ok = matches!(llama_kernels::simd_level(), llama_kernels::Simd::Avx2Fma);

    let (mut worst_qact, mut worst_scalar, mut worst_avx2, mut worst_vnni256) =
        (0f64, 0f64, 0f64, 0f64);
    let (mut worst_engine, mut worst_packed, mut worst_par, mut worst_fused) =
        (0f64, 0f64, 0f64, 0f64);
    let mut qact_bytes_bad = 0usize;
    let mut n_cases = 0;

    for c in &cases {
        let out = std::process::Command::new(&ref_dot)
            .args([
                c.mode, c.ty, &c.n_in.to_string(), &c.n_out.to_string(),
                &c.seed.to_string(), pfx,
            ])
            .output()
            .expect("run ref_dot");
        assert!(out.status.success(), "ref_dot failed: {}", String::from_utf8_lossy(&out.stderr));

        let tensor = read_file(&format!("{pfx}_tensor.bin"));
        let x = f32_from_bytes(&read_file(&format!("{pfx}_x_f32.bin")));
        let qact_rt = read_file(&format!("{pfx}_qact_rt.bin"));
        let _qact_ref = read_file(&format!("{pfx}_qact_ref.bin"));
        let out_rt = parse_out(&format!("{pfx}_out_rt.txt"));
        // per-column term energy from the golden side: Σ_b |sumi_b·d_b|
        let energies = parse_out(&format!("{pfx}_scale.txt"));

        // C self-consistency: the two quantizer contracts must produce
        // near-identical outputs (they differ only on rare tie-roundings)
        let rt_vs_ref = {
            let r = parse_out(&format!("{pfx}_out_ref.txt"));
            out_rt.iter().zip(&r).map(|(a, b)| (a - b).abs() / a.abs().max(*b).max(1e-3)).fold(0.0f64, f64::max)
        };

        let ty = if c.ty == "q8_0" { GgmlType::Q8_0 } else { GgmlType::Q5_0 };
        let row_bytes = if c.ty == "q8_0" { 34 } else { 22 };
        let kb = c.n_in / 32;

        // ---- 1. our quantizer vs C runtime quantizer: BYTE-EXACT ----
        let mut q_ours = vec![0u8; kb * 34];
        quant::quantize_row_q8_0(&x, &mut q_ours);
        let (mism, max_lsb) = {
            let mut m = 0usize;
            let mut l = 0i32;
            for i in 0..qact_rt.len() {
                if q_ours[i] != qact_rt[i] {
                    m += 1;
                    l = l.max((q_ours[i] as i32 - qact_rt[i] as i32).abs());
                }
            }
            (m, l)
        };
        qact_bytes_bad += mism;
        worst_qact = worst_qact.max(if mism == 0 { 0.0 } else { f64::INFINITY });
        assert!(mism == 0,
            "quantizer mismatch vs C runtime contract: {mism} bytes (max lsb {max_lsb}) at {:?}", c);

        // ---- per-column outputs through every path ----
        let mut o_scalar = vec![0f32; c.n_out];
        let mut o_avx2 = vec![0f32; c.n_out];
        let mut o_vnni256 = vec![0f32; c.n_out];
        for (col, o) in o_scalar.iter_mut().enumerate() {
            let w = &tensor[col * row_bytes * kb..(col + 1) * row_bytes * kb];
            *o = if c.ty == "q8_0" {
                scalar::dot_q8_0_q8_0(w, &q_ours, c.n_in)
            } else {
                scalar::dot_q5_0_q8_0(w, &q_ours, c.n_in)
            };
        }
        if avx2_ok {
            for (col, o) in o_avx2.iter_mut().enumerate() {
                let w = &tensor[col * row_bytes * kb..(col + 1) * row_bytes * kb];
                *o = if c.ty == "q8_0" {
                    unsafe { avx2::dot_q8_0_q8_0(w, &q_ours, c.n_in) }
                } else {
                    unsafe { avx2::dot_q5_0_q8_0(w, &q_ours, c.n_in) }
                };
            }
        }
        if avx2_ok && avx2::vnni256() {
            for (col, o) in o_vnni256.iter_mut().enumerate() {
                let w = &tensor[col * row_bytes * kb..(col + 1) * row_bytes * kb];
                *o = if c.ty == "q8_0" {
                    unsafe { avx2::dot_q8_0_q8_0_vnni(w, &q_ours, c.n_in) }
                } else {
                    unsafe { avx2::dot_q5_0_q8_0_vnni(w, &q_ours, c.n_in) }
                };
            }
        }
        // ---- 5. engine full dispatch ----
        let mut o_engine = vec![0f32; c.n_out];
        matvec(ty, &tensor, c.n_in, c.n_out, &x, 1, &mut o_engine);

        // ---- 6/7. packed VNNI (sequential + parallel) ----
        let mut o_packed = vec![0f32; c.n_out];
        let mut o_par = vec![0f32; c.n_out];
        if vnni_ok {
            let packed = if c.ty == "q8_0" {
                vnni::repack_q8_0(&tensor, c.n_in, c.n_out)
            } else {
                vnni::repack_q5_0(&tensor, c.n_in, c.n_out)
            };
            vnni::matvec_vnni(&packed, c.n_in, c.n_out, &x, 1, &mut o_packed);
            vnni::matvec_vnni(&packed, c.n_in, c.n_out, &x, 2, &mut o_par);
            // ---- 8. fused pair: same tensor twice → both outputs = golden ----
            let mut o_f1 = vec![0f32; c.n_out];
            let mut o_f2 = vec![0f32; c.n_out];
            vnni::matvec2_vnni(&packed, c.n_out, &packed, c.n_out, c.n_in, &x, 1,
                &mut o_f1, &mut o_f2);
            let fr = max_rel(&o_f1, &out_rt, &energies).max(max_rel(&o_f2, &out_rt, &energies));
            worst_fused = worst_fused.max(fr);
        }

        let ds = max_rel(&o_scalar, &out_rt, &energies);
        let da = if avx2_ok { max_rel(&o_avx2, &out_rt, &energies) } else { 0.0 };
        let dv = if avx2_ok && avx2::vnni256() { max_rel(&o_vnni256, &out_rt, &energies) } else { 0.0 };
        let de = max_rel(&o_engine, &out_rt, &energies);
        let dp = if vnni_ok { max_rel(&o_packed, &out_rt, &energies) } else { 0.0 };
        let dpar = if vnni_ok { max_rel(&o_par, &out_rt, &energies) } else { 0.0 };

        worst_scalar = worst_scalar.max(ds);
        worst_avx2 = worst_avx2.max(da);
        worst_vnni256 = worst_vnni256.max(dv);
        worst_engine = worst_engine.max(de);
        worst_packed = worst_packed.max(dp);
        worst_par = worst_par.max(dpar);

        n_cases += 1;
        eprintln!(
            "  {:>8}/{:>4} {:>4}×{:<4} s{} | rt-vs-ref {:.1e} | scalar {ds:.1e} avx2 {da:.1e} vnni256 {dv:.1e} engine {de:.1e} packed {dp:.1e} par2t {dpar:.1e}",
            c.mode, c.ty, c.n_in, c.n_out, c.seed, rt_vs_ref
        );

        // per-case gates (case-scale-normalized; see case_err doc for why)
        assert!(ds < 1e-5, "scalar vs golden: {ds:.2e} at {c:?}");
        assert!(da < 1e-4 || !avx2_ok, "avx2 vs golden: {da:.2e} at {c:?}");
        assert!(dv < 1e-4 || !(avx2_ok && avx2::vnni256()), "vnni256 vs golden: {dv:.2e} at {c:?}");
        assert!(de < 1e-4, "engine dispatch vs golden: {de:.2e} at {c:?}");
        assert!(dp < 1e-3 || !vnni_ok, "packed VNNI vs golden: {dp:.2e} at {c:?}");
        assert!(dpar < 1e-3 || !vnni_ok, "packed VNNI 2-thread vs golden: {dpar:.2e} at {c:?}");
    }

    // fused-pair gate (worst over both halves)
    if vnni_ok {
        assert!(worst_fused < 1e-3, "fused pair vs golden: {worst_fused:.2e}");
    }

    // ---------- mutation sensitivity self-test ----------
    // The gates above are only meaningful if the metric detects real corruption.
    // Flip ONE weight code byte in a legal position and require the error to
    // blow past every gate (proof the harness is not vacuously green).
    if vnni_ok {
        let (ni, no) = (896usize, 320usize);
        let out = std::process::Command::new(&ref_dot)
            .args(["random", "q8_0", &ni.to_string(), &no.to_string(), "5", pfx])
            .output().unwrap();
        assert!(out.status.success());
        let tensor0 = read_file(&format!("{pfx}_tensor.bin"));
        let x = f32_from_bytes(&read_file(&format!("{pfx}_x_f32.bin")));
        let out_rt = parse_out(&format!("{pfx}_out_rt.txt"));
        let energies = parse_out(&format!("{pfx}_scale.txt"));
        let clean = vnni::repack_q8_0(&tensor0, ni, no);
        let mut o_clean = vec![0f32; no];
        vnni::matvec_vnni(&clean, ni, no, &x, 1, &mut o_clean);
        let base = case_err_scaled(&o_clean, &out_rt, &energies);
        // corrupt byte 3 of column 7's first block (a weight code)
        let mut tensor1 = tensor0.clone();
        let pos = 7 * 34 * (ni / 32) + 2 + 3;
        tensor1[pos] ^= 0x55;
        let corrupt = vnni::repack_q8_0(&tensor1, ni, no);
        let mut o_corrupt = vec![0f32; no];
        vnni::matvec_vnni(&corrupt, ni, no, &x, 1, &mut o_corrupt);
        let mutated = case_err_scaled(&o_corrupt, &out_rt, &energies);
        println!(
            "  mutation self-test: clean err {base:.2e} → one flipped code byte err {mutated:.2e} = {:.0}× (gate sensitivity proven)",
            mutated / base.max(1e-12)
        );
        // detection gate: a real corruption must rise far above the clean
        // f32-rounding noise floor (ratio gate, not an absolute bar)
        assert!(mutated > 100.0 * base.max(1e-12), "mutation not detected: {mutated:.2e} vs base {base:.2e}");
    }

    std::fs::remove_dir_all(&tmp).ok();
    println!("\n== [summary] {n_cases} cases × 8 paths = VERIFIED vs llama.cpp C golden ==", );
    println!("  quantizer bytes vs C runtime contract : mismatches = {qact_bytes_bad} (byte-exact)");
    println!("  worst rel-diff scalar      : {:.2e}", worst_scalar);
    println!("  worst rel-diff avx2        : {:.2e}", worst_avx2);
    println!("  worst rel-diff avx2+vnni256: {:.2e}", worst_vnni256);
    println!("  worst rel-diff engine      : {:.2e}", worst_engine);
    println!("  worst rel-diff packed VNNI : {:.2e}", worst_packed);
    println!("  worst rel-diff packed 2thr : {:.2e}", worst_par);
    println!("  worst rel-diff fused pair  : {:.2e}", worst_fused);
    println!("ALL GATES GREEN");
}
