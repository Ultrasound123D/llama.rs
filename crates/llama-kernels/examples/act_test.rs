// numeric self-test for the Phase 7 activation kernels
fn main() {
    // 1. silu_mul vs scalar reference
    let g: Vec<f32> = (0..64).map(|i| (i as f32) * 0.3 - 9.0).collect();
    let u: Vec<f32> = (0..64).map(|i| ((i % 7) as f32) * 0.11 - 0.3).collect();
    let mut out = vec![0f32; 64];
    llama_kernels::batch::silu_mul(&g, &u, &mut out);
    let mut maxd = 0f32;
    for i in 0..64 {
        let silu_ref = g[i] / (1.0 + (-(g[i] as f64)).exp()) as f32;
        let want = silu_ref * u[i];
        let d = (out[i] - want).abs();
        if d > maxd { maxd = d; }
    }
    println!("silu_mul max abs diff vs scalar f64 ref: {maxd:.3e}");

    // 2. rmsnorm_into vs scalar f64
    let x: Vec<f32> = (0..2048).map(|i| ((i * 2654435761u32 as usize) % 1000) as f32 / 300.0 - 1.7).collect();
    let w: Vec<f32> = (0..2048).map(|i| ((i * 40503u32 as usize) % 900) as f32 / 500.0).collect();
    let mut out2 = vec![0f32; 2048];
    llama_kernels::batch::rmsnorm_into(&x, &w, 1e-5, &mut out2);
    let mut ss = 0f64; for v in &x { ss += (*v as f64) * (*v as f64); }
    let mean = ss / 2048.0;
    let scale = 1.0 / (mean + 1e-5).sqrt();
    let mut maxd2 = 0f32;
    for i in 0..2048 {
        let want = (x[i] as f64 * scale * (w[i] as f64)) as f32;
        let d = (out2[i] - want).abs();
        if d > maxd2 { maxd2 = d; }
    }
    println!("rmsnorm max abs diff vs scalar f64 ref: {maxd2:.3e}");

    // 3. conv_tap8 vs sequential scalar
    let w8: Vec<f32> = (0..8).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let k8: Vec<f32> = (0..8).map(|i| if i < 7 { (i as f32) * 0.25 - 0.8 } else { 0.0 }).collect();
    let got = llama_kernels::batch::conv_tap8(&w8, &k8);
    let mut want = 0f32;
    for j in 0..8 { want += k8[j] * w8[j]; }
    println!("conv_tap8 = {got:.6} scalar = {want:.6} diff = {:.3e}", (got - want).abs());

    // 4. vec_add
    let mut a: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..64).map(|i| 100.0 - i as f32).collect();
    llama_kernels::batch::vec_add(&mut a, &b);
    let ok = a.iter().zip(b.iter()).enumerate().all(|(i, (av, bv))| (av - (i as f32 * 0.1 + bv)).abs() < 1e-5);
    println!("vec_add ok: {ok}");
}
