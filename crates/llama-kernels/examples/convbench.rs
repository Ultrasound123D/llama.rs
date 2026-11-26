// Phase 7 conv_core micro-bench (dead-code-proofed with checksums)
fn main() {
    let n_e = 2048usize;
    let m = 512usize;
    let d_conv = 6usize;
    let span = d_conv + m;
    let layers = 16usize;

    // vary inputs per layer so nothing is const-foldable
    let mut in_proj = vec![0f32; m * 3 * n_e];
    for i in 0..in_proj.len() { in_proj[i] = ((i as i64 % 977) as f32) * 0.001 - 0.4; }
    let mut state = vec![0.02f32; n_e * d_conv];
    let kernel8: Vec<f32> = (0..n_e * 8).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
    let mut conv_w = vec![0f32; n_e * (span + 1)];
    let mut y = vec![0f32; m * n_e];

    let mut checksum = 0f32;
    let t0 = std::time::Instant::now();
    for l in 0..layers {
        let st = &state[..];
        for ch in 0..n_e {
            let wrow = &mut conv_w[ch * (span + 1)..ch * (span + 1) + span];
            for k in 0..d_conv {
                wrow[k] = st[ch * d_conv + k] + l as f32;
            }
            for r in 0..m {
                let bch = &in_proj[r * 3 * n_e..][..n_e];
                let xch = &in_proj[r * 3 * n_e + 2 * n_e..][..n_e];
                wrow[d_conv + r] = bch[ch] * xch[ch];
            }
        }
    }
    let t1 = std::time::Instant::now();
    for _ in 0..layers {
        for r in 0..m {
            let cch = &in_proj[r * 3 * n_e + n_e..][..n_e];
            for ch in 0..n_e {
                let wrow = &conv_w[ch * (span + 1)..ch * (span + 1) + span + 1];
                let acc = llama_kernels::batch::conv_tap8(&wrow[r..r + 8], &kernel8[ch * 8..ch * 8 + 8]);
                y[r * n_e + ch] = cch[ch] * acc;
            }
        }
    }
    let t2 = std::time::Instant::now();
    for _ in 0..layers {
        for ch in 0..n_e {
            let wrow = &conv_w[ch * (span + 1)..ch * (span + 1) + span];
            for k in 0..d_conv {
                state[ch * d_conv + k] = wrow[m + k];
            }
        }
    }
    let t3 = std::time::Instant::now();
    for v in &y { checksum += *v; }
    for v in &state { checksum += *v; }
    for v in &conv_w { checksum += *v; }
    let per_tok = |t: std::time::Duration| t.as_secs_f32() / 512.0 * 1e6;
    println!("checksum {:.3}", checksum);
    println!("window build: {:>8.2} ms  ({:>7.1} µs/token)", (t1 - t0).as_secs_f32() * 1000.0, per_tok(t1 - t0));
    println!("tap dot+cmul: {:>8.2} ms  ({:>7.1} µs/token)", (t2 - t1).as_secs_f32() * 1000.0, per_tok(t2 - t1));
    println!("state roll:   {:>8.2} ms  ({:>7.1} µs/token)", (t3 - t2).as_secs_f32() * 1000.0, per_tok(t3 - t2));
}
