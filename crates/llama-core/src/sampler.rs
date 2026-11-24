//! أخذ العينات — نقل llama-sampling.cpp الأساسي: greedy/temperature/top-k/top-p
pub struct SamplerCfg {
    pub temp: f32,
    pub top_k: usize,   // 0 = الكل
    pub top_p: f32,     // 1.0 = الكل
    pub seed: u64,
}

impl Default for SamplerCfg {
    fn default() -> Self { SamplerCfg { temp: 0.0, top_k: 0, top_p: 1.0, seed: 42 } }
}

/// عينة توكن واحدة من logits (temp=0 → greedy argmax)
pub fn sample(logits: &[f32], cfg: &SamplerCfg, rng: &mut SimpleRng) -> u32 {
    if cfg.temp <= 1e-6 {
        // greedy
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > bv { bv = v; best = i; }
        }
        return best as u32;
    }
    let mut probs: Vec<(usize, f64)> = logits.iter().map(|v| *v as f64).enumerate()
        .map(|(i, v)| (i, (v / cfg.temp as f64).exp())).collect();
    let mut sum: f64 = 0.0;
    for p in probs.iter_mut() { sum += p.1; }
    if sum <= 0.0 { return 0; }
    for p in probs.iter_mut() { p.1 /= sum; }

    // top-k
    if cfg.top_k > 0 && cfg.top_k < probs.len() {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        probs.truncate(cfg.top_k);
        let s: f64 = probs.iter().map(|p| p.1).sum();
        for p in probs.iter_mut() { p.1 /= s; }
    }
    // top-p
    if cfg.top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut cum = 0f64;
        let mut cut = probs.len();
        for (i, item) in probs.iter().enumerate() {
            cum += item.1;
            if cum >= cfg.top_p as f64 { cut = i + 1; break; }
        }
        probs.truncate(cut);
        let s: f64 = probs.iter().map(|p| p.1).sum();
        for p in probs.iter_mut() { p.1 /= s; }
    }
    let r = rng.next_f64();
    let mut cum = 0f64;
    for &(idx, prob) in probs.iter() {
        cum += prob;
        if r < cum { return idx as u32; }
    }
    probs.last().map(|p| p.0 as u32).unwrap_or(0)
}

/// مولد عشوائي بسيط (xorshift64*)
pub struct SimpleRng { state: u64 }
impl SimpleRng {
    pub fn new(seed: u64) -> SimpleRng { SimpleRng { state: seed.max(1) } }
    pub fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.state = x;
        ((x.wrapping_mul(0x2545F4914F6CDD1D)) >> 11) as f64 / (1u64 << 53) as f64
    }
}
