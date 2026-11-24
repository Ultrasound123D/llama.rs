//! llama-core — the inference engine (Rust port of llama.cpp b12502 behavior)
pub mod model;
pub mod tokenizer;
pub mod sampler;
pub mod chat;

use llama_gguf::{GgmlType, GgufFile, TensorInfo};

/// Weight-tensor reference [n_in, n_out] in ggml order (output column = n_in consecutive values)
pub struct W<'a> {
    pub ty: GgmlType,
    pub data: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> W<'a> {
    /// Fetch a tensor by name from the file (dims [n_in, n_out])
    pub fn get(gguf: &'a GgufFile, name: &str) -> W<'a> {
        let t: &TensorInfo = gguf.tensor(name)
            .unwrap_or_else(|| panic!("missing tensor: {name}"));
        assert!(t.dims.len() == 2, "tensor {name} dims {:?}", t.dims);
        W {
            ty: t.ty,
            data: gguf.tensor_bytes(t),
            n_in: t.dims[0] as usize,
            n_out: t.dims[1] as usize,
        }
    }
    /// Parallel matvec: out[o] = Σ_i x[i]·W[i,o]
    pub fn matvec(&self, x: &[f32], out: &mut [f32], n_threads: usize) {
        debug_assert_eq!(x.len(), self.n_in);
        debug_assert_eq!(out.len(), self.n_out);
        llama_quant::matvec(self.ty, self.data, self.n_in, self.n_out, x, n_threads, out);
    }
    /// One column (useful for the embedding lookup: column token_id)
    pub fn column(&self, col: usize, n: usize) -> Vec<f32> {
        llama_quant::column_f32(self.ty, self.data, col, n)
    }
}

/// One-dimensional tensor [n] → f32 weights
pub fn weights1d(gguf: &GgufFile, name: &str) -> Vec<f32> {
    let t = gguf.tensor(name).unwrap_or_else(|| panic!("missing tensor: {name}"));
    assert_eq!(t.dims.len(), 1, "{name} dims {:?}", t.dims);
    let n = t.dims[0] as usize;
    llama_quant::weights1d(t.ty, gguf.tensor_bytes(t), n)
}

// ---------- ops (same formulas as ggml-cpu/ops.cpp) ----------

/// RMSNorm: x * w / sqrt(mean(x²)+eps)  [ggml_compute_forward_rms_norm]
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ss = 0f64;
    for v in x { ss += (*v as f64) * (*v as f64); }
    let mean = ss / n as f64;
    let scale = 1.0 / (mean + eps as f64).sqrt();
    x.iter().zip(w).map(|(v, wv)| (*v as f64 * scale * (*wv as f64)) as f32).collect()
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x as f64).exp()) as f32
}

/// Stable softmax
pub fn softmax_inplace(x: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for v in x.iter() { if *v > max { max = *v; } }
    let mut sum = 0f64;
    for v in x.iter_mut() {
        *v = ((*v - max) as f64).exp() as f32;
        sum += *v as f64;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() { *v = (*v as f64 * inv) as f32; }
}

/// SiLU-gated FFN: down( silu(gate) ⊙ up )  [build_ffn LLM_FFN_SILU/LLM_FFN_PAR]
pub fn swiglu_ffn(down: &W, gate_out: &[f32], up_out: &[f32], scratch: &mut Vec<f32>, n_threads: usize) -> Vec<f32> {
    let n_ff = gate_out.len();
    scratch.resize(n_ff, 0.0);
    for i in 0..n_ff {
        scratch[i] = silu(gate_out[i]) * up_out[i];
    }
    let mut out = vec![0f32; down.n_out];
    down.matvec(scratch, &mut out, n_threads);
    out
}

/// NeoX RoPE with optional frequency factors (port of ggml-cpu/ops.cpp:6148-6252):
/// angle_j = pos · base^(−2j/d) / ff_j, pairs (j, j+d/2)
/// Operates on a whole vec split into heads [n_head][head_dim]
pub fn rope_neox_inplace(heads: &mut [f32], pos: usize, base: f64, factors: Option<&[f32]>) {
    let d = heads.len(); // head_dim (this operates on a single head)
    let half = d / 2;
    // theta_scale = base^(-2/d); theta_j = pos·base^(-2j/d)
    let theta_scale = base.powf(-2.0 / d as f64);
    let mut theta = pos as f64;
    for j in 0..half {
        let ff = factors.map(|f| f[j] as f64).unwrap_or(1.0);
        let angle = theta / ff;
        let (s, c) = (angle.sin(), angle.cos());
        let x0 = heads[j] as f64;
        let x1 = heads[j + half] as f64;
        heads[j] = (x0 * c - x1 * s) as f32;
        heads[j + half] = (x0 * s + x1 * c) as f32;
        theta *= theta_scale;
    }
}

/// Softmax over the score vector with a mask (causal and/or SWA), then value weighting
/// Returns the resulting output for one head
pub fn attention_one_head(
    q: &[f32],            // head_dim
    k_all: &[f32],        // [n_kv, head_dim] (one kv head)
    v_all: &[f32],
    n_kv: usize,
    head_dim: usize,
    swa_window: Option<usize>, // None = fully causal
    out: &mut [f32],
    scratch_scores: &mut Vec<f32>,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    scratch_scores.clear();
    for p in 0..n_kv {
        let k = &k_all[p * head_dim..(p + 1) * head_dim];
        let mut dot = 0f32;
        for i in 0..head_dim {
            dot += q[i] * k[i];
        }
        scratch_scores.push(dot * scale);
    }
    // mask: causal (all positions here ≤ pos because KV grows) + SWA window
    if let Some(w) = swa_window {
        // allowed p >= pos-(w-1); current pos = n_kv-1
        let pos = n_kv - 1;
        let lo = pos.saturating_sub(w - 1);
        for (p, s) in scratch_scores.iter_mut().enumerate() {
            if p < lo { *s = f32::NEG_INFINITY; }
        }
    }
    softmax_inplace(scratch_scores);
    for o in out.iter_mut() { *o = 0.0; }
    for (p, wgt) in scratch_scores.iter().enumerate() {
        let v = &v_all[p * head_dim..(p + 1) * head_dim];
        let w = *wgt;
        if w == 0.0 { continue; }
        for i in 0..head_dim {
            out[i] += w * v[i];
        }
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x as f64).exp()) as f32
}
