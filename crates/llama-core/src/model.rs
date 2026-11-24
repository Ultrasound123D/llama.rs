//! Models: LFM3 (hybrid shortconv+attention with a per-head gate and learned RoPE)
//! + the llama family (llama1/2/3, qwen2/qwen2.5) — port of src/models/lfm3.cpp and models/llama.cpp
use llama_gguf::{GgufFile, Value};

// ============ LFM3 ============

pub struct Lfm3 {
    pub gguf: GgufFile,
    pub n_embd: usize,     // 2048
    pub n_head: usize,     // 16
    pub n_head_kv: usize,  // 4
    pub head_dim: usize,   // 128
    pub n_ff: usize,       // 7680
    pub n_layer: usize,    // 24
    pub eps: f32,
    pub l_cache: usize,    // 7 (taps)
    pub d_conv: usize,     // 6 (state rows)
    pub swa_window: usize, // 256
    pub rope_base_full: f64, // 1e6
    pub rope_base_swa: f64, // 1e4
    pub rope_freqs: Option<Vec<f32>>, // 64 learned (full-attention layers only)
    pub is_recr: Vec<bool>,    // 16 conv
    pub is_swa: Vec<bool>,     // 4 SWA
    pub has_post_norms: bool,
    // KV cache per attention layer: [max_ctx][n_head_kv*head_dim]
    kv_k: Vec<Vec<f32>>,
    kv_v: Vec<Vec<f32>>,
    kv_len: Vec<usize>,
    // conv state per recurrent layer: [d_conv][n_embd] (channel-major: [ch][d_conv])
    conv_state: Vec<Vec<f32>>,
    // conv kernel F32 [n_embd][l_cache] (channel-major taps)
    conv_kernels: Vec<Vec<f32>>,
    pub n_ctx: usize,
    pub pos: usize,
    pub n_threads: usize,
    // scratch buffers
    s_out: Vec<f32>,
    s_gate: Vec<f32>, s_up: Vec<f32>, s_ffn: Vec<f32>,
    s_q: Vec<f32>, s_k: Vec<f32>, s_v: Vec<f32>, s_gate_h: Vec<f32>,
    s_scores: Vec<f32>, s_attn: Vec<f32>,
    logits: Vec<f32>,
    // small preloaded f32 weights
    w_attn_norm: Vec<Vec<f32>>,
    w_ffn_norm: Vec<Vec<f32>>,
    w_post_anorm: Vec<Option<Vec<f32>>>,
    w_post_fnorm: Vec<Option<Vec<f32>>>,
    w_q_norm: Vec<Vec<f32>>,
    w_k_norm: Vec<Vec<f32>>,
    w_output_norm: Vec<f32>,
}

impl Lfm3 {
    pub fn load(path: &str, n_ctx: usize, n_threads: usize) -> Lfm3 {
        let gguf = GgufFile::open(path).unwrap_or_else(|e| panic!("GGUF load: {e}"));
        assert_eq!(gguf.arch(), Some("lfm3"), "not an lfm3 model");
        let g = |k: &str| -> f64 { gguf.get_meta(k).and_then(|v| v.as_f32()).expect(k) as f64 };
        let gi = |k: &str| -> usize { gguf.get_meta(k).and_then(|v| v.as_u64()).expect(k) as usize };

        let n_embd = gi("embedding_length");
        let n_layer = gi("block_count");
        let n_head = gi("attention.head_count");
        let head_dim = gi("attention.key_length");
        let n_ff = gi("feed_forward_length");
        let eps = g("attention.layer_norm_rms_epsilon") as f32;
        let l_cache = gi("shortconv.l_cache");
        let swa = gi("attention.sliding_window");
        let rope_base_full = g("rope.freq_base");
        let rope_base_swa = g("rope.freq_base_swa");

        // head_count_kv: array of length 24 — only specific layers
        let kv_arr: Vec<u64> = gguf.get_meta("attention.head_count_kv").unwrap()
            .as_arr().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let swa_arr: Vec<bool> = gguf.get_meta("attention.sliding_window_pattern").unwrap()
            .as_arr().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
        let is_recr: Vec<bool> = (0..n_layer).map(|il| kv_arr[il] == 0).collect();
        let is_swa: Vec<bool> = (0..n_layer).map(|il| swa_arr[il]).collect();
        let n_head_kv = *kv_arr.iter().filter(|&&v| v > 0).max().unwrap() as usize;

        let rope_freqs = if gguf.tensor("rope_freqs.weight").is_some() {
            Some(weights1d(&gguf, "rope_freqs.weight"))
        } else { None };

        // state
        let n_attn = is_recr.iter().filter(|&&r| !r).count();
        let kv_k = vec![vec![0f32; n_ctx * n_head_kv * head_dim]; n_attn];
        let kv_v = vec![vec![0f32; n_ctx * n_head_kv * head_dim]; n_attn];
        let kv_len = vec![0usize; n_attn];
        let conv_state = vec![vec![0f32; n_embd * (l_cache - 1)]; is_recr.iter().filter(|&&r| r).count()];
        let conv_kernels: Vec<Vec<f32>> = (0..n_layer).filter(|&il| is_recr[il]).map(|il| {
            // conv kernel [l_cache, n_embd] F32 — ne[0]=l_cache fastest → channel ch: taps at ch*l_cache+k
            let t = gguf.tensor(&format!("blk.{il}.shortconv.conv.weight")).unwrap();
            let data = gguf.tensor_bytes(t);
            let mut v = vec![0f32; n_embd * l_cache];
            for ch in 0..n_embd {
                for k in 0..l_cache {
                    let o = (ch * l_cache + k) * 4;
                    v[ch * l_cache + k] = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
                }
            }
            v
        }).collect();

        // norm weights loaded as f32
        let w_attn_norm = (0..n_layer).map(|il| weights1d(&gguf, &format!("blk.{il}.attn_norm.weight"))).collect();
        let w_ffn_norm = (0..n_layer).map(|il| weights1d(&gguf, &format!("blk.{il}.ffn_norm.weight"))).collect();
        let has_post = gguf.tensor("blk.0.post_attention_norm.weight").is_some();
        let w_post_anorm = (0..n_layer).map(|il| {
            if has_post { Some(weights1d(&gguf, &format!("blk.{il}.post_attention_norm.weight"))) } else { None }
        }).collect();
        let w_post_fnorm = (0..n_layer).map(|il| {
            if has_post { Some(weights1d(&gguf, &format!("blk.{il}.post_ffw_norm.weight"))) } else { None }
        }).collect();
        let w_q_norm = (0..n_layer).map(|il| {
            if is_recr[il] { Vec::new() } else { weights1d(&gguf, &format!("blk.{il}.attn_q_norm.weight")) }
        }).collect();
        let w_k_norm = (0..n_layer).map(|il| {
            if is_recr[il] { Vec::new() } else { weights1d(&gguf, &format!("blk.{il}.attn_k_norm.weight")) }
        }).collect();
        let w_output_norm = weights1d(&gguf, "output_norm.weight");

        Lfm3 {
            n_embd, n_head, n_head_kv, head_dim, n_ff, n_layer, eps, l_cache,
            d_conv: l_cache - 1, swa_window: swa, rope_base_full, rope_base_swa,
            rope_freqs, is_recr, is_swa, has_post_norms: has_post,
            kv_k, kv_v, kv_len, conv_state, conv_kernels,
            n_ctx, pos: 0, n_threads,
            s_out: vec![0f32; n_embd.max(3 * n_embd)],
            s_gate: vec![0f32; n_ff], s_up: vec![0f32; n_ff], s_ffn: vec![0f32; n_ff],
            s_q: vec![0f32; n_embd], s_k: vec![0f32; n_head_kv * head_dim],
            s_v: vec![0f32; n_head_kv * head_dim], s_gate_h: vec![0f32; n_head],
            s_scores: Vec::new(), s_attn: vec![0f32; n_head * head_dim],
            logits: vec![0f32; 0],
            w_attn_norm, w_ffn_norm, w_post_anorm, w_post_fnorm,
            w_q_norm, w_k_norm, w_output_norm,
            gguf,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.gguf.tensor("token_embd.weight").unwrap().dims[1] as usize
    }

    /// Forward one token → logits (length = vocab)
    pub fn forward(&mut self, token: u32) -> &[f32] {
        let tok_emb = W::get(&self.gguf, "token_embd.weight");
        let x0 = tok_emb.column(token as usize, self.n_embd);
        let mut x = x0;

        let n_e = self.n_embd;
        let mut attn_idx = 0usize;
        let mut recr_idx = 0usize;
        let pos = self.pos;

        for il in 0..self.n_layer {
            // ----- norm + attention/conv branch -----
            let h = rmsnorm(&x, &self.w_attn_norm[il], self.eps);
            let branch: Vec<f32> = if self.is_recr[il] {
                self.shortconv_block(il, recr_idx, &h);
                recr_idx += 1;
                self.s_out[..n_e].to_vec()
            } else {
                self.attn_block(il, attn_idx, pos, &h);
                attn_idx += 1;
                self.s_attn.clone()
            };
            let mut branch = branch;
            if let Some(w) = &self.w_post_anorm[il] {
                branch = rmsnorm(&branch, w, self.eps);
            }
            // residual: x = branch + x (un-normalized prev)
            for i in 0..n_e { x[i] = branch[i] + x[i]; }

            // ----- FFN (swiglu) -----
            let h2 = rmsnorm(&x, &self.w_ffn_norm[il], self.eps);
            let gate = W::get(&self.gguf, &format!("blk.{il}.ffn_gate.weight"));
            let up = W::get(&self.gguf, &format!("blk.{il}.ffn_up.weight"));
            let down = W::get(&self.gguf, &format!("blk.{il}.ffn_down.weight"));
            gate.matvec(&h2, &mut self.s_gate, self.n_threads);
            up.matvec(&h2, &mut self.s_up, self.n_threads);
            let mut ffn_out = swiglu_ffn(&down, &self.s_gate, &self.s_up, &mut self.s_ffn, self.n_threads);
            if let Some(w) = &self.w_post_fnorm[il] {
                ffn_out = rmsnorm(&ffn_out, w, self.eps);
            }
            for i in 0..n_e { x[i] = ffn_out[i] + x[i]; }
        }

        // ----- output head -----
        let xn = rmsnorm(&x, &self.w_output_norm, self.eps);
        let out_w_name = if self.gguf.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
        let out_w = W::get(&self.gguf, out_w_name);
        if self.logits.len() != out_w.n_out {
            self.logits = vec![0f32; out_w.n_out];
        }
        out_w.matvec(&xn, &mut self.logits, self.n_threads);
        self.pos += 1;
        &self.logits
    }

    /// One attention layer (head gate + q/k norm + rope + SWA/full)
    fn attn_block(&mut self, il: usize, attn_idx: usize, pos: usize, h: &[f32]) {
        let hd = self.head_dim;
        let nh = self.n_head;
        let nkv = self.n_head_kv;
        let n_e = self.n_embd;
        let rep = nh / nkv; // 4

        let q_w = W::get(&self.gguf, &format!("blk.{il}.attn_q.weight"));
        let k_w = W::get(&self.gguf, &format!("blk.{il}.attn_k.weight"));
        let v_w = W::get(&self.gguf, &format!("blk.{il}.attn_v.weight"));
        q_w.matvec(h, &mut self.s_q, self.n_threads);
        k_w.matvec(h, &mut self.s_k, self.n_threads);
        v_w.matvec(h, &mut self.s_v, self.n_threads);

        // q/k norm per head (q heads: 16 × 128; k heads: 4 × 128)
        for hq in 0..nh {
            let s = hq * hd;
            let mut head: Vec<f32> = self.s_q[s..s + hd].to_vec();
            let mut normed = rmsnorm(&head, &self.w_q_norm[il], self.eps);
            // rope: full → base 1e6 + freqs; swa → base 1e4 without freqs
            if !self.is_swa[il] {
                rope_neox_inplace(normed.as_mut_slice(), pos, self.rope_base_full, self.rope_freqs.as_deref());
            } else {
                rope_neox_inplace(normed.as_mut_slice(), pos, self.rope_base_swa, None);
            }
            self.s_q[s..s + hd].copy_from_slice(&normed);
            head.clear();
        }
        for hk in 0..nkv {
            let s = hk * hd;
            let head = self.s_k[s..s + hd].to_vec();
            let mut normed = rmsnorm(&head, &self.w_k_norm[il], self.eps);
            if !self.is_swa[il] {
                rope_neox_inplace(normed.as_mut_slice(), pos, self.rope_base_full, self.rope_freqs.as_deref());
            } else {
                rope_neox_inplace(normed.as_mut_slice(), pos, self.rope_base_swa, None);
            }
            self.s_k[s..s + hd].copy_from_slice(&normed);
        }

        // KV cache
        let len = self.kv_len[attn_idx];
        let off = len * nkv * hd;
        self.kv_k[attn_idx][off..off + nkv * hd].copy_from_slice(&self.s_k);
        self.kv_v[attn_idx][off..off + nkv * hd].copy_from_slice(&self.s_v);
        self.kv_len[attn_idx] = len + 1;
        let n_kv = len + 1;

        // head gate: sigmoid(attn_gate(h)) per head
        let gate_w = W::get(&self.gguf, &format!("blk.{il}.attn_gate.weight"));
        gate_w.matvec(h, &mut self.s_gate_h, self.n_threads);
        for g in self.s_gate_h.iter_mut() { *g = sigmoid(*g); }

        // attention per q head
        let kcache = &self.kv_k[attn_idx];
        let vcache = &self.kv_v[attn_idx];
        let swa = if self.is_swa[il] { Some(self.swa_window) } else { None };
        let mut out_scratch = vec![0f32; hd];
        for hq in 0..nh {
            let kvh = hq / rep;
            let qs = &self.s_q[hq * hd..(hq + 1) * hd];
            // we pass the rows of one kv head
            let (s, v) = (&kcache[..], &vcache[..]);
            attention_one_head_sliced(qs, s, v, n_kv, nkv, kvh, hd, swa, &mut out_scratch, &mut self.s_scores);
            let g = self.s_gate_h[hq];
            for i in 0..hd {
                self.s_attn[hq * hd + i] = out_scratch[i] * g;
            }
        }

        // output projection
        let o_w = W::get(&self.gguf, &format!("blk.{il}.attn_output.weight"));
        o_w.matvec(&self.s_attn, &mut self.s_out[..n_e], self.n_threads);
        // (self.s_attn is the actual output of the layer)
        self.s_attn.copy_from_slice(&self.s_out[..n_e]);
    }

    /// One shortconv layer: b,c,x = in_proj(h); bx=b⊙x; window [state, bx]; y=c⊙conv(bx); out_proj
    fn shortconv_block(&mut self, il: usize, recr_idx: usize, h: &[f32]) {
        let n_e = self.n_embd;
        let taps = self.l_cache;         // 7
        let d_conv = self.d_conv;        // 6

        let in_w = W::get(&self.gguf, &format!("blk.{il}.shortconv.in_proj.weight"));
        assert_eq!(in_w.n_out, 3 * n_e);
        in_w.matvec(h, &mut self.s_out[..3 * n_e], self.n_threads);
        let bcx = &self.s_out[..3 * n_e];
        let (b, c, xx) = (&bcx[0..n_e], &bcx[n_e..2 * n_e], &bcx[2 * n_e..3 * n_e]);

        let state = &mut self.conv_state[recr_idx]; // [n_embd][6] channel-major
        let kernel = &self.conv_kernels[recr_idx];   // [n_embd][7]

        let mut y = vec![0f32; n_e];
        for ch in 0..n_e {
            let bx = b[ch] * xx[ch];
            // window: state(6) + bx → sum of taps against the kernel
            let s_base = ch * d_conv;
            let k_base = ch * taps;
            let mut acc = 0f32;
            for k in 0..d_conv {
                acc += kernel[k_base + k] * state[s_base + k];
            }
            acc += kernel[k_base + d_conv] * bx; // last tap = current bx
            y[ch] = c[ch] * acc;
            // state update: shift, then bx as the last element
            for k in 0..d_conv - 1 {
                state[s_base + k] = state[s_base + k + 1];
            }
            state[s_base + d_conv - 1] = bx;
        }

        let out_w = W::get(&self.gguf, &format!("blk.{il}.shortconv.out_proj.weight"));
        out_w.matvec(&y, &mut self.s_out[..n_e], self.n_threads);
    }

    pub fn reset(&mut self) {
        for k in self.kv_k.iter_mut() { k.iter_mut().for_each(|v| *v = 0.0); }
        for v in self.kv_v.iter_mut() { v.iter_mut().for_each(|v| *v = 0.0); }
        for l in self.kv_len.iter_mut() { *l = 0; }
        for s in self.conv_state.iter_mut() { s.iter_mut().for_each(|v| *v = 0.0); }
        self.pos = 0;
    }
}

/// One attention head reading the rows of a given kv head from [n_kv][nkv*hd] buffers
pub fn attention_one_head_sliced(
    q: &[f32],
    k_all: &[f32], v_all: &[f32],
    n_kv: usize, nkv: usize, kv_head: usize,
    hd: usize,
    swa_window: Option<usize>,
    out: &mut [f32],
    scores: &mut Vec<f32>,
) {
    let scale = 1.0 / (hd as f32).sqrt();
    scores.clear();
    for p in 0..n_kv {
        let k = &k_all[p * nkv * hd + kv_head * hd..p * nkv * hd + (kv_head + 1) * hd];
        let mut dot = 0f32;
        for i in 0..hd {
            dot += q[i] * k[i];
        }
        scores.push(dot * scale);
    }
    if let Some(w) = swa_window {
        let pos = n_kv - 1;
        let lo = pos.saturating_sub(w - 1);
        for (p, s) in scores.iter_mut().enumerate() {
            if p < lo { *s = f32::NEG_INFINITY; }
        }
    }
    softmax_inplace(scores);
    for o in out.iter_mut() { *o = 0.0; }
    for p in 0..n_kv {
        let wgt = scores[p];
        if wgt == 0.0 { continue; }
        let v = &v_all[p * nkv * hd + kv_head * hd..p * nkv * hd + (kv_head + 1) * hd];
        for i in 0..hd {
            out[i] += wgt * v[i];
        }
    }
}

// ============ llama family (llama1/2/3, qwen2, qwen2.5) ============

pub struct LlamaFamily {
    pub gguf: GgufFile,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_layer: usize,
    pub eps: f32,
    pub rope_base: f64,
    pub n_ctx: usize,
    pub pos: usize,
    pub n_threads: usize,
    kv_k: Vec<Vec<f32>>,
    kv_v: Vec<Vec<f32>>,
    kv_len: Vec<usize>,
    s_q: Vec<f32>, s_k: Vec<f32>, s_v: Vec<f32>, s_attn: Vec<f32>,
    s_gate: Vec<f32>, s_up: Vec<f32>, s_ffn: Vec<f32>, s_scores: Vec<f32>,
    logits: Vec<f32>,
    w_attn_norm: Vec<Vec<f32>>,
    w_ffn_norm: Vec<Vec<f32>>,
    w_output_norm: Vec<f32>,
    // Q/K/V biases (qwen2)
    b_q: Vec<Option<Vec<f32>>>,
    b_k: Vec<Option<Vec<f32>>>,
    b_v: Vec<Option<Vec<f32>>>,
}

impl LlamaFamily {
    pub fn load(path: &str, n_ctx: usize, n_threads: usize) -> LlamaFamily {
        let gguf = GgufFile::open(path).unwrap_or_else(|e| panic!("{e}"));
        let arch = gguf.arch().expect("arch");
        assert!(matches!(arch, "llama" | "qwen2" | "qwen3" | "mistral"), "unsupported arch {arch}");
        let g = |k: &str| gguf.get_meta(k).and_then(|v| v.as_f32()).expect(k) as f64;
        let gi = |k: &str| gguf.get_meta(k).and_then(|v| v.as_u64()).expect(k) as usize;

        let n_embd = gi("embedding_length");
        let n_layer = gi("block_count");
        let n_head = gi("attention.head_count");
        let n_head_kv = gguf.get_meta("attention.head_count_kv").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(n_head);
        let head_dim = gguf.get_meta("attention.key_length").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(n_embd / n_head);
        let n_ff = gguf.get_meta("feed_forward_length").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(0);
        let eps = g("attention.layer_norm_rms_epsilon") as f32;
        let rope_base = g("rope.freq_base");

        let kv_k = vec![vec![0f32; n_ctx * n_head_kv * head_dim]; n_layer];
        let kv_v = vec![vec![0f32; n_ctx * n_head_kv * head_dim]; n_layer];
        let w_attn_norm = (0..n_layer).map(|il| weights1d(&gguf, &format!("blk.{il}.attn_norm.weight"))).collect();
        let w_ffn_norm = (0..n_layer).map(|il| weights1d(&gguf, &format!("blk.{il}.ffn_norm.weight"))).collect();
        let w_output_norm = weights1d(&gguf, "output_norm.weight");
        let b_q = (0..n_layer).map(|il| {
            gguf.tensor(&format!("blk.{il}.attn_q.bias")).map(|_| weights1d(&gguf, &format!("blk.{il}.attn_q.bias")))
        }).collect();
        let b_k = (0..n_layer).map(|il| {
            gguf.tensor(&format!("blk.{il}.attn_k.bias")).map(|_| weights1d(&gguf, &format!("blk.{il}.attn_k.bias")))
        }).collect();
        let b_v = (0..n_layer).map(|il| {
            gguf.tensor(&format!("blk.{il}.attn_v.bias")).map(|_| weights1d(&gguf, &format!("blk.{il}.attn_v.bias")))
        }).collect();

        LlamaFamily {
            n_embd, n_head, n_head_kv, head_dim, n_ff, n_layer, eps, rope_base,
            n_ctx, pos: 0, n_threads,
            kv_k, kv_v, kv_len: vec![0; n_layer],
            s_q: vec![0f32; n_embd], s_k: vec![0f32; n_head_kv * head_dim],
            s_v: vec![0f32; n_head_kv * head_dim], s_attn: vec![0f32; n_embd],
            s_gate: vec![0f32; n_ff], s_up: vec![0f32; n_ff], s_ffn: vec![0f32; n_ff],
            s_scores: Vec::new(),
            logits: vec![],
            w_attn_norm, w_ffn_norm, w_output_norm,
            b_q, b_k, b_v,
            gguf,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.gguf.tensor("token_embd.weight").unwrap().dims[1] as usize
    }

    pub fn forward(&mut self, token: u32) -> &[f32] {
        let tok_emb = W::get(&self.gguf, "token_embd.weight");
        let mut x = tok_emb.column(token as usize, self.n_embd);
        let n_e = self.n_embd;
        let pos = self.pos;

        for il in 0..self.n_layer {
            let h = rmsnorm(&x, &self.w_attn_norm[il], self.eps);
            let q_w = W::get(&self.gguf, &format!("blk.{il}.attn_q.weight"));
            let k_w = W::get(&self.gguf, &format!("blk.{il}.attn_k.weight"));
            let v_w = W::get(&self.gguf, &format!("blk.{il}.attn_v.weight"));
            q_w.matvec(&h, &mut self.s_q, self.n_threads);
            k_w.matvec(&h, &mut self.s_k, self.n_threads);
            v_w.matvec(&h, &mut self.s_v, self.n_threads);
            if let Some(b) = &self.b_q[il] { for i in 0..self.s_q.len() { self.s_q[i] += b[i]; } }
            if let Some(b) = &self.b_k[il] { for i in 0..self.s_k.len() { self.s_k[i] += b[i]; } }
            if let Some(b) = &self.b_v[il] { for i in 0..self.s_v.len() { self.s_v[i] += b[i]; } }

            // neox rope on q/k
            for hq in 0..self.n_head {
                let s = hq * self.head_dim;
                let mut head: Vec<f32> = self.s_q[s..s + self.head_dim].to_vec();
                rope_neox_inplace(head.as_mut_slice(), pos, self.rope_base, None);
                self.s_q[s..s + self.head_dim].copy_from_slice(&head);
            }
            for hk in 0..self.n_head_kv {
                let s = hk * self.head_dim;
                let mut head: Vec<f32> = self.s_k[s..s + self.head_dim].to_vec();
                rope_neox_inplace(head.as_mut_slice(), pos, self.rope_base, None);
                self.s_k[s..s + self.head_dim].copy_from_slice(&head);
            }

            // cache
            let len = self.kv_len[il];
            let off = len * self.n_head_kv * self.head_dim;
            self.kv_k[il][off..off + self.n_head_kv * self.head_dim].copy_from_slice(&self.s_k);
            self.kv_v[il][off..off + self.n_head_kv * self.head_dim].copy_from_slice(&self.s_v);
            self.kv_len[il] = len + 1;
            let n_kv = len + 1;

            // causal GQA attention
            let rep = self.n_head / self.n_head_kv;
            let (kcache, vcache) = (&self.kv_k[il], &self.kv_v[il]);
            let mut out_scratch = vec![0f32; self.head_dim];
            for hq in 0..self.n_head {
                let kvh = hq / rep;
                let qs = &self.s_q[hq * self.head_dim..(hq + 1) * self.head_dim];
                attention_one_head_sliced(qs, kcache, vcache, n_kv, self.n_head_kv, kvh,
                    self.head_dim, None, &mut out_scratch, &mut self.s_scores);
                self.s_attn[hq * self.head_dim..(hq + 1) * self.head_dim].copy_from_slice(&out_scratch);
            }
            let o_w = W::get(&self.gguf, &format!("blk.{il}.attn_output.weight"));
            let mut attn_out = vec![0f32; n_e];
            o_w.matvec(&self.s_attn, &mut attn_out, self.n_threads);
            for i in 0..n_e { x[i] += attn_out[i]; }

            // FFN
            let h2 = rmsnorm(&x, &self.w_ffn_norm[il], self.eps);
            let gate = W::get(&self.gguf, &format!("blk.{il}.ffn_gate.weight"));
            let up = W::get(&self.gguf, &format!("blk.{il}.ffn_up.weight"));
            let down = W::get(&self.gguf, &format!("blk.{il}.ffn_down.weight"));
            gate.matvec(&h2, &mut self.s_gate, self.n_threads);
            up.matvec(&h2, &mut self.s_up, self.n_threads);
            let ffn_out = swiglu_ffn(&down, &self.s_gate, &self.s_up, &mut self.s_ffn, self.n_threads);
            for i in 0..n_e { x[i] += ffn_out[i]; }
        }

        let xn = rmsnorm(&x, &self.w_output_norm, self.eps);
        let out_w_name = if self.gguf.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
        let out_w = W::get(&self.gguf, out_w_name);
        if self.logits.len() != out_w.n_out {
            self.logits = vec![0f32; out_w.n_out];
        }
        out_w.matvec(&xn, &mut self.logits, self.n_threads);
        self.pos += 1;
        &self.logits
    }

    pub fn reset(&mut self) {
        for k in self.kv_k.iter_mut() { k.iter_mut().for_each(|v| *v = 0.0); }
        for v in self.kv_v.iter_mut() { v.iter_mut().for_each(|v| *v = 0.0); }
        for l in self.kv_len.iter_mut() { *l = 0; }
        self.pos = 0;
    }
}

// imports (aggregated)
use crate::{W, weights1d, rmsnorm, rope_neox_inplace, softmax_inplace, swiglu_ffn, sigmoid};
#[allow(unused_imports)]
use Value as _;
