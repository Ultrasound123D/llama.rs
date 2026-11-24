// Deep diagnostics: print intermediate values after the first layer
use llama_core::model::LlamaFamily;
use llama_core::{rmsnorm, W, weights1d, silu};

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let token: u32 = std::env::args().nth(2).unwrap().parse().unwrap();
    let m = LlamaFamily::load(&path, 512, 1);

    let tok_emb = W::get(&m.gguf, "token_embd.weight");
    let x = tok_emb.column(token as usize, m.n_embd);
    println!("embed[0..5]: {:?}", &x[..5]);
    let n0 = weights1d(&m.gguf, "blk.0.attn_norm.weight");
    let h = rmsnorm(&x, &n0, m.eps);
    println!("rmsnormed[0..5]: {:?}", &h[..5]);

    let q_w = W::get(&m.gguf, "blk.0.attn_q.weight");
    let b_q = m.gguf.tensor("blk.0.attn_q.bias").map(|_| weights1d(&m.gguf, "blk.0.attn_q.bias"));
    let mut q = vec![0f32; q_w.n_out];
    q_w.matvec(&h, &mut q, 1);
    if let Some(b) = &b_q { for i in 0..q.len() { q[i] += b[i]; } }
    println!("q[0..6]: {:?}", &q[..6]);

    let v_w = W::get(&m.gguf, "blk.0.attn_v.weight");
    let mut v = vec![0f32; v_w.n_out];
    v_w.matvec(&h, &mut v, 1);
    if let Some(b) = &m.gguf.tensor("blk.0.attn_v.bias").map(|_| weights1d(&m.gguf, "blk.0.attn_v.bias")) {
        for i in 0..v.len() { v[i] += b[i]; }
    }
    println!("v[0..6]: {:?}", &v[..6]);

    // GQA: 14 q heads × 2 kv heads → repeat v (single token: attention = repeated v)
    let rep = m.n_head / m.n_head_kv;
    let mut v_rep = vec![0f32; m.n_head * m.head_dim];
    for hq in 0..m.n_head {
        let kvh = hq / rep;
        for i in 0..m.head_dim {
            v_rep[hq * m.head_dim + i] = v[kvh * m.head_dim + i];
        }
    }
    let o_w = W::get(&m.gguf, "blk.0.attn_output.weight");
    let mut attn_out = vec![0f32; o_w.n_out];
    o_w.matvec(&v_rep, &mut attn_out, 1);
    println!("wo(vrep)[0..6]: {:?}", &attn_out[..6]);

    let x1: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();
    let n1 = weights1d(&m.gguf, "blk.0.ffn_norm.weight");
    let h2 = rmsnorm(&x1, &n1, m.eps);
    let g_w = W::get(&m.gguf, "blk.0.ffn_gate.weight");
    let mut g = vec![0f32; g_w.n_out];
    g_w.matvec(&h2, &mut g, 1);
    let u_w = W::get(&m.gguf, "blk.0.ffn_up.weight");
    let mut u = vec![0f32; u_w.n_out];
    u_w.matvec(&h2, &mut u, 1);
    println!("gate[0..4]: {:?}  up[0..4]: {:?}", &g[..4], &u[..4]);
    let d_w = W::get(&m.gguf, "blk.0.ffn_down.weight");
    let mut fused = vec![0f32; g.len()];
    for i in 0..g.len() {
        fused[i] = silu(g[i]) * u[i];
    }
    let mut ffn_out = vec![0f32; d_w.n_out];
    d_w.matvec(&fused, &mut ffn_out, 1);
    println!("ffn_out[0..6]: {:?}", &ffn_out[..6]);
}
