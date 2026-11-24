//! llama-rs-cli — command-line interface for the engine
//! modes: generate | chat | tokenize | bench
use llama_core::model::{Lfm3, LlamaFamily};
use llama_core::tokenizer::BpeTokenizer;
use llama_core::sampler::{sample, SamplerCfg, SimpleRng};
use llama_core::chat::{ChatMsg, format_lfm3, format_chatml, THINK_CLOSE, THINK_OPEN};
use llama_gguf::GgufFile;
use std::io::Write;

enum Model {
    Lfm3(Lfm3),
    Llama(LlamaFamily),
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = String::new();
    let mut mode = "generate".to_string();
    let mut prompt = String::new();
    let mut n_gen = 48usize;
    let mut threads = 2usize;
    let mut temp = 0.0f32;
    let mut ctx = 4096usize;
    let mut no_bos = false;
    let mut thinking = false;

    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--model" | "-m" => { model_path = args[i + 1].clone(); i += 2; }
            "--mode" => { mode = args[i + 1].clone(); i += 2; }
            "--prompt" | "-p" => { prompt = args[i + 1].clone(); i += 2; }
            "--n" | "-n" => { n_gen = args[i + 1].parse().unwrap_or(48); i += 2; }
            "--threads" | "-t" => { threads = args[i + 1].parse().unwrap_or(2); i += 2; }
            "--temp" => { temp = args[i + 1].parse().unwrap_or(0.0); i += 2; }
            "--ctx" | "-c" => { ctx = args[i + 1].parse().unwrap_or(4096); i += 2; }
            "--no-bos" => { no_bos = true; i += 1; }
            "--thinking" => { thinking = true; i += 1; }
            _ => { eprintln!("unknown arg {a}"); std::process::exit(1); }
        }
    }
    if model_path.is_empty() {
        eprintln!("usage: llama-rs-cli -m model.gguf --mode generate|chat|tokenize|bench -p \"prompt\" [-n N] [-t T] [--temp T] [--thinking]");
        std::process::exit(1);
    }

    let gguf = GgufFile::open(&model_path).expect("open gguf");
    let arch = gguf.arch().unwrap_or("").to_string();
    let tokenizer = BpeTokenizer::from_gguf(&gguf);
    drop(gguf);

    let t0 = std::time::Instant::now();
    let mut model = match arch.as_str() {
        "lfm3" => Model::Lfm3(Lfm3::load(&model_path, ctx, threads)),
        "llama" | "qwen2" | "qwen3" | "mistral" => Model::Llama(LlamaFamily::load(&model_path, ctx, threads)),
        other => { eprintln!("unsupported arch: {other}"); std::process::exit(1); }
    };
    let load_s = t0.elapsed().as_secs_f32();
    eprintln!("[llama.rs] arch={arch} loaded in {load_s:.1}s, threads={threads}, ctx={ctx}");

    match mode.as_str() {
        "tokenize" => {
            let ids = tokenizer.tokenize(&prompt, Some(!no_bos), false);
            println!("tokens({}): {:?}", ids.len(), ids);
        }
        "generate" => {
            // no chat template — matches raw llama-completion
            let ids = tokenizer.tokenize(&prompt, Some(!no_bos), false);
            let (tok_out, text) = generate(&mut model, &tokenizer, &ids, n_gen, temp);
            let _ = tok_out;
            println!("{text}");
        }
        "chat" => {
            let msgs = vec![ChatMsg { role: "user".into(), content: prompt.clone() }];
            let formatted = match &model {
                Model::Lfm3(_) => format_lfm3(&msgs, thinking, true),
                Model::Llama(_) => format_chatml(&msgs, true),
            };
            // replacing the bos marker with its special token happens via fragmentation
            let ids = tokenizer.tokenize(&formatted, None, false);
            let (out_ids, text) = generate_stop(&mut model, &tokenizer, &ids, n_gen, temp);
            let _ = out_ids;
            // print without the thinking tags
            let shown = text
                .replace(THINK_OPEN, "")
                .replace(THINK_CLOSE, "");
            println!("{}", shown.trim());
        }
        "debug" => {
            let ids = tokenizer.tokenize(&prompt, Some(!no_bos), false);
            // prefill, then print the top-5
            let mut logits: Vec<f32>;
            {
                let l = forward_token(&mut model, ids[0]);
                logits = l.to_vec();
            }
            for &id in &ids[1..] {
                let l = forward_token(&mut model, id);
                logits = l.to_vec();
            }
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
            for (r, i) in idx.iter().take(5).enumerate() {
                println!("top{}: id={} logit={:.4} text={:?}", r + 1, i, logits[*i], tokenizer.detokenize_one(*i as u32));
            }
        }
        "bench" => {
            let ids = tokenizer.tokenize(&prompt, Some(!no_bos), false);
            let (n_tok, secs) = bench(&mut model, &tokenizer, &ids, n_gen);
            println!("bench: {n_tok} tokens in {secs:.2}s = {:.2} tok/s", n_tok as f32 / secs);
            if llama_core::prof_enabled() {
                let secs: f64 = secs as f64;
                let fwds = (ids.len() + n_tok) as f64; // prefill + generated forwards
                let mv_ns = llama_core::MATVEC_NS.load(std::sync::atomic::Ordering::Relaxed) as f64;
                println!(
                    "prof: matvec {mv_ms:.1} ms/token of {ms:.1} ms/token total ({pct:.0}%), weights {gb:.2} GB/token streamed at {gbs:.1} GB/s",
                    mv_ms = mv_ns / 1e6 / fwds,
                    ms = secs * 1000.0 / n_tok as f64,
                    pct = 100.0 * (mv_ns / 1e6 / fwds) / (secs * 1000.0 / n_tok as f64),
                    gb = llama_core::MATVEC_BYTES.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9 / fwds,
                    gbs = llama_core::MATVEC_BYTES.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9 / (secs * n_tok as f64),
                );
            }
        }
        _ => { eprintln!("unknown mode {mode}"); std::process::exit(1); }
    }
}

fn generate(model: &mut Model, tok: &BpeTokenizer, ids: &[u32], n: usize, temp: f32) -> (Vec<u32>, String) {
    generate_stop(model, tok, ids, n, temp)
}

fn generate_stop(model: &mut Model, tok: &BpeTokenizer, ids: &[u32], n: usize, temp: f32) -> (Vec<u32>, String) {
    let cfg = SamplerCfg { temp, ..Default::default() };
    let mut rng = SimpleRng::new(cfg.seed);
    let mut out_ids = Vec::new();
    let mut text = String::new();

    // prefill
    let mut logits: Vec<f32>;
    {
        let l = forward_token(model, ids[0]);
        logits = l.to_vec();
    }
    for &id in &ids[1..] {
        let l = forward_token(model, id);
        logits = l.to_vec();
    }

    let t_start = std::time::Instant::now();
    for _ in 0..n {
        let next = sample(&logits, &cfg, &mut rng);
        if next == tok.eos_id { break; }
        out_ids.push(next);
        text.push_str(&tok.detokenize_one(next));
        print!("{}", tok.detokenize_one(next));
        std::io::stdout().flush().ok();
        let l = forward_token(model, next);
        logits = l.to_vec();
    }
    eprintln!("\n[llama.rs] {} tokens in {:.2}s", out_ids.len(), t_start.elapsed().as_secs_f32());
    (out_ids, text)
}

fn bench(model: &mut Model, tok: &BpeTokenizer, ids: &[u32], n: usize) -> (usize, f32) {
    // prefill, then time the greedy generation of n tokens
    let mut logits: Vec<f32>;
    {
        let l = forward_token(model, ids[0]);
        logits = l.to_vec();
    }
    for &id in &ids[1..] {
        let l = forward_token(model, id);
        logits = l.to_vec();
    }
    let t0 = std::time::Instant::now();
    let mut count = 0usize;
    for _ in 0..n {
        let next = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap_or(0);
        if next == tok.eos_id { break; }
        count += 1;
        let l = forward_token(model, next);
        logits = l.to_vec();
    }
    (count, t0.elapsed().as_secs_f32())
}

fn forward_token(model: &mut Model, token: u32) -> Vec<f32> {
    match model {
        Model::Lfm3(m) => {
            let l = m.forward(token);
            l.to_vec()
        }
        Model::Llama(m) => {
            let l = m.forward(token);
            l.to_vec()
        }
    }
}
