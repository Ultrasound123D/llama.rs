//! llama-rs-cli — command-line interface for the engine
//! modes: generate | chat | tokenize | bench | batchcmp | mem
use llama_core::model::{Lfm3, LlamaFamily};
use llama_core::tokenizer::BpeTokenizer;
use llama_core::sampler::{sample, SamplerCfg, SimpleRng};
use llama_core::chat::{ChatMsg, format_lfm3, format_chatml, THINK_CLOSE, THINK_OPEN};
use llama_core::prof;
use llama_gguf::GgufFile;
use std::io::Write;

enum Model {
    Lfm3(Lfm3),
    Llama(LlamaFamily),
}

/// Batched prefill (Phase 6.5) — routes M=1 chunks through the decode path.
fn prefill_model(model: &mut Model, ids: &[u32]) -> Vec<f32> {
    let l = match model {
        Model::Lfm3(m) => m.prefill(ids),
        Model::Llama(m) => m.prefill(ids),
    };
    l.to_vec()
}

fn reset_model(model: &mut Model) {
    match model {
        Model::Lfm3(m) => m.reset(),
        Model::Llama(m) => m.reset(),
    }
}

/// (VmRSS, VmHWM) in KB from /proc/self/status
fn read_rss() -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut rss = 0u64;
    let mut hwm = 0u64;
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("VmRSS:") {
            rss = v.trim().split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("VmHWM:") {
            hwm = v.trim().split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(0);
        }
    }
    (rss, hwm)
}

fn rss_line(tag: &str) {
    let (rss, hwm) = read_rss();
    eprintln!("[llama.rs] rss {tag}: VmRSS={:.1} MB VmHWM={:.1} MB", rss as f64 / 1024.0, hwm as f64 / 1024.0);
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
    prof::init();

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
            // batched prefill (Phase 6.5), then print the top-5 — the same
            // debug output the parity script greps (top1 unchanged)
            let logits = prefill_model(&mut model, &ids);
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
            for (r, i) in idx.iter().take(5).enumerate() {
                println!("top{}: id={} logit={:.4} text={:?}", r + 1, i, logits[*i], tokenizer.detokenize_one(*i as u32));
            }
        }
        "batchcmp" => {
            // batched-prefill self-check: token-by-token (decode path) vs
            // batched prefill logits + greedy agreement on the real model
            let ids = tokenizer.tokenize(&prompt, Some(!no_bos), false);
            std::env::set_var("LLAMA_RS_PREFILL_M", "1");
            let ref_logits = prefill_model(&mut model, &ids);
            reset_model(&mut model);
            std::env::set_var("LLAMA_RS_PREFILL_M", "512");
            let batch_logits = prefill_model(&mut model, &ids);
            let mut max_d = 0f64;
            let mut max_rel = 0f64;
            let top = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
            let (t1, t2) = (top(&ref_logits), top(&batch_logits));
            for i in 0..ref_logits.len().min(batch_logits.len()) {
                let d = (ref_logits[i] - batch_logits[i]).abs() as f64;
                if d > max_d {
                    max_d = d;
                }
                let r = d / (ref_logits[i].abs().max(1.0) as f64);
                if r > max_rel {
                    max_rel = r;
                }
            }
            println!("batchcmp: tokens={} max|dlogit|={max_d:.6} max_rel={max_rel:.6} top1: ref={t1} batch={t2} {}", ids.len(), if t1 == t2 { "MATCH" } else { "DIFFER" });
        }
        "mem" => {
            // memory study: RSS/peak at load → after batched prefill → after decode
            let ids = tokenizer.tokenize(&prompt, Some(!no_bos), false);
            rss_line("after-load");
            let _ = prefill_model(&mut model, &ids);
            rss_line("after-prefill");
            let _ = forward_token(&mut model, 1);
            rss_line("after-1-decode");
        }
        "bench" => {
            let ids = tokenizer.tokenize(&prompt, Some(!no_bos), false);
            let (n_tok, secs, pp_tok, pp_secs) = bench(&mut model, &tokenizer, &ids, n_gen);
            let m = llama_core::model::prefill_batch_max();
            println!("bench-pp: {pp_tok} prompt tokens in {pp_secs:.2}s = {:.2} tok/s (prefill, batch M={m})", pp_tok as f32 / pp_secs);
            println!("bench: {n_tok} tokens in {secs:.2}s = {:.2} tok/s (decode)", n_tok as f32 / secs);
            let (rss, hwm) = read_rss();
            println!("rss: VmRSS={:.1} MB VmHWM={:.1} MB", rss as f64 / 1024.0, hwm as f64 / 1024.0);
            if prof::prof_enabled() {
                print_kstats_summary(n_tok);
                print!("{}", prof::report());
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

    // prefill — batched (Phase 6.5); M=1 falls back to the decode path
    let mut logits = prefill_model(model, ids);

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

/// Bench one run: prefill is timed separately from decode (review gate —
/// "measure prefill and decode separately"). Returns (decode_tokens,
/// decode_secs, prefill_tokens, prefill_secs). Prefill runs the Phase 6.5
/// batched path (LLAMA_RS_PREFILL_M controls the batch size; =1 restores
/// the token-at-a-time behavior for A/B).
fn bench(model: &mut Model, tok: &BpeTokenizer, ids: &[u32], n: usize) -> (usize, f32, usize, f32) {
    // prefill, timed separately (includes first-touch page warmup of the
    // weights; across N process repetitions the median absorbs the cold rep).
    // LLAMA_RS_PROF_PP=1 profiles the prefill phase (op table + kstats).
    let prof_pp = std::env::var("LLAMA_RS_PROF_PP").is_ok();
    if prof_pp {
        prof::reset();
        llama_kernels::kstats::reset();
    }
    let tp0 = std::time::Instant::now();
    let mut logits = prefill_model(model, ids);
    let pp_secs = tp0.elapsed().as_secs_f32();
    let pp_tok = ids.len();
    if prof_pp {
        eprintln!("--- prefill profile ({} tokens) ---", pp_tok);
        print_kstats_summary(pp_tok);
        print!("{}", prof::report());
    }

    let t0 = std::time::Instant::now();
    // separate prefill from decode in the profile: reset all counters now
    prof::reset();
    llama_kernels::kstats::reset();
    let tsc0 = prof::decode_begin();
    let mut count = 0usize;
    for _ in 0..n {
        let ts0 = if prof::prof_enabled() { prof::tsc() } else { 0 };
        let next = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap_or(0);
        if prof::prof_enabled() {
            prof::record(prof::bucket("sample(argmax)"), ts0, 0);
        }
        if next == tok.eos_id { break; }
        count += 1;
        let l = forward_token(model, next);
        logits = l.to_vec();
    }
    let secs = t0.elapsed().as_secs_f32();
    prof::decode_end(tsc0, count as u64);
    (count, secs, pp_tok, pp_secs)
}

fn forward_token(model: &mut Model, token: u32) -> Vec<f32> {
    // NOTE: the forward pass itself is timed by its internal op buckets;
    // only the logits copy is timed here (exclusively).
    let l = match model {
        Model::Lfm3(m) => m.forward(token),
        Model::Llama(m) => m.forward(token),
    };
    let t0 = if prof::prof_enabled() { prof::tsc() } else { 0 };
    let out = l.to_vec();
    if prof::prof_enabled() {
        let b = prof::bucket("logits_copy(to_vec)");
        prof::record(b, t0, 0);
        prof::note_alloc(b, out.len() * 4);
    }
    out
}

/// Kernel-side counters (kstats) summary — printed alongside the op report.
fn print_kstats_summary(n_tok: usize) {
    use llama_kernels::kstats as ks;
    // reuse the TSC calibration from prof (ticks→ns)
    let ghz = prof::tsc_ghz();
    if ghz <= 0.0 {
        return;
    }
    let ns = |t: u64| t as f64 / ghz;
    let n = n_tok as f64;
    let matvecs = ks::MATVECS.load(std::sync::atomic::Ordering::Relaxed) as f64 / n;
    let quant = ns(ks::QUANT_TICKS.load(std::sync::atomic::Ordering::Relaxed)) / 1e3 / n;
    let dot = ns(ks::DOT_TICKS.load(std::sync::atomic::Ordering::Relaxed)) / 1e3 / n;
    let enq = ns(ks::POOL_ENQ_TICKS.load(std::sync::atomic::Ordering::Relaxed)) / 1e3 / n;
    let wait = ns(ks::POOL_WAIT_TICKS.load(std::sync::atomic::Ordering::Relaxed)) / 1e3 / n;
    let wake = ks::WAKE_TICKS.load(std::sync::atomic::Ordering::Relaxed) as f64;
    let wake_jobs = ks::WAKE_JOBS.load(std::sync::atomic::Ordering::Relaxed).max(1) as f64;
    let wake_us = ns(wake as u64) / 1e3 / wake_jobs;
    let parks = ks::PARKS.load(std::sync::atomic::Ordering::Relaxed);
    let spins = ks::SPIN_HITS.load(std::sync::atomic::Ordering::Relaxed);
    let c_spin_wins = ks::CALLER_SPIN_WINS.load(std::sync::atomic::Ordering::Relaxed);
    let c_parks = ks::CALLER_PARKS.load(std::sync::atomic::Ordering::Relaxed);
    let q_allocs = ks::QUANT_ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed) as f64 / n;
    let q_alloc_kb = ks::QUANT_ALLOC_BYTES.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1024.0 / n;
    let wall_us = ns(ks::MATVEC_WALL_TICKS.load(std::sync::atomic::Ordering::Relaxed)) / 1e3 / n;
    let caller_band_us = ns(ks::CALLER_BAND_TICKS.load(std::sync::atomic::Ordering::Relaxed)) / 1e3 / n;
    let n_jobs = ks::WAKE_JOBS.load(std::sync::atomic::Ordering::Relaxed) as f64 / n;
    println!(
        "kstats: {matvecs:.0} matvecs/token | matvec wall {wall_us:.1} µs/tok | quantize {quant:.1} µs/tok ({q_allocs:.0} allocs, {q_alloc_kb:.0} KB) | dots {dot:.1} µs CPU/tok (both threads) | caller band {caller_band_us:.1} µs/tok | pool enqueue {enq:.1} + caller wait {wait:.1} µs/tok | worker wake {wake_us:.2} µs avg over {n_jobs:.0} jobs/token ({parks} parks / {spins} spin-hits; caller barrier: {c_spin_wins} spin-wins / {c_parks} parks)",
        matvecs = matvecs, wall_us = wall_us, quant = quant, q_allocs = q_allocs, q_alloc_kb = q_alloc_kb,
        dot = dot, caller_band_us = caller_band_us, enq = enq, wait = wait, wake_us = wake_us,
        n_jobs = n_jobs, parks = parks, spins = spins, c_spin_wins = c_spin_wins, c_parks = c_parks,
    );
}
