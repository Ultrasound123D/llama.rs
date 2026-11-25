//! Op-level profiler (Phase 6.1) — low-overhead TSC-based timing of every
//! engine operation, with allocation accounting.
//!
//! Design constraints (measured on this VM before building it):
//! - `perf_event_open` returns EPERM for hardware counters in this KVM guest
//!   (verified: cycles/instructions/cache-miss fds all = -1), so hardware
//!   counters are NOT available. The cycle source is the invariant TSC
//!   (constant_tsc + nonstop_tsc confirmed), calibrated once against
//!   CLOCK_MONOTONIC.
//! - Per-op overhead is two `_rdtsc()` reads (~10 ns each) plus relaxed
//!   atomic adds on a thread-shared bucket: ~50-60 ns per timed op.
//!   A decode token issues ~150 matvecs + ~500 small ops => < 40 µs/token
//!   of overhead on a ~110 ms budget (< 0.04%). Profiling therefore does
//!   not perturb the measurement.
//!
//! Buckets are interned by string key (`bucket("name")`); hot paths that can
//! afford it (W::matvec) resolve the bucket once in `W::get` and store it.
//!
//! Activation: `LLAMA_RS_PROF=1` (op-level report at the end of `bench`).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

// ---------- timer ----------

/// Calibrated TSC ticks per nanosecond (set once when profiling starts).
static TSC_GHZ: AtomicU64 = AtomicU64::new(0);
/// Fixed-point scale: ticks * TSC_NS_NUM / TSC_NS_DEN = ns (10^6 scale).
pub const TSC_NS_NUM: u64 = 1_000_000;

#[inline(always)]
pub fn tsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { std::arch::x86_64::_rdtsc() }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

/// Calibrate the TSC frequency against CLOCK_MONOTONIC (~120 ms).
fn calibrate() -> f64 {
    let t0 = std::time::Instant::now();
    let c0 = tsc();
    std::thread::sleep(std::time::Duration::from_millis(120));
    let c1 = tsc();
    let dt = t0.elapsed().as_secs_f64();
    (c1 - c0) as f64 / dt / 1e9 // GHz
}

pub fn prof_enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("LLAMA_RS_PROF").is_ok())
}

/// Machine-readable profile output for scripts (Phase 6.6): tab-separated
/// rows instead of the human table — `scripts/audit.sh` parses these.
pub fn prof_tsv() -> bool {
    static T: OnceLock<bool> = OnceLock::new();
    *T.get_or_init(|| std::env::var("LLAMA_RS_PROF_TSV").is_ok())
}

/// Calibrated TSC frequency in GHz (0 if not initialized).
pub fn tsc_ghz() -> f64 {
    TSC_GHZ.load(Ordering::Relaxed) as f64 / TSC_NS_NUM as f64
}

/// Must be called once before any timed section (cheap; idempotent).
pub fn init() {
    static I: OnceLock<()> = OnceLock::new();
    I.get_or_init(|| {
        if !prof_enabled() {
            return;
        }
        let ghz = calibrate();
        TSC_GHZ.store((ghz * TSC_NS_NUM as f64) as u64, Ordering::Relaxed);
    });
}

#[inline(always)]
fn tsc_to_ns(t: u64) -> u64 {
    let scale = TSC_NS_NUM as f64 / TSC_GHZ.load(Ordering::Relaxed) as f64;
    (t as f64 * scale) as u64
}

// ---------- buckets ----------

pub struct Bucket {
    pub name: String,
    pub ns: AtomicU64,   // total time (ns)
    pub ticks: AtomicU64,  // total time (raw TSC ticks — cycle-level source)
    pub calls: AtomicU64,
    pub bytes: AtomicU64, // traffic attributed to the op (e.g. weight bytes)
    pub allocs: AtomicU64,
    pub alloc_bytes: AtomicU64,
}

impl Bucket {
    fn new(name: String) -> Bucket {
        Bucket {
            name,
            ns: AtomicU64::new(0),
            ticks: AtomicU64::new(0),
            calls: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            allocs: AtomicU64::new(0),
            alloc_bytes: AtomicU64::new(0),
        }
    }
}

fn registry() -> &'static Mutex<Vec<&'static Bucket>> {
    static R: OnceLock<Mutex<Vec<&'static Bucket>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Vec::new()))
}

/// Intern a bucket by name (hash lookup per call — fine for op-level use;
/// hot paths cache the returned reference).
pub fn bucket(name: &str) -> &'static Bucket {
    // small process-wide cache keyed by name
    static CACHE: OnceLock<Mutex<std::collections::HashMap<String, &'static Bucket>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if let Some(b) = cache.lock().unwrap().get(name) {
        return b;
    }
    let b: &'static Bucket = Box::leak(Box::new(Bucket::new(name.to_string())));
    registry().lock().unwrap().push(b);
    cache.lock().unwrap().insert(name.to_string(), b);
    b
}

/// Timing guard: records elapsed TSC ticks into `b` on drop.
/// `bytes` (optional) is added once per completed call.
#[must_use = "the guard records the timing on drop"]
pub struct Timed {
    t0: u64,
    b: &'static Bucket,
}

#[inline(always)]
pub fn timed(b: &'static Bucket) -> Timed {
    Timed { t0: tsc(), b }
}

impl Drop for Timed {
    #[inline(always)]
    fn drop(&mut self) {
        let t = tsc().wrapping_sub(self.t0);
        self.b.ticks.fetch_add(t, Ordering::Relaxed);
        self.b.calls.fetch_add(1, Ordering::Relaxed);
    }
}

/// Record a completed call with explicit start (avoids guard when the
/// payload is a plain closure) + traffic bytes.
#[inline(always)]
pub fn record(b: &'static Bucket, t0: u64, bytes: u64) {
    let t = tsc().wrapping_sub(t0);
    b.ticks.fetch_add(t, Ordering::Relaxed);
    b.calls.fetch_add(1, Ordering::Relaxed);
    b.bytes.fetch_add(bytes, Ordering::Relaxed);
}

#[inline(always)]
pub fn note_alloc(b: &'static Bucket, n_bytes: usize) {
    b.allocs.fetch_add(1, Ordering::Relaxed);
    b.alloc_bytes.fetch_add(n_bytes as u64, Ordering::Relaxed);
}

// ---------- report ----------

pub struct BucketSnapshot {
    pub name: String,
    pub ns: u64,
    pub ticks: u64,
    pub calls: u64,
    pub bytes: u64,
    pub allocs: u64,
    pub alloc_bytes: u64,
}

pub fn snapshot() -> Vec<BucketSnapshot> {
    registry()
        .lock()
        .unwrap()
        .iter()
        .map(|b| BucketSnapshot {
            name: b.name.clone(),
            ns: tsc_to_ns(b.ticks.load(Ordering::Relaxed)),
            ticks: b.ticks.load(Ordering::Relaxed),
            calls: b.calls.load(Ordering::Relaxed),
            bytes: b.bytes.load(Ordering::Relaxed),
            allocs: b.allocs.load(Ordering::Relaxed),
            alloc_bytes: b.alloc_bytes.load(Ordering::Relaxed),
        })
        .collect()
}

/// Total decode wall time in TSC ticks (fed by the bench harness).
pub static DECODE_TICKS: AtomicU64 = AtomicU64::new(0);
pub static DECODE_TOKENS: AtomicU64 = AtomicU64::new(0);

/// Clear every bucket (used to separate prefill from decode phases).
pub fn reset() {
    if let Ok(reg) = registry().try_lock() {
        for b in reg.iter() {
            b.ns.store(0, Ordering::Relaxed);
            b.ticks.store(0, Ordering::Relaxed);
            b.calls.store(0, Ordering::Relaxed);
            b.bytes.store(0, Ordering::Relaxed);
            b.allocs.store(0, Ordering::Relaxed);
            b.alloc_bytes.store(0, Ordering::Relaxed);
        }
    }
    DECODE_TICKS.store(0, Ordering::Relaxed);
    DECODE_TOKENS.store(0, Ordering::Relaxed);
}

pub fn decode_begin() -> u64 {
    tsc()
}

pub fn decode_end(t0: u64, n_tokens: u64) {
    DECODE_TICKS.fetch_add(tsc().wrapping_sub(t0), Ordering::Relaxed);
    DECODE_TOKENS.fetch_add(n_tokens, Ordering::Relaxed);
}

/// Human-readable report: per-op time per token, % of phase wall time,
/// effective bandwidth for traffic-carrying ops, MB of traffic per token,
/// allocation counts. TSV mode (`LLAMA_RS_PROF_TSV=1`) emits machine-
/// readable rows for the Phase 6.6 audit scripts.
pub fn report() -> String {
    let ghz = TSC_GHZ.load(Ordering::Relaxed) as f64 / TSC_NS_NUM as f64;
    let mut out = String::new();
    let snaps = snapshot();
    let n_tok = DECODE_TOKENS.load(Ordering::Relaxed).max(1) as f64;
    let wall_ns = tsc_to_ns(DECODE_TICKS.load(Ordering::Relaxed));
    let wall_ms_per_tok = wall_ns as f64 / 1e6 / n_tok;
    let mut s = snaps;
    s.sort_by(|a, b| b.ticks.cmp(&a.ticks));
    if prof_tsv() {
        out.push_str("#prof-tsv\tname\tus_per_tok\tpct_wall\tGBps\tMB_per_tok\tcalls_per_tok\tallocs_per_tok\n");
        for b in &s {
            if b.ticks == 0 && b.calls == 0 && b.allocs == 0 {
                continue;
            }
            let us = b.ns as f64 / 1e3 / n_tok;
            let pct = 100.0 * b.ns as f64 / wall_ns.max(1) as f64;
            let mb = b.bytes as f64 / 1e6 / n_tok;
            let gbs = if b.bytes > 0 {
                b.bytes as f64 / 1e9 / (b.ns as f64 / 1e9).max(1e-12)
            } else {
                0.0
            };
            out.push_str(&format!(
                "prof-tsv\t{}\t{:.3}\t{:.2}\t{:.3}\t{:.4}\t{:.1}\t{:.1}\n",
                b.name, us, pct, gbs, mb, b.calls as f64 / n_tok, b.allocs as f64 / n_tok
            ));
        }
        out.push_str(&format!("#prof-tsv-wall\t{:.4}\tms_per_tok\ttokens\t{}\n", wall_ms_per_tok, DECODE_TOKENS.load(Ordering::Relaxed)));
        return out;
    }
    out.push_str(&format!(
        "\n==== llama.rs op profile (TSC calibration: {ghz:.4} GHz) ====\n"
    ));
    out.push_str("op                              calls/tok   µs/tok    %wall   GB/s   MB/tok  allocs/tok  allocKB/tok\n");
    out.push_str("---------------------------------------------------------------------------------------------------------\n");
    for b in &s {
        if b.ticks == 0 && b.calls == 0 && b.allocs == 0 {
            continue;
        }
        let us = b.ns as f64 / 1e3 / n_tok;
        let pct = 100.0 * b.ns as f64 / wall_ns.max(1) as f64;
        let calls = b.calls as f64 / n_tok;
        let gbs = if b.bytes > 0 {
            b.bytes as f64 / 1e9 / (b.ns as f64 / 1e9).max(1e-12)
        } else {
            0.0
        };
        let mb = b.bytes as f64 / 1e6 / n_tok;
        let allocs = b.allocs as f64 / n_tok;
        let akb = b.alloc_bytes as f64 / 1024.0 / n_tok;
        out.push_str(&format!(
            "{:<32} {:>7.1} {:>8.2} {:>6.1}% {:>6.2} {:>6.2} {:>8.1} {:>9.1}\n",
            truncate(&b.name, 32),
            calls,
            us,
            pct,
            gbs,
            mb,
            allocs,
            akb
        ));
    }
    out.push_str(&format!(
        "phase wall: {} tokens, {:.1} ms/token\n",
        n_tok as u64,
        wall_ms_per_tok
    ));
    out
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        let mut t = s[..n - 1].to_string();
        t.push('…');
        t
    }
}

/// Tensor-class name: normalize "blk.17.ffn_gate.weight" → "blk.*.ffn_gate.weight"
pub fn tensor_class(name: &str) -> String {
    // replace any '.<digits>.' component with '.*.'
    let mut out = String::with_capacity(name.len() + 2);
    let mut num_run = String::new();
    let mut prev_dot = false;
    for ch in name.chars() {
        if ch.is_ascii_digit() {
            num_run.push(ch);
            continue;
        }
        if !num_run.is_empty() {
            if prev_dot && (ch == '.') {
                out.push('*');
            } else {
                out.push_str(&num_run);
            }
            num_run.clear();
        }
        prev_dot = ch == '.';
        out.push(ch);
    }
    if !num_run.is_empty() {
        if prev_dot {
            out.push('*');
        } else {
            out.push_str(&num_run);
        }
    }
    out
}
