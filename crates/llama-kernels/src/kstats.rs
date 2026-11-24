//! Kernel-internal profiling counters (Phase 6.1).
//!
//! These split the matvec wall time into its three phases and expose
//! worker-pool scheduling behavior. All relaxed atomics — the accounting
//! overhead (~4 relaxed RMW per matvec) is < 0.2 µs on a ~750 µs matvec.
//!
//! Counters (TSC ticks; converted to ns by the reporting side):
//! - QUANT_TICKS: activation quantization (`QAct::quantize` = llama.cpp's
//!   `from_float` on wdata) + its heap allocation.
//! - DOT_TICKS:   the vec_dot loop itself (caller + worker threads both add).
//! - POOL_ENQ_TICKS / POOL_WAIT_TICKS: caller-side enqueue & join cost
//!   (mutex + condvar + scheduling imbalance).
//! - WAKE_TICKS:  worker wakeup latency = tsc(pickup) - tsc(enqueue-release),
//!   summed per job; counts park-wakeups vs spin-catches.
//! - PARKS / SPIN_HITS: how the worker observed the job.

use std::sync::atomic::{AtomicU64, Ordering};

pub static QUANT_TICKS: AtomicU64 = AtomicU64::new(0);
pub static QUANT_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
pub static QUANT_ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
pub static DOT_TICKS: AtomicU64 = AtomicU64::new(0);
pub static POOL_ENQ_TICKS: AtomicU64 = AtomicU64::new(0);
pub static POOL_WAIT_TICKS: AtomicU64 = AtomicU64::new(0);
pub static WAKE_TICKS: AtomicU64 = AtomicU64::new(0);
pub static WAKE_JOBS: AtomicU64 = AtomicU64::new(0);
pub static PARKS: AtomicU64 = AtomicU64::new(0);
pub static SPIN_HITS: AtomicU64 = AtomicU64::new(0);
/// caller-side barrier resolution: spin-win vs condvar fallback
pub static CALLER_SPIN_WINS: AtomicU64 = AtomicU64::new(0);
pub static CALLER_PARKS: AtomicU64 = AtomicU64::new(0);
/// jobs executed through the lock-free slot (par_chunks2)
pub static SLOT_JOBS: AtomicU64 = AtomicU64::new(0);
/// caller-side wait time on the slot spin barrier
pub static SLOT_WAIT_TICKS: AtomicU64 = AtomicU64::new(0);
pub static MATVECS: AtomicU64 = AtomicU64::new(0);
/// whole-matvec wall time (quantize + dispatch + dots + join)
pub static MATVEC_WALL_TICKS: AtomicU64 = AtomicU64::new(0);
/// caller-side band only (chunk 0)
pub static CALLER_BAND_TICKS: AtomicU64 = AtomicU64::new(0);
/// sum over jobs of (job exec ticks) — worker-side dot time lands here too.
pub static JOB_TICKS: AtomicU64 = AtomicU64::new(0);

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

/// A timestamp published by the caller at enqueue-release; workers compute
/// their pickup latency against it (cross-core invariant TSC).
pub static ENQUEUE_TSC: AtomicU64 = AtomicU64::new(0);

#[inline(always)]
pub fn enabled() -> bool {
    // enabled together with the engine profiler
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("LLAMA_RS_PROF").is_ok())
}

#[inline(always)]
pub fn add(a: &AtomicU64, v: u64) {
    if enabled() {
        a.fetch_add(v, Ordering::Relaxed);
    }
}

/// Clear all counters (used between benchmark repetitions).
pub fn reset() {
    for a in [&QUANT_TICKS, &DOT_TICKS, &POOL_ENQ_TICKS, &POOL_WAIT_TICKS, &WAKE_TICKS] {
        a.store(0, Ordering::Relaxed);
    }
    for a in [&WAKE_JOBS, &PARKS, &SPIN_HITS, &MATVECS, &JOB_TICKS,
              &QUANT_ALLOC_COUNT, &QUANT_ALLOC_BYTES,
              &MATVEC_WALL_TICKS, &CALLER_BAND_TICKS,
              &CALLER_SPIN_WINS, &CALLER_PARKS, &SLOT_JOBS,
              &SLOT_WAIT_TICKS] {
        a.store(0, Ordering::Relaxed);
    }
}
