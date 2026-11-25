//! Phase 6.6: achievable DRAM read-bandwidth ceiling for this machine.
//!
//! Streams a buffer through a 4-accumulator u64 folding sum (the compiler
//! cannot elide the reads; 2 independent loads per iteration to avoid being
//! add-port-limited) with 1 and 2 threads, taking 3 passes per config:
//! pass 1 may still hit L3 (504 MB here); the steady-state passes are the
//! honest DRAM number. The lowest pass is what decode matvecs are compared
//! against in the audit — a decode token streams the whole packed weight
//! set once, so tok/s × model-bytes cannot beat this number.
//!
//! Run: cargo run --release --example drambw --manifest-path crates/llama-kernels/Cargo.toml [-- 1]

use std::convert::TryInto;

fn fold(b: &[u8]) -> u64 {
    // 4 independent accumulators, 32 bytes per iteration
    let mut acc = [0u64; 4];
    for c in b.chunks_exact(32) {
        acc[0] = acc[0].wrapping_add(u64::from_le_bytes(c[0..8].try_into().unwrap()));
        acc[1] = acc[1].wrapping_add(u64::from_le_bytes(c[8..16].try_into().unwrap()));
        acc[2] = acc[2].wrapping_add(u64::from_le_bytes(c[16..24].try_into().unwrap()));
        acc[3] = acc[3].wrapping_add(u64::from_le_bytes(c[24..32].try_into().unwrap()));
    }
    acc[0] ^ acc[1] ^ acc[2] ^ acc[3]
}

fn bench(gb: usize, threads: usize, passes: usize) -> f64 {
    let n = gb * 1024 * 1024 * 1024;
    let buf = vec![1u8; n];
    let chunk = n / threads;
    let mut worst = f64::MAX;
    std::thread::scope(|s| {
        for p in 1..=passes {
            let mut sums = vec![];
            let t0 = std::time::Instant::now();
            for t in 0..threads {
                let b = &buf[t * chunk..(t + 1) * chunk];
                sums.push(s.spawn(move || fold(b)));
            }
            let total: u64 = sums.into_iter().map(|h| h.join().unwrap()).sum::<u64>();
            let secs = t0.elapsed().as_secs_f64();
            let gbs = n as f64 / 1e9 / secs;
            println!(
                "threads={threads} pass={p}/{passes}: {secs:.3}s = {gbs:.2} GB/s over {gb} GiB (acc={total:x})"
            );
            worst = worst.min(gbs);
        }
    });
    worst
}

fn main() {
    let gb: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    println!("DRAM streaming-read ceiling (linear, 4-acc u64 fold, 3 passes; L3 = 504 MB):");
    let _one = bench(gb, 1, 3);
    println!();
    let two = bench(gb, 2, 3);
    println!("#audit dram_ceiling_1t={_one:.2} dram_ceiling_2t={two:.2} GB/s (min pass)");
}
