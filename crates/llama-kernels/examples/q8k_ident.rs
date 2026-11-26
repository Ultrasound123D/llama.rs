// value-identity test: SIMD q8_K quantizer vs the scalar reference algorithm
use llama_kernels::quant::{quantize_row_q8_k, quantize_row_q8_k_scalar};

fn nearest_int(f: f32) -> i32 {
    // ggml's magic-number rounding (bit-equivalent to RNE on this domain)
    const MAGIC: f32 = 12582912.0; // 2^23 + 2^22
    let out = (f + MAGIC) as i64 as i32;
    out
}

fn scalar_ref(x: &[f32], out: &mut [u8]) {
    let nb = x.len() / 256;
    for i in 0..nb {
        let xb = &x[i * 256..(i + 1) * 256];
        let mut max = 0f32;
        let mut amax = 0f32;
        for &v in xb {
            let a = v.abs();
            if a > amax {
                amax = a;
                max = v;
            }
        }
        let base = i * 292;
        if amax == 0.0 {
            out[base..base + 4].copy_from_slice(&0f32.to_le_bytes());
            out[base + 4..base + 260].fill(0);
            out[base + 260..base + 292].fill(0);
            continue;
        }
        let iscale = -127.0f32 / max;
        for (j, &v) in xb.iter().enumerate() {
            let vq = nearest_int(iscale * v);
            out[base + 4 + j] = vq.min(127) as u8;
        }
        for j in 0..16 {
            let mut sum = 0i32;
            for q in &out[base + 4 + j * 16..base + 4 + j * 16 + 16] {
                sum += *q as i8 as i32;
            }
            out[base + 260 + 2 * j..base + 260 + 2 * j + 2].copy_from_slice(&(sum as i16).to_le_bytes());
        }
        let d = 1.0f32 / iscale;
        out[base..base + 4].copy_from_slice(&d.to_le_bytes());
    }
}

fn main() {
    let mut st = 0x12345678u64;
    let mut xr = |st: &mut u64| { let mut x = *st; x ^= x << 13; x ^= x >> 7; x ^= x << 17; *st = x; (x >> 32) as i64 };
    for case in 0..50 {
        let n = (case % 7 + 1) * 256;
        let x: Vec<f32> = (0..n).map(|_| {
            let r = xr(&mut st) % 2000;
            (r as f32 / 400.0) - 2.4
        }).collect();
        let mut a = vec![0u8; n / 256 * 292];
        let mut b = vec![0u8; n / 256 * 292];
        quantize_row_q8_k(&x, &mut a);
        quantize_row_q8_k_scalar(&x, &mut b);
        if a != b {
            let mut diffs = 0;
            for i in 0..a.len() { if a[i] != b[i] { diffs += 1; if diffs < 8 { println!("  case {case} byte {i}: simd={} ref={}", a[i], b[i]); } } }
            println!("case {case}: DIFFER ({diffs} bytes, n={n})");
        } else {
            println!("case {case}: identical (n={n})");
        }
    }
    // edge: all-zero block, extreme-positive, extreme-negative
    for (name, x) in [("all-zero", vec![0f32; 256]), ("pos-ext", {
        let mut v = vec![0.01f32; 256]; v[7] = 3.5; v
    }), ("neg-ext", {
        let mut v = vec![-0.01f32; 256]; v[3] = -5.5; v
    })] {
        let mut a = vec![0u8; 292];
        let mut b = vec![0u8; 292];
        quantize_row_q8_k(&x, &mut a);
        quantize_row_q8_k_scalar(&x, &mut b);
        println!("edge {name}: {}", if a == b { "identical" } else { "DIFFER" });
    }
}
