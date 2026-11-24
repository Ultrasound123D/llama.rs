//! debug: per-column golden vs engine paths for one extreme case
use llama_kernels::{avx2, scalar};

fn main() {
    let pfx = std::env::var("PFX").unwrap_or("/tmp/diag".into());
    let tensor = std::fs::read(format!("{pfx}_tensor.bin")).unwrap();
    let q = std::fs::read(format!("{pfx}_qact_rt.bin")).unwrap();
    let x_b = std::fs::read(format!("{pfx}_x_f32.bin")).unwrap();
    let x: Vec<f32> = x_b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let golden: Vec<f64> = std::fs::read_to_string(format!("{pfx}_out_rt.txt")).unwrap()
        .split_whitespace().map(|t| t.parse().unwrap()).collect();
    let (n_in, n_out, kb) = (96usize, 48usize, 3usize);
    println!("{:>4} {:>18} {:>18} {:>18} {:>18}", "col", "golden", "scalar", "avx2", "vnni256");
    for col in 0..n_out {
        let w = &tensor[col * 34 * kb..(col + 1) * 34 * kb];
        let s = scalar::dot_q8_0_q8_0(w, &q, n_in);
        let a = unsafe { avx2::dot_q8_0_q8_0(w, &q, n_in) };
        let v = unsafe { avx2::dot_q8_0_q8_0_vnni(w, &q, n_in) };
        println!("{col:>4} {:+18.6e} {:+18.6e} {:+18.6e} {:+18.6e}", golden[col], s, a, v);
    }
    eprintln!("x[0..8] = {:?}", &x[0..8]);
}
