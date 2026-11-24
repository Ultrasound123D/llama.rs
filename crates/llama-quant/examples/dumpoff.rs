// Tensor offset diagnostics: print the first tensors with offsets and sizes
use llama_gguf::GgufFile;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    println!("data_start: {}", gguf.data_start);
    let mut cum = 0u64;
    for (i, t) in gguf.tensors.iter().take(6).enumerate() {
        let n_vals: u64 = t.dims.iter().product::<u64>().max(1);
        let per = t.ty.block_vals();
        let bs = t.ty.block_size();
        let blocks = n_vals.div_ceil(per);
        println!("{}: {} dims={:?} ty={:?} t.offset={} computed_size={} cum={}", i, t.name, t.dims, t.ty, t.offset, blocks * bs, cum);
        cum += blocks * bs;
    }
}
