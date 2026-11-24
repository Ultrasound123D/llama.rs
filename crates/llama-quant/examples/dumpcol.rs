// Parity test of dequantization against the C reference (original ggml code)
// Reads one column of a tensor and compares the values
use llama_gguf::GgufFile;
use llama_quant::column_f32;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let tensor = std::env::args().nth(2).unwrap();
    let col: usize = std::env::args().nth(3).unwrap().parse().unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let t = gguf.tensor(&tensor).unwrap();
    let n = t.dims[0] as usize;
    let data = gguf.tensor_bytes(t);
    let vpb = match t.ty {
        llama_gguf::GgmlType::F32 | llama_gguf::GgmlType::F16 | llama_gguf::GgmlType::BF16 => 1,
        _ => llama_quant::vals_per_block(t.ty),
    };
    let col_bytes = (n / vpb) * t.ty.block_size() as usize;
    eprintln!("tensor {} offset_in_file={} dims={:?} ty={:?} col_bytes={}", tensor, gguf.data_start as usize + t.offset as usize, t.dims, t.ty, col_bytes);
    let v = column_f32(t.ty, data, col, n);
    for (i, val) in v.iter().enumerate() {
        println!("{i} {}", format!("{:.9}", val));
    }
}
