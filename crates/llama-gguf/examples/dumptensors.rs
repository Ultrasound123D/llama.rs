// dump tensor name classes: (suffix, type, dims) with counts
use llama_gguf::GgufFile;
use std::collections::BTreeMap;
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let mut seen: BTreeMap<(String, String, u64), usize> = BTreeMap::new();
    for t in &gguf.tensors {
        let name = &t.name;
        let cls: String = if name.starts_with("blk.") {
            let parts: Vec<&str> = name.split('.').collect();
            parts.iter().skip(2).cloned().collect::<Vec<_>>().join(".")
        } else { name.clone() };
        let key = (cls, format!("{:?}", t.ty), t.dims[0]);
        *seen.entry(key).or_insert(0) += 1;
    }
    for ((cls, ty, d0), c) in &seen {
        println!("{:<34} {:>6} n_in={:>6} x{}", cls, ty, d0, c);
    }
}
