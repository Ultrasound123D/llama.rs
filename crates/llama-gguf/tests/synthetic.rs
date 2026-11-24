//! Synthetic test: build a small GGUF v3 file, then read it back and verify every field.
//! Covers: all KV types (including a non-ASCII UTF-8 string and an array), tensor dims in ne
//! order directly (as gguf.cpp:1550 writes info.t.ne[j]), 32-byte data-region alignment,
//! sequential offset computation, and raw byte access.
use llama_gguf::{GgmlType, GgufFile, Value};

fn u32b(v: u32) -> [u8; 4] { v.to_le_bytes() }
fn u64b(v: u64) -> [u8; 8] { v.to_le_bytes() }

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&u64b(s.len() as u64));
    b.extend_from_slice(s.as_bytes());
}

fn build_gguf() -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&u32b(3)); // version
    b.extend_from_slice(&u64b(2));  // n_tensors
    b.extend_from_slice(&u64b(7));  // n_kv

    // KV 1: general.architecture = "test"
    put_str(&mut b, "general.architecture");
    b.extend_from_slice(&u32b(8));
    put_str(&mut b, "test");
    // KV 2: non-ASCII UTF-8 name (UTF-8 check)
    put_str(&mut b, "general.name");
    b.extend_from_slice(&u32b(8));
    put_str(&mut b, "café 東京");
    // KV 3: test.block_count = 1 (u32)
    put_str(&mut b, "test.block_count");
    b.extend_from_slice(&u32b(4));
    b.extend_from_slice(&u32b(1));
    // KV 4: f64
    put_str(&mut b, "general.custom_f64");
    b.extend_from_slice(&u32b(12));
    b.extend_from_slice(&u64b(1.5f64.to_bits()));
    // KV 5: bool
    put_str(&mut b, "general.custom_bool");
    b.extend_from_slice(&u32b(7));
    b.push(1);
    // KV 6: u32 array
    put_str(&mut b, "general.custom_arr");
    b.extend_from_slice(&u32b(9)); // array
    b.extend_from_slice(&u32b(4)); // u32 element
    b.extend_from_slice(&u64b(3)); // count
    for v in [1u32, 2, 3] {
        b.extend_from_slice(&u32b(v));
    }
    // KV 7: u64
    put_str(&mut b, "general.custom_u64");
    b.extend_from_slice(&u32b(10));
    b.extend_from_slice(&u64b(1 << 40));

    // tensor table (dims in direct ne order)
    // t0: emb.weight F32 [ne=4, 2]
    put_str(&mut b, "emb.weight");
    b.extend_from_slice(&u32b(2));
    b.extend_from_slice(&u64b(4));
    b.extend_from_slice(&u64b(2));
    b.extend_from_slice(&u32b(0)); // F32
    b.extend_from_slice(&u64b(0)); // offset
    // t1: wq.weight Q4_0 [ne=32, 1]
    put_str(&mut b, "wq.weight");
    b.extend_from_slice(&u32b(2));
    b.extend_from_slice(&u64b(32));
    b.extend_from_slice(&u64b(1));
    b.extend_from_slice(&u32b(2)); // Q4_0
    b.extend_from_slice(&u64b(0)); // offset

    // 32-byte alignment for the data region (GGUF_ALIGNMENT)
    while b.len() % 32 != 0 {
        b.push(0);
    }

    // t0 data: 8 f32 values
    for i in 1..=8i32 {
        b.extend_from_slice(&(i as f32).to_le_bytes());
    }
    // t1 data: one Q4_0 block: d=f16(1.0)=0x3C00 + qs[16]=0x21
    b.extend_from_slice(&0x3C00u16.to_le_bytes());
    for _ in 0..16 {
        b.push(0x21);
    }
    b
}

#[test]
fn synthetic_gguf_roundtrip() {
    let path = std::env::temp_dir().join("llama_rs_synthetic.gguf");
    std::fs::write(&path, build_gguf()).unwrap();

    let f = GgufFile::open(path.to_str().unwrap()).unwrap();

    // metadata
    assert_eq!(f.version, 3);
    assert_eq!(f.arch(), Some("test"));
    assert_eq!(
        f.metadata.get("general.name").and_then(|v| v.as_str()),
        Some("café 東京")
    );
    // get_meta resolves by architecture first (test.block_count)
    assert_eq!(f.get_meta("block_count").and_then(|v| v.as_u64()), Some(1));
    assert_eq!(f.metadata.get("general.custom_f64"), Some(&Value::F64(1.5)));
    assert_eq!(f.metadata.get("general.custom_bool"), Some(&Value::Bool(true)));
    assert_eq!(f.metadata.get("general.custom_u64"), Some(&Value::U64(1 << 40)));
    let arr: Vec<u64> = f
        .metadata
        .get("general.custom_arr")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(arr, vec![1, 2, 3]);

    // tensors
    assert_eq!(f.tensors.len(), 2);
    let t0 = f.tensor("emb.weight").unwrap();
    assert_eq!(t0.dims, vec![4, 2]);
    assert_eq!(t0.ty, GgmlType::F32);
    let t1 = f.tensor("wq.weight").unwrap();
    assert_eq!(t1.dims, vec![32, 1]);
    assert_eq!(t1.ty, GgmlType::Q4_0);

    // offsets: region aligned, t0 at 0 (32 bytes), t1 at 32 (18 bytes)
    assert_eq!(f.data_start % 32, 0);
    assert!(f.data_start > 0);
    assert_eq!(t0.offset, 0);
    assert_eq!(t1.offset, 32);
    assert_eq!(f.n_bytes_total(), 32 + 18);

    // t0 bytes = the same 8 values
    let b0 = f.tensor_bytes(t0);
    assert_eq!(b0.len(), 32);
    for i in 0..8 {
        let v = f32::from_le_bytes([b0[4 * i], b0[4 * i + 1], b0[4 * i + 2], b0[4 * i + 3]]);
        assert_eq!(v, (i + 1) as f32);
    }
    // t1 bytes: f16(1.0) then 0x21
    let b1 = f.tensor_bytes(t1);
    assert_eq!(b1.len(), 18);
    assert_eq!(u16::from_le_bytes([b1[0], b1[1]]), 0x3C00);
    assert!(b1[2..].iter().all(|&q| q == 0x21));

    std::fs::remove_file(&path).ok();
}

#[test]
fn rejects_bad_magic() {
    let path = std::env::temp_dir().join("llama_rs_not_gguf.bin");
    std::fs::write(&path, b"NOTGGUF...").unwrap();
    assert!(GgufFile::open(path.to_str().unwrap()).is_err());
    std::fs::remove_file(&path).ok();
}
