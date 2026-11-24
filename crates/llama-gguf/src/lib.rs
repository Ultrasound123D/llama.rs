//! llama-gguf — GGUF v3 reader
//! Rust port of the behavior of ggml/src/gguf.cpp + llama-model-loader.cpp (b12502):
//! header {GGUF, version, n_tensors, n_kv}, then KV metadata, then the tensor table,
//! then tensor data aligned to GGUF_GENERAL_ALIGNMENT = 32.
// Quantization type names match ggml.h verbatim (Q4_K, Q5_K, ...) to keep
// reference parity with llama.cpp naming — so this naming is allowed deliberately.
#![allow(non_camel_case_types)]
use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read};

/// Metadata "type" values as in gguf.h
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32),
    F32(f32), Bool(bool), String(String),
    U64(u64), I64(i64), F64(f64),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Value::U32(v) => Some(*v),
            Value::U16(v) => Some(*v as u32),
            Value::I32(v) => Some(*v as u32),
            Value::U64(v) => Some(*v as u32),
            Value::I64(v) => Some(*v as u32),
            Value::U8(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U64(v) => Some(*v),
            Value::U32(v) => Some(*v as u64),
            _ => self.as_u32().map(|v| v as u64),
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::F32(v) => Some(*v),
            Value::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self { Value::String(s) => Some(s.as_str()), _ => None }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self { Value::Bool(b) => Some(*b), _ => None }
    }
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self { Value::Array(a) => Some(a), _ => None }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Array(a) => {
                if a.len() > 16 { write!(f, "ARRAY[{}]", a.len()) }
                else { write!(f, "{:?}", a) }
            }
            Value::String(s) => write!(f, "{:?}", if s.len() > 48 { format!("{}…", &s[..40]) } else { s.clone() }),
            _ => write!(f, "{:?}", self),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32, F16, Q4_0, Q4_1, Q4_1_3BIT, Q5_0, Q5_1, Q8_0, Q8_1,
    Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, I8, BF16, AUTO, BITS,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<GgmlType> {
        // official ggml.h enum: F32=0 F16=1 Q4_0=2 Q4_1=3 Q5_0=6 Q5_1=7 Q8_0=8 Q8_1=9
        // Q2_K=10 Q3_K=11 Q4_K=12 Q5_K=13 Q6_K=14 Q8_K=15 I8=24 BF16=30
        use GgmlType::*;
        Some(match v {
            0 => F32, 1 => F16, 2 => Q4_0, 3 => Q4_1, 6 => Q5_0, 7 => Q5_1,
            8 => Q8_0, 9 => Q8_1, 10 => Q2_K, 11 => Q3_K, 12 => Q4_K,
            13 => Q5_K, 14 => Q6_K, 15 => Q8_K, 24 => I8, 30 => BF16,
            _ => return None,
        })
    }
    pub fn id(&self) -> u32 {
        use GgmlType::*;
        match self {
            F32 => 0, F16 => 1, Q4_0 => 2, Q4_1 => 3, Q5_0 => 6, Q5_1 => 7,
            Q8_0 => 8, Q8_1 => 9, Q2_K => 10, Q3_K => 11, Q4_K => 12,
            Q5_K => 13, Q6_K => 14, Q8_K => 15, I8 => 24, BF16 => 30,
            AUTO | BITS | Q4_1_3BIT => 0,
        }
    }
    /// Size of one block in bytes
    pub fn block_size(&self) -> u64 {
        use GgmlType::*;
        match self {
            F32 => 4, F16 => 2, BF16 => 2, I8 => 1,
            Q4_0 => 2 + 16,             // d + qs[16]
            Q4_1 => 4 + 16,             // d,m + qs[16]
            Q5_0 => 2 + 4 + 16,          // d + qh(4) + qs[16]
            Q5_1 => 4 + 4 + 16,
            Q8_0 => 2 + 32,              // d + qs[32]
            Q8_1 => 4 + 32,
            Q2_K => 2 + 16 + 64,         // d+dmin + scales[16] + qs[64]
            Q3_K => 2 + 12 + 32 + 64,    // d + scales[12] + hmask[32] + qs[64]
            Q4_K => 2 + 2 + 12 + 128,    // d+dmin + scales[12] + qs[128]  (144)
            Q5_K => 2 + 2 + 12 + 32 + 128, // 176
            Q6_K => 2 + 16 + 128 + 64,   // d + scales[16] + ql[128] + qh[64] (210)
            Q8_K => 4 + 256 + 32,
            AUTO | BITS | Q4_1_3BIT => 0,
        }
    }
    /// Number of values in the block
    pub fn block_vals(&self) -> u64 {
        use GgmlType::*;
        match self {
            F32 | F16 | BF16 | I8 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 => 32,
            Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | Q8_K => 256,
            AUTO | BITS | Q4_1_3BIT => 1,
        }
    }
    pub fn name(&self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "F32", F16 => "F16", BF16 => "BF16", I8 => "I8",
            Q4_0 => "Q4_0", Q4_1 => "Q4_1", Q5_0 => "Q5_0", Q5_1 => "Q5_1",
            Q8_0 => "Q8_0", Q8_1 => "Q8_1", Q2_K => "Q2_K", Q3_K => "Q3_K",
            Q4_K => "Q4_K", Q5_K => "Q5_K", Q6_K => "Q6_K", Q8_K => "Q8_K",
            Q4_1_3BIT => "Q4_1_3BIT", AUTO => "AUTO", BITS => "BITS",
        }
    }
}

/// Number of elements per dimension (ggml stores reversed: ne[0] is the fastest dimension)
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// dims[0..n] in gguf order (ne[0] first)
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Offset of the tensor within the file (after the tensor table) — relative to the end of the table
    pub offset: u64,
}

pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
    /// Offset of the first byte after the tensor table (start of the data region)
    pub data_start: u64,
    /// The whole file in memory (a 1GB model is acceptable; the intent is to stay quantized)
    pub data: Vec<u8>,
}

const GGUF_ALIGNMENT: u64 = 32;

impl GgufFile {
    pub fn open(path: &str) -> io::Result<GgufFile> {
        let mut data = Vec::new();
        {
            let mut f = std::fs::File::open(path)?;
            f.read_to_end(&mut data)?;
        }
        let mut r = Cursor { data: &data, pos: 0 };
        let magic = r.read_u32()?;
        if magic != 0x46554747 { // "GGUF" little-endian
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a GGUF file"));
        }
        let version = r.read_u32()?;
        if version < 2 || version > 3 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unsupported GGUF version {version}")));
        }
        let n_tensors = r.read_u64()?;
        let n_kv = r.read_u64()?;
        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let k = r.read_string()?;
            let v = r.read_value()?;
            metadata.insert(k, v);
        }
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = r.read_string()?;
            let n_dims = r.read_u32()?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(r.read_u64()?);
            }
            let ty = GgmlType::from_u32(r.read_u32()?)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad tensor type"))?;
            r.read_u64()?; // offset within the data region (recomputed, since we read sequentially)
            tensors.push(TensorInfo { name, dims, ty, offset: 0 });
        }
        let table_end = r.pos as u64;
        // 32-byte alignment for the data region (same as gguf.cpp: GGUF_ALIGNMENT)
        let data_start = (table_end + GGUF_ALIGNMENT - 1) & !(GGUF_ALIGNMENT - 1);
        let mut offset = 0u64;
        for t in tensors.iter_mut() {
            t.offset = offset;
            offset += tensor_bytes(t);
        }
        let tensor_index = tensors.iter().enumerate()
            .map(|(i, t)| (t.name.clone(), i)).collect();
        Ok(GgufFile { version, metadata, tensors, tensor_index, data_start, data })
    }

    pub fn arch(&self) -> Option<&str> {
        self.metadata.get("general.architecture").and_then(|v| v.as_str())
    }

    pub fn get_meta(&self, key: &str) -> Option<&Value> {
        // look up by architecture first (lfm3.*), then general (ggml.)
        let arch = self.arch().unwrap_or("");
        if !arch.is_empty() {
            if let Some(v) = self.metadata.get(&format!("{arch}.{key}")) {
                return Some(v);
            }
        }
        self.metadata.get(&format!("general.{key}"))
            .or_else(|| self.metadata.get(key))
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    /// Byte slice of a whole tensor (dims may later specify sub-groups)
    pub fn tensor_bytes(&self, t: &TensorInfo) -> &[u8] {
        let start = self.data_start as usize + t.offset as usize;
        let len = tensor_bytes(t) as usize;
        &self.data[start..start + len]
    }

    pub fn n_bytes_total(&self) -> u64 {
        self.tensors.iter().map(tensor_bytes).sum()
    }
}

fn tensor_bytes(t: &TensorInfo) -> u64 {
    let n_vals: u64 = t.dims.iter().product::<u64>().max(1);
    if t.ty == GgmlType::I8 {
        return n_vals; // special case
    }
    let per = t.ty.block_vals();
    let blocks = n_vals.div_ceil(per);
    blocks * t.ty.block_size()
}

struct Cursor<'a> { data: &'a [u8], pos: usize }

impl<'a> Cursor<'a> {
    fn read_u32(&mut self) -> io::Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_u64(&mut self) -> io::Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }
    fn take(&mut self, n: usize) -> io::Result<&[u8]> {
        if self.pos + n > self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "eof"));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn read_string(&mut self) -> io::Result<String> {
        let n = self.read_u64()? as usize;
        if n > 10_000_000 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "string too large"));
        }
        let b = self.take(n)?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }
    fn read_value(&mut self) -> io::Result<Value> {
        let t = self.read_u32()?;
        Ok(match t {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            4 => Value::U32(self.read_u32()?),
            5 => Value::I32(i32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            6 => Value::F32(f32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::String(self.read_string()?),
            9 => {
                let et = self.read_u32()?;
                let n = self.read_u64()? as usize;
                if n > 5_000_000 { return Err(io::Error::new(io::ErrorKind::InvalidData, "array too large")); }
                let mut vals = Vec::with_capacity(n.min(1 << 16));
                for _ in 0..n {
                    // read every element with the same element type — consume it fully to stay in sync
                    let saved = Cursor { data: self.data, pos: self.pos };
                    let mut sub = saved;
                    let v = sub.read_typed(et)?;
                    self.pos = sub.pos;
                    vals.push(v);
                }
                Value::Array(vals)
            }
            10 => Value::U64(self.read_u64()?),
            11 => Value::I64(i64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            12 => Value::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("bad kv type {t}"))),
        })
    }
    fn read_typed(&mut self, t: u32) -> io::Result<Value> {
        Ok(match t {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            4 => Value::U32(self.read_u32()?),
            5 => Value::I32(i32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            6 => Value::F32(f32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::String(self.read_string()?),
            10 => Value::U64(self.read_u64()?),
            11 => Value::I64(i64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            12 => Value::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "bad array elem type")),
        })
    }
}
