//! llama-quant — بنى كتل ggml + فك التكميم + نواة matvec
//! نقل حرفي لخوارزميات ggml/src/ggml-quants.c (b12502):
//! get_scale_min_k4:889, de q4_0:462, de q4_1:482, de q8_0:556,
//! de q4_K:1539, de q5_K:1741, de q6_K:1949.
use llama_gguf::GgmlType;

// ---------- f16 / bf16 ----------

/// IEEE 754 half → f32 (نفس سلوك GGML_FP16_TO_FP32)
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x03FF) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // subnormal: value = mant × 2^-24 — نُطبّع: نزيح حتى البت 10 (0x400)
            let mut s = 0u32;
            let mut m = mant;
            while m & 0x0400 == 0 {
                m <<= 1;
                s += 1;
            }
            // القيمة = 1.f × 2^(10 - s - 24) → حقل الأس = 127 + 10 - s - 24
            let e32 = 127u32 + 10 - s - 24;
            ((sign as u32) << 31) | (e32 << 23) | (((m & 0x03FF) as u32) << 13)
        }
    } else if exp == 0x1F {
        ((sign as u32) << 31) | (0xFF << 23) | (mant << 13)
    } else {
        let e32 = (exp - 15 + 127) as u32;
        ((sign as u32) << 31) | (e32 << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

#[inline]
fn read_f16(b: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
}

/// BF16 → f32
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ---------- get_scale_min_k4 (مصدر: ggml-quants.c:889) ----------

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    let (d, m) = if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | (((q[j - 4] >> 6) as u16) << 4) as u8;
        let m = (q[j + 4] >> 4) | (((q[j] >> 6) as u16) << 4) as u8;
        (d, m)
    };
    (d as f32, m as f32)
}

// ---------- dequant_row: كتلة واحدة → 256 قيمة f32 ----------

/// يفك كتلة واحدة (values_per_block قيمة) في buf
pub fn dequant_block(ty: GgmlType, b: &[u8], out: &mut [f32]) {
    match ty {
        GgmlType::F32 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
            }
        }
        GgmlType::F16 => {
            for i in 0..out.len() {
                out[i] = read_f16(&b[2 * i..]);
            }
        }
        GgmlType::BF16 => {
            for i in 0..out.len() {
                out[i] = bf16_to_f32(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]));
            }
        }
        GgmlType::Q4_0 => {
            // block: d(f16) + qs[16]
            let d = read_f16(b);
            for j in 0..16 {
                let x0 = (b[2 + j] & 0x0F) as i32 - 8;
                let x1 = (b[2 + j] >> 4) as i32 - 8;
                out[j] = x0 as f32 * d;
                out[16 + j] = x1 as f32 * d;
            }
        }
        GgmlType::Q4_1 => {
            let d = read_f16(b);
            let m = read_f16(&b[2..]);
            for j in 0..16 {
                let x0 = (b[4 + j] & 0x0F) as f32;
                let x1 = (b[4 + j] >> 4) as f32;
                out[j] = x0 * d + m;
                out[16 + j] = x1 * d + m;
            }
        }
        GgmlType::Q5_0 => {
            // block(22): d(f16) + qh(u32) + qs[16]  (dequantize_row_q5_0:503)
            let d = read_f16(b);
            let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
            for j in 0..16 {
                let xh_0 = (((qh >> j) << 4) & 0x10) as i32;
                let xh_1 = ((qh >> (j + 12)) & 0x10) as i32;
                let x0 = ((b[6 + j] & 0x0F) as i32 | xh_0) - 16;
                let x1 = ((b[6 + j] >> 4) as i32 | xh_1) - 16;
                out[j] = x0 as f32 * d;
                out[16 + j] = x1 as f32 * d;
            }
        }
        GgmlType::Q5_1 => {
            // block(22): d,m(f16) + qh(u32) + qs[16]  (dequantize_row_q5_1:529)
            let d = read_f16(b);
            let m = read_f16(&b[2..]);
            let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            for j in 0..16 {
                let xh_0 = (((qh >> j) << 4) & 0x10) as i32;
                let xh_1 = ((qh >> (j + 12)) & 0x10) as i32;
                let x0 = ((b[8 + j] & 0x0F) as i32 | xh_0) as f32;
                let x1 = ((b[8 + j] >> 4) as i32 | xh_1) as f32;
                out[j] = x0 * d + m;
                out[16 + j] = x1 * d + m;
            }
        }
        GgmlType::Q8_0 => {
            let d = read_f16(b);
            for j in 0..32 {
                out[j] = (b[2 + j] as i8) as f32 * d;
            }
        }
        GgmlType::Q4_K => {
            // block(144): d, dmin, scales[12], qs[128]
            let d = read_f16(b);
            let dmin = read_f16(&b[2..]);
            let scales = &b[4..16];
            let qs = &b[16..144];
            let mut is = 0usize;
            for g in 0..4usize {
                let (sc, m) = get_scale_min_k4(is, scales);
                let d1 = d * sc;
                let m1 = dmin * m;
                let (sc, m) = get_scale_min_k4(is + 1, scales);
                let d2 = d * sc;
                let m2 = dmin * m;
                let q = &qs[g * 32..g * 32 + 32];
                for l in 0..32 {
                    out[g * 64 + l] = d1 * (q[l] & 0xF) as f32 - m1;
                }
                for l in 0..32 {
                    out[g * 64 + 32 + l] = d2 * (q[l] >> 4) as f32 - m2;
                }
                is += 2;
            }
        }
        GgmlType::Q5_K => {
            // block(176): d, dmin, scales[12], qh[32], qs[128]
            let d = read_f16(b);
            let dmin = read_f16(&b[2..]);
            let scales = &b[4..16];
            let qh = &b[16..48];
            let qs = &b[48..176];
            let mut is = 0usize;
            let mut u1: u8 = 1;
            let mut u2: u8 = 2;
            for g in 0..4usize {
                let (sc, m) = get_scale_min_k4(is, scales);
                let d1 = d * sc;
                let m1 = dmin * m;
                let (sc, m) = get_scale_min_k4(is + 1, scales);
                let d2 = d * sc;
                let m2 = dmin * m;
                let ql = &qs[g * 32..g * 32 + 32];
                for l in 0..32 {
                    let hi = if qh[g * 32 + l] & u1 != 0 { 16 } else { 0 };
                    out[g * 64 + l] = d1 * ((ql[l] & 0xF) as f32 + hi as f32) - m1;
                }
                for l in 0..32 {
                    let hi = if qh[g * 32 + l] & u2 != 0 { 16 } else { 0 };
                    out[g * 64 + 32 + l] = d2 * ((ql[l] >> 4) as f32 + hi as f32) - m2;
                }
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        GgmlType::Q6_K => {
            // block(210): ql[128], qh[64], scales i8[16], d f16
            let ql = &b[0..128];
            let qh = &b[128..192];
            let sc = &b[192..208];
            let d = read_f16(&b[208..]);
            for n in 0..2usize {
                let ql_n = &ql[n * 64..n * 64 + 64];
                let qh_n = &qh[n * 32..n * 32 + 32];
                let sc_n = &sc[n * 8..n * 8 + 8];
                for l in 0..32 {
                    let is_ = l / 16;
                    let q1 = (((ql_n[l] & 0xF) as i32) | (((qh_n[l] & 3) as i32) << 4)) - 32;
                    let q2 = (((ql_n[l + 32] & 0xF) as i32) | ((((qh_n[l] >> 2) & 3) as i32) << 4)) - 32;
                    let q3 = (((ql_n[l] >> 4) as i32) | ((((qh_n[l] >> 4) & 3) as i32) << 4)) - 32;
                    let q4 = (((ql_n[l + 32] >> 4) as i32) | ((((qh_n[l] >> 6) & 3) as i32) << 4)) - 32;
                    let base = n * 128;
                    out[base + l] = d * (sc_n[is_] as i8) as f32 * q1 as f32;
                    out[base + 32 + l] = d * (sc_n[is_ + 2] as i8) as f32 * q2 as f32;
                    out[base + 64 + l] = d * (sc_n[is_ + 4] as i8) as f32 * q3 as f32;
                    out[base + 96 + l] = d * (sc_n[is_ + 6] as i8) as f32 * q4 as f32;
                }
            }
        }
        other => unimplemented!("quant type {other:?} غير مدعوم بعد"),
    }
}

pub fn vals_per_block(ty: GgmlType) -> usize {
    match ty {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => 1,
        GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0 | GgmlType::Q8_1 => 32,
        _ => 256,
    }
}

/// تفكيك عمود كامل من الموتر (n قيمة متتالية على ne[0]) إلى f32
pub fn dequant_slice(ty: GgmlType, data: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    dequant_into(ty, data, &mut out);
    out
}

pub fn dequant_into(ty: GgmlType, data: &[u8], out: &mut [f32]) {
    let n = out.len();
    match ty {
        GgmlType::F32 => {
            for i in 0..n {
                out[i] = f32::from_le_bytes([data[4 * i], data[4 * i + 1], data[4 * i + 2], data[4 * i + 3]]);
            }
        }
        GgmlType::F16 => {
            for i in 0..n {
                out[i] = read_f16(&data[2 * i..]);
            }
        }
        GgmlType::BF16 => {
            for i in 0..n {
                out[i] = bf16_to_f32(u16::from_le_bytes([data[2 * i], data[2 * i + 1]]));
            }
        }
        _ => {
            let vpb = vals_per_block(ty);
            let bs = ty.block_size() as usize;
            let nb = n / vpb;
            let mut buf = [0f32; 256];
            for i in 0..nb {
                dequant_block(ty, &data[i * bs..(i + 1) * bs], &mut buf[..vpb]);
                out[i * vpb..(i + 1) * vpb].copy_from_slice(&buf[..vpb]);
            }
        }
    }
}

// ---------- matvec: out[o] = dot(x, W[:, o]) ----------

/// حاصل ضرب x × عمود o من موتر مكمّم.
/// الموتر بترتيب ggml: ne[0]=n_in (متصل)، ne[1]=n_out؛ عمود كامل متتالٍ من n_in قيمة.
pub fn dot_column(ty: GgmlType, data: &[u8], col: usize, n_in: usize, x: &[f32]) -> f32 {
    match ty {
        GgmlType::F32 => {
            let mut s = 0f32;
            for i in 0..n_in {
                let b = [data[(col * n_in + i) * 4], data[(col * n_in + i) * 4 + 1],
                         data[(col * n_in + i) * 4 + 2], data[(col * n_in + i) * 4 + 3]];
                s += f32::from_le_bytes(b) * x[i];
            }
            s
        }
        GgmlType::F16 => {
            let mut s = 0f32;
            for i in 0..n_in {
                let o = (col * n_in + i) * 2;
                s += read_f16(&data[o..]) * x[i];
            }
            s
        }
        GgmlType::BF16 => {
            let mut s = 0f32;
            for i in 0..n_in {
                let o = (col * n_in + i) * 2;
                s += bf16_to_f32(u16::from_le_bytes([data[o], data[o + 1]])) * x[i];
            }
            s
        }
        _ => {
            let vpb = vals_per_block(ty);
            let bs = ty.block_size() as usize;
            let nb = n_in / vpb;
            let mut buf = [0f32; 256];
            let base = col * nb * bs;
            let mut acc = 0f32;
            for i in 0..nb {
                dequant_block(ty, &data[base + i * bs..base + (i + 1) * bs], &mut buf[..vpb]);
                let xs = &x[i * vpb..(i + 1) * vpb];
                let bs_ = &buf[..vpb];
                for j in 0..vpb {
                    acc += bs_[j] * xs[j];
                }
            }
            acc
        }
    }
}

/// matvec كامل متوازٍ على الأعمدة (نواة v0: إلغاء تكميم داخل الحلقة)
/// `data` شريطة الموتر الكاملة، ne=[n_in, n_out]
pub fn matvec(ty: GgmlType, data: &[u8], n_in: usize, n_out: usize, x: &[f32], n_threads: usize, out: &mut [f32]) {
    let n_threads = n_threads.max(1);
    let chunk = (n_out + n_threads - 1) / n_threads;
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..n_threads {
            let start = t * chunk;
            let end = ((t + 1) * chunk).min(n_out);
            if start >= end { continue; }
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().add(start), end - start) };
            handles.push(s.spawn(move || {
                for o in start..end {
                    out_slice[o - start] = dot_column(ty, data, o, n_in, x);
                }
            }));
        }
        for h in handles { h.join().unwrap(); }
    });
}

/// جلب عمود كامل فك تكميمه (مفيد للموارد الصغيرة: norms, conv kernel)
pub fn column_f32(ty: GgmlType, data: &[u8], col: usize, n: usize) -> Vec<f32> {
    match ty {
        GgmlType::F32 => {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let o = (col * n + i) * 4;
                v.push(f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]));
            }
            v
        }
        GgmlType::F16 | GgmlType::BF16 => {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let o = (col * n + i) * 2;
                if ty == GgmlType::F16 { v.push(read_f16(&data[o..])); }
                else { v.push(bf16_to_f32(u16::from_le_bytes([data[o], data[o + 1]]))); }
            }
            v
        }
        _ => {
            let vpb = vals_per_block(ty);
            let bs = ty.block_size() as usize;
            let nb = n / vpb;
            let base = col * nb * bs;
            let mut buf = [0f32; 256];
            let mut v = Vec::with_capacity(n);
            for i in 0..nb {
                dequant_block(ty, &data[base + i * bs..base + (i + 1) * bs], &mut buf[..vpb]);
                v.extend_from_slice(&buf[..vpb]);
            }
            v
        }
    }
}

/// تفكيك متجه أوزان [n] كامل (norms مثلاً) — موتر أحادي البعد
pub fn weights1d(ty: GgmlType, data: &[u8], n: usize) -> Vec<f32> {
    let n_eff = if n % 256 != 0 && n < 256 { n } else { n };
    match ty {
        GgmlType::F32 => {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(f32::from_le_bytes([data[4 * i], data[4 * i + 1], data[4 * i + 2], data[4 * i + 3]]));
            }
            v
        }
        GgmlType::F16 => {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(read_f16(&data[2 * i..]));
            }
            v
        }
        GgmlType::BF16 => {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(bf16_to_f32(u16::from_le_bytes([data[2 * i], data[2 * i + 1]])));
            }
            v
        }
        _ => {
            let vpb = vals_per_block(ty);
            if n_eff % vpb != 0 {
                // أبعاد غير من مضاعفات 256 نادرة؛ نستخدم dequant_to مع تمدد
                let padded = n_eff.div_ceil(vpb) * vpb;
                let mut out = vec![0f32; padded];
                dequant_into(ty, data, &mut out);
                out.truncate(n);
                return out;
            }
            let mut out = vec![0f32; n];
            dequant_into(ty, data, &mut out);
            out
        }
    }
}
