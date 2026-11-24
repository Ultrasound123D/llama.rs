//! اختبار فك التكميم على كتل اصطناعية بقيم محسوبة يدوياً + تحويلات f16/BF16
use llama_gguf::GgmlType;
use llama_quant::{bf16_to_f32, dequant_into, f16_to_f32};

#[test]
fn f16_bf16_conversions() {
    assert_eq!(f16_to_f32(0x3C00), 1.0); // 1.0
    assert_eq!(f16_to_f32(0x3800), 0.5); // 0.5
    assert_eq!(f16_to_f32(0x8000), -0.0);
    assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24)); // subnormal
    assert!(f16_to_f32(0x7C01).is_nan()); // NaN
    assert!(f16_to_f32(0x7C00).is_infinite()); // +inf
    assert_eq!(bf16_to_f32(0x3F80), 1.0);
    assert_eq!(bf16_to_f32(0xBF80), -1.0);
}

#[test]
fn dequant_q4_0_synth() {
    // كتلة واحدة: d = f16(1.0)، qs[j] = 0x21 → منخفض=1، مرتفع=2
    // → القيم (1-8)×1.0 و (2-8)×1.0
    let mut data = Vec::new();
    data.extend_from_slice(&0x3C00u16.to_le_bytes());
    for _ in 0..16 {
        data.push(0x21);
    }
    let mut out = vec![0f32; 32];
    dequant_into(GgmlType::Q4_0, &data, &mut out);
    for j in 0..16 {
        assert_eq!(out[j], -7.0);
    }
    for j in 16..32 {
        assert_eq!(out[j], -6.0);
    }
}

#[test]
fn dequant_q8_0_synth() {
    let mut data = Vec::new();
    data.extend_from_slice(&0x4000u16.to_le_bytes()); // f16(2.0)
    let qs: Vec<i8> = (0..32i8).collect();
    for v in &qs {
        data.push(*v as u8);
    }
    let mut out = vec![0f32; 32];
    dequant_into(GgmlType::Q8_0, &data, &mut out);
    for i in 0..32 {
        assert_eq!(out[i], i as f32 * 2.0);
    }
}

#[test]
fn dequant_f32_passthrough() {
    let data: Vec<u8> = [0.5f32, -1.25, 3.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut out = vec![0f32; 3];
    dequant_into(GgmlType::F32, &data, &mut out);
    assert_eq!(out, vec![0.5, -1.25, 3.0]);
}

#[test]
fn dequant_f16_synth() {
    // f16(1.0) = 0x3C00، f16(0.5) = 0x3800
    let mut data = Vec::new();
    data.extend_from_slice(&0x3C00u16.to_le_bytes());
    data.extend_from_slice(&0x3800u16.to_le_bytes());
    let mut out = vec![0f32; 2];
    dequant_into(GgmlType::F16, &data, &mut out);
    assert_eq!(out, vec![1.0, 0.5]);
}
