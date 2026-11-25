//! Phase 6.7 audit probe: the flash kernel vs wide/narrow score spreads —
//! the FTZ/DAZ evidence (denormal exps trigger FP assists: 665 vs 131
//! c/key with/without the MXCSR guard) plus per-phase attribution.
use llama_kernels::avx2;
use std::arch::x86_64::*;

fn tsc() -> u64 { unsafe { std::arch::x86_64::_rdtsc() } }

const FLASH_BK: usize = 64;

#[target_feature(enable = "avx2,fma")]
unsafe fn flash_instr(
    q: &[f32], k_all: &[f32], v_all: &[f32],
    nkv: usize, kv_head: usize, lo: usize, n_kv: usize,
    hd: usize, scale: f32, out: &mut [f32],
    t_dot: &mut u64, t_exp: &mut u64, t_v: &mut u64, t_misc: &mut u64,
) {
    let stride = nkv * hd;
    let koff = kv_head * hd;
    let mut s = [0f32; FLASH_BK];
    let mut e = [0f32; FLASH_BK];
    let mut m_run = f32::NEG_INFINITY;
    let mut l_run = 0f32;
    let mut p = lo;
    while p < n_kv {
        let bsz = usize::min(FLASH_BK, n_kv - p);
        // ---- dot phase
        let t0 = tsc();
        for j in 0..bsz {
            let krow = std::slice::from_raw_parts(
                k_all.as_ptr().add((p + j) * stride + koff), hd);
            s[j] = avx2::dot_f32(q, krow) * scale;
        }
        let t1 = tsc();
        // ---- block max + corr + rescale (misc)
        let mut m_b = f32::NEG_INFINITY;
        for j in 0..bsz {
            if s[j] > m_b { m_b = s[j]; }
        }
        let m_new = if m_run > m_b { m_run } else { m_b };
        let corr = (m_run - m_new).exp();
        let vc = _mm256_set1_ps(corr);
        let mut i = 0;
        let np = hd & !7;
        while i < np {
            let o_ = _mm256_loadu_ps(out.as_ptr().add(i) as *const f32);
            _mm256_storeu_ps(out.as_mut_ptr().add(i) as *mut f32,
                _mm256_mul_ps(o_, vc));
            i += 8;
        }
        while i < hd {
            out[i] *= corr;
            i += 1;
        }
        let t2 = tsc();
        // ---- exp phase
        let mut l_b = 0f32;
        let mn = _mm256_set1_ps(m_new);
        let mut j = 0;
        while j + 8 <= bsz {
            let v = _mm256_loadu_ps(s.as_ptr().add(j) as *const f32);
            let ex = exp8_local(_mm256_sub_ps(v, mn));
            _mm256_storeu_ps(e.as_mut_ptr().add(j), ex);
            l_b += avx2::hsum_float_8(ex);
            j += 8;
        }
        for jj in j..bsz {
            let ex = (s[jj] - m_new).exp();
            e[jj] = ex;
            l_b += ex;
        }
        l_run = l_run * corr + l_b;
        let t3 = tsc();
        // ---- V phase (4-key register-blocked)
        let mut j = 0;
        while j + 4 <= bsz {
            let e0 = _mm256_set1_ps(e[j]);
            let e1 = _mm256_set1_ps(e[j + 1]);
            let e2 = _mm256_set1_ps(e[j + 2]);
            let e3 = _mm256_set1_ps(e[j + 3]);
            let v0 = v_all.as_ptr().add((p + j) * stride + koff);
            let v1 = v_all.as_ptr().add((p + j + 1) * stride + koff);
            let v2 = v_all.as_ptr().add((p + j + 2) * stride + koff);
            let v3 = v_all.as_ptr().add((p + j + 3) * stride + koff);
            let mut i = 0;
            while i < np {
                let acc = _mm256_loadu_ps(out.as_ptr().add(i) as *const f32);
                let a = _mm256_fmadd_ps(e0, _mm256_loadu_ps(v0.add(i)), acc);
                let a = _mm256_fmadd_ps(e1, _mm256_loadu_ps(v1.add(i)), a);
                let a = _mm256_fmadd_ps(e2, _mm256_loadu_ps(v2.add(i)), a);
                let a = _mm256_fmadd_ps(e3, _mm256_loadu_ps(v3.add(i)), a);
                _mm256_storeu_ps(out.as_mut_ptr().add(i) as *mut f32, a);
                i += 8;
            }
            while i < hd {
                out[i] += e[j] * *v0.add(i) + e[j + 1] * *v1.add(i)
                    + e[j + 2] * *v2.add(i) + e[j + 3] * *v3.add(i);
                i += 1;
            }
            j += 4;
        }
        for jj in j..bsz {
            let ej = _mm256_set1_ps(e[jj]);
            let vj = v_all.as_ptr().add((p + jj) * stride + koff);
            let mut i = 0;
            while i < np {
                let acc = _mm256_loadu_ps(out.as_ptr().add(i) as *const f32);
                let a = _mm256_fmadd_ps(ej, _mm256_loadu_ps(vj.add(i)), acc);
                _mm256_storeu_ps(out.as_mut_ptr().add(i) as *mut f32, a);
                i += 8;
            }
            while i < hd {
                out[i] += e[jj] * *vj.add(i);
                i += 1;
            }
        }
        let t4 = tsc();
        *t_dot += t1 - t0;
        *t_misc += t2 - t1;
        *t_exp += t3 - t2;
        *t_v += t4 - t3;
        m_run = m_new;
        p += bsz;
    }
    if l_run > 0.0 {
        let inv = 1.0 / l_run;
        let vi = _mm256_set1_ps(inv);
        let mut i = 0;
        let np = hd & !7;
        while i < np {
            let o_ = _mm256_loadu_ps(out.as_ptr().add(i) as *const f32);
            _mm256_storeu_ps(out.as_mut_ptr().add(i) as *mut f32,
                _mm256_mul_ps(o_, vi));
            i += 8;
        }
        while i < hd {
            out[i] *= inv;
            i += 1;
        }
    }
}

// local copy of exp8 (same polynomial as batch.rs)
#[target_feature(enable = "avx2,fma")]
unsafe fn exp8_local(x: __m256) -> __m256 {
    const LOG2E: f32 = 1.4426950408889634;
    const LN2: f32 = 0.6931471805599453;
    let t = _mm256_mul_ps(x, _mm256_set1_ps(LOG2E));
    let k = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let r = _mm256_mul_ps(_mm256_sub_ps(t, k), _mm256_set1_ps(LN2));
    let p = _mm256_set1_ps(1.0 / 5040.0);
    let p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0 / 720.0));
    let p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0 / 120.0));
    let p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0 / 24.0));
    let p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0 / 6.0));
    let p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0 / 2.0));
    let p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0));
    let p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0));
    let ki = _mm256_cvtps_epi32(k);
    let ki = _mm256_max_epi32(ki, _mm256_set1_epi32(-126));
    let bits = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    let two_k = _mm256_castsi256_ps(bits);
    _mm256_mul_ps(p, two_k)
}

fn main() {
    // ---- 0) dispatch truth: which branch does the REAL library kernel take?
    {
        use llama_kernels::batch as bf;
        // dot_f32 must be real SIMD if the dispatch is healthy:
        // time it over 4050 keys and compare to raw avx2::dot_f32.
        let keys0 = 4050usize;
        let k0: Vec<f32> = (0..keys0 * 64).map(|i| (i as f32) * 1e-4).collect();
        let q0: Vec<f32> = (0..64).map(|i| (i as f32) * 1e-4).collect();
        let mut sink = 0f32;
        let ta = tsc();
        for p in 0..keys0 {
            sink += bf::dot_f32(&q0, &k0[p * 64..p * 64 + 64]);
        }
        let tb = tsc();
        println!("lib batch::dot_f32 ×4050: {} cycles ({} /key) sink={sink:.3}", tb - ta, (tb - ta) / keys0 as u64);
    }

    let keys = 4050usize;
    let nkv = 2usize;
    let hd = 64usize;
    let stride = nkv * hd;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut kcache = vec![0f32; keys * stride];
    let mut vcache = vec![0f32; keys * stride];
    let mut st = 42u64;
    for v in kcache.iter_mut().chain(vcache.iter_mut()) {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((st >> 40) as i32) as f32 / 1e6;
    }
    let q: Vec<f32> = (0..hd).map(|i| (i as f32) * 0.001 - 0.032).collect();
    let mut out = vec![0f32; hd];

    // ---- 0b) the REAL library kernel, timed directly, same data
    // TWO data distributions: (i) small q → narrow scores → normal exps,
    // (ii) wide q (the attnbench distribution) → wide score spread →
    // subnormal / underflowed exps. If (ii) collapses, the denormal
    // hypothesis is confirmed.
    use llama_kernels::batch::attn_head_flash;
    {
        let mut q_wide: Vec<f32> = vec![0f32; hd];
        let mut stw = 999u64;
        for v in q_wide.iter_mut() {
            stw = stw.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((stw >> 40) as i32) as f32 / 1e6;
        }
        unsafe {
            attn_head_flash(&q_wide, &kcache, &vcache, nkv, 0, 0, keys, hd, scale, &mut out);
            let ta = tsc();
            for _ in 0..14 {
                attn_head_flash(&q_wide, &kcache, &vcache, nkv, 0, 0, keys, hd, scale, &mut out);
            }
            let tb = tsc();
            println!("REAL flash ×14, WIDE-q scores: {} cycles ({}/head, {}/key) out0={}",
                tb - ta, (tb - ta) / 14, (tb - ta) / 14 / keys as u64, out[0]);
        }
    }
    unsafe {
        attn_head_flash(&q, &kcache, &vcache, nkv, 0, 0, keys, hd, scale, &mut out);
        let ta = tsc();
        for _ in 0..14 {
            attn_head_flash(&q, &kcache, &vcache, nkv, 0, 0, keys, hd, scale, &mut out);
        }
        let tb = tsc();
        println!("REAL attn_head_flash ×14: {} cycles ({}/head, {}/key) out0={}",
            tb - ta, (tb - ta) / 14, (tb - ta) / 14 / keys as u64, out[0]);
    }

    let (mut td, mut tm, mut te, mut tv) = (0u64, 0u64, 0u64, 0u64);
    unsafe {
        // warm
        flash_instr(&q, &kcache, &vcache, nkv, 0, 0, keys, hd, scale, &mut out, &mut td, &mut tm, &mut te, &mut tv);
        td = 0; tm = 0; te = 0; tv = 0;
        for _ in 0..14 {
            flash_instr(&q, &kcache, &vcache, nkv, 0, 0, keys, hd, scale, &mut out, &mut td, &mut tm, &mut te, &mut tv);
        }
    }
    let total = td + tm + te + tv;
    println!("14 heads total: {total} cycles ({}/head, {}/key)", total / 14, total / 14 / keys as u64);
    println!("  dot phase : {:>10} cycles → {:>7}/head ({} /key)", td, td / 14, td / 14 / keys as u64);
    println!("  misc      : {:>10} cycles → {:>7}/head ({} /key)", tm, tm / 14, tm / 14 / keys as u64);
    println!("  exp phase : {:>10} cycles → {:>7}/head ({} /key)", te, te / 14, te / 14 / keys as u64);
    println!("  V phase   : {:>10} cycles → {:>7}/head ({} /key)", tv, tv / 14, tv / 14 / keys as u64);
    println!("out[0..4] = {:?}", &out[0..4]);
}

// appended: compare against the REAL library kernel in the same process
#[no_mangle]
pub extern "C" fn rust_oxygen_breaker() {}
