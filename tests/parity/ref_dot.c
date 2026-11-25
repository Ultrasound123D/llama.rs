// Golden reference for the packed-VNNI differential tests (Phase 6.3/6.4 verification).
//
// Computes Q8_0 / Q5_0 / Q4_K / Q6_K mat-vec products with *actual* llama.cpp
// (b12502) code, transcribed verbatim from:
//   - ggml/src/ggml-quants.c        :: quantize_row_q8_0_ref (id = 1/d, roundf)
//   - ggml/src/ggml-cpu/arch/x86/quants.c:302 :: quantize_row_q8_0 — the code
//     llama.cpp actually EXECUTES on x86 at runtime (id = 127.f/max, hardware
//     round-to-nearest-even). Both quantizer contracts are emitted because
//     they differ (llama.cpp's own ref-vs-runtime divergence): the runtime
//     one is the byte-exact gate for llama.rs; the ref one validates the
//     transcription against the published reference.
//   - ggml/src/ggml-quants.c:2784 :: quantize_row_q8_K_ref — the K-quant
//     activation quantizer; on x86 the runtime quantize_row_q8_K literally
//     forwards to this _ref (arch/x86/quants.c), so the K-quant contract is
//     single and this is the byte-exact gate.
//   - ggml/src/ggml-cpu/quants.c     :: ggml_vec_dot_q8_0_q8_0_generic
//   - ggml/src/ggml-cpu/quants.c     :: ggml_vec_dot_q5_0_q8_0_generic
//   - ggml/src/ggml-cpu/quants.c:744 :: ggml_vec_dot_q4_K_q8_K_generic
//   - ggml/src/ggml-cpu/quants.c:899 :: ggml_vec_dot_q6_K_q8_K_generic
//
// Modes:
//   random   — LCG-random tensor blocks (safe positive f16 scales) + f32 act
//   extremes — extreme f16 scales (0 / min-normal / 1 / 65504) × edge code
//              bytes {00,01,7F,80,FE,FF} + activations that quantize to ±127
//              (K-quants: full-range i8 scales + walked scale/min nibble bytes)
//   qhwalk   — Q5_0: column j uses qh with single bit (j % 32) set;
//              Q6_K: qh bytes each carry a single walked bit (systematic
//              sweep of every 2-bit high-part position)
//
// Output files (for the Rust driver), prefix given:
//   <p>_tensor.bin  weight tensor bytes (shared ground-truth input)
//   <p>_x_f32.bin   f32 activation
//   <p>_qact_rt.bin runtime-contract quantized activation (q8_0 or q8_K)
//   <p>_qact_ref.bin reference-contract quantized activation (q8_0 only;
//                    for K-quants identical bytes — single contract)
//   <p>_out_rt.txt  golden outputs computed from qact_rt (runtime contract)
//   <p>_out_ref.txt golden outputs computed from qact_ref (reference contract)
//   <p>_scale.txt   per-column term energy: sum_b |sumi_b * d_b| — the honest
//                  error-normalization denominator for dot products whose
//                  outputs may catastrophically cancel to ~0 (adversarial
//                  data makes outputs pure f32-rounding residues; the
//                  meaningful precision statement is error / term energy).
//
// usage: ref_dot <mode> <q8_0|q5_0|q4_k|q6_k> <n_in> <n_out> <seed> <prefix>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef uint16_t ggml_fp16_t;

// ---- GGML fp16 conversion (ggml-impl.h semantics; _Float16 cast is the
// compiler's IEEE round-to-nearest-even conversion = GGML_CPU_FP32_TO_FP16) ----
static float fp16_to_fp32(ggml_fp16_t h) {
    _Float16 fh; memcpy(&fh, &h, 2);
    return (float)fh;
}
static ggml_fp16_t fp32_to_fp16(float f) {
    _Float16 fh = (_Float16)f; ggml_fp16_t h; memcpy(&h, &fh, 2);
    return h;
}

// ============ transcribed llama.cpp quantizers (verbatim math) ==============

// ggml-quants.c:273 — reference quantizer (file-creation contract)
static void quantize_row_q8_0_ref(const float *x, uint8_t *y, int64_t k) {
    const int qk = 32, nb = k / qk;
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            amax = fmaxf(amax, fabsf(v));
        }
        const float d = amax / 127.0f;
        const float id = d ? 1.0f/d : 0.0f;
        uint8_t *blk = y + i*34;
        memcpy(blk, &(ggml_fp16_t){fp32_to_fp16(d)}, 2);
        for (int j = 0; j < qk; ++j) {
            const float x0 = x[i*qk + j]*id;
            blk[2 + j] = (int8_t)roundf(x0);        // roundf: half away from zero
        }
    }
}

// arch/x86/quants.c:302 — the quantizer llama.cpp RUNS on AVX2 x86 (the
// numeric contract of the live inference path):
//   d = maxScalar/127.f; id = 127.f/maxScalar; codes = round-to-nearest-even
// (the SSE/AVX intrinsic `_MM_ROUND_NEAREST` == nearbyint under the default
// FE_TONEAREST mode). max of abs is exact in both impls.
static void quantize_row_q8_0_rt(const float *x, uint8_t *y, int64_t k) {
    const int qk = 32, nb = k / qk;
    for (int i = 0; i < nb; i++) {
        float maxScalar = 0.0f;
        for (int j = 0; j < qk; j++) {
            const float v = fabsf(x[i*qk + j]);
            if (v > maxScalar) maxScalar = v;
        }
        const float d = maxScalar / 127.f;
        uint8_t *blk = y + i*34;
        memcpy(blk, &(ggml_fp16_t){fp32_to_fp16(d)}, 2);
        const float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        for (int j = 0; j < qk; ++j) {
            const float x0 = x[i*qk + j]*id;
            int r = (int)nearbyintf(x0);            // FE_TONEAREST: ties-to-even
            if (r > 127) r = 127; if (r < -127) r = -127;
            blk[2 + j] = (int8_t)r;
        }
    }
}

// ============ transcribed llama.cpp vec_dots (verbatim math) =================

/// `nearest_int` (ggml-common.h) — the magic-number round-to-nearest-even
/// used by quantize_row_q8_K_ref.
static int nearest_int(float fval) {
    const float val = fval + 12582912.0f;
    int i; memcpy(&i, &val, 4);
    return (i & 0x007fffff) - 0x00400000;
}

// ggml-quants.c:2784 — quantize_row_q8_K_ref. NOTE (single contract): on x86
// the runtime `quantize_row_q8_K` literally calls this _ref (arch/x86/quants.c
// "placeholder"), so this IS the byte-exact gate for llama.rs's q8_K
// quantizer. Block layout: float d (4) | int8 qs[256] | int16 bsums[16] = 292 B.
static void quantize_row_q8_K_ref(const float *x, uint8_t *y, int64_t k) {
    const int qk = 256, nb = k / qk;
    for (int i = 0; i < nb; i++) {
        float max = 0, amax = 0;
        for (int j = 0; j < qk; j++) {
            float ax = fabsf(x[i*qk + j]);
            if (ax > amax) { amax = ax; max = x[i*qk + j]; }
        }
        uint8_t *blk = y + i*292;
        if (!amax) {
            memset(blk, 0, 292);
            continue;
        }
        const float iscale = -127.0f/max;
        for (int j = 0; j < qk; j++) {
            const int v = nearest_int(iscale*x[i*qk + j]);
            blk[4 + j] = (int8_t)(v < 127 ? v : 127);
        }
        for (int j = 0; j < qk/16; j++) {
            int sum = 0;
            for (int ii = 0; ii < 16; ii++) sum += (int8_t)blk[4 + j*16 + ii];
            int16_t s = (int16_t)sum;
            memcpy(blk + 260 + 2*j, &s, 2);
        }
        const float d = 1.0f/iscale;
        memcpy(blk, &d, 4);
    }
}

// ggml-cpu/quants.c:744 — ggml_vec_dot_q4_K_q8_K_generic (nrc==1), verbatim
// math (8-lane aux32 accumulation + per-16 bsum/min term).
static float vec_dot_q4_K_q8_K(const uint8_t *x, const uint8_t *y, int n) {
    const int nb = n / 256;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8_t *scales = (const uint8_t*)&utmp[0];
    const uint8_t *mins   = (const uint8_t*)&utmp[2];
    int8_t  aux8[256];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * q4 = x + i*144 + 16;   // qs
        const uint8_t * s12 = x + i*144 + 4;  // 12 scale bytes
        const  int8_t * q8 = (const int8_t*)(y + i*292 + 4);
        const int16_t * bs = (const int16_t*)(y + i*292 + 260);
        float yd; memcpy(&yd, y + i*292, 4);
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * a = aux8;
        for (int j = 0; j < 256/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, s12, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        int sumi = 0;
        for (int j = 0; j < 16; ++j) sumi += bs[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < 256/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = fp16_to_fp32(*(const ggml_fp16_t*)(x + i*144 + 0)) * yd;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = fp16_to_fp32(*(const ggml_fp16_t*)(x + i*144 + 2)) * yd;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

static float vec_dot_q4_K_q8_K_e(const uint8_t *x, const uint8_t *y, int n, float *energy) {
    // same math + per-block term energy for the honest error normalization
    const int nb = n / 256;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    const uint8_t *scales = (const uint8_t*)&utmp[0];
    const uint8_t *mins   = (const uint8_t*)&utmp[2];
    int8_t  aux8[256];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));
    float sumf = 0, en = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * q4 = x + i*144 + 16;
        const uint8_t * s12 = x + i*144 + 4;
        const  int8_t * q8 = (const int8_t*)(y + i*292 + 4);
        const int16_t * bs = (const int16_t*)(y + i*292 + 260);
        float yd; memcpy(&yd, y + i*292, 4);
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * a = aux8;
        for (int j = 0; j < 256/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, s12, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        int sumi = 0;
        for (int j = 0; j < 16; ++j) sumi += bs[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < 256/32; ++j) {
            int32_t scale = scales[is++];
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
                for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
                q8 += 8; a += 8;
            }
        }
        const float d = fp16_to_fp32(*(const ggml_fp16_t*)(x + i*144 + 0)) * yd;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = fp16_to_fp32(*(const ggml_fp16_t*)(x + i*144 + 2)) * yd;
        sumf -= dmin * sumi;
        float enb = 0;
        for (int l = 0; l < 8; ++l) enb += fabsf((float)aux32[l]);
        en += fabsf(d)*enb + fabsf(dmin)*fabsf((float)sumi);
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    if (energy) *energy = en;
    return sumf;
}

// ggml-cpu/quants.c:899 — ggml_vec_dot_q6_K_q8_K_generic (nrc==1), verbatim.
static float vec_dot_q6_K_q8_K(const uint8_t *x, const uint8_t *y, int n) {
    const int nb = n / 256;
    int8_t  aux8[256];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * q4 = x + i*210 + 0;    // ql
        const uint8_t * qh = x + i*210 + 128;  // qh
        const int8_t * scl = (const int8_t*)(x + i*210 + 192);
        const  int8_t * q8 = (const int8_t*)(y + i*292 + 4);
        float yd; memcpy(&yd, y + i*292, 4);
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * a = aux8;
        for (int j = 0; j < 256; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128; q4 += 64; qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < 256/16; ++j) {
            int scale = scl[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = fp16_to_fp32(*(const ggml_fp16_t*)(x + i*210 + 208)) * yd;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

static float vec_dot_q6_K_q8_K_e(const uint8_t *x, const uint8_t *y, int n, float *energy) {
    const int nb = n / 256;
    int8_t  aux8[256];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));
    float sumf = 0, en = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * q4 = x + i*210 + 0;
        const uint8_t * qh = x + i*210 + 128;
        const int8_t * scl = (const int8_t*)(x + i*210 + 192);
        const  int8_t * q8 = (const int8_t*)(y + i*292 + 4);
        float yd; memcpy(&yd, y + i*292, 4);
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * a = aux8;
        for (int j = 0; j < 256; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128; q4 += 64; qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < 256/16; ++j) {
            int scale = scl[is++];
            for (int k = 0; k < 2; ++k) {
                for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
                for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
                q8 += 8; a += 8;
            }
        }
        const float d = fp16_to_fp32(*(const ggml_fp16_t*)(x + i*210 + 208)) * yd;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        float enb = 0;
        for (int l = 0; l < 8; ++l) enb += fabsf((float)aux32[l]);
        en += fabsf(d)*enb;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    if (energy) *energy = en;
    return sumf;
}

// ggml-cpu/quants.c:499 — ggml_vec_dot_q8_0_q8_0_generic (nrc==1)
static float vec_dot_q8_0_q8_0(const uint8_t *x, const uint8_t *y, int n) {
    const int qk = 32, nb = n / qk;
    int ib = 0; float sumf = 0;
    for (; ib < nb; ++ib) {
        int sumi = 0;
        const uint8_t *xb = x + ib*34, *yb = y + ib*34;
        for (int j = 0; j < qk; j++) {
            sumi += (int8_t)xb[2+j] * (int8_t)yb[2+j];
        }
        sumf += sumi * (fp16_to_fp32(*(const ggml_fp16_t*)(xb+0)) *
                        fp16_to_fp32(*(const ggml_fp16_t*)(yb+0)));
    }
    return sumf;
}

static float vec_dot_q8_0_q8_0_e(const uint8_t *x, const uint8_t *y, int n, float *energy) {
    const int qk = 32, nb = n / qk;
    int ib = 0; float sumf = 0; float en = 0;
    for (; ib < nb; ++ib) {
        int sumi = 0;
        const uint8_t *xb = x + ib*34, *yb = y + ib*34;
        for (int j = 0; j < qk; j++) {
            sumi += (int8_t)xb[2+j] * (int8_t)yb[2+j];
        }
        const float d = (fp16_to_fp32(*(const ggml_fp16_t*)(xb+0)) *
                         fp16_to_fp32(*(const ggml_fp16_t*)(yb+0)));
        sumf += sumi*d;
        en += fabsf((float)sumi)*fabsf(d);
    }
    if (energy) *energy = en;
    return sumf;
}

// ggml-cpu/quants.c:413 — ggml_vec_dot_q5_0_q8_0_generic (nrc==1)
static float vec_dot_q5_0_q8_0(const uint8_t *x, const uint8_t *y, int n) {
    const int qk = 32, nb = n / qk;
    int ib = 0; float sumf = 0;
    for (; ib < nb; ++ib) {
        const uint8_t *xb = x + ib*22, *yb = y + ib*34;
        uint32_t qh; memcpy(&qh, xb + 2, 4);
        int sumi0 = 0, sumi1 = 0;
        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));
            const int32_t x0 = (int8_t)(((xb[6+j] & 0x0F) | xh_0) - 16);
            const int32_t x1 = (int8_t)(((xb[6+j] >>   4) | xh_1) - 16);
            sumi0 += x0 * (int8_t)yb[2+j];
            sumi1 += x1 * (int8_t)yb[2+16+j];
        }
        sumf += (sumi0 + sumi1) *
                (fp16_to_fp32(*(const ggml_fp16_t*)(xb+0)) *
                 fp16_to_fp32(*(const ggml_fp16_t*)(yb+0)));
    }
    return sumf;
}

static float vec_dot_q5_0_q8_0_e(const uint8_t *x, const uint8_t *y, int n, float *energy) {
    const int qk = 32, nb = n / qk;
    int ib = 0; float sumf = 0; float en = 0;
    for (; ib < nb; ++ib) {
        const uint8_t *xb = x + ib*22, *yb = y + ib*34;
        uint32_t qh; memcpy(&qh, xb + 2, 4);
        int sumi0 = 0, sumi1 = 0;
        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));
            const int32_t x0 = (int8_t)(((xb[6+j] & 0x0F) | xh_0) - 16);
            const int32_t x1 = (int8_t)(((xb[6+j] >>   4) | xh_1) - 16);
            sumi0 += x0 * (int8_t)yb[2+j];
            sumi1 += x1 * (int8_t)yb[2+16+j];
        }
        const float d = (fp16_to_fp32(*(const ggml_fp16_t*)(xb+0)) *
                         fp16_to_fp32(*(const ggml_fp16_t*)(yb+0)));
        sumf += (sumi0 + sumi1) * d;
        en += (fabsf((float)sumi0) + fabsf((float)sumi1)) * fabsf(d);
    }
    if (energy) *energy = en;
    return sumf;
}

// ============================ data generation ================================

static uint64_t lcg(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return x;
}

static float frand(uint64_t *s, float lo, float hi) {
    return lo + (hi - lo) * ((float)(lcg(s) % 1000000u) / 1000000.0f);
}

static void write_all(const char *pfx, const uint8_t *tensor,
                      size_t tlen, const float *x, int n_in,
                      const uint8_t *qact, const float *out, int n_out,
                      const uint8_t *qact2, const float *out2,
                      const float *scale, size_t qact_bytes) {
    char path[512]; FILE *f;
    snprintf(path, sizeof path, "%s_tensor.bin",  pfx); f = fopen(path, "wb"); fwrite(tensor, 1, tlen, f); fclose(f);
    snprintf(path, sizeof path, "%s_x_f32.bin",  pfx); f = fopen(path, "wb"); fwrite(x, 1, sizeof(float)*n_in, f); fclose(f);
    snprintf(path, sizeof path, "%s_qact_rt.bin",  pfx); f = fopen(path, "wb"); fwrite(qact, 1, qact_bytes, f); fclose(f);
    snprintf(path, sizeof path, "%s_qact_ref.bin", pfx); f = fopen(path, "wb"); fwrite(qact2, 1, qact_bytes, f); fclose(f);
    snprintf(path, sizeof path, "%s_out_rt.txt",  pfx); f = fopen(path, "w");
    for (int c = 0; c < n_out; c++) fprintf(f, "%s%.9g", c ? " " : "", out[c]);
    fprintf(f, "\n"); fclose(f);
    snprintf(path, sizeof path, "%s_out_ref.txt", pfx); f = fopen(path, "w");
    for (int c = 0; c < n_out; c++) fprintf(f, "%s%.9g", c ? " " : "", out2[c]);
    fprintf(f, "\n"); fclose(f);
    snprintf(path, sizeof path, "%s_scale.txt", pfx); f = fopen(path, "w");
    for (int c = 0; c < n_out; c++) fprintf(f, "%s%.9g", c ? " " : "", scale[c]);
    fprintf(f, "\n"); fclose(f);
}

int main(int argc, char **argv) {
    if (argc != 7) { fprintf(stderr, "usage: %s mode type n_in n_out seed prefix\n", argv[0]); return 1; }
    const char *mode = argv[1];
    const char *ty   = argv[2];
    const int n_in  = atoi(argv[3]);
    const int n_out = atoi(argv[4]);
    uint64_t seed   = strtoull(argv[5], NULL, 0);
    const char *pfx = argv[6];

    const int is_q4k = !strcmp(ty, "q4_k");
    const int is_q6k = !strcmp(ty, "q6_k");
    const int is_kq  = is_q4k || is_q6k;
    const int block_vals = is_kq ? 256 : 32;
    const int row_bytes = is_q4k ? 144 : is_q6k ? 210 : (ty[1] == '8') ? 34 : 22;
    const int kb = n_in / block_vals;
    const size_t tlen = (size_t)n_out * kb * row_bytes;

    uint8_t *tensor = malloc(tlen);
    float *x = malloc(sizeof(float) * n_in);

    const uint16_t scales[] = { 0x0000, 0x0400, 0x3C00, 0x7BFF };
    const uint8_t codes[] = { 0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF };

    if (is_kq && n_in % 256) { fprintf(stderr, "n_in must be a multiple of 256 for %s\n", ty); return 1; }

    if (!strcmp(mode, "random")) {
        for (size_t i = 0; i < tlen; i++) tensor[i] = (uint8_t)(lcg(&seed) >> 24);
        for (int c = 0; c < n_out; c++) {
            for (int b = 0; b < kb; b++) {
                uint8_t *blk = tensor + (size_t)c * kb * row_bytes + (size_t)b * row_bytes;
                if (is_q4k) {
                    float d = frand(&seed, 0.05f, 3.0f);
                    float dmin = frand(&seed, 0.01f, 0.5f);
                    memcpy(blk, &(ggml_fp16_t){fp32_to_fp16(d)}, 2);
                    memcpy(blk + 2, &(ggml_fp16_t){fp32_to_fp16(dmin)}, 2);
                } else if (is_q6k) {
                    float d = frand(&seed, 0.05f, 3.0f);
                    memcpy(blk + 208, &(ggml_fp16_t){fp32_to_fp16(d)}, 2);
                    // i8 scales: full signed range
                    for (int j = 0; j < 16; j++) blk[192 + j] = (int8_t)(lcg(&seed) >> 24);
                } else {
                    float d = frand(&seed, 0.05f, 3.0f);
                    memcpy(blk, &(ggml_fp16_t){fp32_to_fp16(d)}, 2);
                }
            }
        }
        for (int i = 0; i < n_in; i++) x[i] = frand(&seed, -1.0f, 1.0f);
    } else if (!strcmp(mode, "extremes")) {
        size_t bi = 0;
        for (int c = 0; c < n_out; c++) {
            for (int b = 0; b < kb; b++) {
                uint8_t *blk = tensor + (size_t)c * kb * row_bytes + (size_t)b * row_bytes;
                uint16_t sc = scales[(bi / 6) % 4];
                if (is_q4k) {
                    memcpy(blk, &sc, 2);
                    memcpy(blk + 2, &sc, 2);
                    for (int j = 0; j < 12; j++) blk[4 + j] = (bi + j) % 2 ? 0x3F : 0x00;
                    for (int j = 0; j < 128; j++) blk[16 + j] = codes[(bi + j) % 6];
                } else if (is_q6k) {
                    memcpy(blk + 208, &sc, 2);
                    for (int j = 0; j < 16; j++) blk[192 + j] = (int8_t)codes[(bi + j) % 6];
                    for (int j = 0; j < 128; j++) blk[j] = codes[(bi + j) % 6];
                    for (int j = 0; j < 64; j++) blk[128 + j] = (bi + j) % 2 ? 0xFF : 0x00;
                } else if (row_bytes == 34) {
                    memcpy(blk, &sc, 2);
                    for (int j = 0; j < 32; j++) blk[2 + j] = codes[(bi + j) % 6];
                } else {
                    uint32_t qh = 0xDEADBEEFu;
                    memcpy(blk + 2, &qh, 4);
                    for (int j = 0; j < 16; j++) blk[6 + j] = codes[(bi + j) % 6];
                }
                bi++;
            }
        }
        for (int i = 0; i < n_in; i++) x[i] = frand(&seed, -1.0f, 1.0f);
        for (int i = 0; i < n_in; i += 8) x[i] = 1e6f * ((i % 16) ? 1.0f : -1.0f);
    } else if (!strcmp(mode, "qhwalk")) {
        if (!is_q6k && ty[1] != '5') { fprintf(stderr, "qhwalk is for q5_0/q6_k\n"); return 1; }
        for (int c = 0; c < n_out; c++) {
            for (int b = 0; b < kb; b++) {
                uint8_t *blk = tensor + (size_t)c * kb * row_bytes + (size_t)b * row_bytes;
                if (is_q6k) {
                    // walk every qh bit position: byte j carries bit (c + j) % 8
                    for (int j = 0; j < 64; j++) blk[128 + j] = (uint8_t)(1u << ((c + j) % 8));
                    memset(blk, 0xA5, 128);
                    for (int j = 0; j < 16; j++) blk[192 + j] = (int8_t)(1 + (c % 7));
                    uint16_t sc = (c % 4 == 3) ? 0x7BFF : 0x3C00;
                    memcpy(blk + 208, &sc, 2);
                } else {
                    uint16_t sc = (c % 4 == 3) ? 0x7BFF : 0x3C00;
                    memcpy(blk, &sc, 2);
                    uint32_t qh = 1u << (c % 32);
                    memcpy(blk + 2, &qh, 4);
                    memset(blk + 6, 0xA5, 16);
                }
            }
        }
        for (int i = 0; i < n_in; i++) x[i] = frand(&seed, -0.5f, 0.5f);
    } else { fprintf(stderr, "bad mode\n"); return 1; }

    // quantized activations (K-quants: q8_K; simple quants: both q8_0 contracts)
    const size_t qact_bytes = is_kq ? (size_t)kb * 292 : (size_t)kb * 34;
    uint8_t *qrt  = malloc(qact_bytes);
    uint8_t *qref = malloc(qact_bytes);
    if (is_kq) {
        quantize_row_q8_K_ref(x, qrt, n_in);
        memcpy(qref, qrt, qact_bytes); // single contract on x86 (rt == ref)
    } else {
        quantize_row_q8_0_rt(x, qrt, n_in);
        quantize_row_q8_0_ref(x, qref, n_in);
    }

    float *out_rt  = malloc(sizeof(float) * n_out);
    float *out_ref = malloc(sizeof(float) * n_out);
    float *scale2  = malloc(sizeof(float) * n_out);
    for (int c = 0; c < n_out; c++) {
        const uint8_t *wcol = tensor + (size_t)c * kb * row_bytes;
        float en = 0;
        if (is_q4k) {
            out_rt[c]  = vec_dot_q4_K_q8_K_e(wcol, qrt, n_in, &en);
            out_ref[c] = vec_dot_q4_K_q8_K(wcol, qref, n_in);
        } else if (is_q6k) {
            out_rt[c]  = vec_dot_q6_K_q8_K_e(wcol, qrt, n_in, &en);
            out_ref[c] = vec_dot_q6_K_q8_K(wcol, qref, n_in);
        } else if (ty[1] == '8') {
            out_rt[c]  = vec_dot_q8_0_q8_0_e(wcol, qrt,  n_in, &en);
            out_ref[c] = vec_dot_q8_0_q8_0(wcol, qref, n_in);
        } else {
            out_rt[c]  = vec_dot_q5_0_q8_0_e(wcol, qrt,  n_in, &en);
            out_ref[c] = vec_dot_q5_0_q8_0(wcol, qref, n_in);
        }
        scale2[c] = en;
    }

    write_all(pfx, tensor, tlen, x, n_in, qrt, out_rt, n_out, qref, out_ref, scale2, qact_bytes);

    // human-readable summary to stderr
    double mean = 0; for (int c = 0; c < n_out; c++) mean += out_rt[c];
    int qdiff = 0;
    for (size_t i = 0; i < qact_bytes; i++) if (qrt[i] != qref[i]) qdiff++;
    double emax = 0; for (int c = 0; c < n_out; c++) if (scale2[c] > emax) emax = scale2[c];
    fprintf(stderr, "ref_dot %s/%s n_in=%d n_out=%d blocks=%d mean_out_rt=%.6g rt_vs_ref_qact_bytes_differ=%zu max_term_energy=%.6g\n",
            mode, ty, n_in, n_out, n_out*kb, mean / (n_out ? n_out : 1), (size_t)qdiff, emax);
    return 0;
}
