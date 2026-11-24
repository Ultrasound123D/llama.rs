// مرجع Q6_K من ggml-quants.c:1949 حرفياً (llama.cpp, MIT) — ground truth للاختبار
// ملاحظة: تقدّم المؤشرات ql/qh/sc بين نصفي الكتلة مطابق للمصدر (ql += 64; qh += 32; sc += 8)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef uint16_t ggml_half;
static inline float ggml_half_to_float(ggml_half h) {
    const uint16_t sign = (h >> 15) & 1;
    const uint16_t exp  = (h >> 10) & 0x1F;
    const uint16_t mant = h & 0x3FF;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) { out = (uint32_t)sign << 31; }
        else { return (sign ? -1 : 1) * ldexpf((float)mant, -24); }
    } else if (exp == 0x1F) {
        out = ((uint32_t)sign << 31) | (0xFFu << 23) | ((uint32_t)mant << 13);
    } else {
        const uint32_t e32 = (uint32_t)(exp - 15 + 127);
        out = ((uint32_t)sign << 31) | (e32 << 23) | ((uint32_t)mant << 13);
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
}

int main(int argc, char **argv) {
    const char *path = argv[1];
    long off = atol(argv[2]);
    int n_bytes = atoi(argv[3]);
    FILE *f = fopen(path, "rb");
    fseek(f, off, SEEK_SET);
    static uint8_t buf[65536];
    fread(buf, 1, n_bytes, f);
    fclose(f);

    static float y[65536];
    const int nb = n_bytes / 210;
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = buf + i * 210;
        const float d = ggml_half_to_float(*(const uint16_t *)(b + 208));
        const uint8_t *ql = b + 0;
        const uint8_t *qh = b + 128;
        const int8_t  *sc = (const int8_t *)(b + 192);
        for (int n = 0; n < 2; ++n) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[i*256 + n*128 + l +  0] = d * sc[is + 0] * q1;
                y[i*256 + n*128 + l + 32] = d * sc[is + 2] * q2;
                y[i*256 + n*128 + l + 64] = d * sc[is + 4] * q3;
                y[i*256 + n*128 + l + 96] = d * sc[is + 6] * q4;
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    for (int k = 0; k < nb * 256; k++) printf("%s%.9g", k ? " " : "", y[k]);
    printf("\n");
    return 0;
}
