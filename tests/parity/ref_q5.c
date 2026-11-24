// Q5_0 dequantization reference — original ggml code + f16 conversion by the compiler itself (ground truth)
// Transcribed from ggml/src/ggml-quants.c (llama.cpp, MIT) — for use in parity tests only
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *path = argv[1];
    long off = atol(argv[2]);
    int n_bytes = atoi(argv[3]);
    FILE *f = fopen(path, "rb");
    fseek(f, off, SEEK_SET);
    static uint8_t buf[4096];
    fread(buf, 1, n_bytes, f);
    fclose(f);

    float y[1024];
    int nb = n_bytes / 22;
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = buf + i * 22;
        _Float16 fh;
        memcpy(&fh, b, 2);
        const float d = (float)fh;
        uint32_t qh;
        memcpy(&qh, b + 2, 4);
        const uint8_t *qs = b + 6;
        for (int j = 0; j < 16; j++) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;
            const int32_t x0 = ((qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((qs[j] >>   4) | xh_1) - 16;
            y[i*32 + j + 0]  = x0 * d;
            y[i*32 + j + 16] = x1 * d;
        }
    }
    for (int k = 0; k < nb * 32; k++) printf("%s%.9g", k ? " " : "", y[k]);
    printf("\n");
    return 0;
}
