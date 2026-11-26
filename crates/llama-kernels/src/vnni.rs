//! AVX-512 VNNI INT8 packed backend (Phase 6.3) — the M=1 (decode) route
//! of llama.cpp's AMX backend, ported from `ggml/src/ggml-cpu/amx/mmq.cpp`
//! (b12502).
//!
//! NAMING NOTE (review-driven correction, 2026-11-25): this module was
//! originally announced as the "AMX INT8 backend". That name was inaccurate
//! and is corrected here: **no AMX tile instruction (LDTILECFG / TILELOADD /
//! TDPBUSD / TILEZERO) is executed anywhere in this engine.** The compute
//! instructions in this module are AVX-512 VNNI (`_mm512_dpbusd_epi32`)
//! operating on weights repacked at load time. Presence of AMX in CPUID
//! leaf-7 and XCR0 bits 17/18 is capability *detection*, not execution; the
//! detection is kept and reported so a future M>1 tile path can be built on
//! verified ground. AMX tiles pay off only for M>1 (batched prefill), which
//! this engine does not issue.
//!
//! What llama.cpp actually does on this class of CPU for DECODE (M=1):
//! the "AMX backend" repacks quantized weights at LOAD time into a VNNI
//! int8 layout (`convert_B_packed_format` / `pack_B`) and the M==1 path
//! runs `tinygemm_kernel_vnni` — AVX512-VNNI 512-bit `_mm512_dpbusd_epi32`
//! over 16 columns per instruction. The AMX *tile* instructions
//! (ldtilecfg / tdpbusd, 16x64 tiles) only serve M>1 (batched prefill),
//! which this engine does not issue (prefill is token-by-token). The
//! tile/XSTATE capability is still detected and reported so the
//! capability picture is honest and ready for a future batched path.
//!
//! Packed layout per (16-column group n, 32-value block k) — TILE=576 B:
//!   [ 512 B quants: VNNI int8, 8 rows x 64 B — row j holds values 4j..4j+3
//!     of columns 0..7 (32 B) then columns 8..15 (32 B) ]
//!   [ 32  B d0: 16 f16 block scales ]
//!   [ 64  B comp: 16 x i32 = 128·Σb per column (s8s8 compensation) ]
//!
//! Numeric contract (mmq.cpp `s8s8_compensation`): dpbusd is u8(a)×s8(b);
//! a' = a + 0x80 makes the activation unsigned; Σ(a+128)·b − 128·Σb = Σ a·b
//! exactly. Block scales are applied in f32 afterwards — the same structure
//! as the AVX2 kernels (integer dot exact, f32 rounding at the scale step).
//!
//! Supported: Q8_0 (as in llama.cpp) and Q5_0 (llama.cpp does not repack
//! Q5_0; codes dequantize to int8 in [−16,15] and the same construction
//! applies verbatim).
//!
//! K-quants (Phase 6.4): **Q4_K and Q6_K** repacked the same way llama.cpp's
//! `pack_B<block_q4_K>/pack_B<block_q6_K>` does — nibbles stay NIBBLE-PACKED
//! (4.5/6.56 bpw preserved, so decode stays bandwidth-neutral) and the
//! kernels unpack them in-register (`and`/`srli`) before `dpbusd`, exactly
//! as `tinygemm_kernel_vnni<block_q8_K, block_q4_K/q6_K>` does. Tile sizes:
//! Q4_K 2368 B (= 16·144 + 64, identical to llama.cpp's TILE_SIZE) and
//! Q6_K 3360 B (= 16·210). Activations are quantized to q8_K (the K-quant
//! vec_dot_type) with the q8_K runtime quantizer; the min term runs through
//! `dpwssds` and the q6 (q−32) offset through the bsums×32 compensation —
//! both straight from the llama.cpp kernels.

use crate::kstats;
use crate::pool;
use crate::quant;

use llama_gguf::GgmlType;

pub const TILE_N: usize = 16; // columns per packed group
pub const TILE_K: usize = 32; // values per block (Q8_0/Q5_0 block size)
pub const TILE_BYTES: usize = TILE_N * TILE_K + TILE_N * 2 + TILE_N * 4; // 576

/// Values per K-quant super-block (QK_K).
pub const QK_K: usize = 256;

/// Packed Q4_K tile (16 columns × one 256-value super-block), 2368 B —
/// the same tile size as llama.cpp's `pack_B<block_q4_K>` (TILE_N·144+64):
///   [0, 2048)   quants: 8 sub-groups × 4 chunks × 64 B — chunk (g, c) byte
///               4n+j holds the nibble pair (value 8c+j, value 8c+4+j) of
///               column n, so in-register `and 0xF` / `>>4` unpack aligns
///               with the broadcast activation dwords (llama.cpp's
///               "pack {n,g,k} → {g, k/4, 4n}" VNNI interleave)
///   [2048, 2176) scales: 8 sub-groups × 16 cols (u8, 6-bit)
///   [2176, 2304) mins: 4 pair-chunks × 32 B — byte 2n = min(2p, n),
///               byte 2n+1 = min(2p+1, n) (the dpwssds pair layout)
///   [2304, 2336) d: 16 f16   [2336, 2368) dmin: 16 f16
pub const TILE_BYTES_Q4K: usize = 2368;
const Q4K_OFF_SCALES: usize = 2048;
const Q4K_OFF_MINS: usize = 2176;
const Q4K_OFF_D: usize = 2304;
const Q4K_OFF_DMIN: usize = 2336;

/// Packed Q6_K tile (16 columns × one super-block), 3360 B = 16·210:
///   [0, 2048)   ql: 16 groups × 2 chunks × 64 B (nibble pairs as Q4_K)
///   [2048, 3072) qh: 16 groups × 64 B — byte 4n+j packs the four 2-bit
///               high parts of values {j, 4+j, 8+j, 12+j} (bits 0..7)
///   [3072, 3328) scales: 16 groups × 16 cols (i8)
///   [3328, 3360) d: 16 f16
pub const TILE_BYTES_Q6K: usize = 3360;
const Q6K_OFF_QH: usize = 2048;
const Q6K_OFF_SCALES: usize = 3072;
const Q6K_OFF_D: usize = 3328;

// ---------- capability detection ----------

/// Raw cpuid via the stable `__cpuid_count` intrinsic.
#[inline]
unsafe fn cpuid(leaf: u32, subleaf: u32) -> (u32, u32, u32, u32) {
    let r = std::arch::x86_64::__cpuid_count(leaf, subleaf);
    (r.eax, r.ebx, r.ecx, r.edx)
}

/// XCR0 via xgetbv (user-mode allowed for ecx=0)
#[inline]
unsafe fn xcr0() -> u64 {
    let (lo, hi): (u32, u32);
    std::arch::asm!(
        "xgetbv",
        lateout("eax") lo,
        lateout("edx") hi,
        in("ecx") 0,
        options(nostack, preserves_flags)
    );
    (hi as u64) << 32 | lo as u64
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AmxCaps {
    /// cpuid leaf-7 ebx bit 24: AMX_INT8 (TDPBUSD tiles)
    pub amx_int8: bool,
    /// cpuid leaf-7 ebx bit 25: AMX_BF16
    pub amx_bf16: bool,
    /// AVX512-VNNI (leaf-7 ecx bit 11) + AVX512-VL (ebx bit 31)
    pub avx512_vnni: bool,
    /// XCR0 bits 17/18 (XTILECFG/XTILEDATA) — OS permission for AMX tiles
    pub xstate_tile: bool,
}

pub fn detect() -> AmxCaps {
    static C: std::sync::OnceLock<AmxCaps> = std::sync::OnceLock::new();
    *C.get_or_init(|| unsafe {
        // leaf 1 ecx bit 27 = OSXSAVE (required for xgetbv)
        let (_, _b1, c1, _d1) = cpuid(1, 0);
        let osxsave = (c1 & (1 << 27)) != 0;
        let (_, b7, c7, _d7) = cpuid(7, 0);
        let amx_int8 = (b7 & (1 << 24)) != 0;
        let amx_bf16 = (b7 & (1 << 25)) != 0;
        let avx512_vnni = (c7 & (1 << 11)) != 0 && (b7 & (1 << 31)) != 0;
        let xstate_tile = if osxsave {
            let x = xcr0();
            (x & (0b11 << 17)) == (0b11 << 17)
        } else {
            false
        };
        AmxCaps { amx_int8, amx_bf16, avx512_vnni, xstate_tile }
    })
}

/// The repacked-VNNI decode kernels need AVX512-VNNI (+VL). AMX tiles are
/// not used by the M==1 path (they serve M>1 batching); detection and XSTATE
/// validation are exported for capability reporting and future tiles.
pub fn vnni_path_available() -> bool {
    static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *A.get_or_init(|| detect().avx512_vnni)
}

/// Full AMX tile readiness (would be required by a batched M>1 path).
pub fn amx_tiles_available() -> bool {
    let c = detect();
    c.amx_int8 && c.avx512_vnni && c.xstate_tile
}

// ---------- repack (load time; scalar interleave = clarity) ----------

/// VNNI interleave of one 16-column × 32-value block.
/// Row j (64 B): [bytes 4j..4j+3 of columns 0..7] ++ [bytes 4j..4j+3 of
/// columns 8..15] — exactly the layout `pack_qs` produces via the 8x8
/// 32-bit transpose in mmq.cpp.
fn pack_group_16x32(dst: &mut [u8], col_codes: &[[u8; TILE_K]; TILE_N]) {
    for j in 0..8usize {
        for n in 0..8usize {
            dst[j * 64 + 4 * n..j * 64 + 4 * n + 4]
                .copy_from_slice(&col_codes[n][4 * j..4 * j + 4]);
        }
        for n in 8..16usize {
            dst[j * 64 + 32 + 4 * (n - 8)..j * 64 + 32 + 4 * (n - 8) + 4]
                .copy_from_slice(&col_codes[n][4 * j..4 * j + 4]);
        }
    }
}

/// Repack a Q8_0 tensor [n_in, n_out] (gguf layout) into packed VNNI.
/// Requires n_in % 32 == 0 and n_out % 16 == 0.
pub fn repack_q8_0(data: &[u8], n_in: usize, n_out: usize) -> Vec<u8> {
    assert!(n_in % TILE_K == 0, "n_in {n_in} not multiple of 32");
    assert!(n_out % TILE_N == 0, "n_out {n_out} not multiple of 16");
    let kb = n_in / TILE_K;
    let ng = n_out / TILE_N;
    let mut packed = vec![0u8; ng * kb * TILE_BYTES];
    let row_bytes = 34usize;
    let mut col_codes = [[0u8; TILE_K]; TILE_N];
    for g in 0..ng {
        for k in 0..kb {
            let dst = &mut packed[(g * kb + k) * TILE_BYTES..][..TILE_BYTES];
            for n in 0..TILE_N {
                let col = g * TILE_N + n;
                let b = &data[col * row_bytes * kb + k * row_bytes..][..row_bytes];
                col_codes[n].copy_from_slice(&b[2..34]);
            }
            pack_group_16x32(dst, &col_codes);
            for n in 0..TILE_N {
                let col = g * TILE_N + n;
                let b = &data[col * row_bytes * kb + k * row_bytes..][..row_bytes];
                dst[512 + n * 2..514 + n * 2].copy_from_slice(&b[0..2]);
            }
            for n in 0..TILE_N {
                let s: i32 = col_codes[n].iter().map(|&b| b as i8 as i32).sum();
                let c = (s as i64 * 128) as i32;
                dst[544 + n * 4..548 + n * 4].copy_from_slice(&c.to_le_bytes());
            }
        }
    }
    packed
}

/// Repack a Q5_0 tensor into the same layout: codes dequantized to
/// s8 in [−16, 15] (raw code − 16), block scale d kept as f16.
pub fn repack_q5_0(data: &[u8], n_in: usize, n_out: usize) -> Vec<u8> {
    assert!(n_in % TILE_K == 0, "n_in {n_in} not multiple of 32");
    assert!(n_out % TILE_N == 0, "n_out {n_out} not multiple of 16");
    let kb = n_in / TILE_K;
    let ng = n_out / TILE_N;
    let mut packed = vec![0u8; ng * kb * TILE_BYTES];
    let row_bytes = 22usize;
    let mut col_codes = [[0u8; TILE_K]; TILE_N];
    for g in 0..ng {
        for k in 0..kb {
            let dst = &mut packed[(g * kb + k) * TILE_BYTES..][..TILE_BYTES];
            for n in 0..TILE_N {
                let col = g * TILE_N + n;
                let b = &data[col * row_bytes * kb + k * row_bytes..][..row_bytes];
                // block_q5_0: { d:2, qh:4, qs:16 } — qh BEFORE qs
                let qh32 = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                let ql = &b[6..22];
                for i in 0..16usize {
                    let lo = (ql[i] & 0xF) as i32;
                    let hi = (((qh32 >> i) & 1) << 4) as i32;
                    col_codes[n][i] = ((lo | hi) - 16) as i8 as u8;
                    let hi2 = (ql[i] >> 4) as i32;
                    let bit2 = (((qh32 >> (i + 16)) & 1) << 4) as i32;
                    col_codes[n][i + 16] = ((hi2 | bit2) - 16) as i8 as u8;
                }
            }
            pack_group_16x32(dst, &col_codes);
            for n in 0..TILE_N {
                let col = g * TILE_N + n;
                let b = &data[col * row_bytes * kb + k * row_bytes..][..row_bytes];
                dst[512 + n * 2..514 + n * 2].copy_from_slice(&b[0..2]);
            }
            for n in 0..TILE_N {
                let s: i32 = col_codes[n].iter().map(|&b| b as i8 as i32).sum();
                let c = (s as i64 * 128) as i32;
                dst[544 + n * 4..548 + n * 4].copy_from_slice(&c.to_le_bytes());
            }
        }
    }
    packed
}

/// Decode the 12 packed 6-bit scale/min bytes of a Q4_K block into 16 bytes:
/// [0..8) = sub-block scales, [8..16) = sub-block mins (the u32 algebra of
/// mmq.cpp `unpack_mins_and_scales`, same result as the generic kernels'
/// `get_scale_min_k4` decode).
fn unpack_mins_and_scales(s12: &[u8]) -> [u8; 16] {
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;
    let mut u = [0u32; 4];
    u[0] = u32::from_le_bytes(s12[0..4].try_into().unwrap());
    u[1] = u32::from_le_bytes(s12[4..8].try_into().unwrap());
    u[2] = u32::from_le_bytes(s12[8..12].try_into().unwrap());
    u[3] = ((u[2] >> 4) & KMASK2) | (((u[1] >> 6) & KMASK3) << 4);
    let uaux = u[1] & KMASK1;
    u[1] = (u[2] & KMASK2) | (((u[0] >> 6) & KMASK3) << 4);
    u[2] = uaux;
    u[0] &= KMASK1;
    let mut out = [0u8; 16];
    for (i, w) in u.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// Repack a Q4_K tensor [n_in, n_out] (gguf layout) into packed VNNI.
/// Requires n_in % 256 == 0 and n_out % 16 == 0.
///
/// Value order inside a super-block (from `ggml_vec_dot_q4_K_q8_K_generic`):
/// sub-group 2j takes the LOW nibbles of qs bytes [32j, 32j+32),
/// sub-group 2j+1 the HIGH nibbles of the same bytes.
pub fn repack_q4_k(data: &[u8], n_in: usize, n_out: usize) -> Vec<u8> {
    assert!(n_in % QK_K == 0, "n_in {n_in} not a multiple of 256");
    assert!(n_out % TILE_N == 0, "n_out {n_out} not a multiple of 16");
    let kb = n_in / QK_K;
    let ng = n_out / TILE_N;
    let mut packed = vec![0u8; ng * kb * TILE_BYTES_Q4K];
    let row_bytes = 144usize;
    let mut vals = [[0u8; 32]; 8]; // code values (0..15) per sub-group
    for g in 0..ng {
        for s in 0..kb {
            let dst = &mut packed[(g * kb + s) * TILE_BYTES_Q4K..][..TILE_BYTES_Q4K];
            for n in 0..TILE_N {
                let col = g * TILE_N + n;
                let b = &data[col * row_bytes * kb + s * row_bytes..][..row_bytes];
                dst[Q4K_OFF_D + 2 * n..Q4K_OFF_D + 2 * n + 2].copy_from_slice(&b[0..2]);
                dst[Q4K_OFF_DMIN + 2 * n..Q4K_OFF_DMIN + 2 * n + 2].copy_from_slice(&b[2..4]);
                let sm = unpack_mins_and_scales(&b[4..16]);
                let qs = &b[16..144];
                for j in 0..4usize {
                    let qs32 = &qs[32 * j..32 * j + 32];
                    for r in 0..32usize {
                        vals[2 * j][r] = qs32[r] & 0xF;
                        vals[2 * j + 1][r] = qs32[r] >> 4;
                    }
                }
                // nibble-pair quants: byte 4n+j = (val[8c+j] | val[8c+4+j] << 4)
                for gg in 0..8usize {
                    for c in 0..4usize {
                        for j in 0..4usize {
                            let lo = vals[gg][8 * c + j];
                            let hi = vals[gg][8 * c + 4 + j];
                            dst[gg * 256 + c * 64 + 4 * n + j] = lo | (hi << 4);
                        }
                    }
                    dst[Q4K_OFF_SCALES + 16 * gg + n] = sm[gg];
                }
                // mins in dpwssds pair layout: byte 2n = min(2p, n), 2n+1 = min(2p+1, n)
                for p in 0..4usize {
                    dst[Q4K_OFF_MINS + 32 * p + 2 * n] = sm[8 + 2 * p];
                    dst[Q4K_OFF_MINS + 32 * p + 2 * n + 1] = sm[8 + 2 * p + 1];
                }
            }
        }
    }
    packed
}

/// Repack a Q6_K tensor [n_in, n_out] into packed VNNI.
/// Requires n_in % 256 == 0 and n_out % 16 == 0.
///
/// Value order (from `ggml_vec_dot_q6_K_q8_K_generic`): per 128-value slab j,
/// value at l+{0,32,64,96} = (ql[64j+{l,32+l,l,l+32}] nibble) | qh 2-bit
/// field — transcribed verbatim below.
pub fn repack_q6_k(data: &[u8], n_in: usize, n_out: usize) -> Vec<u8> {
    assert!(n_in % QK_K == 0, "n_in {n_in} not a multiple of 256");
    assert!(n_out % TILE_N == 0, "n_out {n_out} not a multiple of 16");
    let kb = n_in / QK_K;
    let ng = n_out / TILE_N;
    let mut packed = vec![0u8; ng * kb * TILE_BYTES_Q6K];
    let row_bytes = 210usize;
    let mut vals = [0u8; 256];
    for g in 0..ng {
        for s in 0..kb {
            let dst = &mut packed[(g * kb + s) * TILE_BYTES_Q6K..][..TILE_BYTES_Q6K];
            for n in 0..TILE_N {
                let col = g * TILE_N + n;
                let b = &data[col * row_bytes * kb + s * row_bytes..][..row_bytes];
                dst[Q6K_OFF_D + 2 * n..Q6K_OFF_D + 2 * n + 2].copy_from_slice(&b[208..210]);
                let ql = &b[0..128];
                let qh = &b[128..192];
                for j in 0..2usize {
                    for l in 0..32usize {
                        let v0 = (ql[64 * j + l] & 0xF) | (((qh[32 * j + l] >> 0) & 3) << 4);
                        let v1 = (ql[64 * j + 32 + l] & 0xF) | (((qh[32 * j + l] >> 2) & 3) << 4);
                        let v2 = (ql[64 * j + l] >> 4) | (((qh[32 * j + l] >> 4) & 3) << 4);
                        let v3 = (ql[64 * j + 32 + l] >> 4) | (((qh[32 * j + l] >> 6) & 3) << 4);
                        vals[128 * j + l] = v0;
                        vals[128 * j + 32 + l] = v1;
                        vals[128 * j + 64 + l] = v2;
                        vals[128 * j + 96 + l] = v3;
                    }
                }
                for g16 in 0..16usize {
                    // ql nibble pairs: byte 4n+j = (val[8c+j] & 0xF) | ((val[8c+4+j] & 0xF) << 4)
                    // (values are 6-bit: the low nibble goes to ql, the high 2 bits to qh)
                    for c in 0..2usize {
                        for j in 0..4usize {
                            let lo = vals[16 * g16 + 8 * c + j] & 0xF;
                            let hi = vals[16 * g16 + 8 * c + 4 + j] & 0xF;
                            dst[g16 * 128 + c * 64 + 4 * n + j] = lo | (hi << 4);
                        }
                    }
                    // qh byte 4n+j: four 2-bit high parts of values {j, 4+j, 8+j, 12+j}
                    for j in 0..4usize {
                        let byte = ((vals[16 * g16 + j] >> 4) & 3)
                            | (((vals[16 * g16 + 4 + j] >> 4) & 3) << 2)
                            | (((vals[16 * g16 + 8 + j] >> 4) & 3) << 4)
                            | (((vals[16 * g16 + 12 + j] >> 4) & 3) << 6);
                        dst[Q6K_OFF_QH + 64 * g16 + 4 * n + j] = byte;
                    }
                    dst[Q6K_OFF_SCALES + 16 * g16 + n] = b[192 + g16];
                }
            }
        }
    }
    packed
}

// ---------- decode kernel (M == 1) ----------

/// Process-wide free-list of activation-quantization buffers (Phase 6.4 glue
/// trim: the packed path allocated a fresh `q` Vec per matvec — 154
/// allocations + zeroing + first-touch faults per decode token, ~0.5 ms/tok
/// measured). Buffers are handed out as RAII guards and returned on drop;
/// during a matvec the buffer is only *shared-read* by the caller and the
/// worker band (the join precedes the guard's drop), so cross-thread &[u8]
/// access is sound exactly like the previous stack Vec was.
struct QScratch {
    free: std::sync::Mutex<Vec<Vec<u8>>>,
}
static QSCRATCH: std::sync::OnceLock<QScratch> = std::sync::OnceLock::new();

pub(crate) struct QScratchGuard {
    buf: Vec<u8>,
}

impl std::ops::Deref for QScratchGuard {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.buf
    }
}

impl std::ops::DerefMut for QScratchGuard {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.buf
    }
}

impl Drop for QScratchGuard {
    fn drop(&mut self) {
        let pool = QSCRATCH.get_or_init(|| QScratch { free: std::sync::Mutex::new(Vec::new()) });
        // keep the pool bounded: return at most 8 buffers
        let mut free = pool.free.lock().unwrap();
        if free.len() < 8 {
            free.push(std::mem::take(&mut self.buf));
        }
    }
}

/// Take a reusable activation buffer of `bytes` length (zeroed by the
/// quantizer contract: quantize_row_q8_0/q8_k overwrite every byte they
/// later read; the vec is resized grow-only).
fn take_q(bytes: usize) -> QScratchGuard {
    let pool = QSCRATCH.get_or_init(|| QScratch { free: std::sync::Mutex::new(Vec::new()) });
    let mut free = pool.free.lock().unwrap();
    let mut buf = match free.pop() {
        Some(b) if b.len() >= bytes => b,
        Some(b) => {
            let mut b = b;
            b.resize(bytes, 0);
            b
        }
        None => vec![0u8; bytes],
    };
    if buf.len() != bytes {
        buf.resize(bytes, 0);
    }
    drop(free);
    QScratchGuard { buf }
}

/// f16 → f32 (F16C; the exact GGML_CPU_FP16_TO_FP32)
#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn h_f16(b: [u8; 2]) -> f32 {
    use std::arch::x86_64::*;
    let v = _mm_cvtsi32_si128(u16::from_le_bytes(b) as i32);
    _mm_cvtss_f32(_mm_cvtph_ps(v))
}

/// One (16-column group) × full activation: returns 16 f32 results.
/// The inner compute of `tinygemm_kernel_vnni<q8_0, q8_0>::apply`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn group_dot_q8_0(q: &[u8], kb: usize, packed: &[u8], group: usize) -> [f32; TILE_N] {
    use std::arch::x86_64::*;
    let off = _mm512_set1_epi8(-128i8); // 0x80 as u8 (s8s8 offset)
    let mut acc = _mm512_setzero_ps();
    for k in 0..kb {
        let ab = &q[k * 34..(k + 1) * 34];
        let a_ptr = ab[2..34].as_ptr() as *const i32;
        let mut va = [_mm512_setzero_si512(); 8];
        for w in 0..8 {
            va[w] = _mm512_set1_epi32(*a_ptr.add(w));
            va[w] = _mm512_add_epi8(va[w], off);
        }
        let vd1 = _mm512_set1_ps(h_f16([ab[0], ab[1]]));
        let tile = &packed[(group * kb + k) * TILE_BYTES..][..TILE_BYTES];
        let mut vsum = _mm512_setzero_si512();
        for w in 0..8 {
            let vb = _mm512_loadu_si512(tile.as_ptr().add(w * 64) as *const __m512i);
            vsum = _mm512_dpbusd_epi32(vsum, va[w], vb);
        }
        let vcomp = _mm512_loadu_si512(tile.as_ptr().add(544) as *const __m512i);
        vsum = _mm512_sub_epi32(vsum, vcomp);
        let vd0 = _mm512_cvtph_ps(_mm256_loadu_si256(tile.as_ptr().add(512) as *const __m256i));
        let vd = _mm512_mul_ps(vd0, vd1);
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), vd, acc);
    }
    let mut r = [0f32; TILE_N];
    _mm512_storeu_ps(r.as_mut_ptr() as *mut f32, acc);
    r
}

/// One (16-column group) × full q8_K activation for Q4_K weights: the inner
/// compute of `tinygemm_kernel_vnni<block_q8_K, block_q4_K>::apply`
/// (mmq.cpp:1574) — dpbusd over in-register-unpacked nibbles, integer
/// scale multiply per sub-group, dpwssds min term.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn group_dot_q4_k(q: &[u8], kb: usize, packed: &[u8], group: usize) -> [f32; TILE_N] {
    use std::arch::x86_64::*;
    let m4 = _mm512_set1_epi8(0xF);
    let mut acc = _mm512_setzero_ps();
    for s in 0..kb {
        let ab = &q[s * 292..(s + 1) * 292];
        let qs = ab.as_ptr().add(4);
        // 8 sub-groups of 32 activation bytes, 256-bit each
        let mut va = [_mm512_setzero_si512(); 8];
        for (kg, v) in va.iter_mut().enumerate() {
            *v = _mm512_castsi256_si512(_mm256_loadu_si256(qs.add(32 * kg) as *const __m256i));
        }
        // per-32 activation sums (8 i16) in the low 128 bits
        let bsums = _mm256_loadu_si256(ab.as_ptr().add(260) as *const __m256i);
        let q8s = _mm_hadd_epi16(
            _mm256_castsi256_si128(bsums),
            _mm256_extracti128_si256(bsums, 1),
        );
        let va_bsum = _mm512_castsi128_si512(q8s);
        let vd1 = _mm512_set1_ps(f32::from_le_bytes([ab[0], ab[1], ab[2], ab[3]]));
        let tile = &packed[(group * kb + s) * TILE_BYTES_Q4K..][..TILE_BYTES_Q4K];
        let t = tile.as_ptr();
        // step 1: accumulate the quants (8 sub-groups × 4 chunks × 2 dpbusd)
        let mut acci = _mm512_setzero_si512();
        for kg in 0..8usize {
            let mut vsum = _mm512_setzero_si512();
            for c in 0..4usize {
                let bytes = _mm512_loadu_si512(t.add(256 * kg + 64 * c) as *const __m512i);
                let va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32((2 * c) as i32), va[kg]);
                let va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32((2 * c + 1) as i32), va[kg]);
                // nibbles are the UNSIGNED dpbusd operand; activations the signed one
                vsum = _mm512_dpbusd_epi32(vsum, _mm512_and_si512(bytes, m4), va0);
                vsum = _mm512_dpbusd_epi32(
                    vsum,
                    _mm512_and_si512(_mm512_srli_epi16(bytes, 4), m4),
                    va1,
                );
            }
            let vscale = _mm512_cvtepi8_epi32(
                _mm_loadu_si128(t.add(Q4K_OFF_SCALES + 16 * kg) as *const __m128i),
            );
            acci = _mm512_add_epi32(acci, _mm512_mullo_epi32(vsum, vscale));
        }
        let vd0 = _mm512_cvtph_ps(_mm256_loadu_si256(t.add(Q4K_OFF_D) as *const __m256i));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acci), _mm512_mul_ps(vd0, vd1), acc);
        // step 2: the min term — dpwssds over (bsum pair, min pair) per column
        let mut acc_m = _mm512_setzero_si512();
        for k in 0..4usize {
            let vbs = _mm512_permutexvar_epi32(_mm512_set1_epi32(k as i32), va_bsum);
            let vb = _mm512_cvtepi8_epi16(
                _mm256_loadu_si256(t.add(Q4K_OFF_MINS + 32 * k) as *const __m256i),
            );
            acc_m = _mm512_dpwssds_epi32(acc_m, vbs, vb);
        }
        let vdmin = _mm512_cvtph_ps(_mm256_loadu_si256(t.add(Q4K_OFF_DMIN) as *const __m256i));
        acc = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), acc);
    }
    let mut r = [0f32; TILE_N];
    _mm512_storeu_ps(r.as_mut_ptr() as *mut f32, acc);
    r
}

/// One (16-column group) × full q8_K activation for Q6_K weights: the inner
/// compute of `tinygemm_kernel_vnni<block_q8_K, block_q6_K>::apply`
/// (mmq.cpp:1780) — 4-bit + 2-bit in-register merge, the (q−32) offset
/// compensated from the activation bsums×32, integer scale multiply per
/// 16-value group.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn group_dot_q6_k(q: &[u8], kb: usize, packed: &[u8], group: usize) -> [f32; TILE_N] {
    use std::arch::x86_64::*;
    let m4 = _mm512_set1_epi8(0xF);
    let m3 = _mm512_set1_epi8(0x3);
    let mc = _mm512_set1_epi8(0xC);
    let m30 = _mm512_set1_epi8(0x30);
    let mc0 = _mm512_set1_epi8(0xC0u8 as i8);
    let m32 = _mm512_set1_epi32(32);
    let mut acc = _mm512_setzero_ps();
    for s in 0..kb {
        let ab = &q[s * 292..(s + 1) * 292];
        let qs = ab.as_ptr().add(4);
        // the 256 activation bytes as 4 × 512-bit
        let mut va = [_mm512_setzero_si512(); 4];
        for (r, v) in va.iter_mut().enumerate() {
            *v = _mm512_loadu_si512(qs.add(64 * r) as *const __m512i);
        }
        // per-16 activation sums × 32 — the (q − 32) compensation vector
        let bsums = _mm256_loadu_si256(ab.as_ptr().add(260) as *const __m256i);
        let vcomp = _mm512_mullo_epi32(_mm512_cvtepi16_epi32(bsums), m32);
        let vd1 = _mm512_set1_ps(f32::from_le_bytes([ab[0], ab[1], ab[2], ab[3]]));
        let tile = &packed[(group * kb + s) * TILE_BYTES_Q6K..][..TILE_BYTES_Q6K];
        let t = tile.as_ptr();
        let mut acci = _mm512_setzero_si512();
        for g16 in 0..16usize {
            let r = g16 >> 2;
            let d = (4 * g16 - 16 * r) as i32; // dword base of group g16 within va[r]
            let idx0 = _mm512_set1_epi32(d);
            let idx1 = _mm512_set1_epi32(d + 1);
            let idx2 = _mm512_set1_epi32(d + 2);
            let idx3 = _mm512_set1_epi32(d + 3);
            let mut vsum = _mm512_setzero_si512();
            let hbits = _mm512_loadu_si512(t.add(Q6K_OFF_QH + 64 * g16) as *const __m512i);
            // chunk 0 (values 0..8 of the group)
            let bytes0 = _mm512_loadu_si512(t.add(128 * g16) as *const __m512i);
            let vb0 = _mm512_add_epi8(
                _mm512_and_si512(bytes0, m4),
                _mm512_slli_epi16(_mm512_and_si512(hbits, m3), 4),
            );
            let vb1 = _mm512_add_epi8(
                _mm512_and_si512(_mm512_srli_epi16(bytes0, 4), m4),
                _mm512_slli_epi16(_mm512_and_si512(hbits, mc), 2),
            );
            vsum = _mm512_dpbusd_epi32(vsum, vb0, _mm512_permutexvar_epi32(idx0, va[r]));
            vsum = _mm512_dpbusd_epi32(vsum, vb1, _mm512_permutexvar_epi32(idx1, va[r]));
            // chunk 1 (values 8..16)
            let bytes1 = _mm512_loadu_si512(t.add(128 * g16 + 64) as *const __m512i);
            let vb2 = _mm512_add_epi8(_mm512_and_si512(bytes1, m4), _mm512_and_si512(hbits, m30));
            let vb3 = _mm512_add_epi8(
                _mm512_and_si512(_mm512_srli_epi16(bytes1, 4), m4),
                _mm512_srli_epi16(_mm512_and_si512(hbits, mc0), 2),
            );
            vsum = _mm512_dpbusd_epi32(vsum, vb2, _mm512_permutexvar_epi32(idx2, va[r]));
            vsum = _mm512_dpbusd_epi32(vsum, vb3, _mm512_permutexvar_epi32(idx3, va[r]));
            // − 32·Σ(activation) per 16-group
            vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(_mm512_set1_epi32(g16 as i32), vcomp));
            let vscale = _mm512_cvtepi8_epi32(
                _mm_loadu_si128(t.add(Q6K_OFF_SCALES + 16 * g16) as *const __m128i),
            );
            acci = _mm512_add_epi32(acci, _mm512_mullo_epi32(vsum, vscale));
        }
        let vd0 = _mm512_cvtph_ps(_mm256_loadu_si256(t.add(Q6K_OFF_D) as *const __m256i));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acci), _mm512_mul_ps(vd0, vd1), acc);
    }
    let mut r = [0f32; TILE_N];
    _mm512_storeu_ps(r.as_mut_ptr() as *mut f32, acc);
    r
}

/// Compute the group band [g_lo, g_hi) of a packed Q8_0/Q5_0 matvec into `out`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn vnni_band(packed: &[u8], q: &[u8], kb: usize, g_lo: usize, g_hi: usize, out: *mut f32, is_caller: bool) {
    let t0 = if kstats::enabled() { kstats::tsc() } else { 0 };
    for g in g_lo..g_hi {
        let r = group_dot_q8_0(q, kb, packed, g);
        let dst = out.add(g * TILE_N);
        std::ptr::copy_nonoverlapping(r.as_ptr(), dst, TILE_N);
    }
    if kstats::enabled() {
        let dt = kstats::tsc().wrapping_sub(t0);
        kstats::DOT_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        if is_caller {
            kstats::CALLER_BAND_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Band-job factory for the pool (caller band + worker band).
#[allow(clippy::too_many_arguments)]
fn vnni_band_job<'a>(
    packed: &'a [u8],
    q: &'a [u8],
    kb: usize,
    out: *mut f32,
    g_lo: usize,
    g_hi: usize,
    is_caller: bool,
) -> impl FnMut() + 'a {
    move || unsafe { vnni_band(packed, q, kb, g_lo, g_hi, out, is_caller) }
}

/// Compute the group band [g_lo, g_hi) of a packed K-quant (Q4_K/Q6_K)
/// matvec into `out`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn kq_band(packed: &[u8], ty: GgmlType, q: &[u8], kb: usize, g_lo: usize, g_hi: usize, out: *mut f32, is_caller: bool) {
    let t0 = if kstats::enabled() { kstats::tsc() } else { 0 };
    for g in g_lo..g_hi {
        let r = if ty == GgmlType::Q4_K {
            group_dot_q4_k(q, kb, packed, g)
        } else {
            group_dot_q6_k(q, kb, packed, g)
        };
        let dst = out.add(g * TILE_N);
        std::ptr::copy_nonoverlapping(r.as_ptr(), dst, TILE_N);
    }
    if kstats::enabled() {
        let dt = kstats::tsc().wrapping_sub(t0);
        kstats::DOT_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        if is_caller {
            kstats::CALLER_BAND_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// K-quant band-job factory for the pool.
#[allow(clippy::too_many_arguments)]
fn kq_band_job<'a>(
    packed: &'a [u8],
    ty: GgmlType,
    q: &'a [u8],
    kb: usize,
    out: *mut f32,
    g_lo: usize,
    g_hi: usize,
    is_caller: bool,
) -> impl FnMut() + 'a {
    move || unsafe { kq_band(packed, ty, q, kb, g_lo, g_hi, out, is_caller) }
}

/// Packed matvec entry: `packed` from repack_*, x = f32 activation (n_in),
/// out (n_out). Dispatched on the worker pool with a balanced band split.
/// `ty` selects the packed layout: Q8_0/Q5_0 → 576 B tiles + q8_0
/// activations; Q4_K/Q6_K → K-quant tiles + q8_K activations (the K-quant
/// vec_dot_type — the same contract as the AVX2 path).
pub fn matvec_vnni(
    packed: &[u8],
    ty: GgmlType,
    n_in: usize,
    n_out: usize,
    x: &[f32],
    n_threads: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), n_out);
    let prof = kstats::enabled();
    let tw0 = if prof { kstats::tsc() } else { 0 };
    if prof {
        kstats::MATVECS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    match ty {
        GgmlType::Q8_0 | GgmlType::Q5_0 => {
            // activation quantized ONCE to Q8_0 — the same numeric contract as the
            // AVX2 path (quantize_row_q8_0, round-to-nearest-even); buffer from
            // the scratch pool (no per-matvec allocation)
            assert!(n_in % 32 == 0);
            let kb = n_in / 32;
            let mut q = take_q(kb * 34);
            quant::quantize_row_q8_0(x, &mut *q);
            let ng = n_out / TILE_N;
            let n_threads = n_threads.max(1);
            if n_threads <= 1 || ng < 2 || packed.len() < split_min_bytes() {
                let ptr = out.as_mut_ptr();
                unsafe { vnni_band(packed, &*q, kb, 0, ng, ptr, true) };
            } else {
                let g_mid = (ng / 2).clamp(1, ng - 1);
                let ptr = out.as_mut_ptr();
                let caller = vnni_band_job(packed, &*q, kb, ptr, 0, g_mid, true);
                let worker = vnni_band_job(packed, &*q, kb, ptr, g_mid, ng, false);
                pool::par_chunks2(caller, worker);
            }
        }
        GgmlType::Q4_K | GgmlType::Q6_K => {
            // activation quantized ONCE to q8_K (the K-quant vec_dot_type;
            // quantize_row_q8_K — the quantizer the x86 build actually runs);
            // buffer from the scratch pool (no per-matvec allocation)
            assert!(n_in % QK_K == 0);
            let kb = n_in / QK_K;
            let mut q = take_q(kb * 292);
            quant::quantize_row_q8_k(x, &mut *q);
            let ng = n_out / TILE_N;
            let n_threads = n_threads.max(1);
            if n_threads <= 1 || ng < 2 || packed.len() < split_min_bytes() {
                let ptr = out.as_mut_ptr();
                unsafe { kq_band(packed, ty, &*q, kb, 0, ng, ptr, true) };
            } else {
                let g_mid = (ng / 2).clamp(1, ng - 1);
                let ptr = out.as_mut_ptr();
                let caller = kq_band_job(packed, ty, &*q, kb, ptr, 0, g_mid, true);
                let worker = kq_band_job(packed, ty, &*q, kb, ptr, g_mid, ng, false);
                pool::par_chunks2(caller, worker);
            }
        }
        other => unreachable!("packed VNNI path does not support {other:?}"),
    }
    if prof {
        kstats::MATVEC_WALL_TICKS.fetch_add(kstats::tsc().wrapping_sub(tw0), std::sync::atomic::Ordering::Relaxed);
    }
}

/// Band over two packed tensors (fused gate/up pair): the concatenated group
/// space [0, ng1+ng2); groups [0, ng1) → out1, the rest → out2.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn vnni_band2(
    p1: &[u8], ng1: usize,
    p2: &[u8],
    q: &[u8], kb: usize,
    g_lo: usize, g_hi: usize,
    out1: *mut f32, out2: *mut f32,
    is_caller: bool,
) {
    let t0 = if kstats::enabled() { kstats::tsc() } else { 0 };
    for g in g_lo..g_hi {
        if g < ng1 {
            let r = group_dot_q8_0(q, kb, p1, g);
            std::ptr::copy_nonoverlapping(r.as_ptr(), out1.add(g * TILE_N), TILE_N);
        } else {
            let g2 = g - ng1;
            let r = group_dot_q8_0(q, kb, p2, g2);
            std::ptr::copy_nonoverlapping(r.as_ptr(), out2.add(g2 * TILE_N), TILE_N);
        }
    }
    if kstats::enabled() {
        let dt = kstats::tsc().wrapping_sub(t0);
        kstats::DOT_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        if is_caller {
            kstats::CALLER_BAND_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Band over two packed K-quant tensors (fused gate/up pair): the
/// concatenated group space [0, ng1+ng2); groups [0, ng1) → out1, the rest →
/// out2. Both tensors must be of the same type (asserted by the caller).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn kq_band2(
    p1: &[u8], ty: GgmlType, ng1: usize,
    p2: &[u8],
    q: &[u8], kb: usize,
    g_lo: usize, g_hi: usize,
    out1: *mut f32, out2: *mut f32,
    is_caller: bool,
) {
    let t0 = if kstats::enabled() { kstats::tsc() } else { 0 };
    let dot = |p: &[u8], g: usize| {
        if ty == GgmlType::Q4_K { group_dot_q4_k(q, kb, p, g) } else { group_dot_q6_k(q, kb, p, g) }
    };
    for g in g_lo..g_hi {
        if g < ng1 {
            let r = dot(p1, g);
            std::ptr::copy_nonoverlapping(r.as_ptr(), out1.add(g * TILE_N), TILE_N);
        } else {
            let g2 = g - ng1;
            let r = dot(p2, g2);
            std::ptr::copy_nonoverlapping(r.as_ptr(), out2.add(g2 * TILE_N), TILE_N);
        }
    }
    if kstats::enabled() {
        let dt = kstats::tsc().wrapping_sub(t0);
        kstats::DOT_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        if is_caller {
            kstats::CALLER_BAND_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Band-split policy for small packed matvecs (Phase 6.8 dispatch work).
/// The Phase 6.6 audit measured the per-matvec pool round-trip (slot
/// publish + worker claim + caller spin-join) at ~40 µs — for a small
/// tensor whose whole single-thread compute is ~2× shorter than that,
/// splitting the matvec between caller and worker costs MORE than it
/// saves (the Qwen attention-projection stage ran at 3.29 GB/s vs the
/// 11-19 GB/s of the big tensors: §9.3's "dispatch granularity" term).
/// Below this many packed bytes the matvec runs single-band on the caller
/// (identical results — the band split is a scheduling choice, never a
/// numeric one). Overridable with LLAMA_RS_SPLIT_MIN (bytes; 0 = always
/// split, the previous behavior).
fn split_min_bytes() -> usize {
    static E: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *E.get_or_init(|| match std::env::var("LLAMA_RS_SPLIT_MIN") {
        Ok(v) => v.parse().unwrap_or(0),
        Err(_) => 1_500_000,
    })
}

/// Band over three packed tensors (q/k/v): the concatenated group space
/// [0, ng1+ng2+ng3); groups [0, ng1) → out1, [ng1, ng1+ng2) → out2, the
/// rest → out3. The same `group_dot_any` per group — the per-tensor outputs
/// are bit-identical to three separate `matvec_vnni` calls on the same
/// quantized activation (the quantizer is deterministic, so one shared
/// quantization produces exactly the bytes the three calls would each
/// produce).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn band3_any(
    ty: GgmlType,
    p1: &[u8], ng1: usize,
    p2: &[u8], ng2: usize,
    p3: &[u8],
    q: &[u8], kb: usize,
    g_lo: usize, g_hi: usize,
    out1: *mut f32, out2: *mut f32, out3: *mut f32,
    is_caller: bool,
) {
    let t0 = if kstats::enabled() { kstats::tsc() } else { 0 };
    for g in g_lo..g_hi {
        if g < ng1 {
            let r = group_dot_any(ty, q, kb, p1, g);
            std::ptr::copy_nonoverlapping(r.as_ptr(), out1.add(g * TILE_N), TILE_N);
        } else if g < ng1 + ng2 {
            let g2 = g - ng1;
            let r = group_dot_any(ty, q, kb, p2, g2);
            std::ptr::copy_nonoverlapping(r.as_ptr(), out2.add(g2 * TILE_N), TILE_N);
        } else {
            let g3 = g - ng1 - ng2;
            let r = group_dot_any(ty, q, kb, p3, g3);
            std::ptr::copy_nonoverlapping(r.as_ptr(), out3.add(g3 * TILE_N), TILE_N);
        }
    }
    if kstats::enabled() {
        let dt = kstats::tsc().wrapping_sub(t0);
        kstats::DOT_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        if is_caller {
            kstats::CALLER_BAND_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Fused triple matvec (q/k/v projections): one activation quantization,
/// one dispatch over the concatenated group space of the three packed
/// tensors. All three must share `ty`/`n_in` (the attention q/k/v of both
/// supported architectures do); otherwise the caller falls back to three
/// `matvec_vnni` calls. Per-tensor outputs bit-identical to the separate
/// calls (see `band3_any`).
#[allow(clippy::too_many_arguments)]
pub fn matvec3_vnni(
    p1: &[u8], ty1: GgmlType, n_out1: usize,
    p2: &[u8], ty2: GgmlType, n_out2: usize,
    p3: &[u8], ty3: GgmlType, n_out3: usize,
    n_in: usize,
    x: &[f32],
    n_threads: usize,
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
) {
    debug_assert_eq!(out1.len(), n_out1);
    debug_assert_eq!(out2.len(), n_out2);
    debug_assert_eq!(out3.len(), n_out3);
    if !(ty1 == ty2 && ty1 == ty3) {
        matvec_vnni(p1, ty1, n_in, n_out1, x, n_threads, out1);
        matvec_vnni(p2, ty2, n_in, n_out2, x, n_threads, out2);
        matvec_vnni(p3, ty3, n_in, n_out3, x, n_threads, out3);
        return;
    }
    let ty = ty1;
    let prof = kstats::enabled();
    let tw0 = if prof { kstats::tsc() } else { 0 };
    if prof {
        kstats::MATVECS.fetch_add(3, std::sync::atomic::Ordering::Relaxed);
    }
    let kb = match ty {
        GgmlType::Q8_0 | GgmlType::Q5_0 => {
            assert!(n_in % 32 == 0);
            n_in / 32
        }
        GgmlType::Q4_K | GgmlType::Q6_K => {
            assert!(n_in % QK_K == 0);
            n_in / QK_K
        }
        other => unreachable!("packed VNNI triple does not support {other:?}"),
    };
    let mut q = take_q(match ty {
        GgmlType::Q8_0 | GgmlType::Q5_0 => kb * 34,
        _ => kb * 292,
    });
    if matches!(ty, GgmlType::Q4_K | GgmlType::Q6_K) {
        quant::quantize_row_q8_k(x, &mut *q);
    } else {
        quant::quantize_row_q8_0(x, &mut *q);
    }
    let ng1 = n_out1 / TILE_N;
    let ng2 = n_out2 / TILE_N;
    let ng3 = n_out3 / TILE_N;
    let ng = ng1 + ng2 + ng3;
    let n_threads = n_threads.max(1);
    let total_bytes = p1.len() + p2.len() + p3.len();
    if n_threads <= 1 || ng < 2 || total_bytes < split_min_bytes() {
        let o1 = out1.as_mut_ptr();
        let o2 = out2.as_mut_ptr();
        let o3 = out3.as_mut_ptr();
        unsafe { band3_any(ty, p1, ng1, p2, ng2, p3, &*q, kb, 0, ng, o1, o2, o3, true) };
    } else {
        let g_mid = (ng / 2).clamp(1, ng - 1);
        let o1 = out1.as_mut_ptr();
        let o2 = out2.as_mut_ptr();
        let o3 = out3.as_mut_ptr();
        let qa: &[u8] = &*q;
        let caller = move || unsafe {
            band3_any(ty, p1, ng1, p2, ng2, p3, qa, kb, 0, g_mid, o1, o2, o3, true)
        };
        let worker = move || unsafe {
            band3_any(ty, p1, ng1, p2, ng2, p3, qa, kb, g_mid, ng, o1, o2, o3, false)
        };
        pool::par_chunks2(caller, worker);
    }
    if prof {
        kstats::MATVEC_WALL_TICKS.fetch_add(kstats::tsc().wrapping_sub(tw0), std::sync::atomic::Ordering::Relaxed);
    }
}

/// Fused packed pair matvec (gate/up): one activation quantization, one
/// dispatch. Both tensors must share `ty`/`n_in` (the gate/up pairs of the
/// supported models do); a mixed-type pair is computed as two packed
/// matvecs (correct, one extra activation quantization).
#[allow(clippy::too_many_arguments)]
pub fn matvec2_vnni(
    p1: &[u8], ty1: GgmlType, n_out1: usize,
    p2: &[u8], ty2: GgmlType, n_out2: usize,
    n_in: usize,
    x: &[f32],
    n_threads: usize,
    out1: &mut [f32],
    out2: &mut [f32],
) {
    debug_assert_eq!(out1.len(), n_out1);
    debug_assert_eq!(out2.len(), n_out2);
    if ty1 != ty2 {
        matvec_vnni(p1, ty1, n_in, n_out1, x, n_threads, out1);
        matvec_vnni(p2, ty2, n_in, n_out2, x, n_threads, out2);
        return;
    }
    let ty = ty1;
    let prof = kstats::enabled();
    let tw0 = if prof { kstats::tsc() } else { 0 };
    if prof {
        kstats::MATVECS.fetch_add(2, std::sync::atomic::Ordering::Relaxed);
    }
    let kb = match ty {
        GgmlType::Q8_0 | GgmlType::Q5_0 => {
            assert!(n_in % 32 == 0);
            n_in / 32
        }
        GgmlType::Q4_K | GgmlType::Q6_K => {
            assert!(n_in % QK_K == 0);
            n_in / QK_K
        }
        other => unreachable!("packed VNNI pair does not support {other:?}"),
    };
    let mut q = take_q(match ty {
        GgmlType::Q8_0 | GgmlType::Q5_0 => kb * 34,
        _ => kb * 292,
    });
    if matches!(ty, GgmlType::Q4_K | GgmlType::Q6_K) {
        quant::quantize_row_q8_k(x, &mut *q);
    } else {
        quant::quantize_row_q8_0(x, &mut *q);
    }

    let ng1 = n_out1 / TILE_N;
    let ng2 = n_out2 / TILE_N;
    let ng = ng1 + ng2;
    let n_threads = n_threads.max(1);
    if n_threads <= 1 || ng < 2 || p1.len() + p2.len() < split_min_bytes() {
        let o1 = out1.as_mut_ptr();
        let o2 = out2.as_mut_ptr();
        if ty == GgmlType::Q4_K || ty == GgmlType::Q6_K {
            unsafe { kq_band2(p1, ty, ng1, p2, &*q, kb, 0, ng, o1, o2, true) };
        } else {
            unsafe { vnni_band2(p1, ng1, p2, &*q, kb, 0, ng, o1, o2, true) };
        }
    } else {
        let g_mid = (ng / 2).clamp(1, ng - 1);
        let o1 = out1.as_mut_ptr();
        let o2 = out2.as_mut_ptr();
        let q1: &[u8] = &*q;
        let q2: &[u8] = &*q;
        if ty == GgmlType::Q4_K || ty == GgmlType::Q6_K {
            let caller = move || unsafe { kq_band2(p1, ty, ng1, p2, q1, kb, 0, g_mid, o1, o2, true) };
            let worker = move || unsafe { kq_band2(p1, ty, ng1, p2, q2, kb, g_mid, ng, o1, o2, false) };
            pool::par_chunks2(caller, worker);
        } else {
            let caller = move || unsafe { vnni_band2(p1, ng1, p2, q1, kb, 0, g_mid, o1, o2, true) };
            let worker = move || unsafe { vnni_band2(p1, ng1, p2, q2, kb, g_mid, ng, o1, o2, false) };
            pool::par_chunks2(caller, worker);
        }
    }
    if prof {
        kstats::MATVEC_WALL_TICKS.fetch_add(kstats::tsc().wrapping_sub(tw0), std::sync::atomic::Ordering::Relaxed);
    }
}

// ---------- batched GEMM (Phase 6.5: prefill M>1) ----------
//
// Structure: the activation matrix (M token rows × n_in, row-major f32) is
// quantized ONCE per input (QBatch — one row per token, the same quantizer
// contract as the M=1 path), then each 16-column group of the packed weights
// is streamed against all M rows: for every (group, row) pair the inner
// compute is exactly `group_dot_*` — the same dpbusd sequence, the same
// accumulation order — so every output column of the GEMM is bit-identical
// to running `matvec_vnni` on that single token. Weight tiles are read from
// DRAM once per batch (the group loop is outer, the token loop inner; tiles
// stay L1/L2-resident across the M rows), which is the entire batching win:
// DRAM traffic per weight byte drops from once-per-token to once-per-batch.

/// Quantized activation matrix for the packed GEMM — M rows, one per token,
/// each row exactly what `matvec_vnni` would produce for that token
/// (Q8_0 rows: 34·n_in/32 bytes; Q8_K rows: 292·n_in/256 bytes).
pub struct QBatch {
    data: Vec<u8>,
    row_bytes: usize,
    rows: usize,
}

impl QBatch {
    #[inline]
    pub fn row(&self, m: usize) -> &[u8] {
        &self.data[m * self.row_bytes..(m + 1) * self.row_bytes]
    }
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }
}

impl Drop for QBatch {
    fn drop(&mut self) {
        // return the buffer to the scratch pool (grow-only, bounded)
        let pool = QSCRATCH.get_or_init(|| QScratch { free: std::sync::Mutex::new(Vec::new()) });
        let mut free = pool.free.lock().unwrap();
        if free.len() < 8 {
            free.push(std::mem::take(&mut self.data));
        }
    }
}

fn take_batch(bytes: usize) -> Vec<u8> {
    let pool = QSCRATCH.get_or_init(|| QScratch { free: std::sync::Mutex::new(Vec::new()) });
    let mut free = pool.free.lock().unwrap();
    let mut buf = match free.pop() {
        Some(b) if b.len() >= bytes => b,
        Some(b) => {
            let mut b = b;
            b.resize(bytes, 0);
            b
        }
        None => vec![0u8; bytes],
    };
    if buf.len() != bytes {
        buf.resize(bytes, 0);
    }
    drop(free);
    buf
}

/// Row bytes of a quantized activation row for weight type `ty`.
#[inline]
fn batch_row_bytes(ty: GgmlType, n_in: usize) -> usize {
    match ty {
        GgmlType::Q8_0 | GgmlType::Q5_0 => {
            assert!(n_in % TILE_K == 0);
            (n_in / TILE_K) * 34
        }
        GgmlType::Q4_K | GgmlType::Q6_K => {
            assert!(n_in % QK_K == 0);
            (n_in / QK_K) * 292
        }
        other => unreachable!("packed GEMM does not support {other:?}"),
    }
}

/// Quantize the M×n_in row-major activation matrix `x` for weight type `ty`.
/// Rows are quantized independently (each token = one M=1 contract); with
/// n_threads > 1 the row range is split between caller and worker.
pub fn quantize_batch(
    ty: GgmlType,
    x: &[f32],
    n_in: usize,
    m: usize,
    n_threads: usize,
) -> QBatch {
    debug_assert_eq!(x.len(), n_in * m);
    let row_bytes = batch_row_bytes(ty, n_in);
    let mut data = take_batch(m * row_bytes);
    let quant_row = |mm: usize, dst: &mut [u8]| {
        let src = &x[mm * n_in..(mm + 1) * n_in];
        if matches!(ty, GgmlType::Q4_K | GgmlType::Q6_K) {
            quant::quantize_row_q8_k(src, dst);
        } else {
            quant::quantize_row_q8_0(src, dst);
        }
    };
    let n_threads = n_threads.max(1);
    if n_threads <= 1 || m < 16 {
        for mm in 0..m {
            quant_row(mm, &mut data[mm * row_bytes..(mm + 1) * row_bytes]);
        }
    } else {
        // split the row range between caller and worker through raw pointers
        // (disjoint halves; the pool join precedes the return — same pattern
        // as the matvec band jobs)
        let mid = (m / 2).max(1).min(m - 1);
        let base = data.as_mut_ptr();
        let lo = move || unsafe {
            for mm in 0..mid {
                let dst = std::slice::from_raw_parts_mut(base.add(mm * row_bytes), row_bytes);
                quant_row(mm, dst);
            }
        };
        let hi = move || unsafe {
            for mm in mid..m {
                let dst = std::slice::from_raw_parts_mut(base.add(mm * row_bytes), row_bytes);
                quant_row(mm, dst);
            }
        };
        pool::par_chunks2(lo, hi);
    }
    QBatch { data, row_bytes, rows: m }
}

/// One (group, row) dot for any supported packed type — dispatch by `ty`.
/// Returns the 16 outputs of tile `group` against quantized row `q_row`.
#[inline]
unsafe fn group_dot_any(ty: GgmlType, q_row: &[u8], kb: usize, packed: &[u8], group: usize) -> [f32; TILE_N] {
    match ty {
        GgmlType::Q4_K => group_dot_q4_k(q_row, kb, packed, group),
        GgmlType::Q6_K => group_dot_q6_k(q_row, kb, packed, group),
        // Q8_0/Q5_0 share the 576 B tile + q8_0-activation kernel
        _ => group_dot_q8_0(q_row, kb, packed, group),
    }
}

// ---------- register-blocked GEMM inner kernels (Phase 6.8) ----------
//
// The Phase 6.6 audit ranked the GEMM as the binding term of prefill after
// Phase 6.7 (LFM3 GEMM 82 % of prefill wall at 0.55 GB/s; §9.4/§9.8 item 2)
// and §8.1 root-caused it: `gemm_band` called `group_dot_*` once per
// (group, row), so every row re-LOADED and re-UNPACKED the same weight tile
// and ran one serial 8-deep dpbusd dependency chain (~17 % of the VNNI
// issue-rate ceiling). The kernels below process a block of RB rows per
// pass: tile bytes are loaded and nibbles unpacked ONCE per (group, k) and
// streamed into RB independent accumulator chains — both the load/unpack
// traffic and the latency exposure divide by RB. A 16x pre-broadcast
// activation layout was evaluated and rejected by arithmetic: the
// broadcast bytes would be re-swept once per 16-column group (ng sweeps of
// a 16x M*kb buffer = GB-scale L3 traffic per GEMM call), so the broadcast
// stays an in-register `vpermd` — the same 1-shuffle-per-dpbusd ratio
// llama.cpp's own non-AMX VNNI path runs at.
//
// Bit-identity contract (unchanged from Phase 6.5): per row the instruction
// sequence — activation loads, broadcast indices, dpbusd order, integer
// scale multiplies, f32 FMA/FMA order — is IDENTICAL to `group_dot_*` on
// that row, so every GEMM column stays bit-identical to `matvec_vnni` run
// on the single token.

/// Rows per register block.
const RB: usize = 4;

/// Q8_0/Q5_0 family, RB rows per tile pass.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,f16c")]
unsafe fn group_dot4_q8_0(
    q: &QBatch,
    m0: usize,
    kb: usize,
    packed: &[u8],
    group: usize,
    out: *mut f32,
    out_stride: usize,
) {
    use std::arch::x86_64::*;
    let off = _mm512_set1_epi8(-128i8); // 0x80 as u8 (s8s8 offset)
    let mut acc = [_mm512_setzero_ps(); RB];
    let rowp: [*const u8; RB] = std::array::from_fn(|r| q.row(m0 + r).as_ptr());
    for k in 0..kb {
        let tile = &packed[(group * kb + k) * TILE_BYTES..][..TILE_BYTES];
        let t = tile.as_ptr();
        let vd0 = _mm512_cvtph_ps(_mm256_loadu_si256(t.add(512) as *const __m256i));
        let vcomp = _mm512_loadu_si512(t.add(544) as *const __m512i);
        let mut vsum = [_mm512_setzero_si512(); RB];
        for w in 0..8usize {
            // the weight row is loaded once and fed to all RB rows
            let vb = _mm512_loadu_si512(t.add(w * 64) as *const __m512i);
            for r in 0..RB {
                let a = std::ptr::read_unaligned(rowp[r].add(k * 34 + 2 + 4 * w) as *const i32);
                let va = _mm512_add_epi8(_mm512_set1_epi32(a), off);
                vsum[r] = _mm512_dpbusd_epi32(vsum[r], va, vb);
            }
        }
        for r in 0..RB {
            let v = _mm512_sub_epi32(vsum[r], vcomp);
            let d = std::ptr::read_unaligned(rowp[r].add(k * 34) as *const [u8; 2]);
            let vd1 = _mm512_set1_ps(h_f16(d));
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(v), _mm512_mul_ps(vd0, vd1), acc[r]);
        }
    }
    for r in 0..RB {
        _mm512_storeu_ps(out.add((m0 + r) * out_stride + group * TILE_N) as *mut f32, acc[r]);
    }
}

/// Q4_K, RB rows per tile pass (tile nibble unpack hoisted out of the row
/// loop; per-row dpbusd chains independent).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn group_dot4_q4_k(
    q: &QBatch,
    m0: usize,
    kb: usize,
    packed: &[u8],
    group: usize,
    out: *mut f32,
    out_stride: usize,
) {
    use std::arch::x86_64::*;
    let m4 = _mm512_set1_epi8(0xF);
    let mut acc = [_mm512_setzero_ps(); RB];
    let rowp: [*const u8; RB] = std::array::from_fn(|r| q.row(m0 + r).as_ptr());
    for s in 0..kb {
        let tile = &packed[(group * kb + s) * TILE_BYTES_Q4K..][..TILE_BYTES_Q4K];
        let t = tile.as_ptr();
        let mut acci = [_mm512_setzero_si512(); RB];
        let mut va_bsum = [_mm512_setzero_si512(); RB];
        let mut vd1 = [_mm512_setzero_ps(); RB];
        for r in 0..RB {
            let ab = rowp[r].add(s * 292);
            let bsums = _mm256_loadu_si256(ab.add(260) as *const __m256i);
            let q8s = _mm_hadd_epi16(
                _mm256_castsi256_si128(bsums),
                _mm256_extracti128_si256(bsums, 1),
            );
            va_bsum[r] = _mm512_castsi128_si512(q8s);
            vd1[r] = _mm512_set1_ps(f32::from_le_bytes(std::ptr::read_unaligned(ab as *const [u8; 4])));
        }
        // step 1: accumulate the quants (8 sub-groups x 4 chunks x 2 dpbusd)
        for kg in 0..8usize {
            let mut va = [_mm512_setzero_si512(); RB];
            for r in 0..RB {
                va[r] = _mm512_castsi256_si512(
                    _mm256_loadu_si256(rowp[r].add(s * 292 + 4 + 32 * kg) as *const __m256i),
                );
            }
            let mut vsum = [_mm512_setzero_si512(); RB];
            for c in 0..4usize {
                // tile chunk loaded + nibbles unpacked ONCE for the whole row block
                let bytes = _mm512_loadu_si512(t.add(256 * kg + 64 * c) as *const __m512i);
                let nb0 = _mm512_and_si512(bytes, m4);
                let nb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), m4);
                let i0 = _mm512_set1_epi32((2 * c) as i32);
                let i1 = _mm512_set1_epi32((2 * c + 1) as i32);
                for r in 0..RB {
                    // nibbles are the UNSIGNED dpbusd operand; activations the signed one
                    vsum[r] = _mm512_dpbusd_epi32(
                        vsum[r],
                        nb0,
                        _mm512_permutexvar_epi32(i0, va[r]),
                    );
                    vsum[r] = _mm512_dpbusd_epi32(
                        vsum[r],
                        nb1,
                        _mm512_permutexvar_epi32(i1, va[r]),
                    );
                }
            }
            let vscale = _mm512_cvtepi8_epi32(
                _mm_loadu_si128(t.add(Q4K_OFF_SCALES + 16 * kg) as *const __m128i),
            );
            for r in 0..RB {
                acci[r] = _mm512_add_epi32(acci[r], _mm512_mullo_epi32(vsum[r], vscale));
            }
        }
        let vd0 = _mm512_cvtph_ps(_mm256_loadu_si256(t.add(Q4K_OFF_D) as *const __m256i));
        for r in 0..RB {
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acci[r]), _mm512_mul_ps(vd0, vd1[r]), acc[r]);
        }
        // step 2: the min term — dpwssds over (bsum pair, min pair) per column;
        // the four min chunks are widened once and shared by the whole row block
        let mut vbm = [_mm512_setzero_si512(); 4];
        for k in 0..4usize {
            vbm[k] = _mm512_cvtepi8_epi16(
                _mm256_loadu_si256(t.add(Q4K_OFF_MINS + 32 * k) as *const __m256i),
            );
        }
        let vdmin = _mm512_cvtph_ps(_mm256_loadu_si256(t.add(Q4K_OFF_DMIN) as *const __m256i));
        for r in 0..RB {
            let mut acc_m = _mm512_setzero_si512();
            for k in 0..4usize {
                let vbs = _mm512_permutexvar_epi32(_mm512_set1_epi32(k as i32), va_bsum[r]);
                acc_m = _mm512_dpwssds_epi32(acc_m, vbs, vbm[k]);
            }
            acc[r] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1[r]), acc[r]);
        }
    }
    for r in 0..RB {
        _mm512_storeu_ps(out.add((m0 + r) * out_stride + group * TILE_N) as *mut f32, acc[r]);
    }
}

/// Q6_K, RB rows per tile pass (4-bit + 2-bit merge hoisted; per-row
/// (q-32) compensation from the precomputed vcomp[r]).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn group_dot4_q6_k(
    q: &QBatch,
    m0: usize,
    kb: usize,
    packed: &[u8],
    group: usize,
    out: *mut f32,
    out_stride: usize,
) {
    use std::arch::x86_64::*;
    let m4 = _mm512_set1_epi8(0xF);
    let m3 = _mm512_set1_epi8(0x3);
    let mc = _mm512_set1_epi8(0xC);
    let m30 = _mm512_set1_epi8(0x30);
    let mc0 = _mm512_set1_epi8(0xC0u8 as i8);
    let m32 = _mm512_set1_epi32(32);
    let mut acc = [_mm512_setzero_ps(); RB];
    let rowp: [*const u8; RB] = std::array::from_fn(|r| q.row(m0 + r).as_ptr());
    for s in 0..kb {
        let tile = &packed[(group * kb + s) * TILE_BYTES_Q6K..][..TILE_BYTES_Q6K];
        let t = tile.as_ptr();
        let mut vcomp = [_mm512_setzero_si512(); RB];
        let mut vd1 = [_mm512_setzero_ps(); RB];
        for r in 0..RB {
            let ab = rowp[r].add(s * 292);
            let bsums = _mm256_loadu_si256(ab.add(260) as *const __m256i);
            vcomp[r] = _mm512_mullo_epi32(_mm512_cvtepi16_epi32(bsums), m32);
            vd1[r] = _mm512_set1_ps(f32::from_le_bytes(std::ptr::read_unaligned(ab as *const [u8; 4])));
        }
        let mut acci = [_mm512_setzero_si512(); RB];
        for g16 in 0..16usize {
            let r_ = g16 >> 2;
            let d = (4 * g16 - 16 * r_) as i32; // dword base of group g16 within the activation
            let idx0 = _mm512_set1_epi32(d);
            let idx1 = _mm512_set1_epi32(d + 1);
            let idx2 = _mm512_set1_epi32(d + 2);
            let idx3 = _mm512_set1_epi32(d + 3);
            // tile loads + the 4-bit/2-bit merge: once per (g16), shared by RB rows
            let hbits = _mm512_loadu_si512(t.add(Q6K_OFF_QH + 64 * g16) as *const __m512i);
            let bytes0 = _mm512_loadu_si512(t.add(128 * g16) as *const __m512i);
            let bytes1 = _mm512_loadu_si512(t.add(128 * g16 + 64) as *const __m512i);
            let vb0 = _mm512_add_epi8(
                _mm512_and_si512(bytes0, m4),
                _mm512_slli_epi16(_mm512_and_si512(hbits, m3), 4),
            );
            let vb1 = _mm512_add_epi8(
                _mm512_and_si512(_mm512_srli_epi16(bytes0, 4), m4),
                _mm512_slli_epi16(_mm512_and_si512(hbits, mc), 2),
            );
            let vb2 = _mm512_add_epi8(_mm512_and_si512(bytes1, m4), _mm512_and_si512(hbits, m30));
            let vb3 = _mm512_add_epi8(
                _mm512_and_si512(_mm512_srli_epi16(bytes1, 4), m4),
                _mm512_srli_epi16(_mm512_and_si512(hbits, mc0), 2),
            );
            let vscale = _mm512_cvtepi8_epi32(
                _mm_loadu_si128(t.add(Q6K_OFF_SCALES + 16 * g16) as *const __m128i),
            );
            let ig16 = _mm512_set1_epi32(g16 as i32);
            for r in 0..RB {
                let va = _mm512_loadu_si512(rowp[r].add(s * 292 + 4 + 64 * r_) as *const __m512i);
                let mut vsum = _mm512_setzero_si512();
                vsum = _mm512_dpbusd_epi32(vsum, vb0, _mm512_permutexvar_epi32(idx0, va));
                vsum = _mm512_dpbusd_epi32(vsum, vb1, _mm512_permutexvar_epi32(idx1, va));
                vsum = _mm512_dpbusd_epi32(vsum, vb2, _mm512_permutexvar_epi32(idx2, va));
                vsum = _mm512_dpbusd_epi32(vsum, vb3, _mm512_permutexvar_epi32(idx3, va));
                // - 32*sum(activation) per 16-group
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(ig16, vcomp[r]));
                acci[r] = _mm512_add_epi32(acci[r], _mm512_mullo_epi32(vsum, vscale));
            }
        }
        let vd0 = _mm512_cvtph_ps(_mm256_loadu_si256(t.add(Q6K_OFF_D) as *const __m256i));
        for r in 0..RB {
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acci[r]), _mm512_mul_ps(vd0, vd1[r]), acc[r]);
        }
    }
    for r in 0..RB {
        _mm512_storeu_ps(out.add((m0 + r) * out_stride + group * TILE_N) as *mut f32, acc[r]);
    }
}

/// Dispatch a RB-row block for any supported packed type.
#[inline]
unsafe fn group_dot4_any(
    ty: GgmlType,
    q: &QBatch,
    m0: usize,
    kb: usize,
    packed: &[u8],
    group: usize,
    out: *mut f32,
    out_stride: usize,
) {
    match ty {
        GgmlType::Q4_K => group_dot4_q4_k(q, m0, kb, packed, group, out, out_stride),
        GgmlType::Q6_K => group_dot4_q6_k(q, m0, kb, packed, group, out, out_stride),
        _ => group_dot4_q8_0(q, m0, kb, packed, group, out, out_stride),
    }
}

/// Band of a packed GEMM: groups [g_lo, g_hi) × all M rows, writing
/// out[m * out_stride + g*16 .. +16]. Weights stream once; rows run in
/// RB-row register blocks (tile load + unpack amortized over RB rows,
/// RB independent accumulator chains), tails through the per-row path.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_band(
    packed: &[u8],
    ty: GgmlType,
    q: &QBatch,
    kb: usize,
    g_lo: usize,
    g_hi: usize,
    out: *mut f32,
    out_stride: usize,
    is_caller: bool,
) {
    let t0 = if kstats::enabled() { kstats::tsc() } else { 0 };
    let rows = q.rows();
    for g in g_lo..g_hi {
        let mut m = 0usize;
        while m + RB <= rows {
            // the kernel adds group*TILE_N itself — pass the row-0 base
            group_dot4_any(ty, q, m, kb, packed, g, out, out_stride);
            m += RB;
        }
        while m < rows {
            let obase = out.add(g * TILE_N);
            let r = group_dot_any(ty, q.row(m), kb, packed, g);
            std::ptr::copy_nonoverlapping(r.as_ptr(), obase.add(m * out_stride), TILE_N);
            m += 1;
        }
    }
    if kstats::enabled() {
        let dt = kstats::tsc().wrapping_sub(t0);
        kstats::DOT_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        if is_caller {
            kstats::CALLER_BAND_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Packed GEMM with a pre-quantized activation batch:
/// `out[m * n_out + o] = Σ_i x[m, i] · W[i, o]` for every token row m.
/// `out` is M×n_out row-major (the next op reads token rows contiguously).
pub fn gemm_prepared(
    packed: &PackedTensor,
    q: &QBatch,
    n_threads: usize,
    out: &mut [f32],
) {
    // Phase 7: AMX tile path for M > 1 (Q4_K/Q6_K) — llama.cpp's
    // tinygemm_kernel_amx, bit-identical accumulation order (see amx.rs).
    // M == 1 (decode) and non-AMX machines stay on the VNNI GEMM below.
    #[cfg(target_arch = "x86_64")]
    {
        if q.rows() >= 2
            && matches!(packed.ty, GgmlType::Q4_K | GgmlType::Q6_K)
            && packed.n_out % (2 * 16) == 0
            && crate::amx::amx_ready()
        {
            crate::amx::gemm_amx(packed, q, n_threads, out);
            return;
        }
    }
    let n_out = packed.n_out;
    debug_assert_eq!(out.len(), q.rows() * n_out);
    let kb = match packed.ty {
        GgmlType::Q8_0 | GgmlType::Q5_0 => packed.n_in / TILE_K,
        GgmlType::Q4_K | GgmlType::Q6_K => packed.n_in / QK_K,
        other => unreachable!("packed GEMM does not support {other:?}"),
    };
    if q.rows() == 0 {
        return;
    }
    if kstats::enabled() {
        kstats::MATVECS.fetch_add(q.rows() as u64, std::sync::atomic::Ordering::Relaxed);
    }
    let tw0 = if kstats::enabled() { kstats::tsc() } else { 0 };
    let ng = n_out / TILE_N;
    let n_threads = n_threads.max(1);
    let stride = n_out;
    if n_threads <= 1 || ng < 2 || q.rows() < 4 {
        let ptr = out.as_mut_ptr();
        unsafe { gemm_band(&packed.data, packed.ty, q, kb, 0, ng, ptr, stride, true) };
    } else {
        let g_mid = (ng / 2).clamp(1, ng - 1);
        let ptr = out.as_mut_ptr();
        let caller = gemm_band_job(&packed.data, packed.ty, q, kb, ptr, stride, 0, g_mid, true);
        let worker = gemm_band_job(&packed.data, packed.ty, q, kb, ptr, stride, g_mid, ng, false);
        pool::par_chunks2(caller, worker);
    }
    if kstats::enabled() {
        kstats::MATVEC_WALL_TICKS.fetch_add(kstats::tsc().wrapping_sub(tw0), std::sync::atomic::Ordering::Relaxed);
    }
}

/// Band-job factory for the GEMM bands.
#[allow(clippy::too_many_arguments)]
fn gemm_band_job<'a>(
    packed: &'a [u8],
    ty: GgmlType,
    q: &'a QBatch,
    kb: usize,
    out: *mut f32,
    out_stride: usize,
    g_lo: usize,
    g_hi: usize,
    is_caller: bool,
) -> impl FnMut() + 'a {
    move || unsafe { gemm_band(packed, ty, q, kb, g_lo, g_hi, out, out_stride, is_caller) }
}

/// Fused packed GEMM pair (gate/up) over the concatenated group space —
/// one QBatch, one dispatch, both outputs. `out1`/`out2` are M×n_out1/n_out2
/// row-major. Both tensors must share `ty`/`n_in` (asserted by callers).
#[allow(clippy::too_many_arguments)]
pub fn gemm2_prepared(
    p1: &PackedTensor,
    p2: &PackedTensor,
    q: &QBatch,
    n_threads: usize,
    out1: &mut [f32],
    out2: &mut [f32],
) {
    debug_assert_eq!(p1.ty, p2.ty);
    debug_assert_eq!(p1.n_in, p2.n_in);
    if p1.ty != p2.ty {
        // mixed pair: two separate GEMMs (correct, one extra quantization)
        gemm_prepared(p1, q, n_threads, out1);
        gemm_prepared(p2, q, n_threads, out2);
        return;
    }
    let ng1 = p1.n_out / TILE_N;
    let ng2 = p2.n_out / TILE_N;
    let ng = ng1 + ng2;
    debug_assert_eq!(out1.len(), q.rows() * p1.n_out);
    debug_assert_eq!(out2.len(), q.rows() * p2.n_out);
    if q.rows() == 0 {
        return;
    }
    // Phase 7: AMX tile path per tensor (the shared quantization is kept —
    // one QBatch, two gemm_amx calls; bit-identical to two VNNI GEMMs).
    #[cfg(target_arch = "x86_64")]
    {
        let amx_ok = crate::amx::amx_ready()
            && matches!(p1.ty, GgmlType::Q4_K | GgmlType::Q6_K)
            && p1.ty == p2.ty
            && p1.n_out % (2 * 16) == 0
            && p2.n_out % (2 * 16) == 0
            && q.rows() >= 2;
        if amx_ok {
            crate::amx::gemm_amx(p1, q, n_threads, out1);
            crate::amx::gemm_amx(p2, q, n_threads, out2);
            return;
        }
    }
    if kstats::enabled() {
        kstats::MATVECS.fetch_add(2 * q.rows() as u64, std::sync::atomic::Ordering::Relaxed);
    }
    let tw0 = if kstats::enabled() { kstats::tsc() } else { 0 };
    let kb = match p1.ty {
        GgmlType::Q8_0 | GgmlType::Q5_0 => p1.n_in / TILE_K,
        _ => p1.n_in / QK_K,
    };
    let stride1 = p1.n_out;
    let stride2 = p2.n_out;
    let n_threads = n_threads.max(1);
    if n_threads <= 1 || ng < 2 || q.rows() < 4 {
        let o1 = out1.as_mut_ptr();
        let o2 = out2.as_mut_ptr();
        unsafe { gemm2_band(&p1.data, p1.ty, ng1, &p2.data, q, kb, 0, ng, o1, o2, stride1, stride2, true) };
    } else {
        let g_mid = (ng / 2).clamp(1, ng - 1);
        let o1 = out1.as_mut_ptr();
        let o2 = out2.as_mut_ptr();
        let caller = move || unsafe {
            gemm2_band(&p1.data, p1.ty, ng1, &p2.data, q, kb, 0, g_mid, o1, o2, stride1, stride2, true)
        };
        let worker = move || unsafe {
            gemm2_band(&p1.data, p1.ty, ng1, &p2.data, q, kb, g_mid, ng, o1, o2, stride1, stride2, false)
        };
        pool::par_chunks2(caller, worker);
    }
    if kstats::enabled() {
        kstats::MATVEC_WALL_TICKS.fetch_add(kstats::tsc().wrapping_sub(tw0), std::sync::atomic::Ordering::Relaxed);
    }
}

/// Band of the fused GEMM pair (concatenated group space [0, ng1+ng2));
/// per group the row block runs through the RB-row kernel.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn gemm2_band(
    p1: &[u8],
    ty: GgmlType,
    ng1: usize,
    p2: &[u8],
    q: &QBatch,
    kb: usize,
    g_lo: usize,
    g_hi: usize,
    out1: *mut f32,
    out2: *mut f32,
    stride1: usize,
    stride2: usize,
    is_caller: bool,
) {
    let t0 = if kstats::enabled() { kstats::tsc() } else { 0 };
    let rows = q.rows();
    for g in g_lo..g_hi {
        let (pp, o_ptr, o_stride, gg) = if g < ng1 {
            (p1, out1, stride1, g)
        } else {
            (p2, out2, stride2, g - ng1)
        };
        let obase = o_ptr.add(gg * TILE_N);
        let mut m = 0usize;
        while m + RB <= rows {
            group_dot4_any(ty, q, m, kb, pp, gg, o_ptr, o_stride);
            m += RB;
        }
        while m < rows {
            let r = group_dot_any(ty, q.row(m), kb, pp, gg);
            std::ptr::copy_nonoverlapping(r.as_ptr(), obase.add(m * o_stride), TILE_N);
            m += 1;
        }
    }
    if kstats::enabled() {
        let dt = kstats::tsc().wrapping_sub(t0);
        kstats::DOT_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        if is_caller {
            kstats::CALLER_BAND_TICKS.fetch_add(dt, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// A repacked weight tensor (the engine's analog of the packed buffer of
/// llama.cpp's AMX backend — VNNI layout, no tile registers involved).
#[derive(Clone)]
pub struct PackedTensor {
    pub data: Vec<u8>,
    pub ty: GgmlType,
    pub n_in: usize,
    pub n_out: usize,
}

/// Repack a tensor by type; returns None when the type/shape has no packed
/// path (caller falls back to the AVX2 kernels).
pub fn repack_tensor(
    ty: GgmlType,
    data: &[u8],
    n_in: usize,
    n_out: usize,
) -> Option<PackedTensor> {
    if !vnni_path_available() {
        return None;
    }
    repack_tensor_gated(ty, data, n_in, n_out, true)
}

/// Dispatch-gate-exposed variant: `cpu_ok == false` simulates a machine
/// without AVX512-VNNI (or the LLAMA_RS_VNNI=0 kill switch) and must return
/// None for EVERY type/shape — the caller then takes the AVX2/scalar fallback
/// and no AVX-512 instruction is ever reached. Separated from
/// `repack_tensor` so the dispatch *decision* is unit-testable without
/// physically removing CPU features.
pub fn repack_tensor_gated(
    ty: GgmlType,
    data: &[u8],
    n_in: usize,
    n_out: usize,
    cpu_ok: bool,
) -> Option<PackedTensor> {
    if !cpu_ok {
        return None;
    }
    if n_out % TILE_N != 0 {
        return None;
    }
    let k_mult = match ty {
        GgmlType::Q8_0 | GgmlType::Q5_0 => TILE_K,
        // K-quants repack one 256-value super-block per tile
        GgmlType::Q4_K | GgmlType::Q6_K => QK_K as usize,
        _ => return None,
    };
    if n_in % k_mult != 0 {
        return None;
    }
    let data = match ty {
        GgmlType::Q8_0 => repack_q8_0(data, n_in, n_out),
        GgmlType::Q5_0 => repack_q5_0(data, n_in, n_out),
        GgmlType::Q4_K => repack_q4_k(data, n_in, n_out),
        GgmlType::Q6_K => repack_q6_k(data, n_in, n_out),
        _ => return None,
    };
    Some(PackedTensor { data, ty, n_in, n_out })
}
