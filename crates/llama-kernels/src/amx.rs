//! Phase 7: AMX tile GEMM for batched prefill (M > 1) — a line-by-line
//! Rust port of llama.cpp b12502's `tinygemm_kernel_amx<block_q8_K,
//! block_q4_K / block_q6_K>` (ggml/src/ggml-cpu/amx/mmq.cpp), including
//! the Linux XTILEDATA permission request (`arch_prctl(ARCH_REQ_XCOMP_PERM,
//! 18)` — the same syscall llama.cpp's `ggml_amx_init()` issues).
//!
//! Why raw `asm!` on stable Rust: the tile instructions (LDTILECFG /
//! TILELOADD / TILEZERO / TDPBSSD / TILESTORED) exist in `core::arch` only
//! behind the unstable `x86_amx_intrinsics` feature (rust issue #126622).
//! rustc 1.100's LLVM assembler accepts them from `asm!` in **AT&T
//! syntax** (the Intel-operand-order form of TILESTORED is rejected by
//! the assembler; AT&T is byte-identical to what gcc emits for llama.cpp's
//! intrinsics — verified empirically on this CPU).
//!
//! Packed-layout compatibility: the AMX kernels read the SAME packed
//! tiles llama.rs already repacks at load time (Q4_K 2368 B, Q6_K 3360 B).
//! llama.cpp's `pack_B<block_q4_K/q6_K>` produces the identical byte
//! layout — established by the vnni_golden differential harness — so the
//! AMX path needs no extra memory and no second repack.
//!
//! Numerics: per (row, sub-group) the TMUL computes the same integer dot
//! the VNNI kernel computes with `_mm512_dpbusd_epi32` — s8×s8 vs u8×s8
//! is value-safe because Q4_K nibbles are 0..15 (non-negative as s8), and
//! Q6_K's (q−32) offset is baked into the B tile exactly as llama.cpp
//! does. The accumulation order per output element — per-sub-group
//! integer dot, ×scale, i32 accumulate over sub-groups, one f32 FMA per
//! super-block, then the dpwssds min term — is IDENTICAL to the VNNI
//! GEMM's, so every AMX GEMM output is bit-identical to the Phase 6.8
//! GEMM (differential-tested by the CLI's batchcmp mode).
//!
//! Dispatch: `gemm_prepared`/`gemm2_prepared` route M ≥ 2 calls here when
//! `amx_ready()` (CPUID AMX_INT8 + XCR0 tile bits + permission syscall
//! success). M = 1 (decode) stays on the VNNI matvec — exactly the split
//! llama.cpp's CPU backend makes (AMX tiles only pay off for M > 1; the
//! M==1 path runs `tinygemm_kernel_vnni`). `LLAMA_RS_AMX=0` is the
//! same-binary A/B kill switch.

use crate::pool;
use crate::vnni::{PackedTensor, QBatch};
use llama_gguf::GgmlType;

pub const TILE_M: usize = 16;
pub const TILE_N: usize = 16;
pub const TILE_K: usize = 32;
pub const QK_K: usize = 256;

/// Packed-tile offsets (must stay in sync with vnni.rs's repack layout).
const Q4K_TILE_BYTES: usize = 2368;
const Q4K_OFF_SCALES: usize = 2048;
const Q4K_OFF_MINS: usize = 2176;
const Q4K_OFF_D: usize = 2304;
const Q4K_OFF_DMIN: usize = 2336;

const Q6K_TILE_BYTES: usize = 3360;
const Q6K_OFF_QH: usize = 2048;
const Q6K_OFF_SCALES: usize = 3072;
const Q6K_OFF_D: usize = 3328;

/// q8_K block byte layout: d(f32) | qs(256 s8) | bsums(16 i16) = 292 B.
const Q8K_BLOCK_BYTES: usize = 292;

// ---------- permission & readiness ----------

/// Request XTILEDATA permission (Linux) — llama.cpp's ggml_amx_init().
#[cfg(target_os = "linux")]
unsafe fn amx_permission() -> i64 {
    const ARCH_REQ_XCOMP_PERM: usize = 0x1023;
    const XFEATURE_XTILEDATA: usize = 18;
    let ret: i64;
    std::arch::asm!(
        "syscall",
        inlateout("rax") 158i64 => ret, // SYS_arch_prctl
        in("rdi") ARCH_REQ_XCOMP_PERM,
        in("rsi") XFEATURE_XTILEDATA,
        lateout("rcx") _, lateout("r11") _,
        options(nostack)
    );
    ret
}

#[cfg(not(target_os = "linux"))]
unsafe fn amx_permission() -> i64 {
    -1 // AMX permission is Linux-only in this port
}

/// Process-wide AMX readiness (kill switch: `LLAMA_RS_AMX=0`).
pub fn amx_ready() -> bool {
    static R: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        if let Ok(v) = std::env::var("LLAMA_RS_AMX") {
            if v == "0" {
                return false;
            }
        }
        let c = crate::vnni::detect();
        if !(c.amx_int8 && c.avx512_vnni && c.xstate_tile) {
            return false;
        }
        unsafe { amx_permission() == 0 }
    })
}

/// 64-byte tile config, identical to llama.cpp's `ggml_tile_config_init`:
/// TMM0/1 = B tiles (8 rows x 64 B), TMM2/3 = A tiles (16 x 32 B),
/// TMM4-7 = C tiles (16 x 64 B = 16 int32).
fn tile_config() -> [u8; 64] {
    let mut c = [0u8; 64];
    c[0] = 1; // palette id 1
    let set = |c: &mut [u8; 64], i: usize, rows: u8, colsb: u16| {
        c[16 + 2 * i..18 + 2 * i].copy_from_slice(&colsb.to_le_bytes());
        c[48 + i] = rows;
    };
    set(&mut c, 0, 8, 64);
    set(&mut c, 1, 8, 64);
    set(&mut c, 2, 16, 32);
    set(&mut c, 3, 16, 32);
    for i in 4..8 {
        set(&mut c, i, 16, 64);
    }
    c
}

/// Per-thread LDTILECFG (idempotent; the config survives until the thread
/// exits — same thread_local `done` flag llama.cpp uses).
unsafe fn tile_config_init() {
    use std::cell::Cell;
    thread_local! {
        static DONE: Cell<bool> = const { Cell::new(false) };
    }
    if DONE.with(|d| d.get()) {
        return;
    }
    #[repr(align(64))]
    struct Al64<T>(T);
    let cfg = Al64(tile_config());
    std::arch::asm!("ldtilecfg (%rcx)", in("rcx") cfg.0.as_ptr(), options(preserves_flags, att_syntax));
    DONE.with(|d| d.set(true));
}

// ---------- tile instruction wrappers (AT&T syntax, explicit regs) ----------
//
// The row stride of a tile memory operand is carried by the SIB index
// register (rcx = base, rdx = stride); rbx must not appear and base must
// differ from index (rustc's LLVM assembler rules — see module docs).
// The stride is always a runtime VALUE: constant operands with explicit
// registers make rustc inject a syntax-conflicting materialization line
// into the template (the <inline asm>:2:2 "invalid operand" trap).

#[inline(always)]
unsafe fn lo0(p: *const u8, s: usize) {
    std::arch::asm!("tileloadd (%rcx,%rdx,1), %tmm0", in("rcx") p, in("rdx") s, options(preserves_flags, att_syntax));
}
#[inline(always)]
unsafe fn lo1(p: *const u8, s: usize) {
    std::arch::asm!("tileloadd (%rcx,%rdx,1), %tmm1", in("rcx") p, in("rdx") s, options(preserves_flags, att_syntax));
}
#[inline(always)]
unsafe fn lo2(p: *const u8, s: usize) {
    std::arch::asm!("tileloadd (%rcx,%rdx,1), %tmm2", in("rcx") p, in("rdx") s, options(preserves_flags, att_syntax));
}
#[inline(always)]
unsafe fn lo3(p: *const u8, s: usize) {
    std::arch::asm!("tileloadd (%rcx,%rdx,1), %tmm3", in("rcx") p, in("rdx") s, options(preserves_flags, att_syntax));
}
#[inline(always)]
unsafe fn z4() {
    std::arch::asm!("tilezero %tmm4", options(preserves_flags, att_syntax));
}
#[inline(always)]
unsafe fn z5() {
    std::arch::asm!("tilezero %tmm5", options(preserves_flags, att_syntax));
}
#[inline(always)]
unsafe fn z6() {
    std::arch::asm!("tilezero %tmm6", options(preserves_flags, att_syntax));
}
#[inline(always)]
unsafe fn z7() {
    std::arch::asm!("tilezero %tmm7", options(preserves_flags, att_syntax));
}
/// C(m0 rows) x B(cols tile A): tdpbssd tmm4, tmm2, tmm0
#[inline(always)]
unsafe fn d4() {
    std::arch::asm!("tdpbssd %tmm0, %tmm2, %tmm4", options(preserves_flags, att_syntax));
}
/// C(m1 rows) x B(cols tile A): tdpbssd tmm5, tmm3, tmm0
#[inline(always)]
unsafe fn d5() {
    std::arch::asm!("tdpbssd %tmm0, %tmm3, %tmm5", options(preserves_flags, att_syntax));
}
/// C(m0 rows) x B(cols tile B): tdpbssd tmm6, tmm2, tmm1
#[inline(always)]
unsafe fn d6() {
    std::arch::asm!("tdpbssd %tmm1, %tmm2, %tmm6", options(preserves_flags, att_syntax));
}
/// C(m1 rows) x B(cols tile B): tdpbssd tmm7, tmm3, tmm1
#[inline(always)]
unsafe fn d7() {
    std::arch::asm!("tdpbssd %tmm1, %tmm3, %tmm7", options(preserves_flags, att_syntax));
}
#[inline(always)]
unsafe fn st4(p: *mut u8, s: usize) {
    std::arch::asm!("tilestored %tmm4, (%rcx,%rdx,1)", in("rcx") p, in("rdx") s, options(preserves_flags, att_syntax));
}
#[inline(always)]
unsafe fn st5(p: *mut u8, s: usize) {
    std::arch::asm!("tilestored %tmm5, (%rcx,%rdx,1)", in("rcx") p, in("rdx") s, options(preserves_flags, att_syntax));
}
#[inline(always)]
unsafe fn st6(p: *mut u8, s: usize) {
    std::arch::asm!("tilestored %tmm6, (%rcx,%rdx,1)", in("rcx") p, in("rdx") s, options(preserves_flags, att_syntax));
}
#[inline(always)]
unsafe fn st7(p: *mut u8, s: usize) {
    std::arch::asm!("tilestored %tmm7, (%rcx,%rdx,1)", in("rcx") p, in("rdx") s, options(preserves_flags, att_syntax));
}

// ---------- unpack helpers (AVX-512, ported from mmq.cpp) ----------

/// unpack_B for Q4_K: sub-group k (32 values x 16 columns) of one packed
/// tile -> the 8x64 VNNI B-tile buffer (row 2i = values 8c+j, row 2i+1 =
/// values 8c+4+j of chunk c — nibble-split of the packed quants).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn unpack_b_q4k(tile: &mut [u8; 512], packed_tile: &[u8], k: usize) {
    use std::arch::x86_64::*;
    let group = packed_tile.as_ptr().add(k * 256);
    let m4 = _mm512_set1_epi8(0xF);
    for n in 0..8usize {
        if n % 2 == 0 {
            let bytes = _mm512_loadu_si512(group.add(n * 32) as *const __m512i);
            let r0 = _mm512_and_si512(bytes, m4);
            let r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), m4);
            _mm512_storeu_si512(tile.as_mut_ptr().add(n * 64) as *mut __m512i, r0);
            _mm512_storeu_si512(tile.as_mut_ptr().add(n * 64 + 64) as *mut __m512i, r1);
        }
    }
}

/// unpack_B for Q6_K: 16-value group k of one packed tile -> rows 0..3 of
/// the B tile (rows 4..7 are multiplied by A's zero padding — never read).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn unpack_b_q6k(tile: &mut [u8; 512], packed_tile: &[u8], k: usize) {
    use std::arch::x86_64::*;
    // lower nibbles: k-th 128-byte ql chunk; higher 2 bits: k-th 64-byte qh chunk
    let pb = packed_tile.as_ptr().add(k * 128);
    let ph = packed_tile.as_ptr().add(Q6K_OFF_QH + k * 64);
    let off = _mm512_set1_epi8(32);
    let low = _mm512_set1_epi8(0xF);
    let hmask0 = _mm512_set1_epi8(0x03);
    let hmask1 = _mm512_set1_epi8(0x0C);
    let hbits = _mm512_loadu_si512(ph as *const __m512i);
    let bytes = _mm512_loadu_si512(pb as *const __m512i);
    let r0 = _mm512_sub_epi8(
        _mm512_add_epi8(_mm512_and_si512(bytes, low), _mm512_slli_epi16(_mm512_and_si512(hbits, hmask0), 4)),
        off,
    );
    let r1 = _mm512_sub_epi8(
        _mm512_add_epi8(
            _mm512_and_si512(_mm512_srli_epi16(bytes, 4), low),
            _mm512_slli_epi16(_mm512_and_si512(hbits, hmask1), 2),
        ),
        off,
    );
    _mm512_storeu_si512(tile.as_mut_ptr().add(0) as *mut __m512i, r0);
    _mm512_storeu_si512(tile.as_mut_ptr().add(64) as *mut __m512i, r1);
    let hmask0 = _mm512_set1_epi8(0x30);
    let hmask1 = _mm512_set1_epi8(0xC0u8 as i8);
    let bytes = _mm512_loadu_si512(pb.add(64) as *const __m512i);
    let r2 = _mm512_sub_epi8(
        _mm512_add_epi8(_mm512_and_si512(bytes, low), _mm512_and_si512(hbits, hmask0)),
        off,
    );
    let r3 = _mm512_sub_epi8(
        _mm512_add_epi8(
            _mm512_and_si512(_mm512_srli_epi16(bytes, 4), low),
            _mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), 2),
        ),
        off,
    );
    _mm512_storeu_si512(tile.as_mut_ptr().add(128) as *mut __m512i, r2);
    _mm512_storeu_si512(tile.as_mut_ptr().add(192) as *mut __m512i, r3);
}

/// unpack_A: `nr` activation rows' sub-group k (Q4_K: 32 bytes; Q6_K: 16
/// bytes zero-padded to 32) into the 16x32 A-tile buffer.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn unpack_a(tile: &mut [u8; 512], q: &QBatch, m_glob: usize, i: usize, k: usize, nr: usize, q6: bool) {
    use std::arch::x86_64::*;
    let n = if q6 { 16 } else { 32 };
    let src_step = Q8K_BLOCK_BYTES;
    for m in 0..nr {
        let row = q.row(m_glob + m);
        let blk = row.as_ptr().add(i * src_step + 4 + k * n);
        let dst = tile.as_mut_ptr().add(m * TILE_K);
        if q6 {
            let v = _mm_loadu_si128(blk as *const __m128i);
            let z = _mm_setzero_si128();
            let r = _mm256_inserti128_si256(_mm256_castsi128_si256(v), z, 1);
            _mm256_storeu_si256(dst as *mut __m256i, r);
        } else {
            let v = _mm256_loadu_si256(blk as *const __m256i);
            _mm256_storeu_si256(dst as *mut __m256i, v);
        }
    }
}

// ---------- scale_C / acc_C (AVX-512 epilogues, ported from mmq.cpp) ----------

/// sumi[m][n] += tile[m][n] * scales[k][n] (int32; 16 lanes per row).
/// `first` (k == 0) REPLACES the sumi content — llama.cpp's is_acc=false —
/// so no zeroing pass over the buffers exists.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn scale_c(sumi: &mut [i32; 256], tile: &[i32; 256], scales: *const u8, k: usize, nr: usize, first: bool) {
    use std::arch::x86_64::*;
    let vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128(scales.add(k * TILE_N) as *const __m128i));
    for m in 0..nr {
        let v = _mm512_loadu_si512(tile.as_ptr().add(m * TILE_N) as *const __m512i);
        let prod = _mm512_mullo_epi32(v, vscale);
        let r = if first {
            prod
        } else {
            let s = _mm512_loadu_si512(sumi.as_ptr().add(m * TILE_N) as *const __m512i);
            _mm512_add_epi32(s, prod)
        };
        _mm512_storeu_si512(sumi.as_mut_ptr().add(m * TILE_N) as *mut __m512i, r);
    }
}

/// acc_C for Q4_K: apply d/dmin + the dpwssds min term of super-block i
/// for `nr` rows starting at global row `m_glob`; accumulate into C rows
/// (is_acc = false on i == 0 initializes the output).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,f16c")]
#[inline]
unsafe fn acc_c_q4k(
    out: *mut f32,
    ldc: usize,
    col_base: usize,
    sumi: &[i32; 256],
    q: &QBatch,
    m_glob: usize,
    i: usize,
    packed_tile: &[u8],
    nr: usize,
    is_acc: bool,
) {
    use std::arch::x86_64::*;
    let vd0 = _mm512_cvtph_ps(_mm256_loadu_si256(packed_tile.as_ptr().add(Q4K_OFF_D) as *const __m256i));
    let vdmin = _mm512_cvtph_ps(_mm256_loadu_si256(packed_tile.as_ptr().add(Q4K_OFF_DMIN) as *const __m256i));
    for m in 0..nr {
        let blk = q.row(m_glob + m).as_ptr().add(i * Q8K_BLOCK_BYTES);
        let d1 = f32::from_le_bytes(std::ptr::read_unaligned(blk as *const [u8; 4]));
        let vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
        let vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
        let vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(sumi.as_ptr().add(m * TILE_N) as *const __m512i));
        let mut vsum = if is_acc {
            _mm512_loadu_ps(out.add((m_glob + m) * ldc + col_base))
        } else {
            _mm512_setzero_ps()
        };
        // the min term: dpwssds over (activation bsums, weight mins)
        let bsums = _mm256_loadu_si256(blk.add(260) as *const __m256i);
        let q8s = _mm_hadd_epi16(
            _mm256_castsi256_si128(bsums),
            _mm256_extracti128_si256(bsums, 1),
        );
        let va_bsum = _mm512_castsi128_si512(q8s);
        let mut acc_m = _mm512_setzero_si512();
        for k in 0..4usize {
            let vbs = _mm512_permutexvar_epi32(_mm512_set1_epi32(k as i32), va_bsum);
            let vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256(packed_tile.as_ptr().add(Q4K_OFF_MINS + 32 * k) as *const __m256i));
            acc_m = _mm512_dpwssds_epi32(acc_m, vbs, vb);
        }
        vsum = _mm512_fmadd_ps(vtile, vd, vsum);
        vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
        _mm512_storeu_ps(out.add((m_glob + m) * ldc + col_base), vsum);
    }
}

/// acc_C for Q6_K: d-only epilogue (the (q−32) compensation lives in the
/// B tile; no min term).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,f16c")]
#[inline]
unsafe fn acc_c_q6k(
    out: *mut f32,
    ldc: usize,
    col_base: usize,
    sumi: &[i32; 256],
    q: &QBatch,
    m_glob: usize,
    i: usize,
    packed_tile: &[u8],
    nr: usize,
    is_acc: bool,
) {
    use std::arch::x86_64::*;
    let vd0 = _mm512_cvtph_ps(_mm256_loadu_si256(packed_tile.as_ptr().add(Q6K_OFF_D) as *const __m256i));
    for m in 0..nr {
        let blk = q.row(m_glob + m).as_ptr().add(i * Q8K_BLOCK_BYTES);
        let d1 = f32::from_le_bytes(std::ptr::read_unaligned(blk as *const [u8; 4]));
        let vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
        let vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(sumi.as_ptr().add(m * TILE_N) as *const __m512i));
        let vsum = if is_acc {
            _mm512_loadu_ps(out.add((m_glob + m) * ldc + col_base))
        } else {
            _mm512_setzero_ps()
        };
        let vsum = _mm512_fmadd_ps(vtile, vd, vsum);
        _mm512_storeu_ps(out.add((m_glob + m) * ldc + col_base), vsum);
    }
}

// ---------- the block kernel (llama.cpp's QKK tinygemm_kernel_amx) ----------

#[repr(align(64))]
struct Al64<T>(T);

/// Per-thread kernel scratch: tile buffers live for the whole band (the
/// 64-byte alignment wrapper zeroes them ONCE per band — the k == 0 pass
/// of scale_c replaces Sumi contents, and tilestored overwrites the C
/// dumps, so no per-block initialization remains).
struct AmxBufs {
    t0: Al64<[u8; 512]>,   // B tile VNNI buffer (column group 2n)
    t1: Al64<[u8; 512]>,   // B tile VNNI buffer (column group 2n+1)
    t23: Al64<[u8; 512]>,  // A tile buffer (16 rows x 32 B)
    d4: Al64<[u8; 1024]>,  // C tile dump (16 rows x 16 i32)
    d5: Al64<[u8; 1024]>,
    d6: Al64<[u8; 1024]>,
    d7: Al64<[u8; 1024]>,
    s4: Al64<[i32; 256]>,  // per-super-block int32 sums (16 rows x 16)
    s5: Al64<[i32; 256]>,
    s6: Al64<[i32; 256]>,
    s7: Al64<[i32; 256]>,
}

impl AmxBufs {
    fn new() -> AmxBufs {
        AmxBufs {
            t0: Al64([0u8; 512]),
            t1: Al64([0u8; 512]),
            t23: Al64([0u8; 512]),
            d4: Al64([0u8; 1024]),
            d5: Al64([0u8; 1024]),
            d6: Al64([0u8; 1024]),
            d7: Al64([0u8; 1024]),
            s4: Al64([0i32; 256]),
            s5: Al64([0i32; 256]),
            s6: Al64([0i32; 256]),
            s7: Al64([0i32; 256]),
        }
    }
}

/// One 32-row x 32-col block: for every K super-block i, every sub-group k
// ---------- driver ----------

/// AMX GEMM entry: `out[m*n_out + c] = <x row m, W col c>` for M >= 2.
/// Dispatched from vnni::gemm_prepared when amx_ready(); the packed
/// layout, QBatch and out contract are unchanged (bit-identical results).
///
/// Phase 7 experiments (both measured, both reverted — documented so the
/// next attempt starts from evidence):
///  * nb-major block order (unpack B once per (nb,i,k), stream all row
///    blocks): −36% — the per-(i,k) working set grows to 64KB+ of Sumi +
///    A rows and thrashes L1; the block-major order keeps one row-block's
///    8KB hot across the k-loop.
///  * per-k dump buffers with a post-k scale pass: −20% — the tile-op
///    dependency chains need the interleaved AVX scale work to overlap.
pub fn gemm_amx(packed: &PackedTensor, q: &QBatch, n_threads: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), q.rows() * packed.n_out);
    let ty = packed.ty;
    debug_assert!(matches!(ty, GgmlType::Q4_K | GgmlType::Q6_K), "amx GEMM supports Q4_K/Q6_K");
    if !matches!(ty, GgmlType::Q4_K | GgmlType::Q6_K) {
        // unreachable via the dispatch gates; keep a safe fallback contract
        crate::vnni::gemm_prepared(packed, q, n_threads, out);
        return;
    }
    let tile_bytes = if ty == GgmlType::Q4_K { Q4K_TILE_BYTES } else { Q6K_TILE_BYTES };
    let m_rows = q.rows();
    let n_out = packed.n_out;
    let kb = packed.n_in / QK_K;
    let ldc = n_out;
    let out_len = out.len();
    let out_ptr = out.as_mut_ptr();

    // 32-row x 32-col blocks; tails (m_size < 32) handled inside the kernel
    let mb = m_rows.div_ceil(2 * TILE_M);
    let nb = n_out / (2 * TILE_N);
    debug_assert_eq!(n_out % (2 * TILE_N), 0, "AMX GEMM requires n_out % 32 == 0");
    let total = mb * nb;

    let run = |b_lo: usize, b_hi: usize| unsafe {
        tile_config_init();
        let mut bufs = AmxBufs::new();
        for blk in b_lo..b_hi {
            let (mbi, nbi) = (blk / nb, blk % nb);
            let m_start = mbi * 2 * TILE_M;
            let m_size = m_rows.saturating_sub(m_start).min(2 * TILE_M);
            let out_slice = std::slice::from_raw_parts_mut(out_ptr, out_len);
            amx_block(ty, &packed.data, q, out_slice, ldc, kb, tile_bytes, m_start, m_size, nbi, &mut bufs);
        }
    };

    let n_threads = n_threads.max(1);
    if n_threads > 1 && total >= 2 {
        let mid = total / 2;
        let caller = move || run(0, mid);
        let worker = move || run(mid, total);
        pool::par_chunks2(caller, worker);
    } else {
        run(0, total);
    }
}

/// One 32-row x 32-col block: for every K super-block i, every sub-group k
/// tile-mul the 16(+16) activation rows against the two 16-column B
/// tiles, apply the integer scales, then the f32 epilogue.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,f16c")]
unsafe fn amx_block(
    ty: GgmlType,
    packed: &[u8],
    q: &QBatch,
    out: &mut [f32],
    ldc: usize,
    kb: usize,
    tile_bytes: usize,
    m_glob: usize, // global first row
    m_size: usize, // rows in this block (1..=32)
    nb: usize,     // 32-column block index
    b: &mut AmxBufs,
) {
    let q6 = ty == GgmlType::Q6_K;
    let n_sub = if q6 { 16 } else { 8 };
    let m0 = m_size.min(TILE_M);
    let m1 = m_size.saturating_sub(TILE_M);
    let scale_off = if q6 { Q6K_OFF_SCALES } else { Q4K_OFF_SCALES };

    // packed tiles of this nb: column groups nb*2 and nb*2+1
    let b0_base = packed.as_ptr().add((nb * 2) * kb * tile_bytes);
    let b1_base = packed.as_ptr().add((nb * 2 + 1) * kb * tile_bytes);
    let (c4p, c5p, c6p, c7p) = (
        b.d4.0.as_mut_ptr(),
        b.d5.0.as_mut_ptr(),
        b.d6.0.as_mut_ptr(),
        b.d7.0.as_mut_ptr(),
    );

    for i in 0..kb {
        for k in 0..n_sub {
            let t0p = b0_base.add(i * tile_bytes);
            let t1p = b1_base.add(i * tile_bytes);
            if q6 {
                unpack_b_q6k(&mut b.t0.0, std::slice::from_raw_parts(t0p, tile_bytes), k);
                unpack_b_q6k(&mut b.t1.0, std::slice::from_raw_parts(t1p, tile_bytes), k);
            } else {
                unpack_b_q4k(&mut b.t0.0, std::slice::from_raw_parts(t0p, tile_bytes), k);
                unpack_b_q4k(&mut b.t1.0, std::slice::from_raw_parts(t1p, tile_bytes), k);
            }
            lo0(b.t0.0.as_ptr(), 64);
            lo1(b.t1.0.as_ptr(), 64);

            unpack_a(&mut b.t23.0, q, m_glob, i, k, m0, q6);
            lo2(b.t23.0.as_ptr(), 32);

            // the AVX scale work of iteration k runs interleaved with the
            // TMUL pipeline (see the driver doc: separating them measured
            // 20% slower — the tile dependency chains need the overlap)
            z4();
            d4();
            st4(c4p, 64);
            z6();
            d6();
            st6(c6p, 64);
            scale_c(&mut b.s4.0, &*(c4p as *const [i32; 256]), t0p.add(scale_off), k, m0, k == 0);
            scale_c(&mut b.s6.0, &*(c6p as *const [i32; 256]), t1p.add(scale_off), k, m0, k == 0);

            if m1 != 0 {
                unpack_a(&mut b.t23.0, q, m_glob + TILE_M, i, k, m1, q6);
                lo3(b.t23.0.as_ptr(), 32);
                z5();
                d5();
                st5(c5p, 64);
                z7();
                d7();
                st7(c7p, 64);
                scale_c(&mut b.s5.0, &*(c5p as *const [i32; 256]), t0p.add(scale_off), k, m1, k == 0);
                scale_c(&mut b.s7.0, &*(c7p as *const [i32; 256]), t1p.add(scale_off), k, m1, k == 0);
            }
        }
        // f32 epilogue for super-block i
        let out_ptr = out.as_mut_ptr();
        let col = nb * 2 * TILE_N;
        let is_acc = i > 0;
        let tile0 = std::slice::from_raw_parts(b0_base.add(i * tile_bytes), tile_bytes);
        let tile1 = std::slice::from_raw_parts(b1_base.add(i * tile_bytes), tile_bytes);
        if q6 {
            acc_c_q6k(out_ptr, ldc, col, &b.s4.0, q, m_glob, i, tile0, m0, is_acc);
            acc_c_q6k(out_ptr, ldc, col + TILE_N, &b.s6.0, q, m_glob, i, tile1, m0, is_acc);
            if m1 != 0 {
                acc_c_q6k(out_ptr, ldc, col, &b.s5.0, q, m_glob + TILE_M, i, tile0, m1, is_acc);
                acc_c_q6k(out_ptr, ldc, col + TILE_N, &b.s7.0, q, m_glob + TILE_M, i, tile1, m1, is_acc);
            }
        } else {
            acc_c_q4k(out_ptr, ldc, col, &b.s4.0, q, m_glob, i, tile0, m0, is_acc);
            acc_c_q4k(out_ptr, ldc, col + TILE_N, &b.s6.0, q, m_glob, i, tile1, m0, is_acc);
            if m1 != 0 {
                acc_c_q4k(out_ptr, ldc, col, &b.s5.0, q, m_glob + TILE_M, i, tile0, m1, is_acc);
                acc_c_q4k(out_ptr, ldc, col + TILE_N, &b.s7.0, q, m_glob + TILE_M, i, tile1, m1, is_acc);
            }
        }
    }
}
