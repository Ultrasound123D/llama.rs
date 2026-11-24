# LLAMA_RS_ARCHITECTURE — وثيقة معمارية لمحرك llama.rs (نقلة Rust أصلية لمحرك llama.cpp)

**الإصدار:** 0.1 (Phase 0–5) · **التاريخ:** 2026-11-24
**المرجع المصدري:** llama.cpp b12502 (commit 398f00d65) — مستنسخ ومفحوص محلياً بالكامل
**الهدف:** محرك استدلال GGUF أصيل بلغة Rust، متوافق المخرجات مع llama.cpp (parity مثبت باختبارات)، دون أي ارتباط وقت تشغيل بـ C/C++.

---

## 1) خريطة المصدر ↔ Rust (المرحلة 0: Source Audit — منجزة)

| مكوّن llama.cpp | الملف المصدري | نظير llama.rs (v0 الحالي) | نظير الخطة النهائية |
|---|---|---|---|
| GGUF reader | `ggml/src/gguf.cpp`, `ggml-common.h` | `crates/llama-gguf` | كما هو |
| Tensor types/quant blocks | `ggml-common.h` (block_q*_K) | `crates/llama-quant` | كما هو |
| العمليات الحسابية | `ggml/src/ggml-cpu/ops.cpp` | `crates/llama-core::ops` | `crates/llama-kernels` + SIMD |
| Backend/جدولة | `ggml/src/ggml-backend.cpp` | — (غير مطلوب لCPU في v0) | `crates/llama-backend` |
| Compute graph | `ggml/src/ggml.c` (بناء حبيبي) | **منهج مباشر بدون graph** في v0 (تنفيذ مستقيم لكل نموذج) | `crates/llama-graph` اختياري |
| معمارية النموذج | `src/llama-arch.cpp` | `llama-core::arch` (جدول موترات لكل معمارية) | `crates/llama-models` |
| بناء حسابات النموذج | `src/models/lfm3.cpp`, `models/lfm-base.cpp`, `models/llama.cpp`… | `llama-core::model_lfm3`, `model_llama` (تشمل qwen2) | توسيع معمارية/معمارية |
| hparams/التحميل | `src/llama-model.cpp` + `llama-model-loader.cpp` | `llama-core::gguf_model` | `crates/llama-runtime` |
| KV/الذاكرة الهجينة | `src/llama-memory-hybrid-iswa.cpp` | KV f32 + conv-state داخل كل نموذج (متطابق دلالياً) | `crates/llama-memory` |
| Tokenizer BPE | `src/llama-vocab.cpp` + `src/unicode.cpp` | `llama-core::tokenizer` (نقل حرفي لخوارزمياتها) | `crates/llama-tokenizer` |
| Sampling | `src/llama-sampling.cpp` | `llama-core::sampler` | `crates/llama-sampler` |
| Chat templates | `common/chat.cpp` (jinja) | `llama-core::chat` (ChatML/LFM3 subset) | jinja مصغّر |
| Server/OpenAI API | `tools/server/` | — (Phase 7) | `crates/llama-server` |

**قرارات معمارية مفصلية:**
1. **إعادة بناء العقود لا ترجمة الأسطر**: نعيد تنفيذ *سلوك* العمليات (نفس صيغ ggml نفسها: rope_yarn/rotate_pairs، ssm_conv، get_scale_min_k4...) بلغة Rust آمنة، بدل محاكاة نظام ggml الديناميكي (context/tensor graph) الذي يخدم البنية الخلفية أكثر من دقة الحساب.
2. **الأوزان تبقى مكمّمة في الذاكرة** (كما llama.cpp) والفك يتم داخل نواة matvec مباشرة من الكتل — لا تفعيم كامل (قد يستهلك 6.8GB) ولا f16 كامل (3.4GB). هذا هو وضع "الصحة أولاً"؛ نوى SIMD (Phase 6) تُضاف لاحقاً في نفس نقطة الاستدعاء.
3. **parity قبل السرعة**: مخرجات التطابق مع llama.cpp هي بوابة كل مرحلة (انظر §4).

## 2) الحسابات المنقولة حرفياً من المصدر (المرجع القانوني للكود)

مواقع المصدر المقروءة والمطابقة لها (كلها من b12502):

| العملية | المصدر | ملاحظات الدقة |
|---|---|---|
| فك Q4_K/Q5_K/Q6_K | `ggml-quants.c:1539/1741/1949` + `get_scale_min_k4:889` | نقل حرفي للبايتات |
| Q8_0/Q4_0/Q5_0 | `ggml-quants.c:462…556` | نقل حرفي |
| f16→f32 | `ggml-common.h` + cpu convert | IEEE half |
| RoPE NEOX مع عوامل مكتسبة | `ggml-cpu/ops.cpp:6148-6252` (rope cache + rotate_pairs) | angle_j = pos · base^(−2j/d) / ff_j؛ أزواج (j, j+d/2) |
| shortconv | `models/lfm-base.cpp:7-93` + `ops.cpp:9859 ssm_conv` | نواة [7,2048]؛ حالة 6 صفوف؛ نافذة [state, x]؛ y=c·conv(b·x) |
| LFM3 block fusion | `models/lfm3.cpp:98-208` | ترتيب sandwich norms + residual من prev غير المُعمّى |
| gated attention | `models/lfm3.cpp:101-136` | q_norm/k_norm لكل رأس + بوابة sigmoid لكل رأس |
| llama3 pre-tokenizer | `unicode.cpp:355-493` (آلة حالات) + `llama-vocab.cpp:293-299` | نقل حرفي |
| BPE merges + ignore_merges | `llama-vocab.cpp:660-820` | طابور أولوية بالرتبة |
| GPT-2 byte encoding | `unicode.cpp:157-234` | جدول 0x21-0x7E،0xA1-0xAC،0xAE-0xFF |
| add_bos من GGUF يعلو على pre | `llama-vocab.cpp:3005-3017` | LFM3: add_bos=false |
| تصنيف الحروف | `unicode.h` flags (\p{L}=\{Lu,Ll,Lt,Lm,Lo\}، \p{N}=\{Nd,Nl,No\}، \s=White_Space) | عبر crate `unicode-general-category` |

## 3) معمارية llama.rs (v0)

```
llama.rs/
├── Cargo.toml                    # workspace
├── crates/
│   ├── llama-gguf/               # قارئ GGUF v3: header, kv-metadata (كل الأنواع), tensor index, data
│   ├── llama-quant/              # block_q*_K + dequant + matvec (نواة f32 accumulation)
│   └── llama-core/
│       ├── tensor.rs             # Tensor {dims, type, data-slice}
│       ├── ops.rs                 # rmsnorm, silu, softmax, matvec, rope
│       ├── model_lfm3.rs          # 24 طبقة: 16 shortconv + 8 attention (4 full+4 SWA window 256)
│       ├── model_llama.rs         # عائلة llama/qwen2 القياسية (GQA + SwiGLU)
│       ├── tokenizer.rs           # BPE كامل (llama3-pre state machine, merges, special fragmentation)
│       ├── sampler.rs            # greedy / temperature / top-k / top-p
│       └── chat.rs                # قوالب ChatML + LFM3 (thinking tags)
└── apps/llama-rs-cli/            # generate / chat / tokenize / bench
```

**نموذج بيانات Tensor**: مصفوفة أوزان بترتيب ggml: `W[in, out]` في الذاكرة تسلسلياً بالصفوف (row-major على dim0)، وmatvec يحسب `out[o] = Σ_i W[i,o]·x[i]` — نفس دلالات `ggml_mul_mat` في llama.cpp للملفات المفحوصة (أوزان [n_embd, out] كما في GGUF أعلاه: مثلاً `ffn_gate [2048, 7680]`).

## 4) بوابات الخروج لكل مرحلة (مع حالة التنفيذ الحالية)

| المرحلة | معيار الخروج | الحالة |
|---|---|---|
| P0 التدقيق | خريطة مصدر↔crate + نقل الحسابات الموثّق أعلاه | **منجزة** |
| P1 قارئ GGUF | قراءة LFM3 وQwen GGUFs: metadata + كل الموترات وأبعادها مطابقة للكشف بـ `gguf_explore.py` | **منجزة** (تحقق آلي: التوكنِزة تعمل على النموذجين) |
| P2 تكميم | dequant كتل عينات من ملفات حقيقية: تطابق قيم f32 مع مرجع نفس الخوارزمية + اختبار ذهبي end-to-end (logits parity يغطي) | **منجزة** (Q5_0+Q6_K vs مرجع C بكود ggml: فرق ≤5e-10) |
| P3 التوليد | تشغيل نموذج من البداية للنهاية على CPU بلا أي مكتبة C++ | **منجزة** (LFM3 + Qwen2.5) |
| P4 التوكِنايزر | توكِنة نصوص عربية/إنجليزية/رموز: تطابق ids مع llama.cpp `/tokenize` 100% | **منجزة: 10/10 نصوص = 100% ids** |
| P5 النموذج الأول | Qwen2.5-0.5B: نص greedy متطابق مع `llama-completion` (نفس seed/temp=0) | **منجزة**: 14 توكن الأولى متطابقة حرفياً؛ ثم تعادل رقمي شبه تام (فجوات <0.1 logprob) يقلب ترتيب المرشحين — يحتاج نوى ggml (P6) |
| P5+ النموذج المرجعي | LFM3-1.7B: نفس بوابة Qwen + بنية هجينة (conv+SWA+بوابة رأس+rope مكتسبة) | **منجزة**: top-1 greedy متطابق ×2 prompts؛ إجابة عربية مطابقة حرفياً لإجابة llama.cpp |
| P6 CPU SIMD | benchmark مقابل llama-bench على نفس الجهاز | **مخططة** (الآن 0.9 tok/s قياسي مقابل 11 لـ llama.cpp) |
| P7 خادم/توازٍ | OpenAI API + streaming + batching | مخططة |
| P8 توسيع معماريات | llama3/gemma/qwen3… | مخططة |

## 5) خطة اختبار التطابق (tests/parity)

1. **وحدات اصطناعية (تعمل في CI بلا نماذج)**: `crates/llama-gguf/tests/synthetic.rs` يبني ملف GGUF v3 صغيراً يدوياً ويتحقق من كل أنواع KV وأبعاد التوترات والمحاذاة؛ `crates/llama-quant/tests/dequant_synth.rs` يفك كتل Q4_0/Q8_0/F16 اصطناعية بقيم محسوبة يدوياً.
2. **tokenizer-ids**: نفس قائمة نصوص على llama.cpp server `/tokenize` وعلى `llama-rs-cli tokenize` → مقارنة تتابع ids (يجب أن تكون متطابقة 100%).
3. **greedy-generation**: `llama-completion -m model -p prompt -n N` (بدون sampling عشوائي) مقابل `llama-rs-cli generate` → مقارنة النص/ids. تطابق greedy يعني تطابقاً في: embedding + كل العمليات + KV + sampler argmax.
4. **فك التكميم**: `ref_q5.c`/`ref_q6.c` (منقولان من ggml-quants.c نفسه) يقارنان قيم فك التكميم من ملف حقيقي عموداً بعمود.
5. **bifurcation search** عند أي اختلاف: طباعة logits للتوكن الأول ومقارنتها (أداة مستقبلية).

## 6) حدود v0 (ما لم يُنقل بعد — بصدق كامل)

- لا SIMD kernels محسّنة (نواة f32 accumulation فقط) → أبطأ من llama.cpp CPU kernels (المتوقع عدة أضعاف).
- لا خادم HTTP، لا batching متوازٍ، لا speculative decoding.
- لا GPU backends.
- قالب محادثة subset (لا jinja كامل): ChatML + LFM3 thinking tags، بلا tools/vision.
- معماريات مدعومة: lfm3 + عائلة llama (llama1/2/3, qwen2, qwen2.5 عبر نموذج llama-family). البقية P8.
- ignore التراخيص: MIT (llama.cpp) — إشعارات حقوق محفوظة في NOTICE بالمستودع، والأجزاء المنقولة منطقياً موثّقة بمصادرها في §2.
