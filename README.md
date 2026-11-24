# llama.rs — نقلة Rust أصلية لمحرك llama.cpp (b12502)

محرك استدلال GGUF مكتوب بلغة Rust خالصة، **بلا أي ارتباط وقت تشغيل بـ C/C++**، متوافق المخرجات مع llama.cpp ومثبت ذلك باختبارات تطابق آلية (16/16 ناجحة).

## الحالة (v0.1 — Phase 0–5 مكتملة)

| الميزة | الحالة | الدليل |
|---|---|---|
| قارئ GGUF v3 كامل | ✅ | ميتاداتا + موترات + إزاحات مطابقة |
| فك تكميم Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/F16/F32/BF16 | ✅ | مطابقة مرجع C (كود ggml نفسه): فرق 5e-10 |
| معمارية LFM3 الهجينة (16 shortconv + 8 attention مُبوَّبة + SWA + RoPE مكتسبة + sandwich norms) | ✅ | top-1 greedy متطابق + إجابة عربية مطابقة لـ llama.cpp |
| عائلة llama/qwen2 (GQA + انحيازات q/k/v + SwiGLU) | ✅ | Qwen2.5-0.5B: 14 توكن أولى متطابقة |
| توكِنايزر BPE (llama3-pre + ignore_merges + byte-encode + merges + special fragmentation) | ✅ | 10/10 نصوص عربية/إنجليزية/رموز = 100% ids متطابقة |
| قالب محادثة LFM3 (tags الدقيقة من القالب الخام) + ChatML | ✅ | وضع chat يعمل على النموذجين |
| أخذ عينات greedy/temp/top-k/top-p | ✅ | — |
| CLI: generate/chat/tokenize/bench/debug | ✅ | — |
| SIMD kernels مطابقة لترتيب جمع ggml | ⏳ Phase 6 | ~ ~0.9-3 tok/s (نوى قياسية f32) مقابل ~11 لـ llama.cpp |

## البناء والتشغيل

```bash
cargo build --release

# توليد خام (مطابق llama-completion)
./target/release/llama-rs-cli -m LFM3-1.7B-Q4_K_M.gguf --mode generate -p "The capital of France is" -n 32 -t 2

# محادثة (قالب LFM3 مع thinking tags)
./target/release/llama-rs-cli -m LFM3-1.7B-Q4_K_M.gguf --mode chat -p "ما هي عاصمة المغرب؟ أجب بإيجاز." -n 60 -t 2

# توكنِزة
./target/release/llama-rs-cli -m model.gguf --mode tokenize --no-bos -p "نص"

# قياس
./target/release/llama-rs-cli -m model.gguf --mode bench -p "warm" -n 16 -t 2
```

## اختبارات التطابق (تقارن مع llama.cpp كمرجع ذهبي)

```bash
bash tests/parity/run_parity.sh
# النتيجة الحالية: PASS=16 FAIL=0
#  - توكنِنة ×10 (نصوص عربية/إنجليزية/أرقام/رموز، على النموذجين)
#  - top-1 greedy بعد prefill ×4 (يشمل LFM3 الهجين)
#  - فك تكميم ×2: Q5_0 و Q6_K مقابل كود ggml C مباشرة (مرجع أرضي)
```

## البنية

```
crates/llama-gguf    قارئ GGUF v3 (metadata أنواع كاملة + جدول موترات + بيانات) + اختبارات اصطناعية
crates/llama-quant   بنى كتل ggml + فك التكميم + matvec (خيوط) + اختبارات اصطناعية
crates/llama-core    ops (rmsnorm/rope/swiglu/attention) + model_lfm3 + model_llama + tokenizer + sampler + chat
apps/llama-rs-cli    CLI
tests/parity/        اختبارات تطابق آلية مع llama.cpp + مراجع C أرضية (ref_q5/ref_q6)
```

اختبارات الوحدة الاصطناعية (GGUF مبني يدوياً + كتل تكميم محسوبة يدوياً) تعمل في CI بلا نماذج، واختبارات التطابق الكاملة تعمل محلياً بمقارنة مباشرة مع binaries llama.cpp.

## حدود v0 (بصدق كامل)

- النوى قياسية f32 scalar: أبطأ من llama.cpp CPU kernels بعدة أضعاف (0.9 tok/s مقابل 11 على LFM3 هنا) — المرحلة 6: نوى dot مدمجة بترتيب جمع ggml نفسه (int8 للـ K-quants) لبلوغ تطابق greedy حرفي وتسريع كبير.
- تباعد greedy بعد توكنات قليلة على المرشحين المتعادلين (فجوات logprob < 0.1) سببه ترتيب جمع f32 — يُحل بنوى المرحلة 6.
- لا خادم HTTP/batching/GPU بعد (Phase 7)؛ معماريتان مدعومتان (Phase 8 للتوسيع).
- قالب محادثة subset (لا jinja كامل).

## الترخيص والإسناد

MIT. أعمال مستندة إلى llama.cpp (MIT) — الإسناد موثّق في `LLAMA_RS_ARCHITECTURE.md` §2 بمواقع المصدر الدقيقة لكل خوارزمية منقولة (b12502, commit 398f00d65).
