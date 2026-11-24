//! قوالب محادثة subset — نقل القالب المضمّن في GGUF:
//! LFM3: <|im_start|>assistant\n +  (تفكير مفعّل) /   (معطّل)
//! التاريخ:  + thinking +  ثم المحتوى مباشرة
//! ChatML القياسي لعائلة qwen/llama.

pub struct ChatMsg {
    pub role: String,       // "system" | "user" | "assistant"
    pub content: String,
}

pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";

/// تنسيق LFM3 (مطابق حرفياً للقالب المضمّن — مستخرج بالبايتات)
pub fn format_lfm3(msgs: &[ChatMsg], enable_thinking: bool, add_bos: bool) -> String {
    let mut s = String::new();
    if add_bos { s.push('\u{1}'); } // يُستبدل بـ bos_token_id عند التوكِنة
    for m in msgs {
        s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
    }
    s.push_str("<|im_start|>assistant\n");
    s.push_str(if enable_thinking { THINK_OPEN } else { "<think></think>" });
    s
}

/// ChatML القياسي (qwen2)
pub fn format_chatml(msgs: &[ChatMsg], add_gen: bool) -> String {
    let mut s = String::new();
    for m in msgs {
        s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
    }
    if add_gen { s.push_str("<|im_start|>assistant\n"); }
    s
}
