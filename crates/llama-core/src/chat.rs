//! Chat template subset — port of the template embedded in the GGUF:
//! LFM3: <|im_start|>assistant\n + THINK_OPEN (thinking enabled) / THINK_CLOSE (disabled)
//! history: THINK_OPEN + thinking + THINK_CLOSE, then the content directly
//! Standard ChatML for the qwen/llama family.

pub struct ChatMsg {
    pub role: String,       // "system" | "user" | "assistant"
    pub content: String,
}

pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";

/// LFM3 formatting (byte-exact match of the embedded template — extracted by bytes)
pub fn format_lfm3(msgs: &[ChatMsg], enable_thinking: bool, add_bos: bool) -> String {
    let mut s = String::new();
    if add_bos { s.push('\u{1}'); } // replaced by bos_token_id at tokenization time
    for m in msgs {
        s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
    }
    s.push_str("<|im_start|>assistant\n");
    s.push_str(if enable_thinking { THINK_OPEN } else { "<think></think>" });
    s
}

/// Standard ChatML (qwen2)
pub fn format_chatml(msgs: &[ChatMsg], add_gen: bool) -> String {
    let mut s = String::new();
    for m in msgs {
        s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
    }
    if add_gen { s.push_str("<|im_start|>assistant\n"); }
    s
}
