//! BPE tokenizer — port of llama-vocab.cpp (llm_tokenizer_bpe) + unicode.cpp:
//! - pre "lfm2"/"llama3" → LLAMA3 pre: the state machine of unicode.cpp:355 (verbatim port)
//! - GPT-2 byte encoding (unicode.cpp:157)
//! - BPE merges by priority rank (llama-vocab.cpp:752)
//! - ignore_merges: a pre-token word found as a whole token → direct id (llama-vocab.cpp:724)
//! - splitting around special tokens (fragmentation), then BPE for the rest
//! - detokenize with byte decoding
use llama_gguf::{GgufFile, Value};
use std::collections::HashMap;
use unicode_general_category::{GeneralCategory, get_general_category};

/// \p{L} = \{Lu,Ll,Lt,Lm,Lo\}, \p{N} = \{Nd,Nl,No\}, \s = White_Space (unicode.h)
fn is_letter(c: char) -> bool {
    matches!(get_general_category(c),
        GeneralCategory::UppercaseLetter | GeneralCategory::LowercaseLetter
        | GeneralCategory::TitlecaseLetter | GeneralCategory::ModifierLetter
        | GeneralCategory::OtherLetter)
}
fn is_number(c: char) -> bool {
    matches!(get_general_category(c),
        GeneralCategory::DecimalNumber | GeneralCategory::LetterNumber | GeneralCategory::OtherNumber)
}
fn is_ws(c: char) -> bool { c.is_whitespace() }
/// practically, every assigned character has flags in the llama.cpp tables (flags.as_uint() != 0)
fn has_flags(c: char) -> bool {
    !matches!(get_general_category(c), GeneralCategory::Unassigned)
}
fn tolower(c: char) -> char { c.to_lowercase().next().unwrap_or(c) }

/// GPT-2 byte-encoding table (unicode.cpp:157)
fn byte_encode_table() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut n = 0u32;
    for ch in 0u16..256 {
        let ok = (0x21..=0x7E).contains(&ch) || (0xA1..=0xAC).contains(&ch) || (0xAE..=0xFF).contains(&ch);
        let cp = if ok { ch as u32 } else { 256 + { n += 1; n - 1 } };
        map[ch as usize] = char::from_u32(cp).unwrap();
    }
    map
}

pub struct BpeTokenizer {
    pub tokens: Vec<String>,
    pub token_type: Vec<u8>, // 1 normal, 2 unknown, 3 control, 4 user_defined, 5 unused, 6 byte
    pub token_to_id: HashMap<String, u32>,
    /// (left_id, right_id) → (rank, merged_id)
    pub merges: HashMap<(u32, u32), (u32, u32)>,
    byte_map: [char; 256],
    byte_unmap: HashMap<char, u8>,
    pub bos_id: u32, pub eos_id: u32,
    pub add_bos: bool, pub add_eos: bool,
    /// special = type ≠ normal (same logic as llama.cpp)
    pub special: Vec<(String, u32)>,
}

impl BpeTokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> BpeTokenizer {
        let tokens_v: Vec<String> = gguf.metadata.get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_arr())
            .map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect())
            .expect("tokens");
        let types_v: Vec<u8> = gguf.metadata.get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_arr())
            .map(|a| a.iter().map(|v| v.as_u32().unwrap() as u8).collect())
            .expect("token_type");
        let mut token_to_id = HashMap::new();
        for (i, t) in tokens_v.iter().enumerate() {
            token_to_id.entry(t.clone()).or_insert(i as u32);
        }
        let merges_v: Vec<String> = gguf.metadata.get("tokenizer.ggml.merges")
            .and_then(|v| v.as_arr())
            .map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect())
            .unwrap_or_default();
        let mut merges = HashMap::new();
        for (rank, m) in merges_v.iter().enumerate() {
            let mut parts = m.splitn(2, ' ');
            let left = parts.next().unwrap_or("");
            let right = match parts.next() { Some(r) => r, None => continue };
            if let (Some(&lid), Some(&rid)) = (token_to_id.get(left), token_to_id.get(right)) {
                if let Some(&mid) = token_to_id.get(&format!("{left}{right}")) {
                    merges.insert((lid, rid), (rank as u32, mid));
                }
            }
        }
        let byte_map = byte_encode_table();
        let mut byte_unmap = HashMap::new();
        for (b, ch) in byte_map.iter().enumerate() {
            byte_unmap.insert(*ch, b as u8);
        }
        let g = |k: &str| gguf.metadata.get(&format!("tokenizer.ggml.{k}")).and_then(|v| v.as_u32());
        let add_bos = gguf.metadata.get("tokenizer.ggml.add_bos_token").and_then(|v| v.as_bool()).unwrap_or(false);
        let add_eos = gguf.metadata.get("tokenizer.ggml.add_eos_token").and_then(|v| v.as_bool()).unwrap_or(false);
        let mut special = Vec::new();
        for (i, t) in tokens_v.iter().enumerate() {
            if types_v[i] != 1 { special.push((t.clone(), i as u32)); }
        }
        special.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        BpeTokenizer {
            tokens: tokens_v, token_type: types_v, token_to_id, merges,
            bos_id: g("bos_token_id").unwrap_or(1),
            eos_id: g("eos_token_id").unwrap_or(2),
            add_bos, add_eos, special, byte_map, byte_unmap,
        }
    }

    fn byte_encode_str(&self, s: &str) -> String {
        s.bytes().map(|b| self.byte_map[b as usize]).collect()
    }

    // -------- llama3-pre split (verbatim port of unicode.cpp:355-493) --------
    fn split_llama3(&self, text: &str) -> Vec<String> {
        let cpts: Vec<char> = text.chars().collect();
        let n = cpts.len();
        let getc = |pos: usize| -> Option<char> { if pos < n { Some(cpts[pos]) } else { None } };
        let mut words = Vec::new();
        let mut pos = 0usize;
        let mut word_start = 0usize;

        macro_rules! add_token {
            ($end:expr) => {{
                let end = $end;
                if end > word_start {
                    words.push(cpts[word_start..end].iter().collect::<String>());
                }
                word_start = end;
            }};
        }

        while pos < n {
            let cpt = cpts[pos];
            let wl = is_letter(cpt);
            let wn = is_number(cpt);

            // (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
            if cpt == '\'' {
                if let Some(cn) = getc(pos + 1) {
                    let cn = tolower(cn);
                    if cn == 's' || cn == 't' || cn == 'm' || cn == 'd' {
                        pos += 2;
                        add_token!(pos);
                        continue;
                    }
                    if let Some(cnn) = getc(pos + 2) {
                        let cnn = tolower(cnn);
                        if (cn == 'r' && cnn == 'e') || (cn == 'v' && cnn == 'e') || (cn == 'l' && cnn == 'l') {
                            pos += 3;
                            add_token!(pos);
                            continue;
                        }
                    }
                }
            }

            // [^\r\n\p{L}\p{N}]?\p{L}+
            if !(cpt == '\r' || cpt == '\n' || wn) && (wl || getc(pos + 1).map(is_letter).unwrap_or(false)) {
                pos += 1;
                while getc(pos).map(is_letter).unwrap_or(false) {
                    pos += 1;
                }
                add_token!(pos);
                continue;
            }

            // \p{N}{1,3}
            if wn {
                let mut ini = pos;
                while getc(pos).map(is_number).unwrap_or(false) {
                    pos += 1;
                    if pos - ini >= 3 {
                        add_token!(pos);
                        ini = pos;
                    }
                }
                add_token!(pos);
                continue;
            }

            //  ?[^\s\p{L}\p{N}]+[\r\n]*
            {
                let probe = if cpt == ' ' { pos + 1 } else { pos };
                let probe_special = match getc(probe) {
                    Some(c) => !is_ws(c) && !is_letter(c) && !is_number(c) && has_flags(c),
                    None => false,
                };
                if probe_special && has_flags(cpt) {
                    if cpt == ' ' { pos += 1; }
                    loop {
                        match getc(pos) {
                            Some(c) if !is_ws(c) && !is_letter(c) && !is_number(c) && has_flags(c) => pos += 1,
                            _ => break,
                        }
                    }
                    while matches!(getc(pos), Some('\r') | Some('\n')) { pos += 1; }
                    add_token!(pos);
                    continue;
                }
            }

            // \s*[\r\n]+ | \s+(?!\S) | \s+
            let mut num_ws = 0usize;
            let mut last_rn = 0usize;
            while getc(pos + num_ws).map(is_ws).unwrap_or(false) {
                if matches!(getc(pos + num_ws), Some('\r') | Some('\n')) {
                    last_rn = pos + num_ws + 1;
                }
                num_ws += 1;
            }
            if last_rn > 0 {
                pos = last_rn;
                add_token!(pos);
                continue;
            }
            if num_ws > 1 && getc(pos + num_ws).is_some() {
                pos += num_ws - 1;
                add_token!(pos);
                continue;
            }
            if num_ws > 0 {
                pos += num_ws;
                add_token!(pos);
                continue;
            }

            // no match: single character
            pos += 1;
            add_token!(pos);
        }
        words
    }

    // -------- BPE for one segment (llama-vocab.cpp:716-820) --------
    fn bpe_word(&self, word: &str) -> Vec<u32> {
        // ignore_merges: the whole word in the dictionary → direct id
        if let Some(&id) = self.token_to_id.get(word) {
            return vec![id];
        }
        let mut symbols: Vec<(String, u32)> = Vec::new();
        for ch in word.chars() {
            let s = ch.to_string();
            let id = self.token_to_id.get(&s).copied().unwrap_or(u32::MAX);
            symbols.push((s, id));
        }
        if symbols.is_empty() { return vec![]; }

        loop {
            let mut best: Option<(u32, usize)> = None; // (rank, idx of left)
            for i in 0..symbols.len().saturating_sub(1) {
                if symbols[i].1 == u32::MAX || symbols[i + 1].1 == u32::MAX { continue; }
                if let Some(&(rank, _)) = self.merges.get(&(symbols[i].1, symbols[i + 1].1)) {
                    if best.map(|b| rank < b.0).unwrap_or(true) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_rank, i)) = best else { break; };
            let merged_text = format!("{}{}", symbols[i].0, symbols[i + 1].0);
            let merged_id = self.token_to_id.get(&merged_text).copied().unwrap_or(u32::MAX);
            symbols[i] = (merged_text, merged_id);
            symbols.remove(i + 1);
        }

        let mut out = Vec::new();
        for (text, id) in symbols {
            if id != u32::MAX {
                out.push(id);
            } else {
                for ch in text.chars() {
                    if let Some(&bid) = self.token_to_id.get(&ch.to_string()) {
                        out.push(bid);
                    }
                }
            }
        }
        out
    }

    /// Tokenize with special-token fragmentation (like llama_tokenize special=true)
    pub fn tokenize(&self, text: &str, add_bos: Option<bool>, add_eos: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_bos.unwrap_or(self.add_bos) { ids.push(self.bos_id); }
        let mut rest = text.to_string();
        loop {
            // first occurrence of any special token (longest on ties)
            let mut best: Option<(usize, &str, u32)> = None;
            for (stext, sid) in self.special.iter() {
                if stext.is_empty() || stext.len() > rest.len() { continue; }
                if let Some(p) = rest.find(stext.as_str()) {
                    let better = match best {
                        Some((bp, bs, _)) => p < bp || (p == bp && stext.len() > bs.len()),
                        None => true,
                    };
                    if better { best = Some((p, stext.as_str(), *sid)); }
                }
            }
            match best {
                Some((p, stext, sid)) => {
                    let (before, after) = rest.split_at(p);
                    if !before.is_empty() {
                        ids.extend(self.tokenize_plain(before));
                    }
                    ids.push(sid);
                    rest = after[stext.len()..].to_string();
                }
                None => {
                    if !rest.is_empty() {
                        ids.extend(self.tokenize_plain(&rest));
                    }
                    break;
                }
            }
        }
        if add_eos { ids.push(self.eos_id); }
        ids
    }

    /// BPE only (llama_tokenize special=false)
    pub fn tokenize_plain(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for w in self.split_llama3(text) {
            let encoded = self.byte_encode_str(&w);
            ids.extend(self.bpe_word(&encoded));
        }
        ids
    }

    /// Detokenize one token: normal → byte-decode; special → raw
    pub fn detokenize_one(&self, id: u32) -> String {
        let text = match self.tokens.get(id as usize) {
            Some(t) => t.clone(),
            None => return "<unk>".into(),
        };
        let ty = self.token_type.get(id as usize).copied().unwrap_or(1);
        if ty != 1 { return text; }
        let mut bytes = Vec::new();
        for ch in text.chars() {
            match self.byte_unmap.get(&ch) {
                Some(&b) => bytes.push(b),
                None => {
                    let mut buf = [0u8; 4];
                    let s = ch.encode_utf8(&mut buf);
                    bytes.extend_from_slice(s.as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn detokenize(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        for &id in ids { s.push_str(&self.detokenize_one(id)); }
        s
    }
}

#[allow(unused_imports)]
use Value as _;
