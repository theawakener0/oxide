use std::path::PathBuf;

use anyhow::Result;
use shimmytok::Tokenizer as ShimmyTokenizer;

pub struct TokenizerWrapper {
    inner: ShimmyTokenizer,
    eos_token_id: u32,
    pending_tokens: Vec<u32>,
}

impl TokenizerWrapper {
    pub fn from_gguf(path: &PathBuf) -> Result<Self> {
        let inner = ShimmyTokenizer::from_gguf_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let eos_token_id = inner.eos_token();

        tracing::info!("Loaded tokenizer from GGUF, EOS={}", eos_token_id);

        Ok(Self {
            inner,
            eos_token_id,
            pending_tokens: Vec::new(),
        })
    }

    pub fn from_file(path: &PathBuf) -> Result<Self> {
        let inner = ShimmyTokenizer::from_gguf_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let eos_token_id = inner.eos_token();

        tracing::info!("Loaded tokenizer from file, EOS={}", eos_token_id);

        Ok(Self {
            inner,
            eos_token_id,
            pending_tokens: Vec::new(),
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Encode failed: {}", e))
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner
            .decode(tokens, false)
            .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn clear_cache(&mut self) {
        self.pending_tokens.clear();
    }

    pub fn decode_next(&mut self, token: u32) -> Result<Option<String>> {
        self.pending_tokens.push(token);

        let decoded = self.decode(&self.pending_tokens)?;

        let prev_decoded: String = {
            let prev_tokens = &self.pending_tokens[..self.pending_tokens.len() - 1];
            if prev_tokens.is_empty() {
                String::new()
            } else {
                self.decode(prev_tokens)?
            }
        };

        if decoded.len() > prev_decoded.len() {
            let new_text = decoded[prev_decoded.len()..].to_string();
            Ok(Some(new_text))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&mut self) -> Result<Option<String>> {
        if self.pending_tokens.is_empty() {
            return Ok(None);
        }

        let decoded = self.decode(&self.pending_tokens)?;

        let prev_decoded: String = {
            if self.pending_tokens.len() > 1 {
                let prev_tokens = &self.pending_tokens[..self.pending_tokens.len() - 1];
                self.decode(prev_tokens)?
            } else {
                String::new()
            }
        };

        if decoded.len() > prev_decoded.len() {
            Ok(Some(decoded[prev_decoded.len()..].to_string()))
        } else {
            Ok(None)
        }
    }
}
