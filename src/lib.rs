pub mod bpe;
pub mod regex;
pub mod vocab;

use bpe::bpe_encode_chunk;
use regex::{pattern_for_encoding, SplitPattern};
use vocab::Vocab;

use lru::LruCache;
use rustc_hash::FxHasher;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::path::Path;

/// A complete tokenizer for a specific encoding with multi-level caching.
pub struct Tokenizer {
    pub vocab: Vocab,
    pub pattern: SplitPattern,
    pub encoding_name: String,
    /// Level 1: chunk-level cache (chunk bytes → encoded tokens)
    chunk_cache: RefCell<LruCache<Vec<u8>, Vec<u32>>>,
    /// Level 2: full-text cache (text hash → encoded tokens)
    text_cache: RefCell<LruCache<u64, Vec<u32>>>,
}

/// Fast hash of a string using FxHash
#[inline]
fn fx_hash_str(s: &str) -> u64 {
    let mut hasher = FxHasher::default();
    s.hash(&mut hasher);
    hasher.finish()
}

impl Tokenizer {
    /// Create a tokenizer from an encoding name (cl100k_base, o200k_base, p50k_base).
    pub fn new(encoding_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let vocab_path = format!("vocab/{}.tiktoken", encoding_name);
        Self::from_file(encoding_name, Path::new(&vocab_path))
    }

    /// Create a tokenizer from an encoding name and a specific vocab file path.
    pub fn from_file(encoding_name: &str, vocab_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let vocab = Vocab::from_tiktoken_file(vocab_path)?;
        let pattern = pattern_for_encoding(encoding_name);
        Ok(Tokenizer {
            vocab,
            pattern,
            encoding_name: encoding_name.to_string(),
            chunk_cache: RefCell::new(LruCache::new(NonZeroUsize::new(8192).unwrap())),
            text_cache: RefCell::new(LruCache::new(NonZeroUsize::new(1024).unwrap())),
        })
    }

    /// Encode text into token IDs (with full text-level + chunk-level caching).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Level 2: check text-level cache first
        let text_hash = fx_hash_str(text);
        {
            let mut tc = self.text_cache.borrow_mut();
            if let Some(cached) = tc.get(&text_hash) {
                return cached.clone();
            }
        }

        // Cache miss: do the work
        let chunks = self.pattern.split(text);
        let mut tokens = Vec::new();
        let mut cc = self.chunk_cache.borrow_mut();
        for chunk in chunks {
            let chunk_bytes = chunk.as_bytes();
            if let Some(entry) = cc.get(chunk_bytes) {
                tokens.extend_from_slice(entry);
            } else {
                let chunk_tokens = bpe_encode_chunk(chunk_bytes, &self.vocab);
                cc.put(chunk_bytes.to_vec(), chunk_tokens.clone());
                tokens.extend(chunk_tokens);
            }
        }

        // Store in text-level cache
        self.text_cache.borrow_mut().put(text_hash, tokens.clone());
        tokens
    }

    /// Count tokens (with caching).
    pub fn count(&self, text: &str) -> usize {
        // Use encode which is cached — the allocation overhead is minimal
        // compared to regex splitting
        self.encode(text).len()
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &token in tokens {
            if (token as usize) < self.vocab.decoder.len() {
                bytes.extend_from_slice(&self.vocab.decoder[token as usize]);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Multi-model tokenizer registry — load once, use for any model.
pub struct TokenizerRegistry {
    tokenizers: std::collections::HashMap<String, Tokenizer>,
}

impl TokenizerRegistry {
    pub fn new() -> Self {
        TokenizerRegistry {
            tokenizers: std::collections::HashMap::new(),
        }
    }

    /// Load a tokenizer and register it.
    pub fn load(&mut self, encoding_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let tokenizer = Tokenizer::new(encoding_name)?;
        self.tokenizers.insert(encoding_name.to_string(), tokenizer);
        Ok(())
    }

    /// Get a reference to a loaded tokenizer.
    pub fn get(&self, encoding_name: &str) -> Option<&Tokenizer> {
        self.tokenizers.get(encoding_name)
    }

    /// Encode using a specific encoding.
    pub fn encode(&self, text: &str, encoding_name: &str) -> Option<Vec<u32>> {
        self.get(encoding_name).map(|t| t.encode(text))
    }

    /// Count tokens using a specific encoding.
    pub fn count(&self, text: &str, encoding_name: &str) -> Option<usize> {
        self.get(encoding_name).map(|t| t.count(text))
    }

    /// Map model name to encoding name.
    pub fn encoding_for_model(model: &str) -> &'static str {
        match model {
            m if m.starts_with("gpt-4o") => "o200k_base",
            m if m.starts_with("o1") => "o200k_base",
            m if m.starts_with("o3") => "o200k_base",
            m if m.starts_with("gpt-4") => "cl100k_base",
            m if m.starts_with("gpt-3.5") => "cl100k_base",
            m if m.contains("embedding") => "cl100k_base",
            m if m.starts_with("claude") => "cl100k_base",
            m if m.starts_with("text-davinci") => "p50k_base",
            m if m.starts_with("code-davinci") => "p50k_base",
            _ => "cl100k_base",
        }
    }
}

impl Default for TokenizerRegistry {
    fn default() -> Self {
        Self::new()
    }
}
