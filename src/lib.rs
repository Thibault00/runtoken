pub mod bpe;
pub mod regex;
pub mod vocab;

use bpe::bpe_encode_chunk;
use regex::{pattern_for_encoding, SplitPattern};
use vocab::Vocab;

use lru::LruCache;
use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::path::Path;

/// Cache entry: stores both the token IDs and the count
#[derive(Clone)]
struct CacheEntry {
    tokens: Vec<u32>,
}

/// A complete tokenizer for a specific encoding with chunk-level caching.
pub struct Tokenizer {
    pub vocab: Vocab,
    pub pattern: SplitPattern,
    pub encoding_name: String,
    /// LRU cache: chunk bytes → encoded tokens
    cache: RefCell<LruCache<Vec<u8>, CacheEntry>>,
}

impl Tokenizer {
    /// Create a tokenizer from an encoding name (cl100k_base, o200k_base, p50k_base).
    /// Looks for vocab files in the `vocab/` directory.
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
            cache: RefCell::new(LruCache::new(NonZeroUsize::new(8192).unwrap())),
        })
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let chunks = self.pattern.split(text);
        let mut tokens = Vec::new();
        let mut cache = self.cache.borrow_mut();
        for chunk in chunks {
            let chunk_bytes = chunk.as_bytes();
            if let Some(entry) = cache.get(chunk_bytes) {
                tokens.extend_from_slice(&entry.tokens);
            } else {
                let chunk_tokens = bpe_encode_chunk(chunk_bytes, &self.vocab);
                cache.put(chunk_bytes.to_vec(), CacheEntry { tokens: chunk_tokens.clone() });
                tokens.extend(chunk_tokens);
            }
        }
        tokens
    }

    /// Count tokens without allocating the full token array (fast path).
    pub fn count(&self, text: &str) -> usize {
        let chunks = self.pattern.split(text);
        let mut count = 0;
        let mut cache = self.cache.borrow_mut();
        for chunk in chunks {
            let chunk_bytes = chunk.as_bytes();
            if let Some(entry) = cache.get(chunk_bytes) {
                count += entry.tokens.len();
            } else {
                let chunk_tokens = bpe_encode_chunk(chunk_bytes, &self.vocab);
                let len = chunk_tokens.len();
                cache.put(chunk_bytes.to_vec(), CacheEntry { tokens: chunk_tokens });
                count += len;
            }
        }
        count
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
            // GPT-4o family
            m if m.starts_with("gpt-4o") => "o200k_base",
            m if m.starts_with("o1") => "o200k_base",
            m if m.starts_with("o3") => "o200k_base",
            // GPT-4 family  
            m if m.starts_with("gpt-4") => "cl100k_base",
            // GPT-3.5 family
            m if m.starts_with("gpt-3.5") => "cl100k_base",
            // Embeddings
            m if m.contains("embedding") => "cl100k_base",
            // Claude (approximation — uses cl100k for estimation)
            m if m.starts_with("claude") => "cl100k_base",
            // Old models
            m if m.starts_with("text-davinci") => "p50k_base",
            m if m.starts_with("code-davinci") => "p50k_base",
            // Default to cl100k
            _ => "cl100k_base",
        }
    }
}

impl Default for TokenizerRegistry {
    fn default() -> Self {
        Self::new()
    }
}
