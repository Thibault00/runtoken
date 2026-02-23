use base64::Engine;
use rustc_hash::FxHashMap;
use std::fs;
use std::path::Path;

/// A loaded BPE vocabulary: maps token bytes -> rank (merge priority).
/// Lower rank = higher priority merge.
pub struct Vocab {
    /// token bytes -> rank
    pub encoder: FxHashMap<Vec<u8>, u32>,
    /// rank -> token bytes
    pub decoder: Vec<Vec<u8>>,
    /// Total number of tokens
    pub n_vocab: usize,
    /// Direct lookup for single-byte tokens (avoids HashMap overhead)
    pub single_byte_ranks: [u32; 256],
}

impl Vocab {
    /// Load a .tiktoken file format: each line is `base64(token_bytes) rank`
    pub fn from_tiktoken_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        Self::from_tiktoken_str(&content)
    }

    pub fn from_tiktoken_str(content: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let engine = base64::engine::general_purpose::STANDARD;
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        let mut max_rank: u32 = 0;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let b64 = parts.next().ok_or("missing base64")?;
            let rank_str = parts.next().ok_or("missing rank")?;
            let rank: u32 = rank_str.parse()?;
            let token_bytes = engine.decode(b64)?;
            encoder.insert(token_bytes, rank);
            if rank > max_rank {
                max_rank = rank;
            }
        }

        let n_vocab = encoder.len();
        // Build decoder: rank -> bytes
        let mut decoder: Vec<Vec<u8>> = vec![Vec::new(); (max_rank + 1) as usize];
        for (bytes, &rank) in &encoder {
            decoder[rank as usize] = bytes.clone();
        }

        // Build single-byte rank lookup
        let mut single_byte_ranks = [u32::MAX; 256];
        for byte_val in 0u8..=255 {
            if let Some(&rank) = encoder.get(&vec![byte_val]) {
                single_byte_ranks[byte_val as usize] = rank;
            }
        }

        Ok(Vocab {
            encoder,
            decoder,
            n_vocab,
            single_byte_ranks,
        })
    }

    /// Look up the rank of a byte sequence. Returns None if not in vocab.
    #[inline]
    pub fn rank(&self, bytes: &[u8]) -> Option<u32> {
        self.encoder.get(bytes).copied()
    }

    /// Fast single-byte rank lookup (no HashMap overhead).
    #[inline(always)]
    pub fn rank_single_byte(&self, byte: u8) -> u32 {
        self.single_byte_ranks[byte as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_cl100k() {
        let path = Path::new("vocab/cl100k_base.tiktoken");
        if !path.exists() {
            return;
        }
        let vocab = Vocab::from_tiktoken_file(path).unwrap();
        assert_eq!(vocab.n_vocab, 100256);
        let hello_rank = vocab.rank(b"hello");
        assert!(hello_rank.is_some());
    }

    #[test]
    fn test_single_byte_ranks() {
        let path = Path::new("vocab/cl100k_base.tiktoken");
        if !path.exists() {
            return;
        }
        let vocab = Vocab::from_tiktoken_file(path).unwrap();
        // Space should have a valid rank
        assert_ne!(vocab.rank_single_byte(b' '), u32::MAX);
        // Should match HashMap lookup
        assert_eq!(vocab.rank_single_byte(b' '), vocab.rank(b" ").unwrap());
    }
}
