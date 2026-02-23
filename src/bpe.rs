use crate::vocab::Vocab;

/// Core BPE merge algorithm.
/// Takes a byte slice (one regex chunk) and merges according to vocab ranks.
/// Returns the list of token IDs (ranks).
pub fn bpe_encode_chunk(piece: &[u8], vocab: &Vocab) -> Vec<u32> {
    if piece.is_empty() {
        return vec![];
    }
    if piece.len() == 1 {
        // Single byte — must be in vocab as a byte-level token
        return match vocab.rank(piece) {
            Some(rank) => vec![rank],
            None => vec![], // shouldn't happen with proper vocab
        };
    }

    // Check if the whole piece is already a single token
    if let Some(rank) = vocab.rank(piece) {
        return vec![rank];
    }

    // Start with individual bytes
    // We use a linked-list style approach for efficient merging
    // parts[i] = (start_byte_index, rank_of_token_starting_here)
    // We store the byte ranges that form each current token
    let mut parts: Vec<ByteRange> = piece
        .iter()
        .enumerate()
        .map(|(i, _)| ByteRange { start: i, end: i + 1 })
        .collect();

    loop {
        if parts.len() <= 1 {
            break;
        }

        // Find the pair with the lowest merge rank
        let mut best_rank = u32::MAX;
        let mut best_idx = usize::MAX;

        for i in 0..parts.len() - 1 {
            let merged_bytes = &piece[parts[i].start..parts[i + 1].end];
            if let Some(rank) = vocab.rank(merged_bytes) {
                if rank < best_rank {
                    best_rank = rank;
                    best_idx = i;
                }
            }
        }

        if best_idx == usize::MAX {
            break; // No more merges possible
        }

        // Merge parts[best_idx] and parts[best_idx + 1]
        let new_end = parts[best_idx + 1].end;
        parts[best_idx].end = new_end;
        parts.remove(best_idx + 1);
    }

    // Convert byte ranges to token IDs
    parts
        .iter()
        .map(|range| {
            let bytes = &piece[range.start..range.end];
            vocab.rank(bytes).unwrap_or(0) // should always exist after merging
        })
        .collect()
}

/// Count-only fast path: same algorithm but only returns the count, no allocation for token IDs.
pub fn bpe_count_chunk(piece: &[u8], vocab: &Vocab) -> usize {
    if piece.is_empty() {
        return 0;
    }
    if piece.len() == 1 {
        return 1;
    }
    if vocab.rank(piece).is_some() {
        return 1;
    }

    let mut parts: Vec<ByteRange> = piece
        .iter()
        .enumerate()
        .map(|(i, _)| ByteRange { start: i, end: i + 1 })
        .collect();

    loop {
        if parts.len() <= 1 {
            break;
        }

        let mut best_rank = u32::MAX;
        let mut best_idx = usize::MAX;

        for i in 0..parts.len() - 1 {
            let merged_bytes = &piece[parts[i].start..parts[i + 1].end];
            if let Some(rank) = vocab.rank(merged_bytes) {
                if rank < best_rank {
                    best_rank = rank;
                    best_idx = i;
                }
            }
        }

        if best_idx == usize::MAX {
            break;
        }

        let new_end = parts[best_idx + 1].end;
        parts[best_idx].end = new_end;
        parts.remove(best_idx + 1);
    }

    parts.len()
}

#[derive(Clone, Debug)]
struct ByteRange {
    start: usize,
    end: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_single_byte() {
        let path = Path::new("vocab/cl100k_base.tiktoken");
        if !path.exists() {
            return;
        }
        let vocab = Vocab::from_tiktoken_file(path).unwrap();
        // Single space should be one token
        let tokens = bpe_encode_chunk(b" ", &vocab);
        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn test_hello() {
        let path = Path::new("vocab/cl100k_base.tiktoken");
        if !path.exists() {
            return;
        }
        let vocab = Vocab::from_tiktoken_file(path).unwrap();
        let tokens = bpe_encode_chunk(b"hello", &vocab);
        assert!(!tokens.is_empty());
        // "hello" should be a single token in cl100k
        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn test_count_matches_encode() {
        let path = Path::new("vocab/cl100k_base.tiktoken");
        if !path.exists() {
            return;
        }
        let vocab = Vocab::from_tiktoken_file(path).unwrap();
        let test_pieces = [b"hello" as &[u8], b"world", b" the", b"abc123"];
        for piece in &test_pieces {
            let tokens = bpe_encode_chunk(piece, &vocab);
            let count = bpe_count_chunk(piece, &vocab);
            assert_eq!(tokens.len(), count, "Mismatch for {:?}", String::from_utf8_lossy(piece));
        }
    }
}
