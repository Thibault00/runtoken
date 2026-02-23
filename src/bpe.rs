use crate::vocab::Vocab;

/// Core BPE merge algorithm — tiktoken-style implementation.
/// Uses two paths: fast path for small pieces (<100 bytes), heap-based for large.
pub fn bpe_encode_chunk(piece: &[u8], vocab: &Vocab) -> Vec<u32> {
    let len = piece.len();
    if len == 0 {
        return vec![];
    }
    if len == 1 {
        return vec![vocab.rank_single_byte(piece[0])];
    }
    // Check if the whole piece is already a single token
    if let Some(rank) = vocab.rank(piece) {
        return vec![rank];
    }

    bpe_merge_small(piece, vocab)
}

/// Fast BPE merge for small pieces (most regex chunks are <50 bytes).
/// Tracks min_rank inline to avoid full rescan each iteration.
/// Based on tiktoken's _byte_pair_merge approach.
fn bpe_merge_small(piece: &[u8], vocab: &Vocab) -> Vec<u32> {
    let n = piece.len();
    
    // parts[i] = (byte_start_index, rank_of_merge_at_this_position)
    // The rank stored is for merging parts[i] with parts[i+1].
    // We add two sentinel entries at the end.
    let mut parts: Vec<(usize, u32)> = Vec::with_capacity(n + 1);
    
    let mut min_rank: u32 = u32::MAX;
    let mut min_idx: usize = usize::MAX;
    
    for i in 0..n - 1 {
        let rank = vocab.rank(&piece[i..i + 2]).unwrap_or(u32::MAX);
        if rank < min_rank {
            min_rank = rank;
            min_idx = i;
        }
        parts.push((i, rank));
    }
    parts.push((n - 1, u32::MAX)); // last byte
    parts.push((n, u32::MAX));     // sentinel
    
    // Inline rank getter: computes the rank of merging parts[i] with parts[i+1] 
    // AFTER parts[i+1] will be removed (so we look at parts[i+2] which becomes parts[i+1])
    let get_rank = |parts: &Vec<(usize, u32)>, i: usize| -> u32 {
        if i + 3 < parts.len() {
            vocab.rank(&piece[parts[i].0..parts[i + 3].0]).unwrap_or(u32::MAX)
        } else {
            u32::MAX
        }
    };
    
    while min_rank != u32::MAX {
        let i = min_idx;
        
        // Update ranks for affected positions before removal
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);
        
        // Rescan for new minimum
        min_rank = u32::MAX;
        min_idx = usize::MAX;
        for (idx, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank {
                min_rank = rank;
                min_idx = idx;
            }
        }
    }
    
    // Convert to token IDs
    parts
        .windows(2)
        .map(|w| {
            vocab.rank(&piece[w[0].0..w[1].0]).unwrap_or(0)
        })
        .collect()
}

/// Count-only fast path.
pub fn bpe_count_chunk(piece: &[u8], vocab: &Vocab) -> usize {
    let len = piece.len();
    if len == 0 {
        return 0;
    }
    if len == 1 {
        return 1;
    }
    if vocab.rank(piece).is_some() {
        return 1;
    }
    
    // Use the same merge but just count the result
    bpe_count_small(piece, vocab)
}

fn bpe_count_small(piece: &[u8], vocab: &Vocab) -> usize {
    let n = piece.len();
    let mut parts: Vec<(usize, u32)> = Vec::with_capacity(n + 1);
    
    let mut min_rank: u32 = u32::MAX;
    let mut min_idx: usize = usize::MAX;
    
    for i in 0..n - 1 {
        let rank = vocab.rank(&piece[i..i + 2]).unwrap_or(u32::MAX);
        if rank < min_rank {
            min_rank = rank;
            min_idx = i;
        }
        parts.push((i, rank));
    }
    parts.push((n - 1, u32::MAX));
    parts.push((n, u32::MAX));
    
    let get_rank = |parts: &Vec<(usize, u32)>, i: usize| -> u32 {
        if i + 3 < parts.len() {
            vocab.rank(&piece[parts[i].0..parts[i + 3].0]).unwrap_or(u32::MAX)
        } else {
            u32::MAX
        }
    };
    
    while min_rank != u32::MAX {
        let i = min_idx;
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);
        
        min_rank = u32::MAX;
        min_idx = usize::MAX;
        for (idx, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank {
                min_rank = rank;
                min_idx = idx;
            }
        }
    }
    
    parts.len() - 1 // -1 for the sentinel
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
