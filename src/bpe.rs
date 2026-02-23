use crate::vocab::Vocab;
use smallvec::SmallVec;

/// Core BPE merge algorithm — tiktoken-style implementation.
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

    bpe_merge(piece, vocab)
}

/// BPE merge using SmallVec to avoid heap allocation for small pieces.
/// Most regex chunks are <30 bytes, so 32 elements fits on the stack.
fn bpe_merge(piece: &[u8], vocab: &Vocab) -> Vec<u32> {
    let n = piece.len();
    
    // parts[i] = (byte_start_index, rank_of_merge_at_this_position)
    // SmallVec avoids heap for pieces up to ~30 bytes (common case)
    let mut parts: SmallVec<[(u32, u32); 32]> = SmallVec::with_capacity(n + 1);
    
    let mut min_rank: u32 = u32::MAX;
    let mut min_idx: usize = usize::MAX;
    
    for i in 0..n - 1 {
        let rank = vocab.rank(&piece[i..i + 2]).unwrap_or(u32::MAX);
        if rank < min_rank {
            min_rank = rank;
            min_idx = i;
        }
        parts.push((i as u32, rank));
    }
    parts.push(((n - 1) as u32, u32::MAX)); // last byte
    parts.push((n as u32, u32::MAX));        // sentinel
    
    let get_rank = |parts: &SmallVec<[(u32, u32); 32]>, i: usize| -> u32 {
        if i + 3 < parts.len() {
            vocab.rank(&piece[parts[i].0 as usize..parts[i + 3].0 as usize]).unwrap_or(u32::MAX)
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
            vocab.rank(&piece[w[0].0 as usize..w[1].0 as usize]).unwrap_or(0)
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
    
    bpe_count_merge(piece, vocab)
}

fn bpe_count_merge(piece: &[u8], vocab: &Vocab) -> usize {
    let n = piece.len();
    let mut parts: SmallVec<[(u32, u32); 32]> = SmallVec::with_capacity(n + 1);
    
    let mut min_rank: u32 = u32::MAX;
    let mut min_idx: usize = usize::MAX;
    
    for i in 0..n - 1 {
        let rank = vocab.rank(&piece[i..i + 2]).unwrap_or(u32::MAX);
        if rank < min_rank {
            min_rank = rank;
            min_idx = i;
        }
        parts.push((i as u32, rank));
    }
    parts.push(((n - 1) as u32, u32::MAX));
    parts.push((n as u32, u32::MAX));
    
    let get_rank = |parts: &SmallVec<[(u32, u32); 32]>, i: usize| -> u32 {
        if i + 3 < parts.len() {
            vocab.rank(&piece[parts[i].0 as usize..parts[i + 3].0 as usize]).unwrap_or(u32::MAX)
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
    
    parts.len() - 1
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
