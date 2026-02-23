use crate::vocab::Vocab;

/// Core BPE merge algorithm using a linked-list approach to avoid O(n) removal.
/// Takes a byte slice (one regex chunk) and merges according to vocab ranks.
/// Returns the list of token IDs (ranks).
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
    // Special case: 2 bytes — either merge or return 2 single-byte tokens
    if len == 2 {
        // Already checked full piece above (no merge possible)
        return vec![
            vocab.rank_single_byte(piece[0]),
            vocab.rank_single_byte(piece[1]),
        ];
    }
    // Special case: 3 bytes
    if len == 3 {
        return bpe_encode_3(piece, vocab);
    }

    bpe_encode_general(piece, vocab)
}

/// Special case for 3-byte chunks — avoids the full merge loop
#[inline]
fn bpe_encode_3(piece: &[u8], vocab: &Vocab) -> Vec<u32> {
    // Try merging [0..2] then result+[2], or [1..3] then [0]+result
    let rank_01 = vocab.rank(&piece[0..2]);
    let rank_12 = vocab.rank(&piece[1..3]);

    match (rank_01, rank_12) {
        (Some(r01), Some(r12)) => {
            if r01 < r12 {
                // Merge 0,1 first
                // Already checked full piece[0..3] at the caller
                vec![r01, vocab.rank_single_byte(piece[2])]
            } else {
                // Merge 1,2 first
                vec![vocab.rank_single_byte(piece[0]), r12]
            }
        }
        (Some(r01), None) => {
            vec![r01, vocab.rank_single_byte(piece[2])]
        }
        (None, Some(r12)) => {
            vec![vocab.rank_single_byte(piece[0]), r12]
        }
        (None, None) => {
            vec![
                vocab.rank_single_byte(piece[0]),
                vocab.rank_single_byte(piece[1]),
                vocab.rank_single_byte(piece[2]),
            ]
        }
    }
}

/// General BPE encode using linked-list for O(1) removal
fn bpe_encode_general(piece: &[u8], vocab: &Vocab) -> Vec<u32> {
    let n = piece.len();
    
    // Doubly-linked list of partition points: 0, 1, 2, ..., n
    // Each consecutive pair (p, succ[p]) defines a token spanning piece[p..succ[p]].
    // Merging = removing the boundary point between two tokens.
    let num_nodes = n + 1;
    let mut succ = vec![0u32; num_nodes];
    let mut pred = vec![0u32; num_nodes]; // pred[i] = prev partition point before i
    for i in 0..num_nodes {
        succ[i] = (i + 1) as u32;
        pred[i] = i.wrapping_sub(1) as u32;
    }
    // Sentinel: succ[n] = u32::MAX (end)
    succ[n] = u32::MAX;
    
    loop {
        // Scan all adjacent pairs to find the best merge
        let mut best_rank = u32::MAX;
        let mut best_pos = u32::MAX; // the partition point to remove
        
        let mut p = 0u32; // first partition point
        loop {
            let q = succ[p as usize]; // end of first token
            if q == u32::MAX { break; }
            let r = succ[q as usize]; // end of second token
            if r == u32::MAX { break; }
            
            // Merged token would be piece[p..r]
            let merged = &piece[p as usize..r as usize];
            if let Some(rank) = vocab.rank(merged) {
                if rank < best_rank {
                    best_rank = rank;
                    best_pos = q; // remove this partition point to merge
                }
            }
            
            p = q; // advance to next token
        }
        
        if best_pos == u32::MAX {
            break; // No more merges
        }
        
        // Remove partition point best_pos (merge the two tokens around it)
        let before = pred[best_pos as usize];
        let after = succ[best_pos as usize];
        succ[before as usize] = after;
        if (after as usize) < num_nodes {
            pred[after as usize] = before;
        }
    }
    
    // Collect results
    let mut tokens = Vec::new();
    let mut p = 0u32;
    loop {
        let q = succ[p as usize];
        if q == u32::MAX { break; }
        let bytes = &piece[p as usize..q as usize];
        tokens.push(vocab.rank(bytes).unwrap_or(0));
        p = q;
    }
    
    tokens
}

/// Count-only fast path: same algorithm but only returns the count.
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
    if len == 2 {
        return 2; // Already checked full piece above
    }
    if len == 3 {
        return bpe_count_3(piece, vocab);
    }
    
    bpe_count_general(piece, vocab)
}

#[inline]
fn bpe_count_3(piece: &[u8], vocab: &Vocab) -> usize {
    let rank_01 = vocab.rank(&piece[0..2]);
    let rank_12 = vocab.rank(&piece[1..3]);

    match (rank_01, rank_12) {
        (Some(r01), Some(r12)) => {
            if r01 < r12 {
                // Merge 0,1 first, then try to merge result with 2
                // We already checked full piece[0..3] at the top
                2
            } else {
                2
            }
        }
        (Some(_), None) => 2,
        (None, Some(_)) => 2,
        (None, None) => 3,
    }
}

fn bpe_count_general(piece: &[u8], vocab: &Vocab) -> usize {
    let n = piece.len();
    let num_nodes = n + 1;
    
    let mut succ = vec![0u32; num_nodes];
    let mut pred = vec![0u32; num_nodes];
    for i in 0..num_nodes {
        succ[i] = (i + 1) as u32;
        pred[i] = i.wrapping_sub(1) as u32;
    }
    succ[n] = u32::MAX;
    
    let mut count = n; // Start with n single-byte tokens
    
    loop {
        let mut best_rank = u32::MAX;
        let mut best_pos = u32::MAX;
        
        let mut p = 0u32;
        loop {
            let q = succ[p as usize];
            if q == u32::MAX { break; }
            let r = succ[q as usize];
            if r == u32::MAX { break; }
            
            let merged = &piece[p as usize..r as usize];
            if let Some(rank) = vocab.rank(merged) {
                if rank < best_rank {
                    best_rank = rank;
                    best_pos = q;
                }
            }
            
            p = q;
        }
        
        if best_pos == u32::MAX {
            break;
        }
        
        let before = pred[best_pos as usize];
        let after = succ[best_pos as usize];
        succ[before as usize] = after;
        if (after as usize) < num_nodes {
            pred[after as usize] = before;
        }
        count -= 1;
    }
    
    count
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
