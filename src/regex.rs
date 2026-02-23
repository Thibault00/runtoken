use fancy_regex::Regex;

/// Pre-compiled regex patterns for different encodings.
pub struct SplitPattern {
    regex: Regex,
}

impl SplitPattern {
    pub fn new(pattern: &str) -> Result<Self, fancy_regex::Error> {
        let regex = Regex::new(pattern)?;
        Ok(SplitPattern { regex })
    }

    /// Split text into chunks that the BPE algorithm processes independently.
    /// Returns byte slices of the original text.
    pub fn split<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut result = Vec::new();
        let mut start = 0;
        // Use find_iter to get all non-overlapping matches
        for mat in self.regex.find_iter(text) {
            match mat {
                Ok(m) => {
                    result.push(&text[m.start()..m.end()]);
                    start = m.end();
                }
                Err(_) => break,
            }
        }
        let _ = start; // suppress warning
        result
    }

    /// Get the cl100k_base regex pattern
    pub fn cl100k_base() -> Self {
        Self::new(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
        ).expect("cl100k_base regex should compile")
    }

    /// Get the o200k_base regex pattern  
    pub fn o200k_base() -> Self {
        Self::new(
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
        ).expect("o200k_base regex should compile")
    }

    /// Get the p50k_base regex pattern
    pub fn p50k_base() -> Self {
        Self::new(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        ).expect("p50k_base regex should compile")
    }
}

/// Return the appropriate pattern for a given encoding name.
pub fn pattern_for_encoding(encoding: &str) -> SplitPattern {
    match encoding {
        "cl100k_base" => SplitPattern::cl100k_base(),
        "o200k_base" => SplitPattern::o200k_base(),
        "p50k_base" => SplitPattern::p50k_base(),
        _ => panic!("Unknown encoding: {}", encoding),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cl100k_split() {
        let pat = SplitPattern::cl100k_base();
        let chunks = pat.split("Hello, world!");
        // Should split into meaningful pieces
        assert!(!chunks.is_empty());
        // Reconstructing should give back the original
        let reconstructed: String = chunks.join("");
        assert_eq!(reconstructed, "Hello, world!");
    }

    #[test]
    fn test_p50k_split() {
        let pat = SplitPattern::p50k_base();
        let chunks = pat.split("Hello, world!");
        let reconstructed: String = chunks.join("");
        assert_eq!(reconstructed, "Hello, world!");
    }
}
