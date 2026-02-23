use fancy_regex::Regex as FancyRegex;

/// Pre-compiled regex patterns for different encodings.
/// fancy_regex handles lookaheads required by BPE patterns.
pub struct SplitPattern {
    regex: FancyRegex,
}

impl SplitPattern {
    pub fn new(pattern: &str) -> Result<Self, fancy_regex::Error> {
        let regex = FancyRegex::new(pattern)?;
        Ok(SplitPattern { regex })
    }

    /// Split text into chunks for BPE processing.
    #[inline]
    pub fn split<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut result = Vec::new();
        self.for_each_chunk(text, |chunk| result.push(chunk));
        result
    }

    /// Process chunks inline without collecting into Vec.
    #[inline]
    pub fn for_each_chunk<'a, F>(&self, text: &'a str, mut f: F)
    where
        F: FnMut(&'a str),
    {
        for mat in self.regex.find_iter(text) {
            match mat {
                Ok(m) => f(&text[m.start()..m.end()]),
                Err(_) => break,
            }
        }
    }

    pub fn cl100k_base() -> Self {
        Self::new(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
        ).expect("cl100k_base regex should compile")
    }

    pub fn o200k_base() -> Self {
        Self::new(
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
        ).expect("o200k_base regex should compile")
    }

    pub fn p50k_base() -> Self {
        Self::new(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        ).expect("p50k_base regex should compile")
    }
}

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
        assert!(!chunks.is_empty());
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

    #[test]
    fn test_for_each_chunk() {
        let pat = SplitPattern::cl100k_base();
        let mut chunks = Vec::new();
        pat.for_each_chunk("Hello, world!", |chunk| chunks.push(chunk));
        let reconstructed: String = chunks.join("");
        assert_eq!(reconstructed, "Hello, world!");
    }
}
