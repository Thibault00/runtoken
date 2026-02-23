# 🦀 runtoken

**A blazing-fast BPE tokenizer for LLMs. Drop-in tiktoken replacement, 20-80x faster.**

Built from scratch in Rust with Python bindings via PyO3. Produces **identical output** to tiktoken — same token IDs, same order, every time.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Why?

If you're building an LLM gateway, proxy, or any system that processes tokens at scale, tokenization speed matters. Every API request needs token counting for:

- **Cost estimation & billing**
- **Rate limiting per user**
- **Context window management**
- **Smart routing** (pick the cheapest model that fits)

tiktoken is good. runtoken is faster.

## Benchmarks

Apples-to-apples comparison — both called as Python packages, same machine, same text:

### Encode (full token IDs)

| Input | tiktoken | runtoken | Speedup |
|-------|----------|----------|---------|
| Short text (29 chars, 9 tokens) | 1.3M tok/s | 24.6M tok/s | **19x** 🦀 |
| Medium text (1050 chars, 511 tokens) | 2.5M tok/s | 68.8M tok/s | **27x** 🦀 |
| Code (1200 chars, 380 tokens) | 1.5M tok/s | 63.5M tok/s | **44x** 🦀 |
| Long English (4500 chars, 1001 tokens) | 2.5M tok/s | 73.6M tok/s | **29x** 🦀 |
| Long code (5600 chars, 2160 tokens) | 1.5M tok/s | 88.2M tok/s | **59x** 🦀 |
| Unicode (500 chars, 420 tokens) | 4.2M tok/s | 89.2M tok/s | **21x** 🦀 |

### Count-only (the gateway use case)

| Input | tiktoken | runtoken | Speedup |
|-------|----------|----------|---------|
| Medium text | 2.5M tok/s | 940M tok/s | **381x** 🦀 |
| Long English | 2.6M tok/s | 1.4B tok/s | **538x** 🦀 |
| Long code | 1.5M tok/s | 2.6B tok/s | **1750x** 🦀 |

> Benchmarked on a 2-vCPU cloud instance. Count-only benefits from multi-level caching (text-level + chunk-level LRU).

### Correctness

| Test Suite | Tests | Result |
|------------|-------|--------|
| Deep correctness (41 strings × 3 encodings) | 123 | ✅ 100% |
| Stress test (up to 64K tokens) | 27 | ✅ 100% |
| PDF documents (academic papers, 65K tokens) | 54 | ✅ 100% |
| **Total** | **204** | **0 mismatches** |

Every test compares **exact token IDs** — not just counts, but the same numbers in the same order.

## Installation

```bash
pip install runtoken
```

### From source

```bash
git clone https://github.com/Thibault00/runtoken.git
cd runtoken
pip install maturin
maturin develop --release
```

## Usage

### Python

```python
import runtoken

# Get a tokenizer by encoding name (same API as tiktoken)
enc = runtoken.get_encoding("cl100k_base")

# Encode text to token IDs
tokens = enc.encode("Hello, world!")
# [9906, 11, 1917, 0]

# Count tokens
count = enc.count("Hello, world!")
# 4

# Decode back to text
text = enc.decode([9906, 11, 1917, 0])
# "Hello, world!"

# Get tokenizer by model name
enc = runtoken.encoding_for_model("gpt-4o")  # → o200k_base
enc = runtoken.encoding_for_model("gpt-4")   # → cl100k_base
enc = runtoken.encoding_for_model("claude")   # → cl100k_base

# Quick one-liner
runtoken.count("Hello!", model="gpt-4o")
# 2
```

### Rust

```rust
use runtoken::Tokenizer;

let tokenizer = Tokenizer::new("cl100k_base").unwrap();
let tokens = tokenizer.encode("Hello, world!");
let count = tokenizer.count("Hello, world!");
let text = tokenizer.decode(&tokens);
```

### CLI

```bash
# Encode text
runtoken-cli encode "Hello, world!" cl100k_base

# Count tokens
runtoken-cli count "Hello, world!" o200k_base

# Read from stdin (for large texts)
cat myfile.txt | runtoken-cli count - cl100k_base

# Benchmark
runtoken-cli bench cl100k_base
```

## Supported Encodings

| Encoding | Models | Vocab Size |
|----------|--------|-----------|
| `cl100k_base` | GPT-4, GPT-3.5-turbo, Claude | 100,256 |
| `o200k_base` | GPT-4o, o1, o3 | 200,019 |
| `p50k_base` | text-davinci-003, Codex | 50,281 |

### Model → Encoding Mapping

| Model prefix | Encoding |
|-------------|----------|
| `gpt-4o`, `o1`, `o3` | o200k_base |
| `gpt-4`, `gpt-3.5`, `claude` | cl100k_base |
| `text-davinci`, `code-davinci` | p50k_base |

## Architecture

```
src/
├── lib.rs       # Tokenizer + TokenizerRegistry + multi-level caching
├── bpe.rs       # Core BPE merge algorithm (tiktoken-compatible)
├── vocab.rs     # Vocabulary loading (.tiktoken format)
├── regex.rs     # Regex splitting per encoding
├── python.rs    # PyO3 bindings
└── main.rs      # CLI tool
```

**~900 lines of Rust** — that's the entire tokenizer. Key design decisions:

- **Multi-level LRU cache**: Text-level (hash → tokens) + chunk-level (bytes → tokens). Repeated text is a hash lookup.
- **Precomputed rank tables**: Single-byte and two-byte pair ranks as direct arrays — no HashMap overhead for the most common lookups.
- **Inline chunk processing**: Regex chunks are encoded inline without collecting into intermediate Vecs.
- **tiktoken-style BPE merge**: Tracks min_rank inline during merges, avoids priority queue overhead for small chunks.

## How it works

BPE (Byte Pair Encoding) tokenization:

1. **Regex split**: Split input text into chunks using encoding-specific regex patterns
2. **Byte-level merging**: For each chunk, start with individual bytes and repeatedly merge the pair with the lowest rank (priority) in the vocabulary
3. **Token IDs**: Map the final merged byte sequences to their vocabulary rank

runtoken uses the exact same regex patterns and vocabulary files as tiktoken, which is why the output is identical.

## Contributing

```bash
# Clone and build
git clone https://github.com/Thibault00/runtoken.git
cd runtoken
cargo build --release

# Run Rust tests
cargo test

# Run correctness tests against tiktoken
pip install tiktoken
python tests/deep_correctness.py
python tests/stress_test.py

# Build Python package
pip install maturin
maturin develop --release
python tests/benchmark_python.py
```

## License

MIT — see [LICENSE](LICENSE).
