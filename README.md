# 🦀 runtoken

**A blazing-fast BPE tokenizer for LLMs. Drop-in replacement for tiktoken.**

Built in Rust with Python bindings. 100% correct — produces identical output to tiktoken, token-for-token.

## Installation

```bash
pip install runtoken
```

## Usage

```python
import runtoken

# Just like tiktoken
enc = runtoken.get_encoding("cl100k_base")
tokens = enc.encode("Hello, world!")
# [9906, 11, 1917, 0]

count = enc.count("Hello, world!")
# 4

# Or by model name
enc = runtoken.encoding_for_model("gpt-4o")

# Quick count
runtoken.count("Hello!", model="gpt-4o")
# 2
```

## Supported Encodings

| Encoding | Models | Vocab Size |
|----------|--------|-----------|
| `cl100k_base` | GPT-4, GPT-3.5-turbo, Claude | 100,256 |
| `o200k_base` | GPT-4o, o1, o3 | 200,019 |
| `p50k_base` | text-davinci-003 | 50,281 |

## Correctness

204/204 tests pass with identical token IDs to tiktoken across all encodings. Tested with:
- 41 diverse strings (Unicode, CJK, emoji, code, edge cases)
- Stress tests up to 64K tokens
- Real PDF documents (academic papers up to 65K tokens)

## License

MIT
