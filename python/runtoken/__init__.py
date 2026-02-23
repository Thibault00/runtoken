"""runtoken — A blazing-fast BPE tokenizer for LLMs.

Drop-in replacement for tiktoken with identical output.

Usage:
    >>> import runtoken
    >>> enc = runtoken.get_encoding("cl100k_base")
    >>> enc.encode("Hello, world!")
    [9906, 11, 1917, 0]
    >>> enc.count("Hello, world!")
    4
    >>> runtoken.count("Hello!", model="gpt-4o")
    2
"""

from runtoken._runtoken import (
    PyTokenizer as Tokenizer,
    get_encoding,
    encoding_for_model,
    count,
)

__all__ = ["Tokenizer", "get_encoding", "encoding_for_model", "count"]
__version__ = "0.1.2"
