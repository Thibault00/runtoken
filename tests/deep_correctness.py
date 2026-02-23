#!/usr/bin/env python3
"""Deep correctness check: compare exact token IDs between runtoken and tiktoken."""
import tiktoken, subprocess, json, sys

test_strings = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def foo():\n    return 42",
    "I'm don't won't can't shouldn't",
    "it's they're we've I'll he'd",
    "こんにちは世界",
    "🎉🎊🎈",
    "café résumé naïve",
    "中文测试 with English mixed in 123",
    "ALLCAPS WORDS HERE and MixedCase too",
    "  Hello   World  ",
    "line1\nline2\nline3\n",
    "!@#$%^&*()_+-=[]{}|;':\",./<>?",
    "http://example.com/path?query=value&other=123#fragment",
    '{"key": "value", "number": 42, "array": [1, 2, 3]}',
    "The quick brown fox jumps over the lazy dog. " * 100,
    "fn main() {\n    let x = 42;\n    println!(\"Hello {}\", x);\n}\n" * 50,
    "a" * 500,
    " " * 100,
    "hello " * 200,
    "ab" * 300,
    "\n\n\n\n\n",
    "\t\t\t",
    "MixedCaseWords XMLParser HTMLElement",
    "snake_case_variable camelCaseVar",
    "user@example.com sent $100.00 on 2024-01-15",
    "<html><body><p>Hello</p></body></html>",
    "SELECT * FROM users WHERE id = 1 AND name = 'test';",
    "import numpy as np\nx = np.array([1, 2, 3])\nprint(x.mean())",
    "#include <stdio.h>\nint main() {\n    printf(\"Hello\\n\");\n    return 0;\n}",
    "The     quick\t\tbrown    fox",
    "3.14159265358979323846",
    "192.168.1.1:8080",
    "2024-01-15T10:30:00Z",
    "hello\r\nworld\r\n",
    "trailing spaces   ",
    "   leading spaces",
    "multiple   spaces   between   words",
    "Mixed 中文 English العربية Hindi हिंदी",
    "Ünïcödé tëst wîth dîacrîtîcs",
    # Longer realistic text
    "In machine learning, a large language model (LLM) is a language model notable for its ability to achieve general-purpose language generation and other natural language processing tasks such as classification. LLMs acquire these abilities by learning statistical relationships from text documents during a computationally intensive self-supervised and semi-supervised training process.",
]

encodings = ["cl100k_base", "o200k_base", "p50k_base"]

total = 0
mismatches = 0
count_only_mismatches = 0

for enc_name in encodings:
    enc = tiktoken.get_encoding(enc_name)
    print(f"\n{'='*60}")
    print(f"  {enc_name}")
    print(f"{'='*60}")
    
    for i, text in enumerate(test_strings):
        tiktoken_ids = enc.encode(text, allowed_special="all")
        tiktoken_count = len(tiktoken_ids)
        
        # Get runtoken full encode
        result = subprocess.run(
            ["./target/release/runtoken-cli", "encode", text, enc_name],
            capture_output=True, text=True, cwd="/tmp/runtoken", timeout=10
        )
        
        runtoken_count = -1
        runtoken_ids = []
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.startswith("Count: "):
                    runtoken_count = int(line.split(": ")[1])
                elif line.startswith("Tokens: "):
                    try:
                        runtoken_ids = json.loads(line.split(": ", 1)[1])
                    except:
                        pass
        
        # Also test count-only path
        count_result = subprocess.run(
            ["./target/release/runtoken-cli", "count", text, enc_name],
            capture_output=True, text=True, cwd="/tmp/runtoken", timeout=10
        )
        count_only = int(count_result.stdout.strip()) if count_result.returncode == 0 else -1
        
        total += 1
        
        ids_match = tiktoken_ids == runtoken_ids
        count_match = tiktoken_count == runtoken_count
        count_only_match = tiktoken_count == count_only
        
        if not ids_match:
            mismatches += 1
            display = repr(text[:50]) + ("..." if len(text) > 50 else "")
            print(f"  ❌ #{i} {display}")
            print(f"     tiktoken: count={tiktoken_count}, ids={tiktoken_ids[:10]}{'...' if len(tiktoken_ids) > 10 else ''}")
            print(f"     runtoken: count={runtoken_count}, ids={runtoken_ids[:10]}{'...' if len(runtoken_ids) > 10 else ''}")
            if count_match and not ids_match:
                # Find first differing position
                for j in range(min(len(tiktoken_ids), len(runtoken_ids))):
                    if tiktoken_ids[j] != runtoken_ids[j]:
                        print(f"     First diff at position {j}: tiktoken={tiktoken_ids[j]} runtoken={runtoken_ids[j]}")
                        break
        else:
            if i < 8 or i % 5 == 0:
                display = repr(text[:40]) + ("..." if len(text) > 40 else "")
                print(f"  ✅ #{i} {display} → {tiktoken_count} tokens, IDs exact match")
        
        if not count_only_match:
            count_only_mismatches += 1
            print(f"  ⚠️  #{i} count-only mismatch: tiktoken={tiktoken_count} count_only={count_only}")

print(f"\n{'='*60}")
print(f"  FINAL RESULTS")
print(f"{'='*60}")
print(f"  Total tests: {total}")
print(f"  Exact ID matches: {total - mismatches}/{total}")
print(f"  Count-only matches: {total - count_only_mismatches}/{total}")
print(f"  Mismatches: {mismatches}")
if mismatches == 0 and count_only_mismatches == 0:
    print(f"  🎉 PERFECT — 100% match across all encodings!")
else:
    print(f"  ⚠️  ISSUES FOUND")

sys.exit(1 if mismatches > 0 else 0)
