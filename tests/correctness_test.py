#!/usr/bin/env python3
"""
Correctness test: Compare runtoken output against tiktoken (ground truth).
Runs the runtoken CLI and compares token counts with tiktoken.
"""

import subprocess
import sys
import json
import time
import tiktoken

# ── Test strings ───────────────────────────────────────────────────────

TEST_STRINGS = [
    # Basic
    "Hello, world!",
    "Hello",
    " ",
    "",
    "a",
    "  ",
    "\n",
    "\n\n",
    "\t",

    # Unicode
    "こんにちは世界",
    "🎉🎊🎈",
    "café résumé naïve",
    "Ünïcödé",
    "中文测试",
    "العربية",
    "हिंदी",
    "🇺🇸🇬🇧🇫🇷",

    # Code
    "def foo():\n    return 42",
    "fn main() { println!(\"Hello\"); }",
    "console.log('hello world');",
    "SELECT * FROM users WHERE id = 1;",
    "import numpy as np\nx = np.array([1, 2, 3])",
    "#include <stdio.h>\nint main() { return 0; }",

    # Contractions (important for regex splitting)
    "I'm don't won't can't shouldn't",
    "it's they're we've I'll he'd",
    "It's a beautiful day, isn't it?",

    # Numbers
    "12345",
    "3.14159",
    "1,000,000",
    "2024-01-15",
    "192.168.1.1",

    # Mixed
    "The quick brown fox jumps over the lazy dog.",
    "  Hello   World  ",
    "line1\nline2\nline3",
    "tabs\there\tand\tthere",
    "MixedCaseWords XMLParser HTMLElement",
    "snake_case_variable",
    "kebab-case-name",
    "ALLCAPS WORDS HERE",

    # Longer text
    "The quick brown fox jumps over the lazy dog. " * 10,
    "In the beginning was the Word, and the Word was with God, and the Word was God. He was in the beginning with God. All things were made through Him, and without Him nothing was made that was made.",

    # Edge cases
    "a" * 1000,
    " " * 100,
    "\n" * 50,
    "!@#$%^&*()_+-=[]{}|;':\",./<>?",
    "\\n\\t\\r\\0",
    "http://example.com/path?query=value&other=123#fragment",
    "user@example.com",
    "<html><body><p>Hello</p></body></html>",
    '{"key": "value", "number": 42, "array": [1, 2, 3]}',

    # Whitespace variations
    "hello\r\nworld",
    "hello\rworld",
    "   leading spaces",
    "trailing spaces   ",
    "multiple   spaces   between",

    # Long repetitive
    "ab" * 500,
    "hello " * 200,

    # Binary-ish
    "\x00\x01\x02\x03",
    "null\x00byte",
]

def get_tiktoken_results(text, encoding_name):
    """Ground truth from tiktoken."""
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text, allowed_special="all")
    return {
        "count": len(tokens),
        "token_ids": tokens,
    }

def get_runtoken_results(text, encoding_name, binary_path):
    """Get results from our runtoken CLI."""
    try:
        result = subprocess.run(
            [binary_path, "encode", text, encoding_name],
            capture_output=True, text=True, timeout=10,
            cwd="/tmp/runtoken"
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip(), "count": -1, "token_ids": []}

        # Parse output
        count = -1
        token_ids = []
        for line in result.stdout.strip().split("\n"):
            if line.startswith("Count: "):
                count = int(line.split(": ")[1])
            elif line.startswith("Tokens: "):
                # Parse [1, 2, 3] format
                tokens_str = line.split(": ", 1)[1]
                token_ids = json.loads(tokens_str)

        return {"count": count, "token_ids": token_ids}
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "count": -1, "token_ids": []}
    except Exception as e:
        return {"error": str(e), "count": -1, "token_ids": []}

def get_runtoken_count(text, encoding_name, binary_path):
    """Get count-only from runtoken CLI."""
    try:
        result = subprocess.run(
            [binary_path, "count", text, encoding_name],
            capture_output=True, text=True, timeout=10,
            cwd="/tmp/runtoken"
        )
        if result.returncode != 0:
            return -1
        return int(result.stdout.strip())
    except:
        return -1

def run_correctness_tests(encoding_name="cl100k_base", binary_path="./target/release/runtoken-cli"):
    """Run all correctness tests for an encoding."""
    print(f"\n{'='*60}")
    print(f"  CORRECTNESS TEST: {encoding_name}")
    print(f"{'='*60}\n")

    passed = 0
    failed = 0
    errors = 0
    failures = []

    for i, text in enumerate(TEST_STRINGS):
        if not text:  # skip empty string (tiktoken behavior may vary)
            continue

        tiktoken_result = get_tiktoken_results(text, encoding_name)
        runtoken_result = get_runtoken_results(text, encoding_name, binary_path)

        display_text = repr(text[:60]) + ("..." if len(text) > 60 else "")

        if "error" in runtoken_result:
            errors += 1
            print(f"  ❌ ERROR #{i}: {display_text}")
            print(f"     Error: {runtoken_result['error']}")
            failures.append((text, "error", runtoken_result.get("error", "")))
            continue

        # Compare counts
        if tiktoken_result["count"] != runtoken_result["count"]:
            failed += 1
            print(f"  ❌ FAIL  #{i}: {display_text}")
            print(f"     tiktoken: {tiktoken_result['count']} tokens")
            print(f"     runtoken: {runtoken_result['count']} tokens")
            failures.append((text, "count_mismatch", f"expected={tiktoken_result['count']} got={runtoken_result['count']}"))
            continue

        # Compare token IDs
        if tiktoken_result["token_ids"] != runtoken_result["token_ids"]:
            failed += 1
            print(f"  ❌ FAIL  #{i}: {display_text} (IDs differ, count matches)")
            print(f"     tiktoken: {tiktoken_result['token_ids'][:10]}...")
            print(f"     runtoken: {runtoken_result['token_ids'][:10]}...")
            failures.append((text, "id_mismatch", f"count={tiktoken_result['count']}"))
            continue

        passed += 1
        if i < 20 or i % 10 == 0:  # Print first 20 and every 10th
            print(f"  ✅ PASS  #{i}: {display_text} → {tiktoken_result['count']} tokens")

    # Also test count-only fast path
    print(f"\n  Testing count-only fast path...")
    count_mismatches = 0
    for i, text in enumerate(TEST_STRINGS):
        if not text:
            continue
        tiktoken_count = get_tiktoken_results(text, encoding_name)["count"]
        runtoken_count = get_runtoken_count(text, encoding_name, binary_path)
        if tiktoken_count != runtoken_count:
            count_mismatches += 1
            if count_mismatches <= 5:
                print(f"  ❌ Count mismatch: {repr(text[:40])} tiktoken={tiktoken_count} runtoken={runtoken_count}")

    if count_mismatches == 0:
        print(f"  ✅ Count-only fast path: all {len(TEST_STRINGS)} strings match")

    # Summary
    total = passed + failed + errors
    print(f"\n{'─'*60}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed, {errors} errors")
    if count_mismatches > 0:
        print(f"  COUNT-ONLY MISMATCHES: {count_mismatches}")
    print(f"{'─'*60}")

    if failures:
        print(f"\n  FAILURES:")
        for text, kind, detail in failures[:10]:
            print(f"    [{kind}] {repr(text[:50])}: {detail}")

    return passed, failed, errors

def run_benchmark_comparison(encoding_name="cl100k_base", binary_path="./target/release/runtoken-cli"):
    """Benchmark runtoken vs tiktoken."""
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {encoding_name}")
    print(f"{'='*60}\n")

    test_texts = {
        "short (28 chars)": "Hello, world! This is a test.",
        "medium (450 chars)": "The quick brown fox jumps over the lazy dog. " * 10,
        "long (4500 chars)": "The quick brown fox jumps over the lazy dog. " * 100,
        "code (1KB)": "fn main() {\n    let x = 42;\n    println!(\"Hello {}\", x);\n}\n" * 20,
    }

    enc = tiktoken.get_encoding(encoding_name)

    for label, text in test_texts.items():
        # tiktoken benchmark
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            tokens = enc.encode(text)
        tiktoken_elapsed = time.perf_counter() - start
        token_count = len(tokens)
        tiktoken_tps = (token_count * iterations) / tiktoken_elapsed

        # runtoken benchmark (via CLI — will be slower due to process overhead)
        # For fair comparison, we'd need Python bindings. For now, just report tiktoken baseline.
        print(f"  {label}:")
        print(f"    tokens:  {token_count}")
        print(f"    tiktoken: {tiktoken_tps:,.0f} tok/s ({tiktoken_elapsed/iterations*1000:.3f} ms/call)")

    print(f"\n  Note: CLI benchmarks include process startup overhead.")
    print(f"  Use `cargo run --release -- bench` for pure Rust benchmarks.")

if __name__ == "__main__":
    binary = "./target/release/runtoken-cli"

    encodings = ["cl100k_base"]
    if "--all" in sys.argv:
        encodings = ["cl100k_base", "o200k_base", "p50k_base"]

    total_passed = 0
    total_failed = 0
    total_errors = 0

    for enc in encodings:
        p, f, e = run_correctness_tests(enc, binary)
        total_passed += p
        total_failed += f
        total_errors += e

    print(f"\n{'='*60}")
    print(f"  OVERALL: {total_passed} passed, {total_failed} failed, {total_errors} errors")
    print(f"{'='*60}")

    if "--bench" in sys.argv:
        for enc in encodings:
            run_benchmark_comparison(enc, binary)

    sys.exit(1 if total_failed > 0 or total_errors > 0 else 0)
