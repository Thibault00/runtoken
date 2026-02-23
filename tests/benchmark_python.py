#!/usr/bin/env python3
"""
Apples-to-apples benchmark: runtoken (Rust+PyO3) vs tiktoken (C+Python).
Both called as Python packages — no subprocess, no CLI overhead.
"""
import time
import tiktoken
import runtoken

def benchmark(name, func, text, warmup=100, iterations=10000):
    """Benchmark a function, return tokens/sec."""
    # Warmup
    for _ in range(warmup):
        func(text)
    
    # Measure
    start = time.perf_counter()
    total_tokens = 0
    for _ in range(iterations):
        result = func(text)
        total_tokens += len(result) if isinstance(result, list) else result
    elapsed = time.perf_counter() - start
    
    tokens_per_call = total_tokens / iterations
    tok_per_sec = total_tokens / elapsed
    ms_per_call = (elapsed / iterations) * 1000
    
    return tokens_per_call, tok_per_sec, ms_per_call

# Test texts
test_cases = {
    "short (29 chars)": "Hello, world! This is a test.",
    "medium (1050 chars)": "The quick brown fox jumps over the lazy dog. " * 50 + "In machine learning, language models process text tokens efficiently.",
    "code (1200 chars)": "fn main() {\n    let x = 42;\n    println!(\"Hello {}\", x);\n}\n" * 20,
    "long english (4500 chars)": "The quick brown fox jumps over the lazy dog. " * 100,
    "long code (5600 chars)": "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\n" * 40,
    "unicode (500 chars)": "こんにちは世界 café résumé 中文测试 🎉🎊🎈 " * 20,
}

encodings = ["cl100k_base", "o200k_base"]

for enc_name in encodings:
    tk_enc = tiktoken.get_encoding(enc_name)
    rt_enc = runtoken.get_encoding(enc_name)
    
    print(f"\n{'='*90}")
    print(f"  {enc_name}")
    print(f"{'='*90}")
    print(f"  {'Test':<25s} | {'Tokens':>6s} | {'tiktoken':>14s} | {'runtoken':>14s} | {'Speedup':>8s}")
    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*14}-+-{'-'*14}-+-{'-'*8}")
    
    for label, text in test_cases.items():
        # tiktoken encode
        tk_tpc, tk_tps, tk_ms = benchmark("tiktoken", tk_enc.encode, text)
        
        # runtoken encode
        rt_tpc, rt_tps, rt_ms = benchmark("runtoken", rt_enc.encode, text)
        
        speedup = rt_tps / tk_tps if tk_tps > 0 else 0
        winner = "⚡" if speedup > 1 else "⏱"
        
        print(f"  {label:<25s} | {int(tk_tpc):>6d} | {tk_tps:>11,.0f}/s | {rt_tps:>11,.0f}/s | {speedup:>5.2f}x {winner}")
    
    # Also benchmark count-only
    print(f"\n  Count-only (runtoken.count vs len(tiktoken.encode)):")
    print(f"  {'Test':<25s} | {'Tokens':>6s} | {'tiktoken':>14s} | {'runtoken':>14s} | {'Speedup':>8s}")
    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*14}-+-{'-'*14}-+-{'-'*8}")
    
    for label, text in test_cases.items():
        # tiktoken (has to encode to count)
        tk_tpc, tk_tps, tk_ms = benchmark("tiktoken", lambda t: tk_enc.encode(t), text)
        
        # runtoken count
        rt_tpc, rt_tps, rt_ms = benchmark("runtoken", lambda t: rt_enc.count(t), text, iterations=10000)
        
        speedup = rt_tps / tk_tps if tk_tps > 0 else 0
        winner = "⚡" if speedup > 1 else "⏱"
        
        print(f"  {label:<25s} | {int(tk_tpc):>6d} | {tk_tps:>11,.0f}/s | {rt_tps:>11,.0f}/s | {speedup:>5.2f}x {winner}")

# Final: verify correctness while we're at it
print(f"\n{'='*90}")
print(f"  CORRECTNESS CHECK")
print(f"{'='*90}")
mismatches = 0
total = 0
for enc_name in ["cl100k_base", "o200k_base", "p50k_base"]:
    tk_enc = tiktoken.get_encoding(enc_name)
    rt_enc = runtoken.get_encoding(enc_name)
    for label, text in test_cases.items():
        tk_ids = tk_enc.encode(text)
        rt_ids = rt_enc.encode(text)
        total += 1
        if tk_ids != rt_ids:
            mismatches += 1
            print(f"  ❌ {enc_name}/{label}: tiktoken={len(tk_ids)} runtoken={len(rt_ids)}")
        
print(f"  {total - mismatches}/{total} exact ID matches ({'🎉 PERFECT' if mismatches == 0 else '⚠️ ISSUES'})")
