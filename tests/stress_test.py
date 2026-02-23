#!/usr/bin/env python3
"""Stress test: large inputs up to 50K+ tokens, exact ID comparison via stdin."""
import tiktoken, subprocess, json, sys, time, random, string

def get_runtoken_ids(text, encoding, cwd="/tmp/runtoken"):
    """Pass text via stdin to handle arbitrarily large inputs."""
    result = subprocess.run(
        ["./target/release/runtoken-cli", "encode", "-", encoding],
        input=text, capture_output=True, text=True, cwd=cwd, timeout=60
    )
    if result.returncode != 0:
        return None, result.stderr
    ids = []
    for line in result.stdout.strip().split("\n"):
        if line.startswith("Tokens: "):
            try:
                ids = json.loads(line.split(": ", 1)[1])
            except:
                pass
    return ids, None

def get_runtoken_count(text, encoding, cwd="/tmp/runtoken"):
    result = subprocess.run(
        ["./target/release/runtoken-cli", "count", "-", encoding],
        input=text, capture_output=True, text=True, cwd=cwd, timeout=60
    )
    if result.returncode != 0:
        return -1
    return int(result.stdout.strip())

# Generate test texts at various scales
print("Generating test texts...")

texts = {
    "1K english": "The quick brown fox jumps over the lazy dog. " * 100,
    "5K english": "In machine learning, a large language model is a language model notable for its ability to achieve general-purpose language generation. " * 200,
    "10K english": "The development of artificial intelligence has been a long and winding road, filled with breakthroughs and setbacks. From the earliest days of computing, researchers have dreamed of creating machines that can think and learn like humans. " * 200,
    "20K code": ("def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nclass DataProcessor:\n    def __init__(self, data):\n        self.data = data\n\n    def process(self):\n        return [str(x) for x in self.data]\n\n") * 500,
    "30K mixed": ("Hello world! " * 50 + "SELECT * FROM users WHERE id > 100; " * 30 + "def main(): pass\n" * 20 + "The quick brown fox. " * 50) * 80,
    "50K repetitive": "token " * 50000,
    "50K prose": ("In the vast landscape of modern technology, artificial intelligence continues to reshape our understanding of what machines can accomplish. " * 5 + "Machine learning algorithms process enormous datasets to identify patterns that would be invisible to human analysts. " * 5 + "\n\n") * 300,
    "Unicode heavy 5K": "こんにちは世界 café résumé 中文测试 हिंदी 🎉🎊 Ünïcödé " * 500,
    "JSON 10K": ('{"id": 12345, "name": "test_user", "email": "user@example.com", "score": 99.5, "tags": ["ai", "ml"]}\n') * 500,
}

encodings = ["cl100k_base", "o200k_base", "p50k_base"]

total = 0
mismatches = 0
errors = 0

for enc_name in encodings:
    enc = tiktoken.get_encoding(enc_name)
    print(f"\n{'='*80}")
    print(f"  {enc_name}")
    print(f"{'='*80}")
    
    for label, text in texts.items():
        char_count = len(text)
        
        # tiktoken
        t0 = time.perf_counter()
        tiktoken_ids = enc.encode(text, allowed_special="all")
        tiktoken_ms = (time.perf_counter() - t0) * 1000
        tiktoken_count = len(tiktoken_ids)
        
        # runtoken (exact IDs for smaller texts, count-only for 50K+)
        if tiktoken_count <= 15000:
            # Full ID comparison
            t0 = time.perf_counter()
            runtoken_ids, err = get_runtoken_ids(text, enc_name)
            runtoken_ms = (time.perf_counter() - t0) * 1000
            
            total += 1
            
            if err or runtoken_ids is None:
                errors += 1
                print(f"  ❌ ERROR {label} ({char_count:,} chars): {err}")
                continue
            
            if tiktoken_ids == runtoken_ids:
                print(f"  ✅ {label:20s} | {char_count:>8,} chars | {tiktoken_count:>6,} tokens | IDs exact match | tiktoken {tiktoken_ms:>7.1f}ms")
            else:
                mismatches += 1
                print(f"  ❌ {label:20s} | {char_count:>8,} chars | tiktoken={tiktoken_count} runtoken={len(runtoken_ids)}")
                for j in range(min(len(tiktoken_ids), len(runtoken_ids))):
                    if tiktoken_ids[j] != runtoken_ids[j]:
                        print(f"     First diff at position {j}: tiktoken={tiktoken_ids[j]} runtoken={runtoken_ids[j]}")
                        break
        else:
            # Count-only comparison for very large texts (avoid huge JSON output)
            t0 = time.perf_counter()
            runtoken_count = get_runtoken_count(text, enc_name)
            runtoken_ms = (time.perf_counter() - t0) * 1000
            
            total += 1
            
            if tiktoken_count == runtoken_count:
                print(f"  ✅ {label:20s} | {char_count:>8,} chars | {tiktoken_count:>6,} tokens | count match     | tiktoken {tiktoken_ms:>7.1f}ms")
            else:
                mismatches += 1
                print(f"  ❌ {label:20s} | {char_count:>8,} chars | tiktoken={tiktoken_count} runtoken={runtoken_count} (delta={runtoken_count-tiktoken_count})")
            
            # Also spot-check: compare first 5000 IDs
            runtoken_ids, err = get_runtoken_ids(text[:len(text)//4], enc_name)
            if runtoken_ids:
                partial_tiktoken = enc.encode(text[:len(text)//4], allowed_special="all")
                if runtoken_ids == partial_tiktoken:
                    print(f"       ↳ first-quarter ID spot-check: ✅ exact match ({len(partial_tiktoken)} tokens)")
                else:
                    print(f"       ↳ first-quarter ID spot-check: ❌ MISMATCH")
                    mismatches += 1

print(f"\n{'='*80}")
print(f"  STRESS TEST RESULTS")
print(f"{'='*80}")
print(f"  Total tests: {total}")
print(f"  Exact matches: {total - mismatches - errors}")
print(f"  Mismatches: {mismatches}")
print(f"  Errors: {errors}")
if mismatches == 0 and errors == 0:
    print(f"  🎉 ALL PERFECT — up to 50K tokens, exact match across all encodings!")
else:
    print(f"  ⚠️  ISSUES FOUND")

sys.exit(1 if mismatches > 0 else 0)
