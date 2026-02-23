#!/usr/bin/env python3
"""PDF stress test: extract text from real PDFs and compare runtoken vs tiktoken."""
import tiktoken, subprocess, json, sys, time, os
from PyPDF2 import PdfReader

def get_runtoken_via_stdin(text, encoding, mode="encode"):
    result = subprocess.run(
        ["./target/release/runtoken-cli", mode, "-", encoding],
        input=text, capture_output=True, text=True, cwd="/tmp/runtoken", timeout=120
    )
    if result.returncode != 0:
        return None, -1, result.stderr
    
    if mode == "count":
        return None, int(result.stdout.strip()), None
    
    ids = []
    count = -1
    for line in result.stdout.strip().split("\n"):
        if line.startswith("Tokens: "):
            try:
                ids = json.loads(line.split(": ", 1)[1])
            except:
                pass
        elif line.startswith("Count: "):
            count = int(line.split(": ")[1])
    return ids, count, None

# Extract text from PDFs
pdf_dir = "/tmp/runtoken/test_data"
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

encodings = ["cl100k_base", "o200k_base", "p50k_base"]
total = 0
mismatches = 0
errors = 0

for pdf_file in sorted(pdf_files):
    pdf_path = os.path.join(pdf_dir, pdf_file)
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    except Exception as e:
        print(f"  ⚠️  Failed to read {pdf_file}: {e}")
        continue
    
    if not full_text.strip():
        print(f"  ⚠️  {pdf_file}: no extractable text")
        continue
    
    print(f"\n{'='*80}")
    print(f"  PDF: {pdf_file} ({len(full_text):,} chars extracted)")
    print(f"{'='*80}")
    
    # Test chunks of various sizes from the PDF
    chunks = {
        "first 1K chars": full_text[:1000],
        "first 5K chars": full_text[:5000],
        "first 20K chars": full_text[:20000],
        "full document": full_text,
        "middle section": full_text[len(full_text)//3 : len(full_text)//3 + 5000],
        "last 5K chars": full_text[-5000:],
    }
    
    for enc_name in encodings:
        enc = tiktoken.get_encoding(enc_name)
        print(f"\n  --- {enc_name} ---")
        
        for label, text in chunks.items():
            if not text.strip():
                continue
            
            # tiktoken
            tiktoken_ids = enc.encode(text, allowed_special="all")
            tiktoken_count = len(tiktoken_ids)
            
            total += 1
            
            if tiktoken_count <= 15000:
                # Full ID comparison
                runtoken_ids, runtoken_count, err = get_runtoken_via_stdin(text, enc_name, "encode")
                
                if err or runtoken_ids is None:
                    errors += 1
                    print(f"    ❌ ERROR {label}: {err[:200] if err else 'no output'}")
                    continue
                
                if tiktoken_ids == runtoken_ids:
                    print(f"    ✅ {label:20s} | {tiktoken_count:>6,} tokens | IDs exact match")
                else:
                    mismatches += 1
                    print(f"    ❌ {label:20s} | tiktoken={tiktoken_count} runtoken={len(runtoken_ids)}")
                    for j in range(min(len(tiktoken_ids), len(runtoken_ids))):
                        if tiktoken_ids[j] != runtoken_ids[j]:
                            # Show surrounding bytes for debugging
                            context_start = max(0, j - 3)
                            context_end = min(len(tiktoken_ids), j + 4)
                            print(f"       First diff at position {j}")
                            print(f"       tiktoken[{context_start}:{context_end}]: {tiktoken_ids[context_start:context_end]}")
                            print(f"       runtoken[{context_start}:{context_end}]: {runtoken_ids[context_start:context_end]}")
                            # Decode the differing tokens to see what text they represent
                            try:
                                tk_bytes = enc.decode_single_token_bytes(tiktoken_ids[j])
                                rt_bytes = enc.decode_single_token_bytes(runtoken_ids[j])
                                print(f"       tiktoken token {tiktoken_ids[j]} = {tk_bytes}")
                                print(f"       runtoken token {runtoken_ids[j]} = {rt_bytes}")
                            except:
                                pass
                            break
            else:
                # Count comparison for large texts
                _, runtoken_count, err = get_runtoken_via_stdin(text, enc_name, "count")
                
                if err:
                    errors += 1
                    print(f"    ❌ ERROR {label}: {err[:200]}")
                    continue
                
                if tiktoken_count == runtoken_count:
                    print(f"    ✅ {label:20s} | {tiktoken_count:>6,} tokens | count match")
                else:
                    mismatches += 1
                    print(f"    ❌ {label:20s} | tiktoken={tiktoken_count} runtoken={runtoken_count} (delta={runtoken_count - tiktoken_count})")

print(f"\n{'='*80}")
print(f"  PDF STRESS TEST RESULTS")
print(f"{'='*80}")
print(f"  PDFs tested: {len(pdf_files)}")
print(f"  Total comparisons: {total}")
print(f"  Exact matches: {total - mismatches - errors}")
print(f"  Mismatches: {mismatches}")
print(f"  Errors: {errors}")
if mismatches == 0 and errors == 0:
    print(f"  🎉 PERFECT — real PDF text matches tiktoken 100%!")
else:
    print(f"  ⚠️  ISSUES FOUND — investigate above")

sys.exit(1 if mismatches > 0 else 0)
