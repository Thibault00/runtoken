use runtoken::Tokenizer;
use std::env;
use std::io::Read;
use std::time::Instant;

fn read_text(args: &[String], arg_idx: usize) -> String {
    // If arg is "-" or missing, read from stdin
    match args.get(arg_idx).map(|s| s.as_str()) {
        Some("-") | None => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).expect("Failed to read stdin");
            buf
        }
        Some(text) => text.to_string(),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: runtoken-cli <command> [args]");
        eprintln!("Commands:");
        eprintln!("  encode <text|-|> [encoding]  - Encode text to tokens (- for stdin)");
        eprintln!("  count <text|-|> [encoding]   - Count tokens (- for stdin)");
        eprintln!("  bench [encoding]             - Run quick benchmark");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "encode" => {
            let text = read_text(&args, 2);
            let encoding = args.get(3).map(|s| s.as_str()).unwrap_or("cl100k_base");
            let tokenizer = Tokenizer::new(encoding).expect("Failed to load tokenizer");
            let tokens = tokenizer.encode(&text);
            println!("Encoding: {}", encoding);
            println!("Tokens: {:?}", tokens);
            println!("Count: {}", tokens.len());
        }
        "count" => {
            let text = read_text(&args, 2);
            let encoding = args.get(3).map(|s| s.as_str()).unwrap_or("cl100k_base");
            let tokenizer = Tokenizer::new(encoding).expect("Failed to load tokenizer");
            let count = tokenizer.count(&text);
            println!("{}", count);
        }
        "bench" => {
            let encoding = args.get(2).map(|s| s.as_str()).unwrap_or("cl100k_base");
            run_benchmark(encoding);
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }
}

fn run_benchmark(encoding: &str) {
    println!("=== RunToken Benchmark ===");
    println!("Encoding: {}", encoding);
    println!();

    let test_cases = [
        ("short", "Hello, world! This is a test."),
        ("medium", &"The quick brown fox jumps over the lazy dog. ".repeat(50)),
        ("code", &"fn main() {\n    let x = 42;\n    println!(\"Hello {}\", x);\n}\n".repeat(20)),
    ];

    // --- CACHED BENCHMARK (same text repeated) ---
    println!("--- Cached (repeated text, measures cache + hash lookup) ---");
    {
        let tokenizer = Tokenizer::new(encoding).expect("Failed to load tokenizer");
        for (label, text) in &test_cases {
            let text = text.as_ref();
            // Warmup (populates cache)
            for _ in 0..100 {
                let _ = tokenizer.encode(text);
            }

            let iterations = 100_000;
            let start = Instant::now();
            let mut total_tokens = 0;
            for _ in 0..iterations {
                total_tokens += tokenizer.encode(text).len();
            }
            let elapsed = start.elapsed();
            let tps = (total_tokens as f64) / elapsed.as_secs_f64();
            let tokens_per_call = total_tokens / iterations;

            println!(
                "{:>8}: {:>4} tokens | {:>12.0} tok/s | {:.4}ms/call",
                label, tokens_per_call, tps,
                elapsed.as_secs_f64() * 1000.0 / iterations as f64,
            );
        }
    }

    println!();
    println!("--- Cold (fresh tokenizer, no cache, measures regex + BPE) ---");
    {
        for (label, text) in &test_cases {
            let text = text.as_ref();
            let iterations = 10000;
            let start = Instant::now();
            let mut total_tokens = 0;
            for _ in 0..iterations {
                // Fresh tokenizer each time = no cache
                let t = Tokenizer::new(encoding).expect("Failed to load tokenizer");
                total_tokens += t.encode(text).len();
            }
            let elapsed = start.elapsed();
            // Subtract tokenizer creation overhead
            let start2 = Instant::now();
            for _ in 0..iterations {
                let _t = Tokenizer::new(encoding).expect("Failed to load tokenizer");
            }
            let create_elapsed = start2.elapsed();
            let encode_only = elapsed.saturating_sub(create_elapsed);
            let tps = (total_tokens as f64) / encode_only.as_secs_f64();
            let tokens_per_call = total_tokens / iterations;

            println!(
                "{:>8}: {:>4} tokens | {:>12.0} tok/s | {:.3}ms/call",
                label, tokens_per_call, tps,
                encode_only.as_secs_f64() * 1000.0 / iterations as f64,
            );
        }
    }
}
