use runtoken::Tokenizer;
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: runtoken-cli <command> [args]");
        eprintln!("Commands:");
        eprintln!("  encode <text> [encoding]   - Encode text to tokens");
        eprintln!("  count <text> [encoding]    - Count tokens");
        eprintln!("  bench [encoding]           - Run quick benchmark");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "encode" => {
            let text = args.get(2).map(|s| s.as_str()).unwrap_or("Hello, world!");
            let encoding = args.get(3).map(|s| s.as_str()).unwrap_or("cl100k_base");
            let tokenizer = Tokenizer::new(encoding).expect("Failed to load tokenizer");
            let tokens = tokenizer.encode(text);
            println!("Text: {:?}", text);
            println!("Encoding: {}", encoding);
            println!("Tokens: {:?}", tokens);
            println!("Count: {}", tokens.len());
        }
        "count" => {
            let text = args.get(2).map(|s| s.as_str()).unwrap_or("Hello, world!");
            let encoding = args.get(3).map(|s| s.as_str()).unwrap_or("cl100k_base");
            let tokenizer = Tokenizer::new(encoding).expect("Failed to load tokenizer");
            let count = tokenizer.count(text);
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

    let tokenizer = Tokenizer::new(encoding).expect("Failed to load tokenizer");

    let test_cases = [
        ("short", "Hello, world! This is a test."),
        ("medium", &"The quick brown fox jumps over the lazy dog. ".repeat(50)),
        ("code", &"fn main() {\n    let x = 42;\n    println!(\"Hello {}\", x);\n}\n".repeat(20)),
    ];

    for (label, text) in &test_cases {
        let text = text.as_ref();
        // Warmup
        for _ in 0..100 {
            let _ = tokenizer.count(text);
        }

        // Benchmark encode
        let iterations = 10000;
        let start = Instant::now();
        let mut total_tokens = 0;
        for _ in 0..iterations {
            total_tokens += tokenizer.encode(text).len();
        }
        let encode_elapsed = start.elapsed();
        let encode_tps = (total_tokens as f64) / encode_elapsed.as_secs_f64();

        // Benchmark count-only
        let start = Instant::now();
        let mut total_count = 0;
        for _ in 0..iterations {
            total_count += tokenizer.count(text);
        }
        let count_elapsed = start.elapsed();
        let count_tps = (total_count as f64) / count_elapsed.as_secs_f64();

        // Benchmark regex only
        let start = Instant::now();
        for _ in 0..iterations {
            let chunks = tokenizer.pattern.split(text);
            std::hint::black_box(chunks);
        }
        let regex_elapsed = start.elapsed();

        let tokens_per_call = total_tokens / iterations;
        println!(
            "{:>8}: {:>4} tokens | encode: {:>10.0} tok/s ({:.3}ms) | count: {:>10.0} tok/s ({:.3}ms) | regex: {:.3}ms ({:.0}%)",
            label,
            tokens_per_call,
            encode_tps,
            encode_elapsed.as_secs_f64() * 1000.0 / iterations as f64,
            count_tps,
            count_elapsed.as_secs_f64() * 1000.0 / iterations as f64,
            regex_elapsed.as_secs_f64() * 1000.0 / iterations as f64,
            regex_elapsed.as_secs_f64() / encode_elapsed.as_secs_f64() * 100.0,
        );
    }
}
