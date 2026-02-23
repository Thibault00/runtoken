use criterion::{black_box, criterion_group, criterion_main, Criterion};
use runtoken::Tokenizer;

fn bench_encode(c: &mut Criterion) {
    let tokenizer = Tokenizer::new("cl100k_base").unwrap();

    let short = "Hello, world! This is a test.";
    let medium = "The quick brown fox jumps over the lazy dog. ".repeat(50);
    let code = "fn main() {\n    let x = 42;\n    println!(\"Hello {}\", x);\n}\n".repeat(20);

    c.bench_function("encode_short_cl100k", |b| {
        b.iter(|| tokenizer.encode(black_box(short)))
    });
    c.bench_function("encode_medium_cl100k", |b| {
        b.iter(|| tokenizer.encode(black_box(&medium)))
    });
    c.bench_function("encode_code_cl100k", |b| {
        b.iter(|| tokenizer.encode(black_box(&code)))
    });
}

fn bench_count(c: &mut Criterion) {
    let tokenizer = Tokenizer::new("cl100k_base").unwrap();

    let medium = "The quick brown fox jumps over the lazy dog. ".repeat(50);
    let code = "fn main() {\n    let x = 42;\n    println!(\"Hello {}\", x);\n}\n".repeat(20);

    c.bench_function("count_medium_cl100k", |b| {
        b.iter(|| tokenizer.count(black_box(&medium)))
    });
    c.bench_function("count_code_cl100k", |b| {
        b.iter(|| tokenizer.count(black_box(&code)))
    });
}

fn bench_o200k(c: &mut Criterion) {
    let tokenizer = Tokenizer::new("o200k_base").unwrap();

    let medium = "The quick brown fox jumps over the lazy dog. ".repeat(50);

    c.bench_function("encode_medium_o200k", |b| {
        b.iter(|| tokenizer.encode(black_box(&medium)))
    });
}

criterion_group!(benches, bench_encode, bench_count, bench_o200k);
criterion_main!(benches);
