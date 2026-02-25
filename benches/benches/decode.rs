use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use crate::{load_model, TEST_PROMPTS};

fn decode_short(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");

    group.bench_function("32_tokens", |b| {
        b.iter(|| {
            let mut model = load_model();

            let result = model.generate(black_box("Write: "));
            black_box(result);
        });
    });

    group.bench_function("64_tokens", |b| {
        b.iter(|| {
            let mut model = load_model();

            let result = model.generate(black_box("Write a story: "));
            black_box(result);
        });
    });

    group.bench_function("128_tokens", |b| {
        b.iter(|| {
            let mut model = load_model();

            let result = model.generate(black_box("Tell me a tale: "));
            black_box(result);
        });
    });

    group.finish();
}

fn decode_with_context(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_with_context");

    let context_sizes = vec![256, 512, 1024, 2048];

    for size in context_sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _size| {
            b.iter(|| {
                let mut model = load_model();

                let prompt = " ".repeat(*size);
                let result = model.generate(black_box(&prompt));

                black_box(result);
            });
        });
    }

    group.finish();
}

fn decode_tokens_per_second(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_tps");

    group.bench_function("measure_tps", |b| {
        b.iter(|| {
            let mut model = load_model();

            let start = std::time::Instant::now();

            let result = model.generate(black_box("Count: 1, 2, 3, "));
            let elapsed = start.elapsed();

            let text = black_box(result).unwrap_or_default();
            let token_count = text.split_whitespace().count();

            let tps = token_count as f64 / elapsed.as_secs_f64();
            black_box(tps);
        });
    });

    group.finish();
}

criterion_group! {
    name = decode;
    config = Criterion::default().sample_size(10);
    targets = decode_short, decode_with_context, decode_tokens_per_second
}
criterion_main!(decode);
