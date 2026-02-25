use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use crate::{load_model, TEST_PROMPTS};

fn batch_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_single");

    group.bench_function("1_prompt_32_tokens", |b| {
        b.iter(|| {
            let mut model = load_model();

            let result = model.generate(black_box(TEST_PROMPTS[0]));
            black_box(result);
        });
    });

    group.finish();
}

fn batch_size_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size");

    let batch_sizes = vec![1, 2, 4, 8];

    for size in batch_sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, batch_size| {
            b.iter(|| {
                let mut model = load_model();

                let prompts: Vec<&str> = TEST_PROMPTS.iter().take(*batch_size).cloned().collect();

                let results: Vec<_> = prompts
                    .iter()
                    .map(|p| model.generate(black_box(*p)))
                    .collect();

                black_box(results);
            });
        });
    }

    group.finish();
}

fn batch_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_throughput");

    group.bench_function("8_prompts_sequential", |b| {
        b.iter(|| {
            let mut model = load_model();

            let mut total_tokens = 0;

            for prompt in TEST_PROMPTS.iter().take(8) {
                if let Ok(text) = model.generate(black_box(*prompt)) {
                    total_tokens += text.split_whitespace().count();
                }
            }

            black_box(total_tokens);
        });
    });

    group.finish();
}

fn batch_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_vs_sequential");

    let prompts: Vec<&str> = TEST_PROMPTS.iter().take(4).cloned().collect();

    group.bench_function("sequential_4", |b| {
        b.iter(|| {
            let mut model = load_model();

            for prompt in &prompts {
                let _ = model.generate(black_box(*prompt));
            }
        });
    });

    group.finish();
}

criterion_group! {
    name = batch;
    config = Criterion::default().sample_size(10);
    targets = batch_single, batch_size_impact, batch_throughput, batch_vs_sequential
}
criterion_main!(batch);
