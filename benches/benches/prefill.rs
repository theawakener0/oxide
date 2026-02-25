use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

use crate::{MODEL_PATH, TEST_PROMPTS, LONG_PROMPT, load_model};

mod crate {
    pub use oxide_rs::*;
}

fn prefill_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefill");
    
    let test_cases = vec![
        ("short", "Hello, how are you?"),
        ("medium", "Write a story about a cat."),
        ("long", LONG_PROMPT),
    ];
    
    for (name, prompt) in test_cases {
        group.bench_with_input(BenchmarkId::from_parameter(name), prompt, |b, prompt| {
            b.iter(|| {
                let mut model = load_model();
                
                let result = model.generate(black_box(*prompt));
                black_box(result);
            });
        });
    }
    
    group.finish();
}

fn prefill_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefill_throughput");
    
    let prompts = vec![
        TEST_PROMPTS[0],
        TEST_PROMPTS[1],
        TEST_PROMPTS[2],
    ];
    
    group.bench_function("3_prompts_sequential", |b| {
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
    name = prefill;
    config = Criterion::default().sample_size(10);
    targets = prefill_benchmark, prefill_throughput
}
criterion_main!(prefill);
