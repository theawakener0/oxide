use oxide_rs::{GenerateOptions, Model};
use std::path::PathBuf;
use std::time::{Duration, Instant};

pub const MODEL_PATH: &str =
    "/home/theawakener/Projects/OpenEye_00/models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf";

pub const TEST_PROMPTS: &[&str] = &[
    "Write a short story about a robot learning to paint.",
    "Explain quantum computing in simple terms.",
    "What are the benefits of exercise?",
    "Describe the water cycle.",
    "Write a haiku about the moon.",
];

pub const LONG_PROMPT: &str = r#"In the year 2157, humanity had spread across three solar systems. The colony ships had departed Earth decades ago, carrying thousands of hopeful settlers seeking new beginnings. On the red sands of Mars, the first dome cities had taken root, their translucent walls glowing soft amber against the rust-colored landscape. Children born on Mars had never seen Earth except in virtual reality simulations—they knew blue skies only as a memory preserved in digital archives. The journey to the outer planets took years, propelled by fusion engines that turned hydrogen into thrust. In the cold depths between worlds, generation ships drifted, their passengers aging and dying while their descendants dreamed of arrival. This is the story of one such ship, the Exodus, and the people who called it home."#;

pub fn create_model() -> Model {
    Model::new(MODEL_PATH).expect("Failed to create model")
}

pub fn load_model() -> Model {
    let mut model = Model::new(MODEL_PATH).expect("Failed to create model");
    model.load().expect("Failed to load model");
    model.warmup(64).expect("Failed to warmup");
    model
}

pub fn load_model_with_options(max_tokens: usize, temperature: f64) -> Model {
    let options = GenerateOptions {
        max_tokens,
        temperature,
        ..Default::default()
    };

    let mut model = Model::new(MODEL_PATH)
        .with_options(options)
        .expect("Failed to create model");
    model.load().expect("Failed to load model");
    model.warmup(64).expect("Failed to warmup");
    model
}

pub fn generate_with_timing(model: &mut Model, prompt: &str) -> (String, Duration) {
    let start = Instant::now();
    let result = model.generate(prompt);
    let elapsed = start.elapsed();

    let text = result.expect("Generation failed");
    (text, elapsed)
}
