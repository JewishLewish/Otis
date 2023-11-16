use onnxruntime::{Environment, GraphOptimizationLevel, SessionOptions, Tensor};

fn main() -> anyhow::Result<()> {
    // Initialize the ONNX Runtime environment
    let environment = Environment::builder()
        .with_name("onnxruntime-rust-sentiment-analysis")
        .with_log_level(onnxruntime::LoggingLevel::Verbose)
        .build()
        .expect("Failed to initialize the environment");

    // Load the ONNX model
    let model_path = "path/to/your/human_sentiment_model.safetensor";
    let session_options = SessionOptions::new().graph_optimization_level(GraphOptimizationLevel::Basic);
    let mut session = environment
        .new_session_builder(session_options)
        .expect("Failed to create session builder")
        .load_model(model_path)
        .expect("Failed to load model");

    // Example input data (replace with your actual input data)
    let input_data: Vec<f32> = ["test"];
    let input_tensor = Tensor::new_from_slice(&input_data, /* shape of your input tensor */)
        .expect("Failed to create input tensor");

    // Run the inference
    let outputs = session.run(vec![input_tensor]).expect("Failed to run inference");

    // Process the output tensor(s) as needed
    for output_tensor in outputs {
        let output_data: Vec<f32> = output_tensor
            .as_slice()
            .expect("Failed to retrieve output tensor data")
            .to_vec();

        // Process output data as needed
        println!("Output Data: {:?}", output_data);
    }

    Ok(())
}
