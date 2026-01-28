use ferrite::lang::interpreter::Interpreter;
use ferrite::dynamics::{RuntimeRules, Device};
use ferrite::dynamics::allocator::TlsfAllocator;
use ferrite::compute::synth::Synthesizer;
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: semantic <script.semantic>");
        return;
    }

    let script_path = &args[1];
    let script = fs::read_to_string(script_path).expect("Failed to read script");

    println!("--- Semantic Runtime v0.1 ---");
    
    // 1. Physical Environment
    let cuda_dev = CudaDevice::new(0).expect("No GPU found");
    let pool_size = 512 * 1024 * 1024;
    let allocator = Arc::new(TlsfAllocator::new(Arc::clone(&cuda_dev), pool_size));
    let synth = Arc::new(Synthesizer::new(Arc::clone(&cuda_dev)));

    let rules = RuntimeRules {
        device: Device::Gpu(allocator),
        synthesizer: Some(synth),
        ..RuntimeRules::default()
    };

    // 2. Interpret Mind
    let mut interpreter = Interpreter::new();
    let mut program = interpreter.run(&script);

    // 3. Command Muscles
    println!("[Brain] Orchestrating script: {}", script_path);
    let start = std::time::Instant::now();
    program.execute(&rules);
    let duration = start.elapsed();

    println!("\n[Success] Program completed in {:?}", duration);
}
