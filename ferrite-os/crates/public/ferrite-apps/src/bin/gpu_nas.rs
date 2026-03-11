//! gpu_nas — GPU-Native Evolutionary Neural Architecture Search
//!
//! Breeds neural network architectures via genetic algorithm entirely on GPU.
//! Population of 128 variable-size networks, parallel cuBLAS evaluation, tournament
//! selection, crossover/mutation with dynamic TLSF alloc/free, VMM paging under
//! pressure, VFS-checkpointed Pareto frontier, circuit breaker for OOM protection.
//!
//! OS primitives exercised: TLSF (variable-size alloc/free every generation),
//! cuBLAS SGEMM (parallel forward passes), VFS (Pareto checkpointing), VMM (pressure
//! paging), SHM (fitness leaderboard), CircuitBreaker + RetryPolicy (OOM protection),
//! streams (parallel evaluation), watchdog/keepalive, tensor ops (relu/gelu/silu/
//! sigmoid/tanh/softmax/fill/sub/sqr/reduce_mean).

use std::time::{Duration, Instant};

use anyhow::Result;
use ptx_runtime::resilience::{CircuitBreaker, CircuitState, RetryPolicy};
use ptx_runtime::{GemmOp, GpuPtr, PtxRuntime};
use rand::Rng;

use ferrite_apps::platform;

// ============================================================================
// Constants
// ============================================================================

const POOL_FRACTION: f32 = 0.55;
const MAX_STREAMS: u32 = 256;
const POP_SIZE: usize = 128;
const BATCH_SIZE: usize = 512;
const INPUT_DIM: usize = 64;
const OUTPUT_DIM: usize = 10;
const MIN_HIDDEN: usize = 2;
const MAX_HIDDEN: usize = 5;
const WIDTHS: [usize; 5] = [64, 128, 256, 512, 1024];
const ELITE_FRAC: f32 = 0.10;
const CROSSOVER_FRAC: f32 = 0.60;
const MUTATION_FRAC: f32 = 0.20;
// RANDOM_FRAC = 0.10 (remainder)
const CHECKPOINT_INTERVAL_SECS: u64 = 60;

// Mutation probabilities
const P_ADD_LAYER: f32 = 0.15;
const P_REMOVE_LAYER: f32 = 0.15;
const P_CHANGE_WIDTH: f32 = 0.30;
const P_CHANGE_ACTIVATION: f32 = 0.20;

// ============================================================================
// Data structures
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
enum Activation {
    Relu,
    Sigmoid,
    Gelu,
    Silu,
    Tanh,
}

impl Activation {
    fn random(rng: &mut impl Rng) -> Self {
        match rng.gen_range(0..5) {
            0 => Activation::Relu,
            1 => Activation::Sigmoid,
            2 => Activation::Gelu,
            3 => Activation::Silu,
            _ => Activation::Tanh,
        }
    }

    fn apply(&self, ptr: *mut f32, n: usize, stream: ptx_sys::cudaStream_t) {
        unsafe {
            match self {
                Activation::Relu => ptx_sys::ptx_tensor_relu_f32(ptr, ptr, n, stream),
                Activation::Sigmoid => ptx_sys::ptx_tensor_sigmoid_f32(ptr, ptr, n, stream),
                Activation::Gelu => ptx_sys::ptx_tensor_gelu_f32(ptr, ptr, n, stream),
                Activation::Silu => ptx_sys::ptx_tensor_silu_f32(ptr, ptr, n, stream),
                Activation::Tanh => ptx_sys::ptx_tensor_tanh_f32(ptr, ptr, n, stream),
            }
        }
    }

    fn code(self) -> f32 {
        match self {
            Activation::Relu => 0.0,
            Activation::Sigmoid => 1.0,
            Activation::Gelu => 2.0,
            Activation::Silu => 3.0,
            Activation::Tanh => 4.0,
        }
    }
}

#[derive(Clone, Debug)]
struct LayerGene {
    in_dim: usize,
    out_dim: usize,
    activation: Activation,
}

#[derive(Clone, Debug)]
struct Genome {
    layers: Vec<LayerGene>,
    id: u64,
    generation: u32,
    parent_ids: (Option<u64>, Option<u64>),
}

#[allow(dead_code)]
impl Genome {
    fn random(rng: &mut impl Rng, id: u64) -> Self {
        let num_hidden = rng.gen_range(MIN_HIDDEN..=MAX_HIDDEN);
        let mut layers = Vec::with_capacity(num_hidden + 1);
        let mut prev_dim = INPUT_DIM;

        for _ in 0..num_hidden {
            let out_dim = WIDTHS[rng.gen_range(0..WIDTHS.len())];
            let activation = Activation::random(rng);
            layers.push(LayerGene { in_dim: prev_dim, out_dim, activation });
            prev_dim = out_dim;
        }

        // Output layer: always softmax (applied separately), use Relu as placeholder
        layers.push(LayerGene {
            in_dim: prev_dim,
            out_dim: OUTPUT_DIM,
            activation: Activation::Relu, // placeholder — softmax applied after
        });

        Genome { layers, id, generation: 0, parent_ids: (None, None) }
    }

    fn total_weight_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.in_dim * l.out_dim * 4).sum()
    }

    fn total_output_bytes(&self) -> usize {
        self.layers.iter().map(|l| BATCH_SIZE * l.out_dim * 4).sum()
    }

    fn total_gpu_bytes(&self) -> usize {
        self.total_weight_bytes() + self.total_output_bytes()
    }

    fn hidden_count(&self) -> usize {
        if self.layers.is_empty() { 0 } else { self.layers.len() - 1 }
    }

    fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.in_dim * l.out_dim).sum()
    }

    fn describe(&self) -> String {
        let dims: Vec<String> = std::iter::once(INPUT_DIM.to_string())
            .chain(self.layers.iter().map(|l| l.out_dim.to_string()))
            .collect();
        format!("[{}]", dims.join("→"))
    }

    fn fix_dimensions(&mut self) {
        let mut prev_dim = INPUT_DIM;
        for layer in &mut self.layers {
            layer.in_dim = prev_dim;
            prev_dim = layer.out_dim;
        }
        // Force output layer
        if let Some(last) = self.layers.last_mut() {
            last.out_dim = OUTPUT_DIM;
        }
    }
}

struct Individual {
    genome: Genome,
    weights: Vec<GpuPtr>,
    outputs: Vec<GpuPtr>,
    fitness: f32,
    param_count: usize,
    alive: bool,
}

#[derive(Clone)]
struct ParetoEntry {
    genome: Genome,
    fitness: f32,
    param_count: usize,
}

// ============================================================================
// Materialization: genome → Individual with GPU allocations
// ============================================================================

/// Pre-generated random pool for fast weight initialization.
/// Generated once at startup, then slices are copied to GPU + scaled per-layer.
struct RandomPool {
    data: Vec<f32>,
}

impl RandomPool {
    fn new(rng: &mut impl Rng, n_floats: usize) -> Self {
        let mut data = vec![0.0f32; n_floats];
        for v in &mut data {
            *v = rng.gen::<f32>() * 2.0 - 1.0; // Uniform [-1, +1]
        }
        RandomPool { data }
    }

    fn init_weights(
        &self,
        dst: *mut f32,
        n_weights: usize,
        weight_bytes: usize,
        xavier_scale: f32,
        rng: &mut impl Rng,
        stream: ptx_sys::cudaStream_t,
    ) {
        // Pick a random offset into the pool, wrapping if needed
        let max_offset = self.data.len().saturating_sub(n_weights);
        let offset = if max_offset > 0 { rng.gen_range(0..max_offset) } else { 0 };
        unsafe {
            ptx_sys::cudaMemcpy(
                dst as *mut libc::c_void,
                self.data[offset..].as_ptr() as *const libc::c_void,
                weight_bytes,
                ptx_sys::cudaMemcpyHostToDevice,
            );
            // Scale to Xavier range on GPU
            ptx_sys::ptx_tensor_mul_scalar_f32(
                dst, xavier_scale, dst, n_weights, stream,
            );
        }
    }
}

fn materialize(
    genome: Genome,
    rt: &PtxRuntime,
    circuit_breaker: &CircuitBreaker,
    retry_policy: &RetryPolicy,
    rng: &mut impl Rng,
    random_pool: &RandomPool,
) -> Option<Individual> {
    let mut weights = Vec::with_capacity(genome.layers.len());
    let mut outputs = Vec::with_capacity(genome.layers.len());
    let param_count = genome.param_count();

    for layer in &genome.layers {
        let weight_bytes = layer.in_dim * layer.out_dim * 4;
        let output_bytes = BATCH_SIZE * layer.out_dim * 4;

        let w = match retry_policy.execute(|| {
            if !circuit_breaker.allow_request() {
                return Err(anyhow::anyhow!("Circuit breaker open"));
            }
            match rt.alloc(weight_bytes) {
                Ok(ptr) => { circuit_breaker.record_success(); Ok(ptr) }
                Err(e) => { circuit_breaker.record_failure(); Err(anyhow::anyhow!("{}", e)) }
            }
        }) {
            Ok(ptr) => ptr,
            Err(_) => return None,
        };

        let o = match rt.alloc(output_bytes) {
            Ok(ptr) => ptr,
            Err(_) => return None,
        };

        // Xavier init: copy random slice from pool + GPU-side scale
        let n_weights = layer.in_dim * layer.out_dim;
        let xavier_scale = (2.0_f32 / (layer.in_dim + layer.out_dim) as f32).sqrt();
        let stream = rt.next_stream();
        random_pool.init_weights(
            w.as_ptr_typed::<f32>(), n_weights, weight_bytes,
            xavier_scale, rng, stream.raw(),
        );
        o.zero().ok();

        weights.push(w);
        outputs.push(o);
    }

    Some(Individual {
        genome,
        weights,
        outputs,
        fitness: 0.0,
        param_count,
        alive: true,
    })
}

// ============================================================================
// Fitness evaluation: parallel cuBLAS forward pass
// ============================================================================

fn evaluate_population(
    pop: &mut [Individual],
    input: &GpuPtr,
    target: &GpuPtr,
    fitness_buf: &GpuPtr,
    rt: &PtxRuntime,
    cublas: &ptx_runtime::CublasHandle,
    gemm_count: &mut u64,
) {
    for (i, ind) in pop.iter_mut().enumerate() {
        if !ind.alive { continue; }

        let stream = rt.stream((i % MAX_STREAMS as usize) as i32)
            .expect("stream ID within pool bounds");

        // Forward pass: SGEMM per layer + activation
        let mut prev_ptr = input.as_ptr_typed::<f32>() as *const f32;
        let mut prev_cols = INPUT_DIM;

        for (li, layer) in ind.genome.layers.iter().enumerate() {
            // Set cuBLAS to use this individual's stream
            let _ = cublas.set_stream(&stream);

            // SGEMM: output[BATCH x out] = input[BATCH x in] * weights[in x out]
            // Row-major via col-major trick
            unsafe {
                let _ = cublas.sgemm(
                    GemmOp::None,
                    GemmOp::None,
                    layer.out_dim as i32,
                    BATCH_SIZE as i32,
                    prev_cols as i32,
                    1.0,
                    ind.weights[li].as_ptr_typed::<f32>(),
                    layer.out_dim as i32,
                    prev_ptr,
                    prev_cols as i32,
                    0.0,
                    ind.outputs[li].as_ptr_typed::<f32>(),
                    layer.out_dim as i32,
                );
            }
            *gemm_count += 1;

            let out_size = BATCH_SIZE * layer.out_dim;

            if li < ind.genome.layers.len() - 1 {
                // Hidden layer: apply activation from gene
                layer.activation.apply(
                    ind.outputs[li].as_ptr_typed::<f32>(),
                    out_size,
                    stream.raw(),
                );
            } else {
                // Output layer: softmax
                unsafe {
                    ptx_sys::ptx_tensor_softmax_f32(
                        ind.outputs[li].as_ptr_typed::<f32>(),
                        ind.outputs[li].as_ptr_typed::<f32>(),
                        BATCH_SIZE,
                        OUTPUT_DIM,
                        stream.raw(),
                    );
                }
            }

            prev_ptr = ind.outputs[li].as_ptr_typed::<f32>() as *const f32;
            prev_cols = layer.out_dim;
        }

        // Loss: sub(pred, target) → sqr → reduce_mean → fitness_buf[i]
        // All in-place on the individual's own output buffer (no shared temp = no race)
        let final_idx = ind.genome.layers.len() - 1;
        let final_size = BATCH_SIZE * OUTPUT_DIM;
        let out_ptr = ind.outputs[final_idx].as_ptr_typed::<f32>();
        let fitness_offset = unsafe {
            (fitness_buf.as_ptr_typed::<f32>()).add(i)
        };

        unsafe {
            // sub in-place: output = output - target
            ptx_sys::ptx_tensor_sub_f32(
                out_ptr,
                target.as_ptr_typed::<f32>(),
                out_ptr,
                final_size,
                stream.raw(),
            );

            // sqr in-place: output = output^2
            ptx_sys::ptx_tensor_sqr_f32(
                out_ptr,
                out_ptr,
                final_size,
                stream.raw(),
            );

            // reduce_mean → single f32 at fitness_offset
            ptx_sys::ptx_tensor_reduce_mean_f32(
                out_ptr,
                fitness_offset,
                1, final_size, 1,
                stream.raw(),
            );
        }
    }

    // Sync all streams, then bulk-read fitness values
    let _ = rt.sync_all();

    let mut fitness_host = vec![0.0f32; POP_SIZE];
    unsafe {
        fitness_buf.copy_to_host(
            fitness_host.as_mut_ptr() as *mut libc::c_void,
            POP_SIZE * 4,
        ).ok();
    }

    for (i, ind) in pop.iter_mut().enumerate() {
        if !ind.alive { continue; }
        let mse = fitness_host[i];
        ind.fitness = 1.0 / (1.0 + mse);
    }
}

// ============================================================================
// Pareto frontier
// ============================================================================

fn update_pareto(pareto: &mut Vec<ParetoEntry>, pop: &[Individual]) {
    for ind in pop {
        if !ind.alive { continue; }

        // Skip if an existing entry has identical objective values
        let duplicate = pareto.iter().any(|p| {
            (p.fitness - ind.fitness).abs() < 1e-6 && p.param_count == ind.param_count
        });
        if duplicate { continue; }

        let dominated = pareto.iter().any(|p| {
            p.fitness >= ind.fitness && p.param_count <= ind.param_count
                && (p.fitness > ind.fitness || p.param_count < ind.param_count)
        });
        if dominated { continue; }

        // Remove entries dominated by this individual
        pareto.retain(|p| {
            !(ind.fitness >= p.fitness && ind.param_count <= p.param_count
                && (ind.fitness > p.fitness || ind.param_count < p.param_count))
        });

        pareto.push(ParetoEntry {
            genome: ind.genome.clone(),
            fitness: ind.fitness,
            param_count: ind.param_count,
        });
    }
}

// ============================================================================
// Selection + Breeding
// ============================================================================

fn tournament_select<'a>(pop: &'a [Individual], rng: &mut impl Rng, k: usize) -> &'a Genome {
    let alive: Vec<usize> = pop.iter().enumerate()
        .filter(|(_, ind)| ind.alive)
        .map(|(i, _)| i)
        .collect();
    if alive.is_empty() {
        // Fallback: pick any
        return &pop[0].genome;
    }
    let mut best_idx = alive[rng.gen_range(0..alive.len())];
    for _ in 1..k {
        let idx = alive[rng.gen_range(0..alive.len())];
        if pop[idx].fitness > pop[best_idx].fitness {
            best_idx = idx;
        }
    }
    &pop[best_idx].genome
}

fn crossover(a: &Genome, b: &Genome, rng: &mut impl Rng, id: u64, gen: u32) -> Genome {
    // Single-point layer splice
    let cut_a = rng.gen_range(1..=a.layers.len().max(1));
    let cut_b = rng.gen_range(0..b.layers.len().max(1));

    let mut layers: Vec<LayerGene> = Vec::new();
    for i in 0..cut_a.min(a.layers.len()) {
        layers.push(a.layers[i].clone());
    }
    for i in cut_b..b.layers.len() {
        layers.push(b.layers[i].clone());
    }

    // Fix splice point dimension mismatch
    if layers.len() > 1 && cut_a < layers.len() {
        layers[cut_a].in_dim = layers[cut_a - 1].out_dim;
    }

    // Clamp hidden count
    let total = layers.len();
    if total > MAX_HIDDEN + 1 {
        layers.truncate(MAX_HIDDEN + 1);
    }
    if layers.is_empty() {
        layers.push(LayerGene {
            in_dim: INPUT_DIM,
            out_dim: OUTPUT_DIM,
            activation: Activation::Relu,
        });
    }

    let mut genome = Genome {
        layers,
        id,
        generation: gen,
        parent_ids: (Some(a.id), Some(b.id)),
    };
    genome.fix_dimensions();
    genome
}

fn mutate(genome: &mut Genome, rng: &mut impl Rng) {
    // Add layer
    if rng.gen::<f32>() < P_ADD_LAYER && genome.hidden_count() < MAX_HIDDEN {
        let pos = rng.gen_range(0..genome.layers.len());
        let width = WIDTHS[rng.gen_range(0..WIDTHS.len())];
        let activation = Activation::random(rng);
        genome.layers.insert(pos, LayerGene {
            in_dim: 0, out_dim: width, activation,
        });
    }

    // Remove layer
    if rng.gen::<f32>() < P_REMOVE_LAYER && genome.hidden_count() > MIN_HIDDEN {
        let pos = rng.gen_range(0..genome.hidden_count());
        genome.layers.remove(pos);
    }

    // Change width of a hidden layer
    if rng.gen::<f32>() < P_CHANGE_WIDTH && genome.hidden_count() > 0 {
        let pos = rng.gen_range(0..genome.hidden_count());
        genome.layers[pos].out_dim = WIDTHS[rng.gen_range(0..WIDTHS.len())];
    }

    // Change activation of a hidden layer
    if rng.gen::<f32>() < P_CHANGE_ACTIVATION && genome.hidden_count() > 0 {
        let pos = rng.gen_range(0..genome.hidden_count());
        genome.layers[pos].activation = Activation::random(rng);
    }

    genome.fix_dimensions();
}

fn breed_next_generation(
    pop: &[Individual],
    rng: &mut impl Rng,
    next_id: &mut u64,
    generation: u32,
) -> Vec<Genome> {
    let n_elite = (POP_SIZE as f32 * ELITE_FRAC) as usize;
    let n_crossover = (POP_SIZE as f32 * CROSSOVER_FRAC) as usize;
    let n_mutation = (POP_SIZE as f32 * MUTATION_FRAC) as usize;
    let n_random = POP_SIZE - n_elite - n_crossover - n_mutation;

    // Sort by fitness descending
    let mut indices: Vec<usize> = (0..pop.len()).collect();
    indices.sort_by(|&a, &b| pop[b].fitness.partial_cmp(&pop[a].fitness).unwrap_or(std::cmp::Ordering::Equal));

    let mut new_genomes: Vec<Genome> = Vec::with_capacity(POP_SIZE);

    // Elites: best survive
    for &i in indices.iter().take(n_elite) {
        let mut g = pop[i].genome.clone();
        g.id = *next_id;
        *next_id += 1;
        g.generation = generation;
        g.parent_ids = (Some(pop[i].genome.id), None);
        new_genomes.push(g);
    }

    // Crossover
    for _ in 0..n_crossover {
        let a = tournament_select(pop, rng, 3);
        let b = tournament_select(pop, rng, 3);
        let mut child = crossover(a, b, rng, *next_id, generation);
        *next_id += 1;
        mutate(&mut child, rng);
        new_genomes.push(child);
    }

    // Mutation only
    for _ in 0..n_mutation {
        let parent = tournament_select(pop, rng, 3);
        let mut child = parent.clone();
        child.id = *next_id;
        *next_id += 1;
        child.generation = generation;
        child.parent_ids = (Some(parent.id), None);
        mutate(&mut child, rng);
        new_genomes.push(child);
    }

    // Random immigrants
    for _ in 0..n_random {
        let g = Genome::random(rng, *next_id);
        *next_id += 1;
        new_genomes.push(Genome { generation, ..g });
    }

    new_genomes
}

// ============================================================================
// VMM pressure management
// ============================================================================

fn vmm_pressure_check(
    pop: &mut [Individual],
    rt: &PtxRuntime,
    vmm: *mut ptx_sys::VMMState,
) {
    let tlsf = rt.tlsf_stats();
    if tlsf.utilization_percent <= 80.0 { return; }

    // Sort alive individuals by fitness ascending (worst first)
    let mut alive_indices: Vec<usize> = pop.iter().enumerate()
        .filter(|(_, ind)| ind.alive)
        .map(|(i, _)| i)
        .collect();
    alive_indices.sort_by(|&a, &b| {
        pop[a].fitness.partial_cmp(&pop[b].fitness).unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_to_page = alive_indices.len() / 4;
    for &i in alive_indices.iter().take(n_to_page) {
        // Track via VMM
        unsafe {
            let page = ptx_sys::vmm_alloc_page(vmm, ptx_sys::VMM_FLAG_READ | ptx_sys::VMM_FLAG_WRITE);
            if !page.is_null() {
                ptx_sys::vmm_swap_out(vmm, page);
                ptx_sys::vmm_free_page(vmm, page);
            }
        }
        // Clear GPU allocations (RAII frees TLSF)
        pop[i].weights.clear();
        pop[i].outputs.clear();
        pop[i].alive = false;
    }
}

// ============================================================================
// VFS checkpointing
// ============================================================================

unsafe fn vfs_checkpoint_pareto(
    vfs: *mut ptx_sys::VFSState,
    pareto: &[ParetoEntry],
    generation: u32,
    vfs_pareto_count: &mut usize,
) {
    // Write generation marker
    let gen_path = format!("/nas/checkpoints/gen_{:05}", generation);
    let gen_shape = [1i32];
    let _ = platform::vfs_safe_create_tensor(vfs, &gen_path, &gen_shape, 0);
    let _ = platform::vfs_safe_sync_tensor(vfs, &gen_path);

    // Write Pareto entries (overwrite old ones)
    // First unlink old entries
    for i in 0..*vfs_pareto_count {
        let path = format!("/nas/pareto/entry_{}", i);
        let fit_path = format!("/nas/pareto/entry_{}_fit", i);
        let _ = platform::vfs_safe_unlink(vfs, &path);
        let _ = platform::vfs_safe_unlink(vfs, &fit_path);
    }

    // Write new entries (limit to avoid VFS node exhaustion)
    let to_write = pareto.len().min(50);
    for i in 0..to_write {
        let entry = &pareto[i];

        // Encode genome as tensor: [num_layers, (in_dim, out_dim, activation) per layer]
        let n_layers = entry.genome.layers.len();
        let tensor_len = 1 + n_layers * 3;
        let shape = [tensor_len as i32];

        let path = format!("/nas/pareto/entry_{}", i);
        let _ = platform::vfs_safe_create_tensor(vfs, &path, &shape, 0);
        if let Ok(mapped) = platform::vfs_safe_mmap_tensor(vfs, &path) {
            let mut encoded: Vec<f32> = Vec::with_capacity(tensor_len);
            encoded.push(n_layers as f32);
            for layer in &entry.genome.layers {
                encoded.push(layer.in_dim as f32);
                encoded.push(layer.out_dim as f32);
                encoded.push(layer.activation.code());
            }
            let rc = ptx_sys::cudaMemcpy(
                mapped,
                encoded.as_ptr() as *const libc::c_void,
                encoded.len() * 4,
                ptx_sys::cudaMemcpyHostToDevice,
            );
            if rc != ptx_sys::cudaSuccess {
                eprintln!("WARNING: failed to write Pareto genome {} (cuda err={})", i, rc);
            }
        }
        let _ = platform::vfs_safe_sync_tensor(vfs, &path);

        // Fitness tensor
        let fit_path = format!("/nas/pareto/entry_{}_fit", i);
        let fit_shape = [1i32];
        let _ = platform::vfs_safe_create_tensor(vfs, &fit_path, &fit_shape, 0);
        if let Ok(mapped_fit) = platform::vfs_safe_mmap_tensor(vfs, &fit_path) {
            let fit_val = entry.fitness;
            let rc = ptx_sys::cudaMemcpy(
                mapped_fit,
                &fit_val as *const f32 as *const libc::c_void,
                4,
                ptx_sys::cudaMemcpyHostToDevice,
            );
            if rc != ptx_sys::cudaSuccess {
                eprintln!("WARNING: failed to write Pareto fitness {} (cuda err={})", i, rc);
            }
        }
        let _ = platform::vfs_safe_sync_tensor(vfs, &fit_path);
    }

    *vfs_pareto_count = to_write;
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let duration_secs = platform::get_duration_secs();
    println!("=== GPU NAS ===");
    println!("GPU-native evolutionary neural architecture search");
    println!("Duration: {}", platform::format_duration(duration_secs));
    println!("Population: {}, Batch: {}, Search: {}→[{}-{} hidden]→{}",
        POP_SIZE, BATCH_SIZE, INPUT_DIM, MIN_HIDDEN, MAX_HIDDEN, OUTPUT_DIM);
    println!("Config: pool_fraction={}, max_streams={}", POOL_FRACTION, MAX_STREAMS);
    println!();

    let rt = platform::init_runtime(POOL_FRACTION, MAX_STREAMS)?;
    let mut reporter = platform::TelemetryReporter::new("gpu_nas", 10);

    // Resilience
    let circuit_breaker = CircuitBreaker::new(5, 3, Duration::from_secs(10));
    let retry_policy = RetryPolicy {
        max_attempts: 3,
        initial_delay: Duration::from_millis(10),
        max_delay: Duration::from_millis(100),
        backoff_multiplier: 2.0,
    };

    // VFS
    let vfs = unsafe { platform::vfs_safe_init(&rt)? };
    unsafe {
        platform::vfs_safe_mkdir(vfs, "/nas")?;
        platform::vfs_safe_mkdir(vfs, "/nas/checkpoints")?;
        platform::vfs_safe_mkdir(vfs, "/nas/pareto")?;
    }

    // VMM
    let vmm = unsafe { platform::vmm_safe_init(&rt, 32 * 1024 * 1024)? }; // 32 MB swap

    // SHM leaderboard: top POP_SIZE fitnesses
    let shm_size = POP_SIZE * 4;
    let shm_ptr = unsafe { platform::shm_safe_alloc(&rt, "nas_leaderboard", shm_size)? };
    unsafe { ptx_sys::cudaMemset(shm_ptr, 0, shm_size); }

    // cuBLAS
    let cublas_guard = rt.cublas()?;
    let cublas = cublas_guard.as_ref()
        .ok_or_else(|| anyhow::anyhow!("cuBLAS not available"))?;

    // Allocate shared buffers
    // Input: BATCH_SIZE x INPUT_DIM — varied per sample for selective pressure
    let input_bytes = BATCH_SIZE * INPUT_DIM * 4;
    let input = rt.alloc(input_bytes)?;
    {
        let mut input_host = vec![0.0f32; BATCH_SIZE * INPUT_DIM];
        for s in 0..BATCH_SIZE {
            for j in 0..INPUT_DIM {
                // Each sample gets a distinct input pattern based on its class
                let class = s % OUTPUT_DIM;
                input_host[s * INPUT_DIM + j] =
                    ((class as f32 + 1.0) * (j as f32 + 1.0) * 0.01).sin() * 0.5 + 0.5;
            }
        }
        unsafe {
            ptx_sys::cudaMemcpy(
                input.as_ptr_typed::<f32>() as *mut libc::c_void,
                input_host.as_ptr() as *const libc::c_void,
                input_bytes,
                ptx_sys::cudaMemcpyHostToDevice,
            );
        }
    }

    // Target: BATCH_SIZE x OUTPUT_DIM — one-hot per sample's class
    let target_bytes = BATCH_SIZE * OUTPUT_DIM * 4;
    let target = rt.alloc(target_bytes)?;
    {
        let mut target_host = vec![0.0f32; BATCH_SIZE * OUTPUT_DIM];
        for s in 0..BATCH_SIZE {
            let class = s % OUTPUT_DIM;
            for c in 0..OUTPUT_DIM {
                target_host[s * OUTPUT_DIM + c] = if c == class { 0.9 } else { 0.01 };
            }
        }
        unsafe {
            ptx_sys::cudaMemcpy(
                target.as_ptr_typed::<f32>() as *mut libc::c_void,
                target_host.as_ptr() as *const libc::c_void,
                target_bytes,
                ptx_sys::cudaMemcpyHostToDevice,
            );
        }
    }

    // Fitness buffer: POP_SIZE f32 (each individual writes its MSE here)
    let fitness_buf = rt.alloc(POP_SIZE * 4)?;

    let mut rng = rand::thread_rng();
    let mut next_id: u64 = 0;
    let mut generation: u32 = 0;
    let mut gemm_count: u64 = 0;
    let mut pareto: Vec<ParetoEntry> = Vec::new();
    let mut vfs_pareto_count: usize = 0;
    let mut vfs_gen_count: u32 = 0;
    let mut last_checkpoint = Instant::now();

    // Pre-generate random weight pool (2M floats = 8 MB, reused every generation)
    let random_pool = RandomPool::new(&mut rng, 2 * 1024 * 1024);
    println!("Random weight pool: {} pre-generated", platform::format_bytes(random_pool.data.len() * 4));

    // Initialize population
    println!("Initializing population of {} individuals...", POP_SIZE);
    let mut pop: Vec<Individual> = Vec::with_capacity(POP_SIZE);
    let mut materialized = 0;
    for _ in 0..POP_SIZE {
        let genome = Genome::random(&mut rng, next_id);
        next_id += 1;
        if let Some(ind) = materialize(genome, &rt, &circuit_breaker, &retry_policy, &mut rng, &random_pool) {
            materialized += 1;
            pop.push(ind);
        }
    }
    println!("  Materialized: {}/{}", materialized, POP_SIZE);

    if pop.is_empty() {
        println!("ERROR: Could not materialize any individuals. Exiting.");
        // Cleanup
        drop(fitness_buf);
        drop(target);
        drop(input);
        drop(cublas_guard);
        unsafe {
            platform::shm_safe_unlink(&rt, "nas_leaderboard", shm_ptr)?;
            ptx_sys::vmm_shutdown(vmm);
            ptx_sys::vfs_shutdown(vfs);
        }
        rt.sync_all()?;
        platform::assert_clean_exit(&rt);
        return Ok(());
    }

    let start = Instant::now();
    let deadline = Duration::from_secs(duration_secs);

    println!("\nStarting evolutionary loop...\n");

    while start.elapsed() < deadline {
        rt.keepalive();
        generation += 1;

        // === 1. EVALUATE ===
        evaluate_population(
            &mut pop, &input, &target, &fitness_buf,
            &rt, cublas, &mut gemm_count,
        );

        // === 2. PARETO UPDATE ===
        update_pareto(&mut pareto, &pop);

        // === 3. SHM LEADERBOARD ===
        {
            let mut fitnesses: Vec<f32> = pop.iter().map(|ind| ind.fitness).collect();
            fitnesses.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            fitnesses.resize(POP_SIZE, 0.0);
            unsafe {
                ptx_sys::cudaMemcpy(
                    shm_ptr,
                    fitnesses.as_ptr() as *const libc::c_void,
                    POP_SIZE * 4,
                    ptx_sys::cudaMemcpyHostToDevice,
                );
            }
        }

        // === 4. TELEMETRY (before breed/replace, while fitness is valid) ===
        if reporter.should_report() {
            let alive_count = pop.iter().filter(|ind| ind.alive).count();
            let best_fitness = pop.iter()
                .filter(|ind| ind.alive)
                .map(|ind| ind.fitness)
                .fold(0.0f32, f32::max);
            let avg_fitness = if alive_count > 0 {
                pop.iter().filter(|ind| ind.alive).map(|ind| ind.fitness).sum::<f32>() / alive_count as f32
            } else { 0.0 };

            let param_range = if alive_count > 0 {
                let min_p = pop.iter().filter(|ind| ind.alive).map(|ind| ind.param_count).min().unwrap_or(0);
                let max_p = pop.iter().filter(|ind| ind.alive).map(|ind| ind.param_count).max().unwrap_or(0);
                format!("{}-{}", min_p, max_p)
            } else {
                "N/A".to_string()
            };

            let circuit_state = match circuit_breaker.state() {
                CircuitState::Closed => "CLOSED",
                CircuitState::Open => "OPEN",
                CircuitState::HalfOpen => "HALF_OPEN",
            };

            reporter.report(&rt, &format!(
                "gen={} | best={:.4} | avg={:.4} | alive={}/{} | pareto={} | params=[{}] | gemms={} | circuit={}",
                generation, best_fitness, avg_fitness,
                alive_count, POP_SIZE, pareto.len(),
                param_range, gemm_count, circuit_state,
            ));
        }

        // === 5. SELECT + BREED ===
        let new_genomes = breed_next_generation(&pop, &mut rng, &mut next_id, generation);

        // === 6. VMM PRESSURE ===
        vmm_pressure_check(&mut pop, &rt, vmm);

        // === 7. REPLACE — drop old population (RAII frees TLSF), install new ===
        while let Some(ind) = pop.pop() {
            drop(ind.outputs);
            drop(ind.weights);
        }

        // Materialize new generation
        let mut new_pop: Vec<Individual> = Vec::with_capacity(POP_SIZE);
        for genome in new_genomes {
            if start.elapsed() >= deadline { break; }
            if let Some(ind) = materialize(genome, &rt, &circuit_breaker, &retry_policy, &mut rng, &random_pool) {
                new_pop.push(ind);
            }
        }
        pop = new_pop;

        // === 8. VFS CHECKPOINT ===
        if last_checkpoint.elapsed() >= Duration::from_secs(CHECKPOINT_INTERVAL_SECS) {
            rt.sync_all()?;
            unsafe {
                vfs_checkpoint_pareto(vfs, &pareto, generation, &mut vfs_pareto_count);
            }
            vfs_gen_count = generation;
            last_checkpoint = Instant::now();
        }
    }

    // === SUMMARY ===
    // Use Pareto frontier for best-individual reporting (always valid, unlike pop
    // which may be unevaluated or empty at the end of the run)
    let best_pareto = pareto.iter().max_by(|a, b| {
        a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("\n=== GPU NAS COMPLETE ===");
    println!("Total generations: {}", generation);
    println!("Total GEMMs: {}", gemm_count);
    println!("Pareto frontier size: {}", pareto.len());
    println!("VFS checkpoints: {} generations saved", vfs_gen_count);
    if let Some(best) = best_pareto {
        println!("Best individual: {} fitness={:.4} params={}",
            best.genome.describe(), best.fitness, best.param_count);
    }
    if !pareto.is_empty() {
        println!("Pareto frontier:");
        let mut sorted = pareto.clone();
        sorted.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
        for (i, entry) in sorted.iter().take(5).enumerate() {
            println!("  #{}: {} fitness={:.4} params={}",
                i + 1, entry.genome.describe(), entry.fitness, entry.param_count);
        }
        if sorted.len() > 5 {
            println!("  ... and {} more", sorted.len() - 5);
        }
    }
    println!("Duration: {:.1}s", reporter.elapsed().as_secs_f64());

    // === CLEANUP (reverse order) ===
    // 1. Drop cuBLAS guard
    drop(cublas_guard);

    // 2. Drop all individuals
    for ind in pop.into_iter().rev() {
        drop(ind.outputs);
        drop(ind.weights);
    }

    // 3. Drop shared buffers
    drop(fitness_buf);
    drop(target);
    drop(input);

    // 4. VFS cleanup
    unsafe {
        // Unlink Pareto entries
        for i in 0..vfs_pareto_count {
            let path = format!("/nas/pareto/entry_{}", i);
            let fit_path = format!("/nas/pareto/entry_{}_fit", i);
            let _ = platform::vfs_safe_unlink(vfs, &path);
            let _ = platform::vfs_safe_unlink(vfs, &fit_path);
        }

        // Unlink generation checkpoints
        for g in 1..=vfs_gen_count {
            let path = format!("/nas/checkpoints/gen_{:05}", g);
            let _ = platform::vfs_safe_unlink(vfs, &path);
        }

        let _ = platform::vfs_safe_rmdir(vfs, "/nas/pareto");
        let _ = platform::vfs_safe_rmdir(vfs, "/nas/checkpoints");
        let _ = platform::vfs_safe_rmdir(vfs, "/nas");
        ptx_sys::vfs_shutdown(vfs);
    }

    // 5. VMM shutdown
    unsafe { ptx_sys::vmm_shutdown(vmm); }

    // 6. SHM unlink
    unsafe { platform::shm_safe_unlink(&rt, "nas_leaderboard", shm_ptr)?; }

    // 7. Final validation
    rt.sync_all()?;
    platform::assert_clean_exit(&rt);

    Ok(())
}
