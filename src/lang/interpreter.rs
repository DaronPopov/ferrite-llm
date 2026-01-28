use crate::lang::{Program, Step, SemanticOp};
use crate::data::Grid;
use regex::Regex;
use std::collections::HashMap;

pub struct Interpreter {
    pub skills: HashMap<String, String>, // Alias -> Path mapped skills
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
        }
    }

    pub fn run(&mut self, script: &str) -> Program {
        let mut program = Program::new();
        let lines: Vec<&str> = script.lines().map(|l| l.trim()).filter(|l| !l.is_empty() && !l.starts_with("#")).collect();

        // Basic patterns
        let re_skill = Regex::new(r"skill (\w+) from (.+)").unwrap();
        let re_assign = Regex::new(r"(\w+) = (\w+)\.(\w+)\((.+)\)").unwrap();
        let re_bind = Regex::new(r"bind (\w+) as zero\[(.+)\]").unwrap();
        let re_bind_randn = Regex::new(r"bind (\w+) as randn\[(.+)\]").unwrap();
        let re_bind_rand = Regex::new(r"bind (\w+) as rand\[(.+)\]").unwrap();
        let re_move = Regex::new(r"move (\w+) to device as (\w+)").unwrap();
        let re_move_host = Regex::new(r"move (\w+) to host as (\w+)").unwrap();

        // Training patterns
        let re_init = Regex::new(r"init (\w+) as (\w+)").unwrap();  // init W as xavier
        let re_linear = Regex::new(r"(\w+) = linear\((\w+), (\w+)\)").unwrap();  // y = linear(x, W)
        let re_linear_bias = Regex::new(r"(\w+) = linear\((\w+), (\w+), (\w+)\)").unwrap();  // y = linear(x, W, b)
        let re_loss_mse = Regex::new(r"(\w+) = mse_loss\((\w+), (\w+)\)").unwrap();
        let re_loss_ce = Regex::new(r"(\w+) = cross_entropy\((\w+), (\w+), (\d+)\)").unwrap();
        let re_loss_bce = Regex::new(r"(\w+) = bce_loss\((\w+), (\w+)\)").unwrap();
        let re_backward_mse = Regex::new(r"(\w+) = mse_backward\((\w+), (\w+)\)").unwrap();
        let re_backward_ce = Regex::new(r"(\w+) = cross_entropy_backward\((\w+), (\w+), (\d+)\)").unwrap();
        let re_backward_act = Regex::new(r"(\w+) = (\w+)_backward\((\w+), (\w+)\)").unwrap();
        let re_linear_backward = Regex::new(r"(\w+) = linear_backward\((\w+), (\w+), (\w+)\)").unwrap();
        let re_zero_grad = Regex::new(r"zero_grad\((\w+)\)").unwrap();
        let re_sgd = Regex::new(r"sgd_step\((\w+), (\w+), (\w+), lr=([0-9.]+), momentum=([0-9.]+)\)").unwrap();
        let re_adam = Regex::new(r"adam_step\((\w+), (\w+), (\w+), (\w+), t=(\d+)\)").unwrap();
        let re_set_adam = Regex::new(r"set_adam\(lr=([0-9.]+), beta1=([0-9.]+), beta2=([0-9.]+), eps=([0-9e.-]+), wd=([0-9.]+)\)").unwrap();

        // Normalization patterns
        let re_batchnorm = Regex::new(r"(\w+) = batch_norm\((\w+), (\w+), (\w+), (\w+), (\w+), training=(\w+)\)").unwrap();
        let re_layernorm = Regex::new(r"(\w+) = layer_norm\((\w+), (\w+), (\w+)\)").unwrap();
        let re_dropout = Regex::new(r"(\w+) = dropout\((\w+), p=([0-9.]+), training=(\w+)\)").unwrap();

        // Attention pattern
        let re_attention = Regex::new(r"(\w+) = attention\((\w+), (\w+), (\w+), scale=([0-9.]+), causal=(\w+)\)").unwrap();

        // Reduce patterns
        let re_reduce = Regex::new(r"(\w+) = reduce_(\w+)\((\w+)\)").unwrap();

        // Fill/scale patterns
        let re_fill = Regex::new(r"fill\((\w+), ([0-9.-]+)\)").unwrap();
        let re_scale = Regex::new(r"scale\((\w+), ([0-9.-]+)\)").unwrap();
        let re_copy = Regex::new(r"copy\((\w+), (\w+)\)").unwrap();

        // Pooling patterns
        let re_maxpool = Regex::new(r"(\w+) = max_pool2d\((\w+), kernel=\[(\d+),(\d+)\], stride=\[(\d+),(\d+)\]\)").unwrap();
        let re_avgpool = Regex::new(r"(\w+) = avg_pool2d\((\w+), kernel=\[(\d+),(\d+)\], stride=\[(\d+),(\d+)\]\)").unwrap();

        // Conv2d pattern
        let re_conv2d = Regex::new(r"(\w+) = conv2d\((\w+), (\w+), stride=\[(\d+),(\d+)\], pad=\[(\d+),(\d+)\]\)").unwrap();

        // Loop pattern for training
        let re_epoch = Regex::new(r"epoch (\d+):").unwrap();
        let re_print = Regex::new(r#"print\("(.+)", (\w+)\)"#).unwrap();

        for line in lines {
            // Skip epoch markers and print statements (handled externally)
            if re_epoch.is_match(line) || re_print.is_match(line) {
                continue;
            }

            if let Some(caps) = re_skill.captures(line) {
                let alias = caps[1].to_string();
                let path = caps[2].trim_matches('"').to_string();
                self.skills.insert(alias, path);
            } else if let Some(caps) = re_bind.captures(line) {
                let name = caps[1].to_string();
                let shape: Vec<usize> = caps[2].split(',').map(|s| s.trim().parse().unwrap()).collect();
                let size: usize = shape.iter().product();
                program.bind(&name, Grid::new(vec![0.0; size], shape));
            } else if let Some(caps) = re_bind_randn.captures(line) {
                let name = caps[1].to_string();
                let shape: Vec<usize> = caps[2].split(',').map(|s| s.trim().parse().unwrap()).collect();
                let size: usize = shape.iter().product();
                // Use random normal init on CPU then move
                let data: Vec<f32> = (0..size).map(|_| rand_normal()).collect();
                program.bind(&name, Grid::new(data, shape));
            } else if let Some(caps) = re_bind_rand.captures(line) {
                let name = caps[1].to_string();
                let shape: Vec<usize> = caps[2].split(',').map(|s| s.trim().parse().unwrap()).collect();
                let size: usize = shape.iter().product();
                let data: Vec<f32> = (0..size).map(|_| rand_uniform()).collect();
                program.bind(&name, Grid::new(data, shape));
            } else if let Some(caps) = re_move.captures(line) {
                program.add_step(Step::MoveToDevice(caps[1].to_string(), caps[2].to_string()));
            } else if let Some(caps) = re_move_host.captures(line) {
                program.add_step(Step::MoveToHost(caps[1].to_string(), caps[2].to_string()));
            } else if let Some(caps) = re_init.captures(line) {
                let name = caps[1].to_string();
                let init_type = caps[2].to_string();
                program.add_step(Step::Op(
                    SemanticOp::InitWeights { init_type },
                    vec![name.clone()],
                    name
                ));
            } else if let Some(caps) = re_linear_bias.captures(line) {
                let out = caps[1].to_string();
                let x = caps[2].to_string();
                let w = caps[3].to_string();
                let b = caps[4].to_string();
                program.add_step(Step::Op(
                    SemanticOp::LinearForward { has_bias: true },
                    vec![x, w, b],
                    out
                ));
            } else if let Some(caps) = re_linear.captures(line) {
                let out = caps[1].to_string();
                let x = caps[2].to_string();
                let w = caps[3].to_string();
                program.add_step(Step::Op(
                    SemanticOp::LinearForward { has_bias: false },
                    vec![x, w],
                    out
                ));
            } else if let Some(caps) = re_loss_mse.captures(line) {
                let out = caps[1].to_string();
                let pred = caps[2].to_string();
                let target = caps[3].to_string();
                program.add_step(Step::Op(SemanticOp::MSELoss, vec![pred, target], out));
            } else if let Some(caps) = re_loss_ce.captures(line) {
                let out = caps[1].to_string();
                let logits = caps[2].to_string();
                let targets = caps[3].to_string();
                let num_classes: usize = caps[4].parse().unwrap();
                program.add_step(Step::Op(
                    SemanticOp::CrossEntropyLoss { num_classes },
                    vec![logits, targets],
                    out
                ));
            } else if let Some(caps) = re_loss_bce.captures(line) {
                let out = caps[1].to_string();
                let pred = caps[2].to_string();
                let target = caps[3].to_string();
                program.add_step(Step::Op(SemanticOp::BCEWithLogitsLoss, vec![pred, target], out));
            } else if let Some(caps) = re_backward_mse.captures(line) {
                let out = caps[1].to_string();
                let pred = caps[2].to_string();
                let target = caps[3].to_string();
                program.add_step(Step::Op(SemanticOp::MSELossBackward, vec![pred, target], out));
            } else if let Some(caps) = re_backward_ce.captures(line) {
                let out = caps[1].to_string();
                let logits = caps[2].to_string();
                let targets = caps[3].to_string();
                let num_classes: usize = caps[4].parse().unwrap();
                program.add_step(Step::Op(
                    SemanticOp::CrossEntropyBackward { num_classes },
                    vec![logits, targets],
                    out
                ));
            } else if let Some(caps) = re_backward_act.captures(line) {
                let out = caps[1].to_string();
                let act = caps[2].to_string();
                let x = caps[3].to_string();
                let grad = caps[4].to_string();

                // Match activation backward functions
                if ["relu", "sigmoid", "tanh", "gelu"].contains(&act.as_str()) {
                    program.add_step(Step::Op(
                        SemanticOp::ActivationBackward(act),
                        vec![x, grad],
                        out
                    ));
                }
            } else if let Some(caps) = re_linear_backward.captures(line) {
                let out = caps[1].to_string();
                let x = caps[2].to_string();
                let w = caps[3].to_string();
                let grad = caps[4].to_string();
                program.add_step(Step::Op(
                    SemanticOp::LinearBackward { has_bias: false },
                    vec![x, w, grad],
                    out
                ));
            } else if let Some(caps) = re_zero_grad.captures(line) {
                let grad = caps[1].to_string();
                program.add_step(Step::Op(SemanticOp::ZeroGrad, vec![grad.clone()], grad));
            } else if let Some(caps) = re_sgd.captures(line) {
                let w = caps[1].to_string();
                let grad = caps[2].to_string();
                let mom = caps[3].to_string();
                let lr: f32 = caps[4].parse().unwrap();
                let momentum: f32 = caps[5].parse().unwrap();
                program.add_step(Step::Op(
                    SemanticOp::SGDStep { lr, momentum, weight_decay: 0.0 },
                    vec![w.clone(), grad, mom],
                    w
                ));
            } else if let Some(caps) = re_adam.captures(line) {
                let w = caps[1].to_string();
                let grad = caps[2].to_string();
                let m = caps[3].to_string();
                let v = caps[4].to_string();
                let t: i32 = caps[5].parse().unwrap();
                program.add_step(Step::Op(
                    SemanticOp::AdamStep { timestep: t },
                    vec![w.clone(), grad, m, v],
                    w
                ));
            } else if let Some(caps) = re_set_adam.captures(line) {
                let lr: f32 = caps[1].parse().unwrap();
                let beta1: f32 = caps[2].parse().unwrap();
                let beta2: f32 = caps[3].parse().unwrap();
                let eps: f32 = caps[4].parse().unwrap();
                let wd: f32 = caps[5].parse().unwrap();
                program.add_step(Step::Op(
                    SemanticOp::SetOptimizerParams { lr, beta1, beta2, eps, weight_decay: wd },
                    vec![],
                    "_optimizer_config".to_string()
                ));
            } else if let Some(caps) = re_batchnorm.captures(line) {
                let out = caps[1].to_string();
                let x = caps[2].to_string();
                let gamma = caps[3].to_string();
                let beta = caps[4].to_string();
                let rm = caps[5].to_string();
                let rv = caps[6].to_string();
                let training = &caps[7] == "true";
                program.add_step(Step::Op(
                    SemanticOp::BatchNormForward { training, momentum: 0.1, eps: 1e-5 },
                    vec![x, gamma, beta, rm, rv],
                    out
                ));
            } else if let Some(caps) = re_layernorm.captures(line) {
                let out = caps[1].to_string();
                let x = caps[2].to_string();
                let gamma = caps[3].to_string();
                let beta = caps[4].to_string();
                program.add_step(Step::Op(
                    SemanticOp::LayerNormForward { eps: 1e-5 },
                    vec![x, gamma, beta],
                    out
                ));
            } else if let Some(caps) = re_dropout.captures(line) {
                let out = caps[1].to_string();
                let x = caps[2].to_string();
                let p: f32 = caps[3].parse().unwrap();
                let training = &caps[4] == "true";
                program.add_step(Step::Op(
                    SemanticOp::DropoutForward { p, training },
                    vec![x],
                    out
                ));
            } else if let Some(caps) = re_attention.captures(line) {
                let out = caps[1].to_string();
                let q = caps[2].to_string();
                let k = caps[3].to_string();
                let v = caps[4].to_string();
                let scale: f32 = caps[5].parse().unwrap();
                let causal = &caps[6] == "true";
                program.add_step(Step::Op(
                    SemanticOp::ScaledDotProductAttention { scale, causal },
                    vec![q, k, v],
                    out
                ));
            } else if let Some(caps) = re_reduce.captures(line) {
                let out = caps[1].to_string();
                let reduce_type = &caps[2];
                let x = caps[3].to_string();
                let op = match reduce_type {
                    "sum" => SemanticOp::ReduceSum,
                    "mean" => SemanticOp::ReduceMean,
                    "max" => SemanticOp::ReduceMax,
                    "min" => SemanticOp::ReduceMin,
                    _ => panic!("Unknown reduce type: {}", reduce_type),
                };
                program.add_step(Step::Op(op, vec![x], out));
            } else if let Some(caps) = re_fill.captures(line) {
                let name = caps[1].to_string();
                let value: f32 = caps[2].parse().unwrap();
                program.add_step(Step::Op(SemanticOp::Fill(value), vec![name.clone()], name));
            } else if let Some(caps) = re_scale.captures(line) {
                let name = caps[1].to_string();
                let factor: f32 = caps[2].parse().unwrap();
                program.add_step(Step::Op(SemanticOp::Scale(factor), vec![name.clone()], name));
            } else if let Some(caps) = re_copy.captures(line) {
                let src = caps[1].to_string();
                let dst = caps[2].to_string();
                program.add_step(Step::Op(SemanticOp::Copy, vec![src, dst.clone()], dst));
            } else if let Some(caps) = re_maxpool.captures(line) {
                let out = caps[1].to_string();
                let x = caps[2].to_string();
                let kh: usize = caps[3].parse().unwrap();
                let kw: usize = caps[4].parse().unwrap();
                let sh: usize = caps[5].parse().unwrap();
                let sw: usize = caps[6].parse().unwrap();
                program.add_step(Step::Op(
                    SemanticOp::MaxPool2d { kernel: (kh, kw), stride: (sh, sw) },
                    vec![x],
                    out
                ));
            } else if let Some(caps) = re_avgpool.captures(line) {
                let out = caps[1].to_string();
                let x = caps[2].to_string();
                let kh: usize = caps[3].parse().unwrap();
                let kw: usize = caps[4].parse().unwrap();
                let sh: usize = caps[5].parse().unwrap();
                let sw: usize = caps[6].parse().unwrap();
                program.add_step(Step::Op(
                    SemanticOp::AvgPool2d { kernel: (kh, kw), stride: (sh, sw) },
                    vec![x],
                    out
                ));
            } else if let Some(caps) = re_conv2d.captures(line) {
                let out = caps[1].to_string();
                let x = caps[2].to_string();
                let w = caps[3].to_string();
                let sh: usize = caps[4].parse().unwrap();
                let sw: usize = caps[5].parse().unwrap();
                let ph: usize = caps[6].parse().unwrap();
                let pw: usize = caps[7].parse().unwrap();
                program.add_step(Step::Op(
                    SemanticOp::Conv2dForward { stride: (sh, sw), padding: (ph, pw), has_bias: false },
                    vec![x, w],
                    out
                ));
            } else if let Some(caps) = re_assign.captures(line) {
                let out = caps[1].to_string();
                let skill = &caps[2];
                let op_name = &caps[3];
                let inputs: Vec<String> = caps[4].split(',').map(|s| s.trim().to_string()).collect();

                let op = match skill {
                    "torch" => {
                        if inputs.len() == 1 { SemanticOp::TorchUnary(op_name.to_string()) }
                        else { SemanticOp::TorchBinary(op_name.to_string()) }
                    }
                    "vision" => SemanticOp::Vision(op_name.to_string()),
                    "math" => SemanticOp::Math(op_name.to_string()),
                    _ => panic!("Unknown skill: {}", skill),
                };

                program.add_step(Step::Op(op, inputs, out));
            }
        }

        program
    }
}

// Simple random number generation for CPU initialization
fn rand_normal() -> f32 {
    use std::f32::consts::PI;
    let u1: f32 = rand_uniform();
    let u2: f32 = rand_uniform();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn rand_uniform() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    static mut SEED: u64 = 0;
    unsafe {
        if SEED == 0 {
            SEED = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
        }
        // Simple LCG
        SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (SEED >> 33) as f32 / (1u64 << 31) as f32
    }
}
