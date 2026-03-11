use std::f64::consts::PI;

use anyhow::Result;

#[derive(Clone, Debug)]
pub enum Schedule {
    Constant { lr: f64 },
    LinearWarmup { base_lr: f64, warmup_steps: usize },
    CosineDecay { base_lr: f64, min_lr: f64, warmup_steps: usize, total_steps: usize },
    StepDecay { base_lr: f64, decay_factor: f64, decay_every: usize },
    LinearDecay { base_lr: f64, min_lr: f64, warmup_steps: usize, total_steps: usize },
    OneCycle { max_lr: f64, total_steps: usize, pct_start: f64, div_factor: f64, final_div: f64 },
}

impl Schedule {
    pub fn lr_at(&self, step: usize) -> f64 {
        match self {
            Schedule::Constant { lr } => *lr,

            Schedule::LinearWarmup { base_lr, warmup_steps } => {
                if *warmup_steps == 0 { return *base_lr; }
                let t = (step as f64 / *warmup_steps as f64).min(1.0);
                base_lr * t
            }

            Schedule::CosineDecay { base_lr, min_lr, warmup_steps, total_steps } => {
                if step < *warmup_steps {
                    if *warmup_steps == 0 { return *base_lr; }
                    return base_lr * (step as f64 / *warmup_steps as f64);
                }
                let decay_steps = total_steps.saturating_sub(*warmup_steps);
                if decay_steps == 0 { return *min_lr; }
                let progress = ((step - warmup_steps) as f64 / decay_steps as f64).min(1.0);
                let cosine = 0.5 * (1.0 + (PI * progress).cos());
                min_lr + (base_lr - min_lr) * cosine
            }

            Schedule::StepDecay { base_lr, decay_factor, decay_every } => {
                if *decay_every == 0 { return *base_lr; }
                let drops = step / decay_every;
                base_lr * decay_factor.powi(drops as i32)
            }

            Schedule::LinearDecay { base_lr, min_lr, warmup_steps, total_steps } => {
                if step < *warmup_steps {
                    if *warmup_steps == 0 { return *base_lr; }
                    return base_lr * (step as f64 / *warmup_steps as f64);
                }
                let decay_steps = total_steps.saturating_sub(*warmup_steps);
                if decay_steps == 0 { return *min_lr; }
                let progress = ((step - warmup_steps) as f64 / decay_steps as f64).min(1.0);
                base_lr + (min_lr - base_lr) * progress
            }

            Schedule::OneCycle { max_lr, total_steps, pct_start, div_factor, final_div } => {
                let initial_lr = max_lr / div_factor;
                let min_lr = initial_lr / final_div;
                let up_steps = (*total_steps as f64 * pct_start) as usize;
                let down_steps = total_steps.saturating_sub(up_steps);

                if step < up_steps {
                    if up_steps == 0 { return *max_lr; }
                    let t = step as f64 / up_steps as f64;
                    let cosine = 0.5 * (1.0 + (PI * (1.0 + t)).cos());
                    initial_lr + (max_lr - initial_lr) * cosine
                } else {
                    if down_steps == 0 { return min_lr; }
                    let t = (step - up_steps) as f64 / down_steps as f64;
                    let cosine = 0.5 * (1.0 + (PI * t).cos());
                    min_lr + (max_lr - min_lr) * cosine
                }
            }
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Schedule::Constant { .. } => "constant",
            Schedule::LinearWarmup { .. } => "linear_warmup",
            Schedule::CosineDecay { .. } => "cosine_decay",
            Schedule::StepDecay { .. } => "step_decay",
            Schedule::LinearDecay { .. } => "linear_decay",
            Schedule::OneCycle { .. } => "one_cycle",
        }
    }
}

pub struct ScheduleState {
    pub schedule: Schedule,
    pub step: usize,
}

impl ScheduleState {
    pub fn new(schedule: Schedule) -> Self {
        Self { schedule, step: 0 }
    }

    pub fn current_lr(&self) -> f64 {
        self.schedule.lr_at(self.step)
    }

    pub fn advance(&mut self) {
        self.step += 1;
    }

    pub fn set_step(&mut self, step: usize) {
        self.step = step;
    }
}

fn parse_args() -> Result<(String, f64, usize, Option<usize>, Option<f64>)> {
    let mut schedule_name = String::from("cosine_decay");
    let mut base_lr = 1e-3;
    let mut total_steps = 200usize;
    let mut warmup_steps = None;
    let mut min_lr = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--schedule" => schedule_name = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?,
            "--lr" => base_lr = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--total-steps" => total_steps = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--warmup-steps" => warmup_steps = Some(args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?),
            "--min-lr" => min_lr = Some(args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?),
            "-h" | "--help" => {
                println!("Usage: lr_schedule.rs [options]");
                println!("  --schedule constant|linear_warmup|cosine_decay|step_decay|linear_decay|one_cycle");
                println!("  --lr BASE_LR --total-steps N --warmup-steps N --min-lr F");
                println!("\nPrints the LR schedule as a table for integration or plotting.");
                std::process::exit(0);
            }
            _ => return Err(anyhow::anyhow!("unknown arg: {arg}")),
        }
    }

    Ok((schedule_name, base_lr, total_steps, warmup_steps, min_lr))
}

fn main() -> Result<()> {
    let (name, base_lr, total_steps, warmup_steps, min_lr) = parse_args()?;
    let warmup = warmup_steps.unwrap_or(total_steps / 10);
    let floor = min_lr.unwrap_or(base_lr * 0.01);

    let schedule = match name.as_str() {
        "constant" => Schedule::Constant { lr: base_lr },
        "linear_warmup" => Schedule::LinearWarmup { base_lr, warmup_steps: warmup },
        "cosine_decay" => Schedule::CosineDecay {
            base_lr, min_lr: floor, warmup_steps: warmup, total_steps,
        },
        "step_decay" => Schedule::StepDecay {
            base_lr, decay_factor: 0.5, decay_every: total_steps / 4,
        },
        "linear_decay" => Schedule::LinearDecay {
            base_lr, min_lr: floor, warmup_steps: warmup, total_steps,
        },
        "one_cycle" => Schedule::OneCycle {
            max_lr: base_lr, total_steps, pct_start: 0.3, div_factor: 25.0, final_div: 1e4,
        },
        _ => return Err(anyhow::anyhow!("unknown schedule: {name}")),
    };

    println!("=== LR Schedule: {} ===\n", schedule.name());
    println!("  base_lr:      {:.6}", base_lr);
    println!("  total_steps:  {}", total_steps);
    if warmup > 0 { println!("  warmup_steps: {}", warmup); }
    println!("  min_lr:       {:.9}", floor);
    println!();

    // Print table at 20 evenly-spaced sample points
    let sample_count = 20.min(total_steps);
    let interval = if sample_count > 1 { (total_steps - 1) / (sample_count - 1) } else { 1 };

    println!("{:>8} {:>14}", "step", "lr");
    println!("{:>8} {:>14}", "----", "--");

    let mut state = ScheduleState::new(schedule.clone());
    let mut min_seen = f64::MAX;
    let mut max_seen = f64::MIN;

    for i in 0..sample_count {
        let step = (i * interval).min(total_steps - 1);
        state.set_step(step);
        let lr = state.current_lr();
        min_seen = min_seen.min(lr);
        max_seen = max_seen.max(lr);
        println!("{:>8} {:>14.9}", step, lr);
    }

    // Emit last step
    state.set_step(total_steps - 1);
    let final_lr = state.current_lr();

    println!();
    println!("RESULT schedule={}", schedule.name());
    println!("RESULT base_lr={:.9}", base_lr);
    println!("RESULT min_lr_seen={:.9}", min_seen);
    println!("RESULT max_lr_seen={:.9}", max_seen);
    println!("RESULT final_lr={:.9}", final_lr);
    println!("RESULT total_steps={}", total_steps);

    Ok(())
}
