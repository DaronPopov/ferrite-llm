use std::time::Instant;

use super::ring::RingBuffer;

/// A single processing stage in a pipeline.
pub trait Stage {
    type Input;
    type Output;
    fn name(&self) -> &str;
    fn process(&mut self, input: Self::Input) -> Option<Self::Output>;
}

/// Per-stage latency metrics collected during pipeline execution.
pub struct StageMetrics {
    pub name: String,
    pub total_us: u128,
    pub invocations: u64,
    pub min_us: u128,
    pub max_us: u128,
}

impl StageMetrics {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            total_us: 0,
            invocations: 0,
            min_us: u128::MAX,
            max_us: 0,
        }
    }

    pub fn record(&mut self, us: u128) {
        self.total_us += us;
        self.invocations += 1;
        self.min_us = self.min_us.min(us);
        self.max_us = self.max_us.max(us);
    }

    pub fn avg_us(&self) -> f64 {
        if self.invocations == 0 {
            0.0
        } else {
            self.total_us as f64 / self.invocations as f64
        }
    }
}

/// Aggregate pipeline statistics for a completed run.
pub struct PipelineStats {
    pub total_ticks: u64,
    pub total_time_ms: f64,
    pub warmup_ticks: u64,
    pub result_frames: u64,
    pub stages: Vec<StageMetrics>,
}

impl PipelineStats {
    pub fn throughput_fps(&self) -> f64 {
        if self.total_time_ms <= 0.0 {
            0.0
        } else {
            self.result_frames as f64 / (self.total_time_ms / 1000.0)
        }
    }
}

// ── Pipeline2 ──────────────────────────────────────────────────

/// Two-stage pipeline: S1 → S2
pub struct Pipeline2<S1: Stage, S2: Stage<Input = S1::Output>> {
    pub s1: S1,
    pub s2: S2,
    ring: RingBuffer<S1::Output>,
    metrics: [StageMetrics; 2],
    total_ticks: u64,
    result_frames: u64,
    start: Option<Instant>,
}

impl<S1: Stage, S2: Stage<Input = S1::Output>> Pipeline2<S1, S2> {
    pub fn new(s1: S1, s2: S2, ring_capacity: usize) -> Self {
        let m0 = StageMetrics::new(s1.name());
        let m1 = StageMetrics::new(s2.name());
        Self {
            s1,
            s2,
            ring: RingBuffer::new(ring_capacity),
            metrics: [m0, m1],
            total_ticks: 0,
            result_frames: 0,
            start: None,
        }
    }

    pub fn tick(&mut self, input: S1::Input) -> Option<S2::Output> {
        if self.start.is_none() {
            self.start = Some(Instant::now());
        }
        self.total_ticks += 1;

        // Dispatch in reverse order: downstream first
        let result = if let Some(s1_out) = self.ring.pop() {
            let t = Instant::now();
            let out = self.s2.process(s1_out);
            self.metrics[1].record(t.elapsed().as_micros());
            if out.is_some() {
                self.result_frames += 1;
            }
            out
        } else {
            None
        };

        let t = Instant::now();
        if let Some(s1_out) = self.s1.process(input) {
            self.ring.push(s1_out);
        }
        self.metrics[0].record(t.elapsed().as_micros());

        result
    }

    pub fn stats(&self) -> PipelineStats {
        let elapsed = self.start.map(|s| s.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        PipelineStats {
            total_ticks: self.total_ticks,
            total_time_ms: elapsed,
            warmup_ticks: 1,
            result_frames: self.result_frames,
            stages: self.metrics.iter().map(|m| StageMetrics {
                name: m.name.clone(),
                total_us: m.total_us,
                invocations: m.invocations,
                min_us: m.min_us,
                max_us: m.max_us,
            }).collect(),
        }
    }
}

// ── Pipeline3 ──────────────────────────────────────────────────

/// Three-stage pipeline: S1 → S2 → S3
pub struct Pipeline3<S1, S2, S3>
where
    S1: Stage,
    S2: Stage<Input = S1::Output>,
    S3: Stage<Input = S2::Output>,
{
    pub s1: S1,
    pub s2: S2,
    pub s3: S3,
    ring_12: RingBuffer<S1::Output>,
    ring_23: RingBuffer<S2::Output>,
    metrics: [StageMetrics; 3],
    total_ticks: u64,
    result_frames: u64,
    start: Option<Instant>,
}

impl<S1, S2, S3> Pipeline3<S1, S2, S3>
where
    S1: Stage,
    S2: Stage<Input = S1::Output>,
    S3: Stage<Input = S2::Output>,
{
    pub fn new(s1: S1, s2: S2, s3: S3, ring_capacity: usize) -> Self {
        let m = [
            StageMetrics::new(s1.name()),
            StageMetrics::new(s2.name()),
            StageMetrics::new(s3.name()),
        ];
        Self {
            s1, s2, s3,
            ring_12: RingBuffer::new(ring_capacity),
            ring_23: RingBuffer::new(ring_capacity),
            metrics: m,
            total_ticks: 0,
            result_frames: 0,
            start: None,
        }
    }

    pub fn tick(&mut self, input: S1::Input) -> Option<S3::Output> {
        if self.start.is_none() {
            self.start = Some(Instant::now());
        }
        self.total_ticks += 1;

        // Stage 3 (downstream first)
        let result = if let Some(s2_out) = self.ring_23.pop() {
            let t = Instant::now();
            let out = self.s3.process(s2_out);
            self.metrics[2].record(t.elapsed().as_micros());
            if out.is_some() { self.result_frames += 1; }
            out
        } else { None };

        // Stage 2
        if let Some(s1_out) = self.ring_12.pop() {
            let t = Instant::now();
            if let Some(s2_out) = self.s2.process(s1_out) {
                self.ring_23.push(s2_out);
            }
            self.metrics[1].record(t.elapsed().as_micros());
        }

        // Stage 1
        let t = Instant::now();
        if let Some(s1_out) = self.s1.process(input) {
            self.ring_12.push(s1_out);
        }
        self.metrics[0].record(t.elapsed().as_micros());

        result
    }

    pub fn stats(&self) -> PipelineStats {
        let elapsed = self.start.map(|s| s.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        PipelineStats {
            total_ticks: self.total_ticks,
            total_time_ms: elapsed,
            warmup_ticks: 2,
            result_frames: self.result_frames,
            stages: self.metrics.iter().map(|m| StageMetrics {
                name: m.name.clone(),
                total_us: m.total_us,
                invocations: m.invocations,
                min_us: m.min_us,
                max_us: m.max_us,
            }).collect(),
        }
    }
}

// ── Pipeline4 ──────────────────────────────────────────────────

/// Four-stage pipeline: S1 → S2 → S3 → S4
pub struct Pipeline4<S1, S2, S3, S4>
where
    S1: Stage,
    S2: Stage<Input = S1::Output>,
    S3: Stage<Input = S2::Output>,
    S4: Stage<Input = S3::Output>,
{
    pub s1: S1,
    pub s2: S2,
    pub s3: S3,
    pub s4: S4,
    ring_12: RingBuffer<S1::Output>,
    ring_23: RingBuffer<S2::Output>,
    ring_34: RingBuffer<S3::Output>,
    metrics: [StageMetrics; 4],
    total_ticks: u64,
    result_frames: u64,
    start: Option<Instant>,
}

impl<S1, S2, S3, S4> Pipeline4<S1, S2, S3, S4>
where
    S1: Stage,
    S2: Stage<Input = S1::Output>,
    S3: Stage<Input = S2::Output>,
    S4: Stage<Input = S3::Output>,
{
    pub fn new(s1: S1, s2: S2, s3: S3, s4: S4, ring_capacity: usize) -> Self {
        let m = [
            StageMetrics::new(s1.name()),
            StageMetrics::new(s2.name()),
            StageMetrics::new(s3.name()),
            StageMetrics::new(s4.name()),
        ];
        Self {
            s1, s2, s3, s4,
            ring_12: RingBuffer::new(ring_capacity),
            ring_23: RingBuffer::new(ring_capacity),
            ring_34: RingBuffer::new(ring_capacity),
            metrics: m,
            total_ticks: 0,
            result_frames: 0,
            start: None,
        }
    }

    pub fn tick(&mut self, input: S1::Input) -> Option<S4::Output> {
        if self.start.is_none() {
            self.start = Some(Instant::now());
        }
        self.total_ticks += 1;

        // Stage 4
        let result = if let Some(s3_out) = self.ring_34.pop() {
            let t = Instant::now();
            let out = self.s4.process(s3_out);
            self.metrics[3].record(t.elapsed().as_micros());
            if out.is_some() { self.result_frames += 1; }
            out
        } else { None };

        // Stage 3
        if let Some(s2_out) = self.ring_23.pop() {
            let t = Instant::now();
            if let Some(s3_out) = self.s3.process(s2_out) {
                self.ring_34.push(s3_out);
            }
            self.metrics[2].record(t.elapsed().as_micros());
        }

        // Stage 2
        if let Some(s1_out) = self.ring_12.pop() {
            let t = Instant::now();
            if let Some(s2_out) = self.s2.process(s1_out) {
                self.ring_23.push(s2_out);
            }
            self.metrics[1].record(t.elapsed().as_micros());
        }

        // Stage 1
        let t = Instant::now();
        if let Some(s1_out) = self.s1.process(input) {
            self.ring_12.push(s1_out);
        }
        self.metrics[0].record(t.elapsed().as_micros());

        result
    }

    pub fn stats(&self) -> PipelineStats {
        let elapsed = self.start.map(|s| s.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        PipelineStats {
            total_ticks: self.total_ticks,
            total_time_ms: elapsed,
            warmup_ticks: 3,
            result_frames: self.result_frames,
            stages: self.metrics.iter().map(|m| StageMetrics {
                name: m.name.clone(),
                total_us: m.total_us,
                invocations: m.invocations,
                min_us: m.min_us,
                max_us: m.max_us,
            }).collect(),
        }
    }
}

// ── Pipeline5 ──────────────────────────────────────────────────

/// Five-stage pipeline: S1 → S2 → S3 → S4 → S5
pub struct Pipeline5<S1, S2, S3, S4, S5>
where
    S1: Stage,
    S2: Stage<Input = S1::Output>,
    S3: Stage<Input = S2::Output>,
    S4: Stage<Input = S3::Output>,
    S5: Stage<Input = S4::Output>,
{
    pub s1: S1,
    pub s2: S2,
    pub s3: S3,
    pub s4: S4,
    pub s5: S5,
    ring_12: RingBuffer<S1::Output>,
    ring_23: RingBuffer<S2::Output>,
    ring_34: RingBuffer<S3::Output>,
    ring_45: RingBuffer<S4::Output>,
    metrics: [StageMetrics; 5],
    total_ticks: u64,
    result_frames: u64,
    start: Option<Instant>,
}

impl<S1, S2, S3, S4, S5> Pipeline5<S1, S2, S3, S4, S5>
where
    S1: Stage,
    S2: Stage<Input = S1::Output>,
    S3: Stage<Input = S2::Output>,
    S4: Stage<Input = S3::Output>,
    S5: Stage<Input = S4::Output>,
{
    pub fn new(s1: S1, s2: S2, s3: S3, s4: S4, s5: S5, ring_capacity: usize) -> Self {
        let m = [
            StageMetrics::new(s1.name()),
            StageMetrics::new(s2.name()),
            StageMetrics::new(s3.name()),
            StageMetrics::new(s4.name()),
            StageMetrics::new(s5.name()),
        ];
        Self {
            s1, s2, s3, s4, s5,
            ring_12: RingBuffer::new(ring_capacity),
            ring_23: RingBuffer::new(ring_capacity),
            ring_34: RingBuffer::new(ring_capacity),
            ring_45: RingBuffer::new(ring_capacity),
            metrics: m,
            total_ticks: 0,
            result_frames: 0,
            start: None,
        }
    }

    pub fn tick(&mut self, input: S1::Input) -> Option<S5::Output> {
        if self.start.is_none() {
            self.start = Some(Instant::now());
        }
        self.total_ticks += 1;

        // Stage 5
        let result = if let Some(s4_out) = self.ring_45.pop() {
            let t = Instant::now();
            let out = self.s5.process(s4_out);
            self.metrics[4].record(t.elapsed().as_micros());
            if out.is_some() { self.result_frames += 1; }
            out
        } else { None };

        // Stage 4
        if let Some(s3_out) = self.ring_34.pop() {
            let t = Instant::now();
            if let Some(s4_out) = self.s4.process(s3_out) {
                self.ring_45.push(s4_out);
            }
            self.metrics[3].record(t.elapsed().as_micros());
        }

        // Stage 3
        if let Some(s2_out) = self.ring_23.pop() {
            let t = Instant::now();
            if let Some(s3_out) = self.s3.process(s2_out) {
                self.ring_34.push(s3_out);
            }
            self.metrics[2].record(t.elapsed().as_micros());
        }

        // Stage 2
        if let Some(s1_out) = self.ring_12.pop() {
            let t = Instant::now();
            if let Some(s2_out) = self.s2.process(s1_out) {
                self.ring_23.push(s2_out);
            }
            self.metrics[1].record(t.elapsed().as_micros());
        }

        // Stage 1
        let t = Instant::now();
        if let Some(s1_out) = self.s1.process(input) {
            self.ring_12.push(s1_out);
        }
        self.metrics[0].record(t.elapsed().as_micros());

        result
    }

    pub fn stats(&self) -> PipelineStats {
        let elapsed = self.start.map(|s| s.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        PipelineStats {
            total_ticks: self.total_ticks,
            total_time_ms: elapsed,
            warmup_ticks: 4,
            result_frames: self.result_frames,
            stages: self.metrics.iter().map(|m| StageMetrics {
                name: m.name.clone(),
                total_us: m.total_us,
                invocations: m.invocations,
                min_us: m.min_us,
                max_us: m.max_us,
            }).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DoubleStage;
    impl Stage for DoubleStage {
        type Input = i32;
        type Output = i32;
        fn name(&self) -> &str { "double" }
        fn process(&mut self, input: i32) -> Option<i32> { Some(input * 2) }
    }

    struct AddTenStage;
    impl Stage for AddTenStage {
        type Input = i32;
        type Output = i32;
        fn name(&self) -> &str { "add_ten" }
        fn process(&mut self, input: i32) -> Option<i32> { Some(input + 10) }
    }

    struct ToStringStage;
    impl Stage for ToStringStage {
        type Input = i32;
        type Output = String;
        fn name(&self) -> &str { "to_string" }
        fn process(&mut self, input: i32) -> Option<String> { Some(format!("{}", input)) }
    }

    #[test]
    fn pipeline2_warmup_and_output() {
        let mut pipe = Pipeline2::new(DoubleStage, AddTenStage, 4);
        // Tick 1: input=5 → double=10 pushed to ring, no output yet (warmup)
        assert_eq!(pipe.tick(5), None);
        // Tick 2: ring has 10, stage2 processes 10→20, stage1 processes 3→6
        assert_eq!(pipe.tick(3), Some(20));
        // Tick 3: ring has 6, stage2 processes 6→16
        assert_eq!(pipe.tick(0), Some(16));

        let stats = pipe.stats();
        assert_eq!(stats.result_frames, 2);
        assert_eq!(stats.warmup_ticks, 1);
    }

    #[test]
    fn pipeline3_warmup() {
        let mut pipe = Pipeline3::new(DoubleStage, AddTenStage, ToStringStage, 4);
        // Warmup: 2 ticks
        assert_eq!(pipe.tick(5), None);  // ring_12=[10]
        assert_eq!(pipe.tick(3), None);  // ring_12=[6], ring_23=[20]
        // Tick 3: output from stage3
        assert_eq!(pipe.tick(1), Some("20".to_string()));

        let stats = pipe.stats();
        assert_eq!(stats.result_frames, 1);
        assert_eq!(stats.warmup_ticks, 2);
    }
}
