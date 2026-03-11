use std::time::{Duration, Instant};

use ptx_runtime::PtxRuntime;

pub struct TelemetryReporter {
    start: Instant,
    last_report: Instant,
    interval: Duration,
    app_name: &'static str,
}

impl TelemetryReporter {
    pub fn new(app_name: &'static str, interval_secs: u64) -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last_report: now,
            interval: Duration::from_secs(interval_secs),
            app_name,
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn should_report(&mut self) -> bool {
        if self.last_report.elapsed() >= self.interval {
            self.last_report = Instant::now();
            true
        } else {
            false
        }
    }

    pub fn report(&self, rt: &PtxRuntime, extra: &str) {
        let elapsed = self.start.elapsed();
        let hh = elapsed.as_secs() / 3600;
        let mm = (elapsed.as_secs() % 3600) / 60;
        let ss = elapsed.as_secs() % 60;

        let tlsf = rt.tlsf_stats();
        let health = rt.validate_pool();

        println!(
            "[{:02}:{:02}:{:02}] {} | pool={:.1}% | frag={:.6} | healthy={} | {}",
            hh,
            mm,
            ss,
            self.app_name,
            tlsf.utilization_percent,
            tlsf.fragmentation_ratio,
            if health.is_valid { "YES" } else { "NO" },
            extra,
        );
    }
}
