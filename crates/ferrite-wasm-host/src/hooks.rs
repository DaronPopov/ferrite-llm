use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use rhai::{AST, Engine, EvalAltResult, Scope};
use rhai::{Array, Dynamic, Map};

#[derive(Clone)]
pub struct ScriptHooks {
    path: PathBuf,
    source: Arc<String>,
    options: HookOptions,
    metrics: Arc<Mutex<BTreeMap<String, HookMetric>>>,
}

#[derive(Clone, Copy)]
struct HookOptions {
    on_error: HookErrorMode,
    timeout_ms: Option<u64>,
    emit_metrics: bool,
}

#[derive(Clone, Copy)]
enum HookErrorMode {
    Strict,
    Warn,
}

#[derive(Clone, Copy, Default)]
struct HookMetric {
    calls: u64,
    errors: u64,
    timeouts: u64,
    total_micros: u128,
}

impl ScriptHooks {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let source = std::fs::read_to_string(&path)
            .map_err(|e| format!("failed to read hook script {}: {e}", path.display()))?;
        let engine = Engine::new();
        engine
            .compile(&source)
            .map_err(|e| format!("failed to compile hook script {}: {e}", path.display()))?;
        Ok(Self {
            path,
            source: Arc::new(source),
            options: HookOptions::from_env(),
            metrics: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn apply_pre_prompt(&self, prompt: &str) -> Result<String, String> {
        self.apply_optional_text_hook("pre_prompt", prompt)
    }

    pub fn apply_post_response(&self, response: &str) -> Result<String, String> {
        self.apply_optional_text_hook("post_response", response)
    }

    pub fn apply_post_chunk(&self, chunk: &str) -> Result<String, String> {
        self.apply_optional_text_hook("post_chunk", chunk)
    }

    pub fn apply_post_logits(
        &self,
        candidates: &[ferrite_core::LogitsCandidate],
    ) -> Result<Option<Vec<ferrite_core::LogitsCandidate>>, String> {
        let original = candidates.to_vec();
        self.run_optional_hook("post_logits", Some(original.clone()), || {
            let (engine, ast) = self.compile()?;
            let mut scope = Scope::new();
            let input = candidates_to_array(candidates);
            match engine.call_fn::<Array>(&mut scope, &ast, "post_logits", (input,)) {
                Ok(output) => Ok(Some(array_to_candidates(&output)?)),
                Err(err) if is_missing_fn(&err) => Ok(None),
                Err(err) => Err(format!(
                    "hook script {} post_logits failed: {err}",
                    self.path.display()
                )),
            }
        })
    }

    pub fn gpu_logits_program(&self) -> Result<Option<String>, String> {
        self.apply_optional_zero_arg_string_hook("gpu_logits_program")
    }

    fn apply_optional_text_hook(&self, fn_name: &str, input: &str) -> Result<String, String> {
        self.run_optional_hook(fn_name, Some(input.to_string()), || {
            let (engine, ast) = self.compile()?;
            let mut scope = Scope::new();
            match engine.call_fn::<String>(&mut scope, &ast, fn_name, (input.to_string(),)) {
                Ok(output) => Ok(Some(output)),
                Err(err) if is_missing_fn(&err) => Ok(Some(input.to_string())),
                Err(err) => Err(format!(
                    "hook script {} {} failed: {err}",
                    self.path.display(),
                    fn_name
                )),
            }
        })
        .map(|value| value.unwrap_or_else(|| input.to_string()))
    }

    fn apply_optional_zero_arg_string_hook(&self, fn_name: &str) -> Result<Option<String>, String> {
        self.run_optional_hook(fn_name, None, || {
            let (engine, ast) = self.compile()?;
            let mut scope = Scope::new();
            match engine.call_fn::<String>(&mut scope, &ast, fn_name, ()) {
                Ok(output) => Ok(Some(output)),
                Err(err) if is_missing_fn(&err) => Ok(None),
                Err(err) => Err(format!(
                    "hook script {} {} failed: {err}",
                    self.path.display(),
                    fn_name
                )),
            }
        })
    }

    fn compile(&self) -> Result<(Engine, AST), String> {
        let engine = Engine::new();
        let ast = engine
            .compile(self.source.as_str())
            .map_err(|e| format!("failed to compile hook script {}: {e}", self.path.display()))?;
        Ok((engine, ast))
    }

    fn run_optional_hook<T: Clone>(
        &self,
        hook_name: &str,
        fallback: Option<T>,
        run: impl FnOnce() -> Result<Option<T>, String>,
    ) -> Result<Option<T>, String> {
        let start = Instant::now();
        let result = run();
        let elapsed = start.elapsed();
        let timed_out = self
            .options
            .timeout_ms
            .is_some_and(|limit| elapsed.as_millis() > u128::from(limit));

        self.record_metric(hook_name, elapsed, result.is_err(), timed_out);

        if timed_out {
            let message = format!(
                "hook script {} {} exceeded timeout budget ({} ms > {} ms)",
                self.path.display(),
                hook_name,
                elapsed.as_millis(),
                self.options.timeout_ms.unwrap_or_default()
            );
            return self.handle_hook_error(hook_name, fallback, message);
        }

        match result {
            Ok(value) => {
                self.maybe_log_metric(hook_name, elapsed, "ok");
                Ok(value)
            }
            Err(err) => self.handle_hook_error(hook_name, fallback, err),
        }
    }

    fn handle_hook_error<T: Clone>(
        &self,
        hook_name: &str,
        fallback: Option<T>,
        message: String,
    ) -> Result<Option<T>, String> {
        match self.options.on_error {
            HookErrorMode::Strict => Err(message),
            HookErrorMode::Warn => {
                tracing::warn!("{message}");
                self.maybe_log_metric(hook_name, std::time::Duration::default(), "fallback");
                Ok(fallback)
            }
        }
    }

    fn record_metric(
        &self,
        hook_name: &str,
        elapsed: std::time::Duration,
        errored: bool,
        timed_out: bool,
    ) {
        let Ok(mut metrics) = self.metrics.lock() else {
            return;
        };
        let metric = metrics.entry(hook_name.to_string()).or_default();
        metric.calls += 1;
        metric.total_micros += elapsed.as_micros();
        if errored {
            metric.errors += 1;
        }
        if timed_out {
            metric.timeouts += 1;
        }
    }

    fn maybe_log_metric(&self, hook_name: &str, elapsed: std::time::Duration, outcome: &str) {
        if !self.options.emit_metrics {
            return;
        }

        let snapshot = self
            .metrics
            .lock()
            .ok()
            .and_then(|metrics| metrics.get(hook_name).copied());
        if let Some(metric) = snapshot {
            tracing::info!(
                target: "ferrite::hooks",
                hook = hook_name,
                outcome,
                elapsed_ms = elapsed.as_secs_f64() * 1000.0,
                calls = metric.calls,
                errors = metric.errors,
                timeouts = metric.timeouts,
                avg_ms = (metric.total_micros as f64 / metric.calls.max(1) as f64) / 1000.0,
                "script hook metrics"
            );
        }
    }
}

impl HookOptions {
    fn from_env() -> Self {
        let on_error = match std::env::var("FERRITE_SCRIPT_HOOK_ON_ERROR")
            .ok()
            .as_deref()
        {
            Some("warn") | Some("fallback") => HookErrorMode::Warn,
            _ => HookErrorMode::Strict,
        };
        let timeout_ms = std::env::var("FERRITE_SCRIPT_HOOK_TIMEOUT_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0);
        let emit_metrics = matches!(
            std::env::var("FERRITE_SCRIPT_HOOK_METRICS").ok().as_deref(),
            Some("1" | "true" | "TRUE" | "True")
        );

        Self {
            on_error,
            timeout_ms,
            emit_metrics,
        }
    }
}

fn is_missing_fn(err: &EvalAltResult) -> bool {
    matches!(err, EvalAltResult::ErrorFunctionNotFound(_, _))
}

fn candidates_to_array(candidates: &[ferrite_core::LogitsCandidate]) -> Array {
    candidates
        .iter()
        .map(|candidate| {
            let mut map = Map::new();
            map.insert("token_id".into(), Dynamic::from_int(candidate.token_id as i64));
            map.insert("logit".into(), Dynamic::from_float(candidate.logit as rhai::FLOAT));
            Dynamic::from_map(map)
        })
        .collect()
}

fn array_to_candidates(array: &Array) -> Result<Vec<ferrite_core::LogitsCandidate>, String> {
    array
        .iter()
        .map(|item| {
            let map = item
                .clone()
                .try_cast::<Map>()
                .ok_or_else(|| "post_logits must return an array of maps".to_string())?;
            let token_id = map
                .get("token_id")
                .and_then(|value| value.clone().try_cast::<i64>())
                .ok_or_else(|| "post_logits candidate missing integer token_id".to_string())?;
            let logit = map
                .get("logit")
                .and_then(|value| value.clone().try_cast::<rhai::FLOAT>())
                .ok_or_else(|| "post_logits candidate missing float logit".to_string())?;
            Ok(ferrite_core::LogitsCandidate {
                token_id: token_id as u32,
                logit: logit as f32,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{HookErrorMode, HookOptions, ScriptHooks};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn pre_prompt_transforms_prompt() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ferrite-hook-test-{nonce}"));
        std::fs::create_dir_all(&dir).unwrap();
        let script_path = dir.join("hook.rhai");
        std::fs::write(
            &script_path,
            r#"
                fn pre_prompt(prompt) {
                    prompt + " ::hooked"
                }
            "#,
        )
        .unwrap();

        let hooks = ScriptHooks::from_file(&script_path).unwrap();
        let transformed = hooks.apply_pre_prompt("hello").unwrap();
        assert_eq!(transformed, "hello ::hooked");

        let _ = std::fs::remove_file(&script_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn missing_hook_function_is_noop() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ferrite-hook-test-noop-{nonce}"));
        std::fs::create_dir_all(&dir).unwrap();
        let script_path = dir.join("hook.rhai");
        std::fs::write(
            &script_path,
            r#"
                fn pre_prompt(prompt) {
                    prompt + "!"
                }
            "#,
        )
        .unwrap();

        let hooks = ScriptHooks::from_file(&script_path).unwrap();
        let transformed = hooks.apply_post_response("done").unwrap();
        assert_eq!(transformed, "done");

        let _ = std::fs::remove_file(&script_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn post_hooks_transform_output() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ferrite-hook-test-post-{nonce}"));
        std::fs::create_dir_all(&dir).unwrap();
        let script_path = dir.join("hook.rhai");
        std::fs::write(
            &script_path,
            r#"
                fn post_response(response) {
                    response + " ::response"
                }

                fn post_chunk(chunk) {
                    "[" + chunk + "]"
                }
            "#,
        )
        .unwrap();

        let hooks = ScriptHooks::from_file(&script_path).unwrap();
        assert_eq!(
            hooks.apply_post_response("done").unwrap(),
            "done ::response"
        );
        assert_eq!(hooks.apply_post_chunk("tok").unwrap(), "[tok]");

        let _ = std::fs::remove_file(&script_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn post_logits_rewrites_candidates() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ferrite-hook-test-logits-{nonce}"));
        std::fs::create_dir_all(&dir).unwrap();
        let script_path = dir.join("hook.rhai");
        std::fs::write(
            &script_path,
            r#"
                fn post_logits(candidates) {
                    let out = candidates;
                    out[0]["logit"] = 42.0;
                    out
                }
            "#,
        )
        .unwrap();

        let hooks = ScriptHooks::from_file(&script_path).unwrap();
        let output = hooks
            .apply_post_logits(&[ferrite_core::LogitsCandidate {
                token_id: 7,
                logit: 1.5,
            }])
            .unwrap()
            .unwrap();
        assert_eq!(
            output,
            vec![ferrite_core::LogitsCandidate {
                token_id: 7,
                logit: 42.0
            }]
        );

        let _ = std::fs::remove_file(&script_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn gpu_logits_program_reads_optional_program() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ferrite-hook-test-gpu-program-{nonce}"));
        std::fs::create_dir_all(&dir).unwrap();
        let script_path = dir.join("hook.rhai");
        std::fs::write(
            &script_path,
            r#"
                fn gpu_logits_program() {
                    "x = input([1, 1, 1, 16])\nreturn x"
                }
            "#,
        )
        .unwrap();

        let hooks = ScriptHooks::from_file(&script_path).unwrap();
        let program = hooks.gpu_logits_program().unwrap().unwrap();
        assert!(program.contains("input([1, 1, 1, 16])"));

        let _ = std::fs::remove_file(&script_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn hook_options_default_to_strict_without_metrics() {
        std::env::remove_var("FERRITE_SCRIPT_HOOK_ON_ERROR");
        std::env::remove_var("FERRITE_SCRIPT_HOOK_TIMEOUT_MS");
        std::env::remove_var("FERRITE_SCRIPT_HOOK_METRICS");

        let options = HookOptions::from_env();
        assert!(matches!(options.on_error, HookErrorMode::Strict));
        assert_eq!(options.timeout_ms, None);
        assert!(!options.emit_metrics);
    }

    #[test]
    fn hook_options_parse_warn_timeout_and_metrics() {
        std::env::set_var("FERRITE_SCRIPT_HOOK_ON_ERROR", "warn");
        std::env::set_var("FERRITE_SCRIPT_HOOK_TIMEOUT_MS", "25");
        std::env::set_var("FERRITE_SCRIPT_HOOK_METRICS", "1");

        let options = HookOptions::from_env();
        assert!(matches!(options.on_error, HookErrorMode::Warn));
        assert_eq!(options.timeout_ms, Some(25));
        assert!(options.emit_metrics);

        std::env::remove_var("FERRITE_SCRIPT_HOOK_ON_ERROR");
        std::env::remove_var("FERRITE_SCRIPT_HOOK_TIMEOUT_MS");
        std::env::remove_var("FERRITE_SCRIPT_HOOK_METRICS");
    }
}
