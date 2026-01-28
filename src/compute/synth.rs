use cudarc::driver::CudaDevice;
use cudarc::nvrtc::{CompileOptions, Ptx};
use std::sync::Arc;
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy)]
pub enum RuntimeKind {
    Nvrtc,
    Nvcc,
}

pub struct Synthesizer {
    device: Arc<CudaDevice>,
    known_funcs: Mutex<HashMap<String, &'static str>>,
    preferred_runtime: RuntimeKind,
}

impl std::fmt::Debug for Synthesizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Synthesizer {{ runtime: {:?}, known_skills: {} }}", self.preferred_runtime, self.known_funcs.lock().unwrap().len())
    }
}

impl Synthesizer {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            known_funcs: Mutex::new(HashMap::new()),
            preferred_runtime: RuntimeKind::Nvrtc,
        }
    }

    pub fn synthesize(&self, name: &str, logic: &str) -> String {
        let ptx_key = format!("fused_mod_{}", name);
        
        if self.device.get_func(&ptx_key, name).is_some() {
            return ptx_key;
        }

        let code = format!(r#"
            extern "C" __global__ void {}(float* data, int n) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) {{
                    float x = data[i];
                    {}
                    data[i] = x;
                }}
            }}
        "#, name, logic);

        let ptx = match self.preferred_runtime {
            RuntimeKind::Nvrtc => {
                let mut options = CompileOptions::default();
                options.arch = Some("compute_75");
                match cudarc::nvrtc::compile_ptx_with_opts(code.clone(), options) {
                    Ok(ptx) => ptx,
                    Err(e) => {
                        println!("[Synthesizer] NVRTC failed, falling back to NVCC: {:?}", e);
                        self.compile_with_nvcc(name, &code)
                    }
                }
            }
            RuntimeKind::Nvcc => self.compile_with_nvcc(name, &code),
        };

        let leaked_name: &'static str = Box::leak(name.to_string().into_boxed_str());
        self.known_funcs.lock().unwrap().insert(name.to_string(), leaked_name);

        match self.device.load_ptx(ptx, &ptx_key, &[leaked_name]) {
            Ok(_) => ptx_key,
            Err(e) => {
                println!("[Synthesizer] Load PTX failed ({:?}). Retrying with lower target...", e);
                // If 86/80 fails with PTX version error, try a very old target
                // This is the "No BS" fallback.
                let ptx = self.compile_with_nvcc_compat(name, &code);
                self.device.load_ptx(ptx, &ptx_key, &[leaked_name]).expect("Physical Body rejected all synthesized skills");
                ptx_key
            }
        }
    }

    fn compile_with_nvcc(&self, name: &str, code: &str) -> Ptx {
        self.nvcc_exec(name, code, "80")
    }

    fn compile_with_nvcc_compat(&self, name: &str, code: &str) -> Ptx {
        // Force an older PTX version that 12.2 driver (535) will accept
        self.nvcc_exec(name, code, "75")
    }

    fn nvcc_exec(&self, name: &str, code: &str, arch: &str) -> Ptx {
        use std::process::Command;
        use std::io::Write;
        
        let mut child = Command::new("nvcc")
            .args(&["-ptx", "-arch", &format!("compute_{}", arch), "-x", "cu", "-", "-o", "-"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("Failed to spawn NVCC; is it installed?");

        {
            let mut stdin = child.stdin.take().expect("Failed to open stdin");
            stdin.write_all(code.as_bytes()).expect("Failed to write to stdin");
        }

        let output = child.wait_with_output().expect("Failed to wait on NVCC");
        if !output.status.success() {
            panic!("NVCC error: {}", String::from_utf8_lossy(&output.stderr));
        }

        let mut ptx_code = String::from_utf8_lossy(&output.stdout).to_string();
        
        // ULTIMATE NO BS: If the driver is old, we forcefully downgrade the PTX header.
        // This works because the instructions we use (sin, exp, relu) haven't changed in years.
        if ptx_code.contains(".version 8.5") || ptx_code.contains(".version 8.4") || ptx_code.contains(".version 8.3") {
            println!("[Synthesizer] Force-downgrading PTX version from 8.x to 8.0 for driver compatibility.");
            ptx_code = ptx_code.replace(".version 8.5", ".version 8.0")
                               .replace(".version 8.4", ".version 8.0")
                               .replace(".version 8.3", ".version 8.0");
        }

        let temp_path = format!("/tmp/{}.ptx", name);
        std::fs::write(&temp_path, ptx_code).expect("Failed to write temp PTX");
        
        cudarc::nvrtc::Ptx::from_file(temp_path)
    }
}
