use libloading::Library;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::path::Path;

pub struct MuscleMemory {
    libraries: Mutex<HashMap<String, Arc<Library>>>,
}

impl MuscleMemory {
    pub fn new() -> Self {
        Self {
            libraries: Mutex::new(HashMap::new()),
        }
    }

    pub fn get(&self, path: &str) -> Arc<Library> {
        let mut libs = self.libraries.lock().unwrap();
        if let Some(lib) = libs.get(path) {
            return Arc::clone(lib);
        }

        println!("[MuscleMemory] Engaging Skill: {}", path);
        let lib_abs = Path::new(path).canonicalize().expect(&format!("Skill not found at: {}", path));
        let lib = unsafe { Library::new(lib_abs).expect("Failed to engage muscle skill") };
        let lib_arc = Arc::new(lib);
        libs.insert(path.to_string(), Arc::clone(&lib_arc));
        lib_arc
    }
}

impl std::fmt::Debug for MuscleMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MuscleMemory {{ active_skills: {} }}", self.libraries.lock().unwrap().len())
    }
}
