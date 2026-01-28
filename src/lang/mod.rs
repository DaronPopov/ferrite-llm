pub mod interpreter;
use crate::data::Grid;
use crate::compute::{compute, Op};
use crate::dynamics::RuntimeRules;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use threadpool::ThreadPool;

#[derive(Debug, Clone)]
pub enum Step {
    Op(Op, Vec<String>, String),
    MoveToDevice(String, String),
    MoveToHost(String, String),
}

pub struct Program {
    steps: Vec<Step>,
    registry: Arc<Mutex<HashMap<String, Grid>>>,
}

impl Program {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            registry: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn bind(&mut self, name: &str, grid: Grid) {
        self.registry.lock().unwrap().insert(name.to_string(), grid);
    }

    pub fn add_step(&mut self, step: Step) {
        self.steps.push(step);
    }

    /// Execute steps in parallel groups where data-flow allows
    pub fn execute(&mut self, rules: &RuntimeRules) {
        let pool = ThreadPool::new(4); // 4 Parallel Workers
        
        let mut executed_names = HashSet::new();
        {
            let reg = self.registry.lock().unwrap();
            for name in reg.keys() {
                executed_names.insert(name.clone());
            }
        }

        let mut remaining_steps = self.steps.clone();
        
        while !remaining_steps.is_empty() {
             // 1. Find independent steps (those whose inputs are ready)
             let mut to_launch = Vec::new();
             let mut still_waiting = Vec::new();

             for step in remaining_steps {
                 let inputs = match &step {
                     Step::Op(_, ins, _) => ins.clone(),
                     Step::MoveToDevice(i, _) => vec![i.clone()],
                     Step::MoveToHost(i, _) => vec![i.clone()],
                 };

                 if inputs.iter().all(|i| executed_names.contains(i)) {
                     to_launch.push(step);
                 } else {
                     still_waiting.push(step);
                 }
             }

             if to_launch.is_empty() && !still_waiting.is_empty() {
                 panic!("Deadlock detected in Program dependencies!");
             }

             // 2. Launch independent steps in parallel
             println!("[Brain] Launching {} steps in parallel...", to_launch.len());
             
             let results = Arc::new(Mutex::new(Vec::new()));
             for step in to_launch {
                 let registry = Arc::clone(&self.registry);
                 let rules = rules.clone();
                 let results = Arc::clone(&results);
                 
                 pool.execute(move || {
                     let res_data = match &step {
                         Step::Op(op, input_names, output_name) => {
                             let reg = registry.lock().unwrap();
                             let inputs: Vec<&Grid> = input_names.iter().map(|n| reg.get(n).unwrap()).collect();
                             let res = compute(op.clone(), inputs, None, &rules);
                             (output_name.clone(), res)
                         }
                         Step::MoveToDevice(input_name, output_name) => {
                             let reg = registry.lock().unwrap();
                             let input = reg.get(input_name).unwrap();
                             let res = compute(Op::MoveToDevice, vec![input], None, &rules);
                             (output_name.clone(), res)
                         }
                         Step::MoveToHost(input_name, output_name) => {
                             let reg = registry.lock().unwrap();
                             let input = reg.get(input_name).unwrap();
                             let res = compute(Op::MoveToHost, vec![input], None, &rules);
                             (output_name.clone(), res)
                         }
                     };
                     results.lock().unwrap().push(res_data);
                 });
             }

             pool.join();

             // 3. Update Registry and Executed set
             let mut group_results = results.lock().unwrap();
             let mut reg = self.registry.lock().unwrap();
             for (name, grid) in group_results.drain(..) {
                 executed_names.insert(name.clone());
                 reg.insert(name, grid);
             }

             remaining_steps = still_waiting;
        }
    }

    pub fn get(&self, name: &str) -> Option<Grid> {
        self.registry.lock().unwrap().get(name).cloned()
    }
}

pub use crate::compute::Op as SemanticOp;
